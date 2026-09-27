#!/usr/bin/env python3
"""
Read crawled markdown files, chunk text, and sync them to a Pinecone integrated inference index.

Only chunks whose text is new or changed are embedded. Unchanged chunks are skipped, and
leftover chunks of pages that got shorter are deleted. A local manifest caches the text hash
of every record in the namespace so unchanged chunks never have to be fetched or re-embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pinecone import Pinecone

MAX_UPSERT_BATCH_SIZE = 96
FETCH_BATCH_SIZE = 100
DELETE_BATCH_SIZE = 1000


def parse_source_url_and_body(markdown_text: str) -> tuple[str | None, str]:
    lines = markdown_text.splitlines()
    if lines and lines[0].startswith("Source URL: "):
        source_url = lines[0].replace("Source URL: ", "", 1).strip() or None
        body = "\n".join(lines[2:]) if len(lines) > 2 else ""
        return source_url, body
    return None, markdown_text


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    tokens = re.findall(r"\S+\s*", text)
    if not tokens:
        return []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    i = 0
    while i < len(tokens):
        chunk = "".join(tokens[i : i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        next_i = i + step
        if next_i >= len(tokens):
            break

        if next_i > 0:
            while next_i < len(tokens):
                prev = tokens[next_i - 1].rstrip()
                if re.search(r"[.!?][\"')\]]*$", prev):
                    break
                next_i += 1
        i = next_i
    return chunks


def url_path_prefixes(source_url: str | None) -> tuple[str | None, list[str]]:
    if not source_url:
        return None, []
    parsed = urlparse(source_url)
    path = parsed.path or "/"
    path = path if path.startswith("/") else f"/{path}"
    if path == "/":
        return path, ["/"]

    parts = [p for p in path.split("/") if p]
    prefixes: list[str] = []
    current = ""
    for part in parts:
        current = f"{current}/{part}"
        prefixes.append(current)
    return path, prefixes


def build_record_id(source_url: str | None, file_path: str, chunk_index: int) -> str:
    base = source_url or file_path
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{chunk_index}"


def iter_markdown_files(input_dir: Path) -> list[Path]:
    files = []
    for path in input_dir.rglob("*.md"):
        if path.name.startswith("."):
            continue
        files.append(path)
    return sorted(files)


def batched(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _extract_ids(items: object) -> list[str]:
    if not isinstance(items, list):
        return []

    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.append(item["id"])
        else:
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str):
                ids.append(item_id)
    return ids


def list_existing_record_ids(index: object, namespace: str) -> set[str]:
    existing_ids: set[str] = set()
    pagination_token: str | None = None

    while True:
        page = index.list_paginated(namespace=namespace, limit=100, pagination_token=pagination_token)

        page_ids = _extract_ids(getattr(page, "vectors", None))
        if not page_ids:
            page_ids = _extract_ids(getattr(page, "records", None))
        if not page_ids:
            page_ids = _extract_ids(getattr(page, "ids", None))
        if not page_ids and isinstance(page, dict):
            page_ids = _extract_ids(page.get("vectors"))
        if not page_ids and isinstance(page, dict):
            page_ids = _extract_ids(page.get("records"))
        if not page_ids and isinstance(page, dict):
            page_ids = _extract_ids(page.get("ids"))

        existing_ids.update(page_ids)

        pagination = getattr(page, "pagination", None)
        if pagination is None and isinstance(page, dict):
            pagination = page.get("pagination")

        next_token = getattr(pagination, "next", None)
        if next_token is None and isinstance(pagination, dict):
            next_token = pagination.get("next")
        if not isinstance(next_token, str) or not next_token:
            break
        pagination_token = next_token

    return existing_ids


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def id_prefix(record_id: str) -> str:
    return record_id.split(":", 1)[0]


def as_int(value: object) -> int | None:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_manifest(path: Path, index_name: str, namespace: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("index_name") != index_name or data.get("namespace") != namespace:
        print(f"Manifest {path} is for a different index/namespace; ignoring it.")
        return {}
    return dict(data.get("records", {}))


def save_manifest(path: Path, index_name: str, namespace: str, records: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"index_name": index_name, "namespace": namespace, "records": records}),
        encoding="utf-8",
    )
    tmp.replace(path)


def build_records(input_dir: Path, chunk_size: int, chunk_overlap: int) -> tuple[list[dict], int]:
    records: list[dict] = []
    markdown_files = iter_markdown_files(input_dir)
    for path in markdown_files:
        raw = path.read_text(encoding="utf-8")
        source_url, body = parse_source_url_and_body(raw)
        chunks = chunk_text(body, chunk_size, chunk_overlap)
        url_path, path_prefixes = url_path_prefixes(source_url)
        rel_path = str(path.relative_to(input_dir))
        date_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": build_record_id(source_url, rel_path, chunk_index),
                    "text": chunk,
                    "source_url": source_url,
                    "url_path": url_path,
                    "path_prefixes": path_prefixes,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "file_path": rel_path,
                    "date_modified": date_modified,
                }
            )
    return records, len(markdown_files)


def with_retries(action, description: str, throttle_seconds: float, max_retries: int):
    attempt = 0
    while True:
        try:
            return action()
        except Exception as exc:
            message = str(exc)
            if "for the current month" in message:
                # Monthly quota exhaustion won't clear by waiting; stop with progress saved.
                raise RuntimeError(f"Pinecone monthly quota exhausted during {description}: {message}") from exc
            is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
            if not is_rate_limited or attempt >= max_retries:
                raise
            wait_seconds = max(1.0, throttle_seconds) * (2**attempt)
            print(
                f"Rate limited on {description}; "
                f"retrying in {wait_seconds:.1f}s (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_seconds)
            attempt += 1


def fetch_remote_state(index: object, namespace: str, ids: list[str]) -> dict[str, dict]:
    """Fetch records (no embedding cost) to learn the text hash of records missing from the manifest."""
    state: dict[str, dict] = {}
    for batch_index in range(0, len(ids), FETCH_BATCH_SIZE):
        batch = ids[batch_index : batch_index + FETCH_BATCH_SIZE]
        response = index.fetch(ids=batch, namespace=namespace)
        for record_id, vector in response.vectors.items():
            metadata = vector.metadata or {}
            state[record_id] = {
                "h": text_hash(str(metadata.get("text", ""))),
                "n": as_int(metadata.get("chunk_count")),
            }
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files and sync new/changed chunks to Pinecone."
    )
    parser.add_argument("--input-dir", default="scrape/site_markdown")
    parser.add_argument("--index-name", default="actualism")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--throttle-seconds", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument(
        "--manifest",
        default=".pinecone_manifest.json",
        help="Local cache of text hashes for records already in Pinecone.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Also delete records for pages that are no longer in --input-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be upserted/updated/deleted without writing to Pinecone.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    effective_batch_size = args.batch_size
    if effective_batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if effective_batch_size > MAX_UPSERT_BATCH_SIZE:
        print(
            f"Requested batch size {effective_batch_size} exceeds Pinecone limit "
            f"({MAX_UPSERT_BATCH_SIZE}); using {MAX_UPSERT_BATCH_SIZE}."
        )
        effective_batch_size = MAX_UPSERT_BATCH_SIZE

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index_name)
    manifest_path = Path(args.manifest)

    local_records, file_count = build_records(input_dir, args.chunk_size, args.chunk_overlap)
    local_by_id = {record["id"]: record for record in local_records}
    local_prefixes = {id_prefix(record_id) for record_id in local_by_id}

    remote_ids = list_existing_record_ids(index, args.namespace)
    manifest = load_manifest(manifest_path, args.index_name, args.namespace)
    manifest = {record_id: entry for record_id, entry in manifest.items() if record_id in remote_ids}
    unknown_ids = sorted(remote_ids - manifest.keys())
    if unknown_ids:
        print(f"Fetching {len(unknown_ids)} records not in the manifest to compare text hashes...")
        manifest.update(fetch_remote_state(index, args.namespace, unknown_ids))
        if not args.dry_run:
            save_manifest(manifest_path, args.index_name, args.namespace, manifest)

    to_upsert: list[dict] = []
    to_update_count: list[dict] = []
    for record_id, record in local_by_id.items():
        remote = manifest.get(record_id)
        if remote is None or remote.get("h") != text_hash(record["text"]):
            to_upsert.append(record)
        elif remote.get("n") != record["chunk_count"]:
            to_update_count.append(record)

    stale_ids = sorted(remote_ids - local_by_id.keys())
    orphan_ids = [record_id for record_id in stale_ids if id_prefix(record_id) not in local_prefixes]
    to_delete = stale_ids if args.prune else [i for i in stale_ids if id_prefix(i) in local_prefixes]

    print(
        f"Local: {len(local_by_id)} chunks from {file_count} files. Remote: {len(remote_ids)} records.\n"
        f"  upsert (new/changed, will embed): {len(to_upsert)}\n"
        f"  unchanged: {len(local_by_id) - len(to_upsert)}"
        f" (of which {len(to_update_count)} need a chunk_count metadata update)\n"
        f"  delete (leftover chunks of shortened pages): {len(stale_ids) - len(orphan_ids)}\n"
        f"  records for pages no longer crawled: {len(orphan_ids)}"
        f" ({'deleting, --prune' if args.prune else 'keeping; pass --prune to delete'})"
    )
    if args.dry_run:
        return

    batches = batched(to_upsert, effective_batch_size)
    total_batches = len(batches)
    for batch_index, batch in enumerate(batches, start=1):
        with_retries(
            lambda: index.upsert_records(namespace=args.namespace, records=batch),
            f"upsert batch {batch_index}/{total_batches}",
            args.throttle_seconds,
            args.max_retries,
        )
        for record in batch:
            manifest[record["id"]] = {"h": text_hash(record["text"]), "n": record["chunk_count"]}
        save_manifest(manifest_path, args.index_name, args.namespace, manifest)
        if batch_index % 50 == 0 or batch_index == total_batches:
            print(f"Upserted batch {batch_index}/{total_batches}")

        if args.throttle_seconds > 0 and batch_index < total_batches:
            time.sleep(args.throttle_seconds)

    for record in to_update_count:
        with_retries(
            lambda: index.update(
                id=record["id"],
                set_metadata={"chunk_count": record["chunk_count"]},
                namespace=args.namespace,
            ),
            f"metadata update {record['id']}",
            args.throttle_seconds,
            args.max_retries,
        )
        manifest[record["id"]]["n"] = record["chunk_count"]
    if to_update_count:
        save_manifest(manifest_path, args.index_name, args.namespace, manifest)

    for batch_index in range(0, len(to_delete), DELETE_BATCH_SIZE):
        batch_ids = to_delete[batch_index : batch_index + DELETE_BATCH_SIZE]
        index.delete(ids=batch_ids, namespace=args.namespace)
        for record_id in batch_ids:
            manifest.pop(record_id, None)
        save_manifest(manifest_path, args.index_name, args.namespace, manifest)

    print(
        f"Upserted {len(to_upsert)} chunks, updated {len(to_update_count)}, deleted {len(to_delete)} "
        f"in index='{args.index_name}' namespace='{args.namespace}'"
    )


if __name__ == "__main__":
    main()
