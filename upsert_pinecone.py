#!/usr/bin/env python3
"""
Read crawled markdown files, chunk text, and upsert to Pinecone integrated inference index.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pinecone import Pinecone

MAX_UPSERT_BATCH_SIZE = 96


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files and upsert to Pinecone."
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
        "--merge-type",
        choices=("incremental", "overwrite"),
        default="incremental",
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

    all_records: list[dict] = []
    markdown_files = iter_markdown_files(input_dir)
    for path in markdown_files:
        raw = path.read_text(encoding="utf-8")
        source_url, body = parse_source_url_and_body(raw)
        chunks = chunk_text(body, args.chunk_size, args.chunk_overlap)
        url_path, path_prefixes = url_path_prefixes(source_url)
        rel_path = str(path.relative_to(input_dir))
        date_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

        for chunk_index, chunk in enumerate(chunks):
            record_id = build_record_id(source_url, rel_path, chunk_index)
            all_records.append(
                {
                    "id": record_id,
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

    if args.merge_type == "incremental":
        existing_ids = list_existing_record_ids(index, args.namespace)
        all_records = [record for record in all_records if record["id"] not in existing_ids]
        print(
            f"Skipping {len(existing_ids)} existing records; "
            f"{len(all_records)} records remain to upsert."
        )

    batches = batched(all_records, effective_batch_size)
    total_batches = len(batches)
    for batch_index, batch in enumerate(batches, start=1):
        attempt = 0
        while True:
            try:
                index.upsert_records(namespace=args.namespace, records=batch)
                break
            except Exception as exc:
                message = str(exc)
                is_rate_limited = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not is_rate_limited or attempt >= args.max_retries:
                    raise
                wait_seconds = max(1.0, args.throttle_seconds) * (2**attempt)
                print(
                    f"Rate limited on batch {batch_index}/{total_batches}; "
                    f"retrying in {wait_seconds:.1f}s (attempt {attempt + 1}/{args.max_retries})"
                )
                time.sleep(wait_seconds)
                attempt += 1

        if args.throttle_seconds > 0 and batch_index < total_batches:
            time.sleep(args.throttle_seconds)

    print(
        f"Upserted {len(all_records)} chunks from {len(markdown_files)} files "
        f"to index='{args.index_name}' namespace='{args.namespace}'"
    )


if __name__ == "__main__":
    main()
