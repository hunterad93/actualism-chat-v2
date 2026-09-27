#!/usr/bin/env python3
"""
Back up a Pinecone namespace (ids, vectors, metadata) to local files, or restore it from them.

A restore upserts the stored vectors directly, so it costs write units only and never
re-embeds text. Record ids are unchanged, which keeps the actualism MCP's
get_chunk_context (sha1(source_url)[:16]:chunk_index) working.

    python backup_pinecone.py backup  --out backups/pinecone-actualism-YYYY-MM-DD
    python backup_pinecone.py restore --src backups/pinecone-actualism-YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

from upsert_pinecone import FETCH_BATCH_SIZE, list_existing_record_ids

RESTORE_BATCH_SIZE = 50  # ~15KB per record as JSON; stays well under Pinecone's 2MB request limit.


def backup(index: object, namespace: str, out: Path) -> None:
    ids = sorted(list_existing_record_ids(index, namespace))
    print(f"Backing up {len(ids)} records from namespace '{namespace}'...")

    vectors: list[np.ndarray] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out.with_suffix(".jsonl.gz"), "wt", encoding="utf-8") as meta_file:
        for batch_index in range(0, len(ids), FETCH_BATCH_SIZE):
            batch = ids[batch_index : batch_index + FETCH_BATCH_SIZE]
            fetched = index.fetch(ids=batch, namespace=namespace).vectors
            missing = [record_id for record_id in batch if record_id not in fetched]
            if missing:
                raise RuntimeError(f"Fetch returned no data for {len(missing)} ids, e.g. {missing[:3]}")
            for record_id in batch:
                record = fetched[record_id]
                vectors.append(np.asarray(record.values, dtype=np.float32))
                meta_file.write(json.dumps({"id": record_id, "metadata": dict(record.metadata or {})}) + "\n")
            if (batch_index // FETCH_BATCH_SIZE) % 50 == 0:
                print(f"  fetched {batch_index + len(batch)}/{len(ids)}")

    np.savez_compressed(out.with_suffix(".npz"), ids=np.array(ids), vectors=np.stack(vectors))
    print(f"Wrote {out.with_suffix('.npz')} and {out.with_suffix('.jsonl.gz')}")


def load_backup(src: Path) -> list[dict]:
    data = np.load(src.with_suffix(".npz"))
    with gzip.open(src.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as meta_file:
        metadata = [json.loads(line) for line in meta_file]
    ids = [str(record_id) for record_id in data["ids"]]
    if ids != [row["id"] for row in metadata]:
        raise RuntimeError("Backup vector ids and metadata ids don't line up")
    return [
        {"id": record_id, "values": vector.tolist(), "metadata": row["metadata"]}
        for record_id, vector, row in zip(ids, data["vectors"], metadata)
    ]


def restore(index: object, namespace: str, src: Path, id_prefix: str, limit: int) -> None:
    records = load_backup(src)
    if limit > 0:
        records = records[:limit]
    for record in records:
        record["id"] = f"{id_prefix}{record['id']}"
    print(f"Restoring {len(records)} records into namespace '{namespace}'...")
    for batch_index in range(0, len(records), RESTORE_BATCH_SIZE):
        index.upsert(vectors=records[batch_index : batch_index + RESTORE_BATCH_SIZE], namespace=namespace)
    print("Restore complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back up or restore a Pinecone namespace locally.")
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("--index-name", default="actualism")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--out", help="backup: output path without extension")
    parser.add_argument("--src", help="restore: backup path without extension")
    parser.add_argument("--id-prefix", default="", help="restore: prefix added to every id (for testing)")
    parser.add_argument("--limit", type=int, default=0, help="restore: only the first N records (0 = all)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")
    index = Pinecone(api_key=api_key).Index(args.index_name)

    if args.action == "backup":
        if not args.out:
            parser.error("backup requires --out")
        backup(index, args.namespace, Path(args.out))
    else:
        if not args.src:
            parser.error("restore requires --src")
        restore(index, args.namespace, Path(args.src), args.id_prefix, args.limit)


if __name__ == "__main__":
    main()
