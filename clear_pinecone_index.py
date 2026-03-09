#!/usr/bin/env python3
"""
Delete all records from a Pinecone index namespace.
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from pinecone import Pinecone


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear a Pinecone namespace.")
    parser.add_argument("--index-name", default="actualism")
    parser.add_argument("--namespace", default="default")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index_name)
    index.delete(delete_all=True, namespace=args.namespace)

    print(
        f"Cleared all vectors in index='{args.index_name}' namespace='{args.namespace}'"
    )


if __name__ == "__main__":
    main()
