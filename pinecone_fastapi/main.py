#!/usr/bin/env python3
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "actualism")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_MAX_RETRIES = int(os.getenv("PINECONE_MAX_RETRIES", "4"))
PINECONE_BACKOFF_SECONDS = float(os.getenv("PINECONE_BACKOFF_SECONDS", "1.0"))

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment/.env")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

app = FastAPI(title="Actualism Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchChunksRequest(BaseModel):
    path_prefix: str | None = Field(
        default=None,
        description="Optional URL path prefix (e.g. /sundry/frequentquestions)",
    )
    query: str = Field(description="Natural-language question")
    top_k: int = Field(default=20, ge=1, le=200)
    namespace: str | None = None


def _extract_matches(search_result: object) -> list[dict[str, object]]:
    raw = search_result.to_dict()
    result = raw.get("result", {})
    hits = result.get("hits", [])
    return hits if isinstance(hits, list) else []


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _merge_with_overlap(chunks: list[str]) -> str:
    if not chunks:
        return ""
    merged = chunks[0]
    for next_chunk in chunks[1:]:
        overlap = min(len(merged), len(next_chunk), 800)
        best = 0
        for size in range(overlap, 0, -1):
            if merged.endswith(next_chunk[:size]):
                best = size
                break
        merged += next_chunk[best:]
    return merged


def _numbered_lines(text: str) -> list[dict[str, object]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts: list[str] = []
    current: list[str] = []

    for index, char in enumerate(normalized):
        current.append(char)
        next_char = normalized[index + 1] if index + 1 < len(normalized) else ""
        if char == "\n":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char in ".!?" and (not next_char or next_char.isspace()):
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []

    trailing = "".join(current).strip()
    if trailing:
        parts.append(trailing)

    lines: list[dict[str, object]] = []
    for part in parts:
        lines.append({"line_number": len(lines) + 1, "text": part})
    return lines


def _search_with_retries(*, namespace: str, query: dict[str, object], fields: list[str]) -> object:
    attempt = 0
    while True:
        try:
            return index.search(namespace=namespace, query=query, fields=fields)
        except Exception as exc:
            message = str(exc)
            is_retryable = (
                "429" in message
                or "RESOURCE_EXHAUSTED" in message
                or "503" in message
                or "500" in message
                or "timed out" in message.lower()
            )
            if not is_retryable or attempt >= PINECONE_MAX_RETRIES:
                raise HTTPException(status_code=503, detail="Pinecone search failed") from exc
            time.sleep(PINECONE_BACKOFF_SECONDS * (2**attempt))
            attempt += 1


@app.post("/search-chunks")
def search_chunks(request: SearchChunksRequest) -> dict[str, object]:
    namespace = request.namespace or PINECONE_NAMESPACE

    query_obj: dict[str, object] = {
        "inputs": {"text": request.query},
        "top_k": request.top_k,
    }
    if request.path_prefix:
        query_obj["filter"] = {"path_prefixes": {"$in": [request.path_prefix]}}

    result = _search_with_retries(
        namespace=namespace,
        query=query_obj,
        fields=["text", "url_path", "chunk_index", "chunk_count"],
    )

    matches = _extract_matches(result)
    output_matches: list[dict[str, object]] = []
    for match in matches:
        fields = match.get("fields", {})
        text = fields.get("text")
        numbered_lines = _numbered_lines(text) if isinstance(text, str) else []
        output_matches.append(
            {
                "url_path": fields.get("url_path"),
                "chunk_index": _as_int(fields.get("chunk_index")),
                "chunk_count": _as_int(fields.get("chunk_count")),
                "lines": numbered_lines,
            }
        )

    return {"matches": output_matches}
