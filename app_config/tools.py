from __future__ import annotations

from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": "Search chunk-level quotations relevant to a user's question. Use a natural-language query, not keyword stuffing. `path_prefix` must be a site path like `/richard/articles`, not a full URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language retrieval query. Prefer a clear question like `What does the site say about naivete and actual freedom?` rather than a keyword list.",
                    },
                    "path_prefix": {
                        "type": "string",
                        "description": "Optional site path filter beginning with `/`, for example `/richard`, `/richard/articles`, or `/library/glossary`.",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quote_selection",
            "description": "Select the final quotations to show the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quotes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url_path": {"type": "string"},
                                "chunk_index": {"type": "integer"},
                                "line_ranges": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                            },
                            "required": ["url_path", "chunk_index", "line_ranges"],
                            "additionalProperties": False,
                        },
                    },
                    "failure_reason": {"type": "string"},
                },
                "required": ["quotes"],
                "additionalProperties": False,
            },
        },
    },
]
