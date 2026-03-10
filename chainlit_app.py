#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import re
from secrets import compare_digest
from typing import Any
from urllib.parse import urljoin, urlparse

import chainlit as cl
import requests
from chainlit.input_widget import Select
from openai import OpenAI

from app_config.config import (
    API_BASE_URL,
    CHAINLIT_AUTH_PASSWORD,
    CHAINLIT_AUTH_USERNAME,
    MAX_CHAT_TURNS,
    MODEL_CONFIG,
    MAX_TOOL_LINE_CHARS,
    MAX_TOOL_LINES_PER_MATCH,
    MAX_TOOL_MATCHES_IN_CONTEXT,
    MAX_TOOL_ROUNDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    SITE_BASE_URL,
)
from app_config.prompts import STANDARD_SYSTEM_PROMPT
from app_config.tools import TOOLS

SUPPORTED_MODELS = tuple(MODEL_CONFIG.keys())
MODEL_SETTING_ID = "model_name"
QUOTE_SELECTION_ONLY_TOOLS = [
    tool for tool in TOOLS if tool.get("function", {}).get("name") == "quote_selection"
]

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
openrouter_client = (
    OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    if OPENROUTER_API_KEY
    else None
)

if not CHAINLIT_AUTH_USERNAME or not CHAINLIT_AUTH_PASSWORD:
    raise RuntimeError("Missing CHAINLIT_AUTH_USERNAME/CHAINLIT_AUTH_PASSWORD in environment/.env")

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    if not compare_digest(username, CHAINLIT_AUTH_USERNAME):
        return None
    if not compare_digest(password, CHAINLIT_AUTH_PASSWORD):
        return None
    return cl.User(identifier=username, metadata={"auth_provider": "password"})


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}{path}",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw": response.text}
    if not response.ok:
        raise RuntimeError(f"{path} failed with status {response.status_code}: {body}")
    if not isinstance(body, dict):
        raise RuntimeError(f"{path} returned unexpected response: {body}")
    return body


def _normalize_site_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme or parsed.netloc:
        cleaned = parsed.path or "/"
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    return cleaned


async def _search_chunks(arguments: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": arguments["query"]}
    if "path_prefix" in arguments:
        path_prefix = arguments["path_prefix"]
        if isinstance(path_prefix, str):
            normalized_path_prefix = _normalize_site_path(path_prefix)
            if normalized_path_prefix != "/":
                payload["path_prefix"] = normalized_path_prefix
    if "top_k" in arguments:
        payload["top_k"] = arguments["top_k"]
    return await asyncio.to_thread(_post_json, "/search-chunks", payload)


def _line_map(lines: list[dict[str, Any]]) -> dict[int, str]:
    mapped: dict[int, str] = {}
    for line in lines:
        line_number = line.get("line_number")
        text = line.get("text")
        if isinstance(line_number, int) and isinstance(text, str):
            mapped[line_number] = text
    return mapped


def _absolutize_markdown_links(text: str, url_path: str) -> str:
    page_url = f"{SITE_BASE_URL}{url_path}"

    def replace_link(match: re.Match[str]) -> str:
        prefix = match.group(1)
        label = match.group(2)
        target = match.group(3).strip()
        if target.startswith(("http://", "https://", "mailto:", "javascript:")):
            return match.group(0)
        return f"{prefix}[{label}]({urljoin(page_url, target)})"

    return re.sub(r'(!?)\[([^\]]+)\]\(([^)]+)\)', replace_link, text)


def _format_quotes(quotes: list[dict[str, Any]], chunk_sources: dict[tuple[str, int], dict[str, Any]]) -> str:
    rendered_blocks: list[str] = []
    for quote in quotes:
        url_path = quote["url_path"]
        chunk_index = quote["chunk_index"]
        ranges = quote["line_ranges"]
        source = chunk_sources[(url_path, chunk_index)]
        line_lookup = _line_map(source["lines"])

        for start_line, end_line in ranges:
            selected_lines = [
                line_lookup[line_number]
                for line_number in range(start_line, end_line + 1)
                if line_number in line_lookup
            ]
            quoted_text = _absolutize_markdown_links(" ".join(selected_lines), url_path)
            rendered_blocks.append(
                f"{quoted_text}\n{SITE_BASE_URL}{url_path}"
            )
    return "\n\n".join(rendered_blocks)


def _validate_quote_selection(
    selection: dict[str, Any],
    chunk_sources: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    quotes = selection.get("quotes")
    failure_reason = selection.get("failure_reason")
    if not isinstance(quotes, list):
        return {"ok": False, "error": "quotes must be an array"}
    if not quotes:
        return {"ok": True, "quotes": [], "failure_reason": failure_reason}

    normalized_quotes: list[dict[str, Any]] = []
    for quote in quotes:
        if not isinstance(quote, dict):
            return {"ok": False, "error": "each quote must be an object"}
        url_path = quote.get("url_path")
        chunk_index = quote.get("chunk_index")
        line_ranges = quote.get("line_ranges")
        if not isinstance(url_path, str) or not isinstance(chunk_index, int) or not isinstance(line_ranges, list):
            return {"ok": False, "error": "quote is missing required fields"}

        source_key = (url_path, chunk_index)
        source = chunk_sources.get(source_key)
        if source is None:
            return {"ok": False, "error": f"unknown quote source: {url_path} chunk {chunk_index}"}

        line_lookup = _line_map(source["lines"])
        normalized_ranges: list[list[int]] = []
        for pair in line_ranges:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], int)
                or not isinstance(pair[1], int)
            ):
                return {"ok": False, "error": "line_ranges must contain [start, end] integer pairs"}
            start_line, end_line = pair
            if start_line < 1 or end_line < start_line:
                return {"ok": False, "error": "line ranges must be ascending positive integers"}
            if any(line_number not in line_lookup for line_number in range(start_line, end_line + 1)):
                return {
                    "ok": False,
                    "error": f"line range {start_line}-{end_line} is unavailable for {url_path} chunk {chunk_index}",
                }
            normalized_ranges.append([start_line, end_line])

        normalized_quotes.append(
            {
                "url_path": url_path,
                "chunk_index": chunk_index,
                "line_ranges": normalized_ranges,
            }
        )

    return {"ok": True, "quotes": normalized_quotes, "failure_reason": failure_reason}


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
    return payload


def _tool_step_output(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "search_chunks":
        matches = result.get("matches", [])
        return f"Returned {len(matches) if isinstance(matches, list) else 0} matches."
    if tool_name == "quote_selection":
        if result.get("ok") is False:
            error = result.get("error")
            return f"Quote selection invalid: {error}" if isinstance(error, str) else "Quote selection invalid."
        quotes = result.get("quotes", [])
        if isinstance(quotes, list) and quotes:
            return f"Selected {len(quotes)} quotes."
        reason = result.get("failure_reason")
        return reason if isinstance(reason, str) and reason.strip() else "No quotes selected."
    return json.dumps(result, ensure_ascii=False)


def _trim_line_text(text: str) -> str:
    if len(text) <= MAX_TOOL_LINE_CHARS:
        return text
    return f"{text[: MAX_TOOL_LINE_CHARS - 3].rstrip()}..."


def _compact_tool_result(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "search_chunks":
        return result

    matches = result.get("matches", [])
    if not isinstance(matches, list):
        return result

    compact_matches: list[dict[str, Any]] = []
    for match in matches[:MAX_TOOL_MATCHES_IN_CONTEXT]:
        if not isinstance(match, dict):
            continue
        lines = match.get("lines", [])
        compact_lines: list[dict[str, Any]] = []
        if isinstance(lines, list):
            for line in lines[:MAX_TOOL_LINES_PER_MATCH]:
                if not isinstance(line, dict):
                    continue
                line_number = line.get("line_number")
                text = line.get("text")
                if isinstance(line_number, int) and isinstance(text, str):
                    compact_lines.append(
                        {
                            "line_number": line_number,
                            "text": _trim_line_text(text),
                        }
                    )
        compact_matches.append(
            {
                "url_path": match.get("url_path"),
                "chunk_index": match.get("chunk_index"),
                "chunk_count": match.get("chunk_count"),
                "lines": compact_lines,
            }
        )

    return {
        "matches": compact_matches,
        "total_matches": len(matches),
        "matches_in_context": len(compact_matches),
        "lines_per_match_in_context": MAX_TOOL_LINES_PER_MATCH,
    }


def _trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    max_messages = MAX_CHAT_TURNS * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def _client_for_model(model_name: str) -> OpenAI:
    config = MODEL_CONFIG.get(model_name)
    if config is None:
        raise RuntimeError(f"Unsupported model: {model_name}")

    provider = config["provider"]
    if provider == "openai":
        if openai_client is None:
            raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")
        return openai_client
    if provider == "openrouter":
        if openrouter_client is None:
            raise RuntimeError("Missing OPENROUTER_API_KEY in environment/.env")
        return openrouter_client
    raise RuntimeError(f"Unsupported provider for model {model_name}: {provider}")


async def _run_agent(history: list[dict[str, str]], model_name: str) -> str:
    client = _client_for_model(model_name)
    messages: list[dict[str, Any]] = [{"role": "system", "content": STANDARD_SYSTEM_PROMPT}, *_trim_history(history)]
    chunk_sources: dict[tuple[str, int], dict[str, Any]] = {}
    debug_events: list[str] = [f"model={model_name}"]
    quote_selection_attempted = False

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        available_tools = TOOLS if round_number < MAX_TOOL_ROUNDS - 1 else QUOTE_SELECTION_ONLY_TOOLS
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model_name,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(_assistant_message_dict(message))

        if not message.tool_calls:
            debug_events.append(f"round={round_number} tool_calls=none")
            break
        debug_events.append(
            f"round={round_number} tool_calls="
            + ",".join(tool_call.function.name for tool_call in message.tool_calls)
        )

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
                debug_events.append(f"round={round_number} {tool_name}=invalid_json_args")

            async with cl.Step(name=tool_name, type="tool") as step:
                step.input = json.dumps(arguments, ensure_ascii=False, indent=2)

                if tool_name == "search_chunks":
                    result = await _search_chunks(arguments)
                    matches = result.get("matches", [])
                    debug_events.append(
                        f"round={round_number} search_chunks_matches={len(matches) if isinstance(matches, list) else 0}"
                    )
                    if isinstance(matches, list):
                        for match in matches:
                            if not isinstance(match, dict):
                                continue
                            url_path = match.get("url_path")
                            chunk_index = match.get("chunk_index")
                            if isinstance(url_path, str) and isinstance(chunk_index, int):
                                chunk_sources[(url_path, chunk_index)] = match
                elif tool_name == "quote_selection":
                    quote_selection_attempted = True
                    result = _validate_quote_selection(arguments, chunk_sources)
                    if result.get("ok"):
                        quotes = result.get("quotes", [])
                        debug_events.append(
                            f"round={round_number} quote_selection_quotes={len(quotes) if isinstance(quotes, list) else 0}"
                        )
                    else:
                        debug_events.append(
                            f"round={round_number} quote_selection_error={result.get('error')}"
                        )
                else:
                    result = {"ok": False, "error": f"unknown tool: {tool_name}"}

                step.output = _tool_step_output(tool_name, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(_compact_tool_result(tool_name, result), ensure_ascii=False),
                }
            )

            if tool_name == "quote_selection":
                if result.get("ok"):
                    quotes = result.get("quotes", [])
                    if not quotes:
                        reason = result.get("failure_reason")
                        return reason if isinstance(reason, str) and reason.strip() else "No supported quotations found."
                    return _format_quotes(quotes, chunk_sources)
                continue

    async with cl.Step(name="debug_trace", type="tool") as step:
        step.output = (
            "No supported quotations found.\n"
            f"quote_selection_attempted={quote_selection_attempted}\n"
            + "\n".join(debug_events[-12:])
        )
    return "No supported quotations found."


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("history", [])
    default_model = (
        OPENROUTER_MODEL
        if OPENROUTER_MODEL in SUPPORTED_MODELS
        else OPENAI_MODEL
        if OPENAI_MODEL in SUPPORTED_MODELS
        else SUPPORTED_MODELS[0]
    )
    cl.user_session.set("selected_model", default_model)
    await cl.ChatSettings(
        [
            Select(
                id=MODEL_SETTING_ID,
                label="Model",
                values=list(SUPPORTED_MODELS),
                initial_value=default_model,
            )
        ]
    ).send()
    await cl.Message(
        content=(
            "Ask a question and I will answer with quotations only.\n"
            "Use the settings panel to choose the model."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    selected_model = settings.get(MODEL_SETTING_ID)
    if isinstance(selected_model, str) and selected_model in SUPPORTED_MODELS:
        cl.user_session.set("selected_model", selected_model)
        await cl.Message(content=f"Model switched to `{selected_model}`.").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})
    history = _trim_history(history)
    selected_model = cl.user_session.get("selected_model") or SUPPORTED_MODELS[0]

    try:
        answer = await _run_agent(history, selected_model)
    except Exception as exc:
        answer = f"Unable to retrieve quotations: {exc}"

    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", _trim_history(history))
    await cl.Message(content=answer).send()
