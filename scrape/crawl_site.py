#!/usr/bin/env python3
"""
Crawl a website from a starting URL and save each HTML page as markdown,
mirroring the site's URL path structure on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"}


def normalize_url(url: str) -> str:
    # Remove fragments and trailing slash consistency for dedupe.
    clean, _fragment = urldefrag(url)
    parsed = urlparse(clean)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(path=path).geturl()


def local_markdown_path(output_root: str, url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")

    if not path:
        path = "index"
    elif path.endswith("/"):
        path = f"{path}index"

    filename = f"{path}.md"
    return os.path.join(output_root, filename)


def extract_links(base_url: str, html: str, allowed_netloc: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for a in soup.select("a[href]"):
        raw = a.get("href", "").strip()
        if not raw:
            continue
        absolute = urljoin(base_url, raw)
        normalized = normalize_url(absolute)
        parsed = urlparse(normalized)
        if parsed.netloc != allowed_netloc:
            continue
        if not is_http_url(normalized):
            continue
        links.append(normalized)

    return links


def strip_surrogates(text: str) -> str:
    return "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)


def response_html(response: requests.Response) -> str:
    encoding = (response.encoding or "").lower()
    if not encoding or encoding == "iso-8859-1":
        apparent_encoding = response.apparent_encoding
        if apparent_encoding:
            encoding = apparent_encoding
    if not encoding:
        encoding = "utf-8"
    return response.content.decode(encoding, errors="replace")


def save_markdown(output_root: str, url: str, html: str) -> str:
    markdown = strip_surrogates(to_markdown(html, heading_style="ATX"))
    path = local_markdown_path(output_root, url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Source URL: {url}\n\n")
        f.write(markdown)
    return path


def load_state(
    state_path: str, start_url: str, output_root: str
) -> tuple[
    set[str],
    list[str],
    dict[str, str],
    dict[str, dict[str, str | None]],
    dict[str, str | None],
]:
    if not os.path.exists(state_path):
        return set(), [start_url], {}, {}, {start_url: None}

    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = set(data.get("seen", []))
    to_scrape = list(data.get("to_scrape", []))
    saved = dict(data.get("saved", {}))
    raw_failed = dict(data.get("failed", {}))
    discovered_from = dict(data.get("discovered_from", {}))

    # Backwards-compatible migration: old failed state used plain strings.
    failed: dict[str, dict[str, str | None]] = {}
    for url, value in raw_failed.items():
        if isinstance(value, dict):
            failed[url] = {
                "reason": value.get("reason"),
                "found_on": value.get("found_on"),
            }
        else:
            failed[url] = {"reason": str(value), "found_on": discovered_from.get(url)}

    # Ensure at least the starting URL is queued for a fresh/invalid state.
    if not to_scrape and not seen:
        to_scrape = [start_url]
    if start_url not in discovered_from:
        discovered_from[start_url] = None

    # If output dir changes between runs, keep previous state but continue.
    _ = output_root
    return seen, to_scrape, saved, failed, discovered_from


def save_state(
    state_path: str,
    start_url: str,
    seen: set[str],
    to_scrape: list[str],
    saved: dict[str, str],
    failed: dict[str, dict[str, str | None]],
    discovered_from: dict[str, str | None],
) -> None:
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = {
        "start_url": start_url,
        "seen": sorted(seen),
        "to_scrape": to_scrape,
        "saved": saved,
        "failed": failed,
        "discovered_from": discovered_from,
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def crawl(start_url: str, output_root: str, delay_seconds: float, max_pages: int) -> None:
    start_url = normalize_url(start_url)
    allowed_netloc = urlparse(start_url).netloc
    state_path = os.path.join(output_root, ".crawl_state.json")

    seen, queued_urls, saved, failed, discovered_from = load_state(
        state_path, start_url, output_root
    )
    work = queue.Queue()
    for queued_url in queued_urls:
        work.put(queued_url)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "actualism-markdown-crawler/0.1 (+local script)",
        }
    )

    count = 0
    print(
        "Loaded state:"
        f" seen={len(seen)},"
        f" to_scrape={work.qsize()},"
        f" saved={len(saved)},"
        f" failed={len(failed)}"
    )

    while not work.empty():
        if max_pages > 0 and count >= max_pages:
            break

        url = work.get()
        if url in seen:
            continue
        seen.add(url)

        try:
            response = session.get(url, timeout=30)
        except requests.RequestException as exc:
            failed[url] = {"reason": str(exc), "found_on": discovered_from.get(url)}
            print(f"[skip] {url} ({exc})")
            save_state(
                state_path=state_path,
                start_url=start_url,
                seen=seen,
                to_scrape=list(work.queue),
                saved=saved,
                failed=failed,
                discovered_from=discovered_from,
            )
            continue

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            failed[url] = {
                "reason": f"non-html: {content_type}",
                "found_on": discovered_from.get(url),
            }
            print(f"[skip] {url} (non-html: {content_type})")
            save_state(
                state_path=state_path,
                start_url=start_url,
                seen=seen,
                to_scrape=list(work.queue),
                saved=saved,
                failed=failed,
                discovered_from=discovered_from,
            )
            continue
        if response.status_code != 200:
            failed[url] = {
                "reason": f"status {response.status_code}",
                "found_on": discovered_from.get(url),
            }
            print(f"[skip] {url} (status {response.status_code})")
            save_state(
                state_path=state_path,
                start_url=start_url,
                seen=seen,
                to_scrape=list(work.queue),
                saved=saved,
                failed=failed,
                discovered_from=discovered_from,
            )
            continue

        html = response_html(response)
        output_path = save_markdown(output_root, url, html)
        saved[url] = output_path
        if url in failed:
            del failed[url]
        count += 1
        print(f"[{count}] {url} -> {output_path}")

        for link in extract_links(url, html, allowed_netloc):
            if link not in discovered_from:
                discovered_from[link] = url
            if link not in seen:
                work.put(link)

        save_state(
            state_path=state_path,
            start_url=start_url,
            seen=seen,
            to_scrape=list(work.queue),
            saved=saved,
            failed=failed,
            discovered_from=discovered_from,
        )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    save_state(
        state_path=state_path,
        start_url=start_url,
        seen=seen,
        to_scrape=list(work.queue),
        saved=saved,
        failed=failed,
        discovered_from=discovered_from,
    )
    print(
        f"Done. Newly saved {count} pages to: {output_root} "
        f"(total saved: {len(saved)}, remaining: {work.qsize()})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl a site and export each page as markdown."
    )
    parser.add_argument(
        "--start-url",
        default="https://www.actualfreedom.com.au/sundry/map.htm",
        help="Starting URL for crawl.",
    )
    parser.add_argument(
        "--output-dir",
        default="site_markdown",
        help="Folder to write markdown files.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Optional cap on crawled pages (0 = no cap).",
    )
    args = parser.parse_args()

    crawl(
        start_url=args.start_url,
        output_root=args.output_dir,
        delay_seconds=args.delay_seconds,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
