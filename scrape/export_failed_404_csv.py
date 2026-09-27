#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse


LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def normalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url)
    parsed = urlparse(clean)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(path=path).geturl()


def is_prepended_external_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc.endswith("actualfreedom.com.au"):
        return False
    path_lower = parsed.path.lower()
    return ("http:/" in path_lower) or ("https:/" in path_lower)

def link_target_from_markdown(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    if ' "' in target:
        target = target.split(' "', 1)[0]
    if " '" in target:
        target = target.split(" '", 1)[0]
    return target

def find_visible_text(markdown_path: Path, found_on_url: str, target_url: str) -> str:
    if not markdown_path.exists():
        return ""

    content = markdown_path.read_text(encoding="utf-8")
    normalized_target = normalize_url(target_url)

    for match in LINK_RE.finditer(content):
        visible_text = match.group(1).strip()
        raw_link = match.group(2).strip()
        resolved = normalize_url(urljoin(found_on_url, link_target_from_markdown(raw_link)))
        if resolved == normalized_target:
            return visible_text

    return ""


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "found_on", "visible_text"])
        writer.writeheader()
        writer.writerows(rows)


def export_failed_404(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    failed = data.get("failed", {})
    saved = data.get("saved", {})
    scrape_root = input_path.parent.parent

    rows = []
    for url, details in failed.items():
        if isinstance(details, dict) and details.get("reason") == "status 404":
            if is_prepended_external_url(url):
                continue
            found_on = details.get("found_on", "")
            markdown_rel_path = saved.get(found_on, "")
            markdown_path = scrape_root / markdown_rel_path if markdown_rel_path else Path("")
            rows.append(
                {
                    "url": url,
                    "found_on": found_on,
                    "visible_text": find_visible_text(markdown_path, found_on, url),
                }
            )

    write_csv(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export failed status 404 links from crawl state JSON to CSV."
    )
    parser.add_argument(
        "--input",
        default="site_markdown/.crawl_state.json",
        help="Path to crawl state JSON file.",
    )
    parser.add_argument(
        "--output",
        default="site_markdown/failed_404_links.csv",
        help="Path to output CSV file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists() and args.input == "site_markdown/.crawl_state.json":
        fallback = Path("scrape/site_markdown/.crawl_state.json")
        if fallback.exists():
            input_path = fallback

    count = export_failed_404(input_path, Path(args.output))
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
