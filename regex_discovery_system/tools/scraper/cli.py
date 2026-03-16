#!/usr/bin/env python3
"""Standalone CLI for the scraper tool.

Usage:
    python -m tools.scraper.cli --topic "usa swift code"
    python -m tools.scraper.cli --url "https://bank-code.net/country/UNITED-STATES-%28US%29/100"
    python -m tools.scraper.cli --topic "usa swift code" --out swift_codes.txt
    python -m tools.scraper.cli --url "https://example.com" --regex "\\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,3}\\b"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="General-purpose web scraper")
    parser.add_argument("--topic", help='Topic to search adapters for (e.g. "usa swift code")')
    parser.add_argument("--url", help="Direct URL to scrape")
    parser.add_argument("--regex", help="Optional regex to filter/extract values")
    parser.add_argument("--out", help="Output file (.txt). If omitted, prints to stdout")
    parser.add_argument("--max-pages", type=int, default=20, help="Max pagination pages")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    if not args.topic and not args.url:
        parser.error("Provide at least --topic or --url")

    from tools.scraper.runner import scrape

    values = scrape(
        topic=args.topic,
        url=args.url,
        value_regex=args.regex,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )

    if not values:
        print("No data found.")
        sys.exit(1)

    print(f"Scraped {len(values)} values")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(values) + "\n")
        print(f"Written to {args.out}")
    else:
        for v in values[:50]:
            print(v)
        if len(values) > 50:
            print(f"... and {len(values) - 50} more")


if __name__ == "__main__":
    main()
