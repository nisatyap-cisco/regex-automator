"""Pluggable site adapters + registry.

Each adapter knows:
  - which topics it supports (keyword matching)
  - which seed URLs to crawl
  - how to paginate
  - how to parse values out of a page

New adapters: subclass Adapter, implement the 4 methods, and add an
instance to ADAPTER_REGISTRY at the bottom of this file.
"""
from __future__ import annotations

import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Iterable

from selectolax.parser import HTMLParser

from tools.scraper.fetch import fetch_text, make_client

logger = logging.getLogger(__name__)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ── base class ────────────────────────────────────────────────────────────────

class Adapter(ABC):
    name: str
    keywords: list[str] = []
    required_keywords: list[str] = []

    def matches_topic(self, topic: str) -> int:
        """Return a score (0 = no match) for how well this adapter fits the topic.

        If required_keywords is set, at least one must appear for a positive
        score — prevents "usa" in "USA Medicare Numbers" from triggering a
        SWIFT adapter.
        """
        t = topic.lower()
        if self.required_keywords:
            if not any(rk.lower() in t for rk in self.required_keywords):
                return 0
        return sum(1 for kw in self.keywords if kw.lower() in t)

    @abstractmethod
    def seed_urls(self) -> list[str]:
        ...

    @abstractmethod
    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        ...

    @abstractmethod
    def parse_values(self, html: str, url: str) -> list[str]:
        ...

    def crawl(self, sleep_s: float = 1.0) -> list[str]:
        """Crawl all pages from all seeds and return deduplicated values."""
        all_values: list[str] = []
        seen: set[str] = set()
        client = make_client()

        try:
            for seed in self.seed_urls():
                first_html = fetch_text(client, seed)
                for page_url in self.iter_page_urls(first_html, seed):
                    html = first_html if page_url == seed else fetch_text(client, page_url)
                    values = self.parse_values(html, page_url)
                    for v in values:
                        if v not in seen:
                            seen.add(v)
                            all_values.append(v)
                    logger.info("[%s] %s → %d new values (total: %d)", self.name, page_url, len(values), len(all_values))
                    if page_url != seed:
                        time.sleep(sleep_s)
        finally:
            client.close()

        return all_values


# ── SWIFT / BIC adapters ─────────────────────────────────────────────────────

BIC_US_RE = re.compile(r"\b[A-Z]{4}US[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")


class TheSwiftCodesAdapter(Adapter):
    name = "theswiftcodes_us"
    keywords = ["swift", "bic", "swift code", "usa", "us", "united states"]
    required_keywords = ["swift", "bic"]

    def seed_urls(self) -> list[str]:
        return ["https://www.theswiftcodes.com/united-states/"]

    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        tree = HTMLParser(first_html)
        nums = []
        for a in tree.css("a"):
            href = a.attributes.get("href") or ""
            m = re.search(r"/page/(\d+)/", href)
            if m:
                nums.append(int(m.group(1)))
        last = max(nums) if nums else 1
        yield seed_url
        for p in range(2, last + 1):
            yield seed_url.rstrip("/") + f"/page/{p}/"

    def parse_values(self, html: str, url: str) -> list[str]:
        tree = HTMLParser(html)
        text = tree.text(separator="\n")
        return list(set(BIC_US_RE.findall(text)))


class BankCodeNetAdapter(Adapter):
    name = "bankcodenet_us"
    keywords = ["swift", "bic", "swift code", "usa", "us", "united states", "bank code"]
    required_keywords = ["swift", "bic", "bank code"]

    _BASE = "https://bank-code.net"
    _PATH = "/country/UNITED-STATES-%28US%29"

    def seed_urls(self) -> list[str]:
        return [f"{self._BASE}{self._PATH}.html"]

    def iter_page_urls(self, first_html: str, seed_url: str) -> Iterable[str]:
        tree = HTMLParser(first_html)
        last_offset = 0
        for a in tree.css("a"):
            href = (a.attributes.get("href") or "").strip()
            if "UNITED-STATES-%28US%29" not in href:
                continue
            m = re.search(r"/UNITED-STATES-%28US%29/(\d+)", href)
            if m:
                last_offset = max(last_offset, int(m.group(1)))

        yield seed_url
        for offset in range(50, last_offset + 1, 50):
            yield f"{self._BASE}{self._PATH}/{offset}"

    def parse_values(self, html: str, url: str) -> list[str]:
        tree = HTMLParser(html)
        text = tree.text(separator="\n")
        return list(set(BIC_US_RE.findall(text)))


# ── registry ──────────────────────────────────────────────────────────────────

ADAPTER_REGISTRY: list[Adapter] = [
    TheSwiftCodesAdapter(),
    BankCodeNetAdapter(),
]


def find_adapters(topic: str) -> list[Adapter]:
    """Return adapters matching the topic, sorted by relevance (best first)."""
    scored = [(ad, ad.matches_topic(topic)) for ad in ADAPTER_REGISTRY]
    matched = [(ad, s) for ad, s in scored if s > 0]
    matched.sort(key=lambda x: x[1], reverse=True)
    return [ad for ad, _ in matched]
