"""Scraper runner — single entry point for Agent 1.

Four modes (tried in order for topic-based scraping):
  1. Adapter-based crawl — registered site adapters for known topics
  2. Tavily search + Playwright/Camoufox stealth scrape (primary)
  3. DuckDuckGo search + httpx fallback (legacy)
  4. Direct URL extraction — generic HTML parsing on a given URL

Usage:
  scrape(topic="usa swift code")               → adapters first, then search
  scrape(url="https://bank-code.net/...")       → generic URL extraction
  scrape(topic="...", config={...})             → adapters → Tavily+Playwright
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

from tools.scraper.adapters import find_adapters
from tools.scraper.fetch import fetch_text, make_client
from tools.scraper.parsers import extract_all_values

logger = logging.getLogger(__name__)

MIN_USEFUL_VALUES = 20

_FALLBACK_PATTERNS: dict[str, str] = {
    "postcode": r"\b\d{3}\s?\d{2}\b",
    "postal code": r"\b\d{3,6}\b",
    "zip code": r"\b\d{5}(?:-\d{4})?\b",
    "zip": r"\b\d{5}(?:-\d{4})?\b",
    "phone": r"\+?\d[\d\s\-().]{7,18}\d",
    "medicare": r"\b[A-Za-z0-9]{11}\b",
    "medicaid": r"\b\d{7,13}\b",
    "swift": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
    "bic": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b",
    "ssn": r"\b\d{3}-?\d{2}-?\d{4}\b",
    "passport": r"\b[A-Z0-9]{6,12}\b",
    "pin code": r"\b\d{6}\b",
    "pincode": r"\b\d{6}\b",
}


_llm_regex_cache: dict[str, re.Pattern] = {}


def _cache_llm_regex(topic: str, regex: re.Pattern) -> None:
    """Store the LLM-provided regex so _sanity_filter can use it."""
    _llm_regex_cache[topic.strip().lower()] = regex


def _get_best_regex(topic: str) -> Optional[re.Pattern]:
    """Return the best available regex for this topic.

    Priority: LLM-provided (cached) > hardcoded fallback.
    """
    key = topic.strip().lower()
    if key in _llm_regex_cache:
        return _llm_regex_cache[key]
    return _get_fallback_regex(topic)


def _get_fallback_regex(topic: str) -> Optional[re.Pattern]:
    """Return a basic value regex inferred from the topic type."""
    topic_lower = topic.lower()
    for keyword, pattern in _FALLBACK_PATTERNS.items():
        if keyword in topic_lower:
            logger.info("[fallback-regex] topic '%s' matched keyword '%s' → %s", topic, keyword, pattern)
            return re.compile(pattern)
    return None


def _generic_url_scrape(
    url: str,
    value_regex: Optional[re.Pattern] = None,
    max_pages: int = 20,
    sleep_s: float = 1.0,
) -> list[str]:
    """Scrape a single URL (or paginated set) using generic extractors + httpx."""
    client = make_client()
    all_values: list[str] = []
    seen: set[str] = set()

    try:
        html = fetch_text(client, url)
        values = extract_all_values(html, value_regex)
        for v in values:
            if v not in seen:
                seen.add(v)
                all_values.append(v)
        logger.info("[generic] %s → %d values", url, len(values))

        from selectolax.parser import HTMLParser
        tree = HTMLParser(html)
        page_urls: set[str] = set()
        for a in tree.css("a"):
            href = a.attributes.get("href") or ""
            if re.search(r"[/&?]page[=/]?\d+", href):
                full = href if href.startswith("http") else url.rstrip("/").rsplit("/", 1)[0] + "/" + href.lstrip("/")
                page_urls.add(full)

        for i, page_url in enumerate(sorted(page_urls)):
            if i >= max_pages:
                break
            if page_url == url:
                continue
            try:
                page_html = fetch_text(client, page_url)
                page_vals = extract_all_values(page_html, value_regex)
                for v in page_vals:
                    if v not in seen:
                        seen.add(v)
                        all_values.append(v)
                logger.info("[generic] %s → %d new values", page_url, len(page_vals))
                time.sleep(sleep_s)
            except Exception as exc:
                logger.warning("[generic] failed on %s: %s", page_url, exc)
                break
    finally:
        client.close()

    return all_values


# ── Playwright + Camoufox search-and-scrape (primary path) ───────────────────

def _playwright_search_and_scrape(
    topic: str,
    config: Optional[dict] = None,
    sleep_s: float = 2.0,
) -> list[str]:
    """Tavily URL discovery → Playwright/Camoufox stealth scrape.

    1. Get LLM hints (search queries + value regex) if config is available
    2. Tavily search (fallback DDG) for top 3 URLs
    3. Scrape each URL with Camoufox stealth browser
    4. Extract values with existing parsers
    """
    from tools.search_tool import search_data_urls
    from tools.scraper.browser import scrape_pages, pick_proxy
    from tools.scraper.parsers import extract_all_values as _extract

    value_regex_pattern: Optional[str] = None

    if config:
        try:
            from tools.scraper.llm_hints import get_scraper_hints
            hints = get_scraper_hints(topic, config)
            if hints:
                if hints.get("value_regex"):
                    value_regex_pattern = hints["value_regex"]
                    logger.info("[pw+tavily] LLM value regex: %s", value_regex_pattern)
                if hints.get("gov_domains"):
                    from tools.scraper.search import set_gov_domains
                    set_gov_domains(topic, hints["gov_domains"])
        except Exception as exc:
            logger.warning("[pw+tavily] LLM hints failed: %s — using generic extraction", exc)

    if value_regex_pattern:
        compiled_regex = re.compile(value_regex_pattern)
        _cache_llm_regex(topic, compiled_regex)
    else:
        compiled_regex = _get_fallback_regex(topic)

    tavily_key = (config or {}).get("tavily_api_key")
    urls = search_data_urls(topic, tavily_api_key=tavily_key, max_results=3)
    if not urls:
        logger.info("[pw+tavily] no URLs found for '%s'", topic)
        return []

    logger.info("[pw+tavily] scraping %d URLs for '%s'", len(urls), topic)
    proxies = (config or {}).get("proxies", [])
    page_results = scrape_pages(urls, proxies=proxies or None, sleep_s=sleep_s)

    all_values: list[str] = []
    seen: set[str] = set()
    for url, html in page_results:
        values = _extract(html, compiled_regex)
        for v in values:
            if v not in seen:
                seen.add(v)
                all_values.append(v)
        logger.info("[pw+tavily] %s → %d values (total: %d)", url, len(values), len(all_values))

    logger.info("[pw+tavily] total scraped: %d unique values", len(all_values))
    return all_values


# ── Legacy DuckDuckGo + httpx search-and-scrape (fallback) ───────────────────

def _search_and_scrape(
    topic: str,
    config: Optional[dict] = None,
    sleep_s: float = 1.5,
) -> list[str]:
    """Legacy path: DuckDuckGo + LLM-guided extraction via httpx."""
    from tools.scraper.search import search_urls, build_search_queries

    search_queries = build_search_queries(topic)
    value_regex_pattern: Optional[str] = None

    if config:
        try:
            from tools.scraper.llm_hints import get_scraper_hints
            hints = get_scraper_hints(topic, config)
            if hints:
                if hints.get("search_queries"):
                    search_queries = hints["search_queries"] + search_queries
                if hints.get("value_regex"):
                    value_regex_pattern = hints["value_regex"]
                if hints.get("gov_domains"):
                    from tools.scraper.search import set_gov_domains
                    set_gov_domains(topic, hints["gov_domains"])
        except Exception:
            pass

    if value_regex_pattern:
        compiled_regex = re.compile(value_regex_pattern)
        _cache_llm_regex(topic, compiled_regex)
    else:
        compiled_regex = _get_fallback_regex(topic)
    urls = search_urls(search_queries, max_results_per_query=8, max_total=12, condition=topic)
    if not urls:
        return []

    all_values: list[str] = []
    seen: set[str] = set()
    for url in urls:
        try:
            values = _generic_url_scrape(url, compiled_regex, max_pages=3, sleep_s=sleep_s)
            for v in values:
                if v not in seen:
                    seen.add(v)
                    all_values.append(v)
            time.sleep(sleep_s)
        except Exception as exc:
            logger.warning("[search+httpx] failed on %s: %s", url, exc)

    return all_values


def scrape(
    topic: Optional[str] = None,
    url: Optional[str] = None,
    value_regex: Optional[str] = None,
    config: Optional[dict] = None,
    max_pages: int = 20,
    sleep_s: float = 1.0,
) -> Optional[list[str]]:
    """Scrape data by topic, URL, or both.

    Returns a list of extracted string values, or None if nothing was found.

    Priority:
      1. Adapter-based crawl (if topic matches a registered adapter)
      2. Tavily + Playwright/Camoufox stealth scrape (primary)
      3. DuckDuckGo + httpx fallback (legacy)
      4. Direct URL extraction (if url provided)

    Args:
        topic:       Search topic (e.g. "usa swift code"). Matched against adapters.
        url:         Direct URL to scrape with generic extractors.
        value_regex: Optional regex string to filter/extract specific values from HTML.
        config:      LLM config dict (for LLM-guided hints). Pass None to skip.
        max_pages:   Max pagination pages to follow for generic URL scraping.
        sleep_s:     Delay between requests.
    """
    compiled_regex = re.compile(value_regex) if value_regex else None
    all_values: list[str] = []

    # Mode 1: adapter-based crawl by topic
    if topic:
        adapters = find_adapters(topic)
        if adapters:
            logger.info("Scraper: found %d adapter(s) for topic '%s'", len(adapters), topic)
            for adapter in adapters:
                try:
                    values = adapter.crawl(sleep_s=sleep_s)
                    if compiled_regex:
                        values = [v for v in values if compiled_regex.fullmatch(v)]
                    all_values.extend(values)
                    logger.info("Scraper: adapter '%s' returned %d values", adapter.name, len(values))
                except Exception as exc:
                    logger.warning("Scraper: adapter '%s' failed: %s", adapter.name, exc)
            if all_values:
                deduped = _dedupe(all_values)
                return _sanity_filter(deduped, topic) if topic else deduped

    # Mode 2: Tavily search + Playwright/Camoufox stealth scrape
    if topic:
        logger.info("Scraper: trying Tavily + Playwright stealth scrape for '%s'", topic)
        try:
            pw_values = _playwright_search_and_scrape(topic, config=config, sleep_s=sleep_s)
            if pw_values and len(pw_values) >= MIN_USEFUL_VALUES:
                logger.info("Scraper: Playwright stealth scrape found %d values", len(pw_values))
                all_values.extend(pw_values)
                deduped = _dedupe(all_values)
                return _sanity_filter(deduped, topic)
            elif pw_values:
                logger.info("Scraper: Playwright found %d values (below threshold %d)", len(pw_values), MIN_USEFUL_VALUES)
                all_values.extend(pw_values)
        except Exception as exc:
            logger.warning("Scraper: Playwright path failed: %s — trying legacy DDG+httpx", exc)

    # Mode 3: legacy DuckDuckGo + httpx fallback
    if topic and len(all_values) < MIN_USEFUL_VALUES:
        logger.info("Scraper: falling back to DDG+httpx search for '%s'", topic)
        search_values = _search_and_scrape(topic, config=config, sleep_s=sleep_s)
        if search_values:
            all_values.extend(search_values)
            deduped = _dedupe(all_values)
            if len(deduped) >= MIN_USEFUL_VALUES:
                return _sanity_filter(deduped, topic)

    # Mode 4: direct URL extraction
    if url:
        logger.info("Scraper: generic extraction from %s", url)
        values = _generic_url_scrape(url, compiled_regex, max_pages, sleep_s)
        all_values.extend(values)

    if all_values:
        deduped = _dedupe(all_values)
        if topic:
            deduped = _sanity_filter(deduped, topic)
        return deduped if deduped else None

    logger.info("Scraper: no data found for topic=%s url=%s", topic, url)
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


_TIMESTAMP_14 = re.compile(
    r"^(19|20)\d{2}"          # YYYY (1900–2099)
    r"(0[1-9]|1[0-2])"       # MM
    r"(0[1-9]|[12]\d|3[01])" # DD
    r"([01]\d|2[0-3])"       # HH
    r"[0-5]\d"               # MM
    r"[0-5]\d$"              # SS
)

_TIMESTAMP_8 = re.compile(
    r"^(19|20)\d{2}"          # YYYY
    r"(0[1-9]|1[0-2])"       # MM
    r"(0[1-9]|[12]\d|3[01])$" # DD
)

_SEQUENTIAL = re.compile(
    r"^(0123456789|1234567890|9876543210|0{5,}|1{5,}|"
    r"01234567|12345678|23456789)$"
)

_FALSE_POSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (_TIMESTAMP_14, "timestamp_YYYYMMDDHHMMSS"),
    (_TIMESTAMP_8,  "date_YYYYMMDD"),
    (_SEQUENTIAL,   "sequential/trivial"),
]


def _is_false_positive(value: str) -> Optional[str]:
    """Check if a numeric-looking value is a common false positive.

    Returns the rejection reason string, or None if the value looks OK.
    """
    stripped = value.strip()
    for pattern, reason in _FALSE_POSITIVE_PATTERNS:
        if pattern.match(stripped):
            return reason
    return None


def _sanity_filter(values: list[str], topic: str) -> list[str]:
    """Drop values that are clearly not identifiers (too long, too short,
    look like prose/navigation text, or match known false-positive patterns
    like timestamps and dates).
    """
    fallback = _get_best_regex(topic)
    filtered: list[str] = []
    fp_rejected = 0
    for v in values:
        v_stripped = v.strip()
        if not v_stripped:
            continue
        if len(v_stripped) > 50:
            continue
        if len(v_stripped) < 2:
            continue
        word_count = len(v_stripped.split())
        if word_count > 6:
            continue
        if fallback and not fallback.search(v_stripped):
            continue
        fp_reason = _is_false_positive(v_stripped)
        if fp_reason:
            fp_rejected += 1
            logger.debug("[sanity] rejected '%s' as false positive (%s)", v_stripped, fp_reason)
            continue
        filtered.append(v_stripped)

    if fp_rejected > 0:
        logger.info(
            "[sanity] rejected %d false-positive values (timestamps/dates/sequential) for '%s'",
            fp_rejected, topic,
        )
    if len(filtered) < len(values):
        logger.info(
            "[sanity] filtered %d → %d values for '%s'",
            len(values), len(filtered), topic,
        )
    return filtered
