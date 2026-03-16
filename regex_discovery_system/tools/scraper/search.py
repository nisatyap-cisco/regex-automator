"""DuckDuckGo HTML search for dynamic URL discovery.

Uses httpx + selectolax (already in the project) to query DuckDuckGo's
HTML interface — no extra SDK dependency needed.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional
from urllib.parse import urlparse, quote_plus

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)

_BLOCKED_DOMAINS = {
    "youtube.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "reddit.com", "linkedin.com",
    "pinterest.com", "amazon.com", "ebay.com",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
}


def _is_useful_url(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    for blocked in _BLOCKED_DOMAINS:
        if blocked in domain:
            return False
    return True


def is_url_relevant(url: str, condition: str) -> bool:
    """Check if a URL is plausibly relevant to the search condition.

    Rejects URLs from unrelated government domains when the condition
    is about a different country. E.g. rejects fam.state.gov for
    'Czech Postcode'.
    """
    url_lower = url.lower()
    cond_lower = condition.lower()
    cond_words = [w for w in re.split(r"[\s_-]+", cond_lower) if len(w) > 2]

    gov_domains = _detect_gov_domains(condition)
    parsed = urlparse(url_lower)
    domain = parsed.netloc
    path = parsed.path

    if gov_domains:
        is_us_gov = domain.endswith(".gov") and not any(
            domain.endswith(f".{d}") or domain.endswith(d) for d in gov_domains
        )
        if is_us_gov:
            if any(w in url_lower for w in cond_words):
                return True
            return False

    if any(w in domain or w in path for w in cond_words):
        return True

    domain_parts = domain.replace("www.", "").split(".")
    generic_data_sites = {
        "wikipedia.org", "worldpostalcode.com", "geopostcodes.com",
        "geonames.org", "zipcodebase.com", "postcodebase.com",
    }
    if any(gs in domain for gs in generic_data_sites):
        return True

    return True


def _extract_urls_from_ddg_html(html: str) -> list[str]:
    """Parse DuckDuckGo HTML results page and extract result URLs."""
    tree = HTMLParser(html)
    urls: list[str] = []

    for a in tree.css("a.result__a, a.result-link, a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href:
            continue
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            from urllib.parse import unquote
            href = unquote(m.group(1))
        if href.startswith("http") and "duckduckgo.com" not in href:
            urls.append(href)

    return urls


def search_urls(
    queries: list[str],
    max_results_per_query: int = 8,
    max_total: int = 15,
    condition: str = "",
) -> list[str]:
    """Run DuckDuckGo HTML searches and return deduplicated URLs.

    Includes a circuit breaker: if the first query exhausts all retries
    without returning results (persistent rate-limit / CAPTCHA), the
    remaining queries are skipped to avoid wasting minutes.
    """
    seen: set[str] = set()
    urls: list[str] = []
    skipped = 0
    ddg_blocked = False
    consecutive_empty = 0

    client = httpx.Client(
        timeout=20,
        follow_redirects=True,
        headers=_HEADERS,
    )

    try:
        for query in queries:
            if len(urls) >= max_total:
                break
            if ddg_blocked:
                logger.info("[search] DDG blocked — skipping query: %s", query)
                continue

            logger.info("[search] querying: %s", query)

            found: list[str] = []
            query_rate_limited = False
            for attempt in range(3):
                try:
                    resp = client.get(
                        "https://html.duckduckgo.com/html/",
                        params={"q": query},
                    )
                    if resp.status_code in (202, 429):
                        wait = 5 * (attempt + 1)
                        logger.warning(
                            "[search] DDG rate-limited (HTTP %d) — waiting %ds (attempt %d/3)",
                            resp.status_code, wait, attempt + 1,
                        )
                        query_rate_limited = True
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    found = _extract_urls_from_ddg_html(resp.text)
                    if not found and "captcha" in resp.text.lower():
                        wait = 8 * (attempt + 1)
                        logger.warning(
                            "[search] DDG returned CAPTCHA — waiting %ds (attempt %d/3)",
                            wait, attempt + 1,
                        )
                        query_rate_limited = True
                        time.sleep(wait)
                        continue
                    break
                except httpx.HTTPStatusError as exc:
                    logger.warning("[search] query '%s' HTTP error: %s", query, exc)
                    break
                except Exception as exc:
                    logger.warning("[search] query '%s' failed: %s", query, exc)
                    break

            if not found and query_rate_limited:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    ddg_blocked = True
                    logger.warning(
                        "[search] DDG circuit breaker OPEN — %d consecutive rate-limited "
                        "queries with 0 results; skipping remaining queries",
                        consecutive_empty,
                    )
            elif found:
                consecutive_empty = 0

            count = 0
            for u in found:
                if count >= max_results_per_query:
                    break
                if u in seen or not _is_useful_url(u):
                    continue
                if condition and not is_url_relevant(u, condition):
                    skipped += 1
                    logger.debug("[search] skipped irrelevant URL: %s", u)
                    continue
                seen.add(u)
                urls.append(u)
                count += 1
                if len(urls) >= max_total:
                    break
            logger.info("[search] '%s' → %d URLs found, %d new", query, len(found), count)
            time.sleep(3 if query_rate_limited else 1.5)
    finally:
        client.close()

    if skipped:
        logger.info("[search] filtered out %d irrelevant URLs", skipped)
    logger.info("[search] total unique URLs: %d", len(urls))
    return urls


_gov_domain_cache: dict[str, list[str]] = {}


def set_gov_domains(condition: str, domains: list[str]) -> None:
    """Store LLM-discovered gov domains for a condition (called from runner)."""
    key = condition.strip().lower()
    _gov_domain_cache[key] = domains
    logger.info("[search] cached gov domains for '%s': %s", condition, domains)


def _detect_gov_domains(condition: str) -> list[str]:
    """Return country-specific government domains for the condition.

    Uses LLM-discovered domains (cached via set_gov_domains) first.
    Falls back to simple heuristic for USA-related topics only.
    """
    key = condition.strip().lower()
    if key in _gov_domain_cache:
        return _gov_domain_cache[key]

    if any(kw in key for kw in ("usa", "us ", "united states", "california", "american")):
        return ["gov"]
    return []


def build_search_queries(condition: str) -> list[str]:
    """Generate search queries: gov sites first, then broad third-party."""
    c = condition.strip()
    gov_domains = _detect_gov_domains(c)

    queries: list[str] = []

    if gov_domains:
        queries.append(f"{c} complete list site:{gov_domains[0]}")

    queries.extend([
        f"{c} complete list database directory",
        f"{c} complete list Wikipedia OR geonames OR worldpostalcode",
        f"all {c} examples data open data",
    ])
    return queries
