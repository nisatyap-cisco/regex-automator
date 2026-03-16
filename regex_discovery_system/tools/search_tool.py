"""URL discovery via Tavily (primary) with DuckDuckGo fallback.

Returns the top N URLs most likely to contain structured data lists
for a given condition.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_BLOCKED_DOMAINS = {
    "youtube.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "reddit.com", "linkedin.com",
    "pinterest.com", "amazon.com", "ebay.com",
}


def _is_useful_url(url: str) -> bool:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.lower()
    return not any(blocked in domain for blocked in _BLOCKED_DOMAINS)


def search_data_urls(
    condition: str,
    tavily_api_key: Optional[str] = None,
    max_results: int = 3,
) -> list[str]:
    """Search for pages that list real data for *condition*.

    Tries Tavily first (needs API key), falls back to DuckDuckGo HTML scraping.
    """
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    if api_key:
        urls = _tavily_search(condition, api_key, max_results)
        if urls:
            return urls
        logger.info("[search] Tavily returned nothing — falling back to DDG")

    return _ddg_fallback(condition, max_results)


def search_policy_urls(
    condition: str,
    tavily_api_key: Optional[str] = None,
    max_results: int = 3,
) -> list[str]:
    """Search for pages describing format rules / policies for *condition*.

    Prioritises government (.gov) and official regulatory body websites.
    """
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
    queries = [
        f"{condition} numbering format rules official .gov",
        f"{condition} allocation policy specification government",
        f"{condition} valid range structure official authority",
        f"{condition} numbering format rules",
    ]

    if api_key:
        urls = _tavily_search_multi(queries, api_key, max_results)
        if urls:
            return urls

    return _ddg_fallback_multi(queries, max_results)


def _tavily_search(condition: str, api_key: str, max_results: int) -> list[str]:
    """Search for data pages, prioritising government and official sources."""
    try:
        from tavily import TavilyClient
        from tools.scraper.search import _detect_gov_domains
        client = TavilyClient(api_key)

        gov_domains = _detect_gov_domains(condition)
        if gov_domains:
            site_q = " OR ".join(f"site:{d}" for d in gov_domains[:2])
            queries = [
                f"{condition} complete list {site_q}",
                f"{condition} complete list data examples",
                f"{condition} database directory official",
            ]
        else:
            queries = [
                f"{condition} complete list official government data",
                f"{condition} database directory",
                f"{condition} complete list data examples",
            ]

        seen: set[str] = set()
        urls: list[str] = []
        for q in queries:
            if len(urls) >= max_results:
                break
            resp = client.search(
                query=q,
                search_depth="advanced",
                max_results=max_results + 2,
                include_raw_content=False,
            )
            for r in resp.get("results", []):
                u = r["url"]
                if u not in seen and _is_useful_url(u):
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= max_results:
                        break

        gov_first = sorted(urls, key=lambda u: (0 if ".gov" in u.lower() else 1))
        logger.info("[tavily] '%s' → %d URLs (gov-prioritised)", condition, len(gov_first))
        return gov_first[:max_results]
    except Exception as exc:
        logger.warning("[tavily] search failed: %s", exc)
        return []


def _tavily_search_multi(queries: list[str], api_key: str, max_results: int) -> list[str]:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key)
        seen: set[str] = set()
        urls: list[str] = []
        for q in queries:
            if len(urls) >= max_results:
                break
            resp = client.search(query=q, search_depth="basic", max_results=3)
            for r in resp.get("results", []):
                u = r["url"]
                if u not in seen and _is_useful_url(u):
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= max_results:
                        break
        logger.info("[tavily-policy] %d URLs from %d queries", len(urls), len(queries))
        return urls
    except Exception as exc:
        logger.warning("[tavily-policy] failed: %s", exc)
        return []


def _ddg_fallback(condition: str, max_results: int) -> list[str]:
    try:
        from tools.scraper.search import search_urls, build_search_queries
        queries = build_search_queries(condition)
        return search_urls(queries, max_total=max_results, condition=condition)
    except Exception as exc:
        logger.warning("[ddg-fallback] failed: %s", exc)
        return []


def _ddg_fallback_multi(queries: list[str], max_results: int, condition: str = "") -> list[str]:
    try:
        from tools.scraper.search import search_urls
        return search_urls(queries, max_total=max_results, condition=condition)
    except Exception as exc:
        logger.warning("[ddg-fallback-policy] failed: %s", exc)
        return []
