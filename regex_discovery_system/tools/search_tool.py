"""URL discovery via Tavily (primary) with DuckDuckGo fallback.

Returns the top N URLs most likely to contain structured data lists
for a given condition.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_SEARCH_TTL = 7 * 24 * 3600  # 7 days

tavily_exhausted: bool = False
tavily_exhausted_msg: str = ""

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
    max_results: int = 20,
) -> list[str]:
    """Search for pages describing format rules / policies for *condition*.

    Casts a wide net with diverse query strategies (official specs, Wikipedia,
    standards bodies, validators) to collect up to *max_results* candidate URLs.
    Downstream ranking and deduplication happen in the policy researcher.
    """
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    queries = [
        f"{condition} format specification rules official",
        f"{condition} numbering policy allocation government authority",
        f"{condition} valid range structure check digit specification",
        f"{condition} format wikipedia",
        f"{condition} validation rules structure format guide",
        f"{condition} format rules digits length example",
        f"{condition} official format documentation",
        f"{condition} numbering system structure ISO standard",
        f"{condition} regex pattern format definition",
        f"{condition} identifier format allocation rules breakdown",
    ]

    if api_key:
        urls = _tavily_search_multi(queries, api_key, max_results)
        if urls:
            return urls

    return _ddg_fallback_multi(queries, max_results, condition=condition)


def _check_tavily_exhaustion(exc: Exception) -> bool:
    """Detect Tavily quota/rate-limit errors and set the global flag."""
    global tavily_exhausted, tavily_exhausted_msg
    err = str(exc).lower()
    if any(k in err for k in ("429", "rate limit", "quota", "exceeded", "exceeds",
                               "limit reached", "usage limit", "too many requests",
                               "insufficient credits", "api limit")):
        tavily_exhausted = True
        tavily_exhausted_msg = str(exc)
        logger.warning("=" * 60)
        logger.warning("  TAVILY API QUOTA EXHAUSTED: %s", exc)
        logger.warning("  All subsequent searches will use DuckDuckGo fallback.")
        logger.warning("  Consider stopping the run and retrying later.")
        logger.warning("=" * 60)
        return True
    return False


def _tavily_search(condition: str, api_key: str, max_results: int) -> list[str]:
    """Search for data pages, prioritising government and official sources."""
    global tavily_exhausted
    if tavily_exhausted:
        logger.info("[tavily] skipping — API quota previously exhausted")
        return []

    cache_key = f"tavily_data|{condition}|{max_results}"
    cached = cache_get("search", cache_key)
    if cached is not None:
        logger.info("[tavily] CACHE HIT for '%s' → %d URLs", condition, len(cached))
        return cached

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
        result = gov_first[:max_results]
        logger.info("[tavily] '%s' → %d URLs (gov-prioritised)", condition, len(result))
        cache_set("search", cache_key, result, ttl=_SEARCH_TTL)
        return result
    except Exception as exc:
        if _check_tavily_exhaustion(exc):
            return []
        logger.warning("[tavily] search failed: %s", exc)
        return []


def _tavily_search_multi(queries: list[str], api_key: str, max_results: int) -> list[str]:
    global tavily_exhausted
    if tavily_exhausted:
        logger.info("[tavily-multi] skipping — API quota previously exhausted")
        return []

    cache_key = f"tavily_multi|{'|'.join(sorted(queries))}|{max_results}"
    cached = cache_get("search", cache_key)
    if cached is not None:
        logger.info("[tavily-policy] CACHE HIT → %d URLs", len(cached))
        return cached

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key)
        seen: set[str] = set()
        urls: list[str] = []
        for q in queries:
            if len(urls) >= max_results:
                break
            resp = client.search(
                query=q,
                search_depth="advanced",
                max_results=5,
            )
            for r in resp.get("results", []):
                u = r["url"]
                if u not in seen and _is_useful_url(u):
                    seen.add(u)
                    urls.append(u)
                    if len(urls) >= max_results:
                        break
            logger.debug("[tavily-policy] query=%r → %d total unique URLs so far", q, len(urls))
        logger.info("[tavily-policy] %d URLs collected from %d queries", len(urls), len(queries))
        cache_set("search", cache_key, urls, ttl=_SEARCH_TTL)
        return urls
    except Exception as exc:
        if _check_tavily_exhaustion(exc):
            return []
        logger.warning("[tavily-policy] failed: %s", exc)
        return []


def _ddg_fallback(condition: str, max_results: int) -> list[str]:
    cache_key = f"ddg_data|{condition}|{max_results}"
    cached = cache_get("search", cache_key)
    if cached is not None:
        logger.info("[ddg] CACHE HIT for '%s' → %d URLs", condition, len(cached))
        return cached
    try:
        from tools.scraper.search import search_urls, build_search_queries
        queries = build_search_queries(condition)
        result = search_urls(queries, max_total=max_results, condition=condition)
        if result:
            cache_set("search", cache_key, result, ttl=_SEARCH_TTL)
        return result
    except Exception as exc:
        logger.warning("[ddg-fallback] failed: %s", exc)
        return []


def _ddg_fallback_multi(queries: list[str], max_results: int, condition: str = "") -> list[str]:
    cache_key = f"ddg_multi|{condition}|{'|'.join(sorted(queries))}|{max_results}"
    cached = cache_get("search", cache_key)
    if cached is not None:
        logger.info("[ddg-policy] CACHE HIT → %d URLs", len(cached))
        return cached
    try:
        from tools.scraper.search import search_urls
        result = search_urls(queries, max_total=max_results, condition=condition)
        if result:
            cache_set("search", cache_key, result, ttl=_SEARCH_TTL)
        return result
    except Exception as exc:
        logger.warning("[ddg-fallback-policy] failed: %s", exc)
        return []
