"""URL discovery via Tavily (primary) with DuckDuckGo fallback.

Returns the top N URLs most likely to contain structured data lists
for a given condition.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

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
    max_results: int = 6,
) -> list[str]:
    """Search for pages describing format rules / policies for *condition*.

    Runs multiple query strategies to maximise coverage of format rules,
    allocation policies, structural specs, and validation logic.
    """
    api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")

    # Broad set of query angles targeting official specs, Wikipedia, validators,
    # and format-guide references — more queries = more rule coverage.
    queries = [
        f"{condition} format specification rules official",
        f"{condition} numbering policy allocation government authority",
        f"{condition} valid range structure check digit specification",
        f"{condition} format wikipedia",
        f"{condition} validation rules structure format guide",
        f"{condition} format rules digits length example",
        f"{condition} official format documentation",
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
    if any(k in err for k in ("429", "rate limit", "quota", "exceeded", "limit reached",
                               "too many requests", "insufficient credits", "api limit")):
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
        if _check_tavily_exhaustion(exc):
            return []
        logger.warning("[tavily] search failed: %s", exc)
        return []


def _tavily_search_multi(queries: list[str], api_key: str, max_results: int) -> list[str]:
    global tavily_exhausted
    if tavily_exhausted:
        logger.info("[tavily-multi] skipping — API quota previously exhausted")
        return []
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
                search_depth="advanced",   # deep search for richer rule pages
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
        return urls
    except Exception as exc:
        if _check_tavily_exhaustion(exc):
            return []
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
