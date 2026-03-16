"""Playwright + Camoufox stealth browser scraper.

Provides a single function to fetch a page's HTML using a real browser
with anti-detection fingerprinting, optional rotating proxy, and
geo-matched locale/timezone.

Each Camoufox call is dispatched to a **fresh daemon thread** so there
is never a leftover asyncio event loop from a prior run (Camoufox /
Playwright leave internal loops attached to the thread, which causes
"Sync API inside asyncio loop" on reuse).

A **circuit breaker** tracks consecutive Camoufox failures.  After
_CIRCUIT_THRESHOLD consecutive failures the breaker opens and all
subsequent URLs are routed straight to httpx for the rest of the
process lifetime — avoiding minutes of wasted retry time when the
browser simply can't reach the network.
"""
from __future__ import annotations

import logging
import queue
import random
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Circuit breaker state ────────────────────────────────────────────────────
_CIRCUIT_THRESHOLD = 2          # trip after 2 consecutive Camoufox failures
_camoufox_consecutive_fails = 0
_camoufox_circuit_open = False


def _trip_circuit() -> None:
    global _camoufox_consecutive_fails, _camoufox_circuit_open
    _camoufox_consecutive_fails += 1
    if _camoufox_consecutive_fails >= _CIRCUIT_THRESHOLD and not _camoufox_circuit_open:
        _camoufox_circuit_open = True
        logger.warning(
            "[browser] circuit breaker OPEN — Camoufox failed %d times in a row; "
            "routing all remaining URLs directly to httpx",
            _camoufox_consecutive_fails,
        )


def _reset_circuit() -> None:
    global _camoufox_consecutive_fails, _camoufox_circuit_open
    if _camoufox_consecutive_fails > 0:
        _camoufox_consecutive_fails = 0
        if _camoufox_circuit_open:
            _camoufox_circuit_open = False
            logger.info("[browser] circuit breaker CLOSED — Camoufox recovered")


# ── Firefox network prefs (fix DNS issues in spawned browser) ────────────────
_FIREFOX_DNS_PREFS = {
    "network.dns.disableIPv6": True,
    "network.trr.mode": 0,
    "network.dns.disablePrefetch": True,
    "network.prefetch-next": False,
    "network.http.speculative-parallel-limit": 0,
}


def _scrape_page_impl(
    url: str,
    proxy: Optional[str],
    headless: bool,
    timeout_ms: int,
    result_q: queue.Queue,
) -> None:
    """Run in a fresh thread — guaranteed clean asyncio state."""
    try:
        from camoufox.sync_api import Camoufox

        proxy_cfg = {"server": proxy} if proxy else None
        geoip_val: object = True if proxy else False

        with Camoufox(
            headless=headless,
            proxy=proxy_cfg,
            geoip=geoip_val,
            humanize=True,
            block_images=True,
            firefox_user_prefs=_FIREFOX_DNS_PREFS,
        ) as browser:
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(1500)
                html = page.content()
            finally:
                page.close()

        logger.info("[browser] fetched %s (%d chars)", url, len(html))
        result_q.put(("ok", html))
    except Exception as exc:
        result_q.put(("err", exc))


def scrape_page(
    url: str,
    proxy: Optional[str] = None,
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> str:
    """Fetch *url* with Camoufox and return the full page HTML.

    Spawns a fresh daemon thread per call so Playwright's sync API never
    sees a pre-existing asyncio event loop.  Retries once on transient
    errors.  Raises on permanent failure.
    """
    import time

    last_exc: Optional[Exception] = None
    for attempt in range(1, 3):                          # max 2 attempts
        result_q: queue.Queue = queue.Queue()
        t = threading.Thread(
            target=_scrape_page_impl,
            args=(url, proxy, headless, timeout_ms, result_q),
            daemon=True,
        )
        t.start()
        deadline_s = timeout_ms / 1000 + 15
        t.join(timeout=deadline_s)

        if result_q.empty():
            last_exc = TimeoutError(
                f"Camoufox timed out after {deadline_s:.0f}s on {url}"
            )
        else:
            status, val = result_q.get_nowait()
            if status == "ok":
                _reset_circuit()
                return val
            last_exc = val

        if attempt < 2:
            logger.info(
                "[browser] attempt %d/2 failed on %s — retrying in 3s", attempt, url,
            )
            time.sleep(3)

    _trip_circuit()
    raise last_exc  # type: ignore[misc]


def _httpx_fetch_all(urls: list[str]) -> list[tuple[str, str]]:
    """Fetch all URLs with httpx. Returns (url, html) pairs for successes."""
    results: list[tuple[str, str]] = []
    try:
        from tools.scraper.fetch import fetch_text, make_client
        client = make_client()
        for url in urls:
            try:
                html = fetch_text(client, url)
                if html and len(html) > 200:
                    results.append((url, html))
                    logger.info("[browser] httpx OK: %s (%d chars)", url, len(html))
                else:
                    logger.warning("[browser] httpx empty for %s", url)
            except Exception as exc:
                logger.warning("[browser] httpx failed on %s: %s", url, exc)
        client.close()
    except ImportError:
        logger.warning("[browser] httpx not available")
    return results


def scrape_pages(
    urls: list[str],
    proxies: Optional[list[str]] = None,
    headless: bool = True,
    sleep_s: float = 2.0,
) -> list[tuple[str, str]]:
    """Scrape multiple URLs, returning (url, html) pairs.

    If the Camoufox circuit breaker is open, skips the browser entirely
    and goes straight to httpx — saving minutes of wasted retries.
    """
    import time

    if _camoufox_circuit_open:
        logger.info(
            "[browser] circuit open — fetching %d URLs directly via httpx", len(urls),
        )
        return _httpx_fetch_all(urls)

    results: list[tuple[str, str]] = []
    failed_urls: list[str] = []

    for i, url in enumerate(urls):
        if _camoufox_circuit_open:
            failed_urls.extend(urls[i:])
            break
        proxy = random.choice(proxies) if proxies else None
        try:
            html = scrape_page(url, proxy=proxy, headless=headless)
            results.append((url, html))
        except Exception as exc:
            logger.warning("[browser] Camoufox failed on %s: %s", url, exc)
            failed_urls.append(url)
        if i < len(urls) - 1 and not _camoufox_circuit_open:
            time.sleep(sleep_s)

    if failed_urls:
        logger.info("[browser] retrying %d failed URLs with httpx", len(failed_urls))
        results.extend(_httpx_fetch_all(failed_urls))

    return results


def pick_proxy(config: dict) -> Optional[str]:
    """Select a random proxy from the config list, or None."""
    proxies = config.get("proxies", [])
    return random.choice(proxies) if proxies else None
