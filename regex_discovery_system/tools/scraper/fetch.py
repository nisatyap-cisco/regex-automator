"""HTTP fetch with retries and exponential backoff."""
from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; regex-scraper/1.0)",
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


class FetchError(Exception):
    pass


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(
        (httpx.TimeoutException, httpx.TransportError, FetchError)
    ),
)
def fetch_text(client: httpx.Client, url: str) -> str:
    logger.debug("Fetching %s", url)
    resp = client.get(url, follow_redirects=True)
    if resp.status_code != 200:
        raise FetchError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(headers=DEFAULT_HEADERS, timeout=httpx.Timeout(timeout))
