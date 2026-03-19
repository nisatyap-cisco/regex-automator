"""HTTP fetch with retries and exponential backoff."""
from __future__ import annotations

import logging
import ssl

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


class NonRetryableError(FetchError):
    """4xx errors that should not be retried."""
    pass


def _is_non_retryable(status: int) -> bool:
    return 400 <= status < 500 and status != 429


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, NonRetryableError):
        return False
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError, FetchError))


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=_should_retry,
)
def fetch_text(client: httpx.Client, url: str) -> str:
    logger.debug("Fetching %s", url)
    resp = client.get(url, follow_redirects=True)
    if _is_non_retryable(resp.status_code):
        raise NonRetryableError(f"HTTP {resp.status_code} for {url}")
    if resp.status_code != 200:
        raise FetchError(f"HTTP {resp.status_code} for {url}")
    content_type = resp.headers.get("content-type", "")
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return resp.content.decode("latin-1")
    return resp.text


def _make_legacy_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that tolerates older TLS servers (e.g. TLSv1.0)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
    ctx.minimum_version = ssl.TLSVersion.TLSv1
    return ctx


def make_client(timeout: float = 30.0, legacy_tls: bool = False) -> httpx.Client:
    verify: ssl.SSLContext | bool = _make_legacy_ssl_context() if legacy_tls else True
    return httpx.Client(
        headers=DEFAULT_HEADERS,
        timeout=httpx.Timeout(timeout),
        verify=verify,
    )
