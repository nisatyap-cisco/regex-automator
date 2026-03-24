"""Redis-backed cache for expensive operations (search, LLM, scraping).

Falls back gracefully to a no-op when Redis is unavailable, so the pipeline
always works — just without deduplication.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_redis_checked = False
_DEFAULT_TTL = 7 * 24 * 3600  # 7 days


def _get_redis():
    """Lazy-connect to Redis.  Returns None if unavailable."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        client.ping()
        _redis_client = client
        logger.info("Cache: Redis connected at %s", url)
    except Exception as exc:
        logger.info("Cache: Redis unavailable (%s) — running without cache", exc)
        _redis_client = None
    return _redis_client


def _make_key(namespace: str, raw: str) -> str:
    """Deterministic cache key: namespace:sha256(raw)."""
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"rgx:{namespace}:{digest}"


def cache_get(namespace: str, key_material: str) -> Optional[Any]:
    """Retrieve a cached value.  Returns None on miss or Redis down."""
    r = _get_redis()
    if r is None:
        return None
    key = _make_key(namespace, key_material)
    try:
        raw = r.get(key)
        if raw is not None:
            logger.debug("Cache HIT %s", key)
            return json.loads(raw)
    except Exception as exc:
        logger.debug("Cache get error: %s", exc)
    return None


def cache_set(
    namespace: str,
    key_material: str,
    value: Any,
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Store a value in cache.  Silently no-ops if Redis is down."""
    r = _get_redis()
    if r is None:
        return
    key = _make_key(namespace, key_material)
    try:
        r.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        logger.debug("Cache SET %s (ttl=%ds)", key, ttl)
    except Exception as exc:
        logger.debug("Cache set error: %s", exc)


def cache_stats() -> dict:
    """Return basic cache stats for diagnostics."""
    r = _get_redis()
    if r is None:
        return {"status": "unavailable"}
    try:
        info = r.info("keyspace")
        keys = sum(v.get("keys", 0) for v in info.values() if isinstance(v, dict))
        return {"status": "connected", "keys": keys}
    except Exception:
        return {"status": "error"}
