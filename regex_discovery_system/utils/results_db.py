"""DynamoDB + Redis backed persistent storage for pipeline run results.

Writes go to both DynamoDB (durable) and Redis (fast reads).
Reads try Redis first, then DynamoDB, then caller falls back to disk.
Falls back gracefully when either service is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "regex_discovery_runs")
_REDIS_PREFIX = "rgx:results:"

_table = None
_table_checked = False

_ARTIFACT_KEYS = (
    "patterns",
    "regex_patterns",
    "validation_report",
    "source_of_truth",
    "final_report",
    "compare",
    "logs",
    "notes",
    "debug_log",
)

_MAX_ITEM_BYTES = 390_000  # stay under DynamoDB 400 KB limit


def _get_table(config: Optional[dict] = None):
    """Lazy-connect to DynamoDB table, creating it if necessary."""
    global _table, _table_checked
    if _table_checked:
        return _table
    _table_checked = True
    try:
        from utils.bedrock_client import _build_session, load_config
        cfg = config or load_config("config.yaml")
        session = _build_session(cfg)
        ddb = session.resource("dynamodb", region_name=cfg.get("region", "us-east-1"))

        try:
            tbl = ddb.Table(TABLE_NAME)
            tbl.load()
            logger.info("ResultsDB: connected to DynamoDB table '%s'", TABLE_NAME)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info("ResultsDB: creating table '%s' …", TABLE_NAME)
                tbl = ddb.create_table(
                    TableName=TABLE_NAME,
                    KeySchema=[{"AttributeName": "slug", "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": "slug", "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST",
                )
                tbl.wait_until_exists()
                logger.info("ResultsDB: table '%s' created", TABLE_NAME)
            else:
                raise

        _table = tbl
    except Exception as exc:
        logger.warning("ResultsDB: DynamoDB unavailable (%s) — running without DB", exc)
        _table = None
    return _table


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truncate_item(item: dict) -> dict:
    """If the serialised item exceeds the DynamoDB limit, drop debug_log first,
    then truncate the largest remaining field."""
    raw = json.dumps(item, ensure_ascii=False).encode("utf-8")
    if len(raw) <= _MAX_ITEM_BYTES:
        return item

    if "debug_log" in item and item["debug_log"]:
        item["debug_log"] = "(truncated — too large for DynamoDB)"
        raw = json.dumps(item, ensure_ascii=False).encode("utf-8")
        if len(raw) <= _MAX_ITEM_BYTES:
            return item

    largest_key = max(
        (k for k in _ARTIFACT_KEYS if k in item and item[k]),
        key=lambda k: len(str(item[k])),
        default=None,
    )
    if largest_key:
        item[largest_key] = "(truncated — too large for DynamoDB)"
    return item


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _get_redis():
    """Reuse the Redis connection from the cache module."""
    try:
        from utils.cache import _get_redis as _cache_redis
        return _cache_redis()
    except Exception:
        return None


def _redis_save_run(slug: str, item: dict) -> bool:
    """Store full run item in Redis as a JSON string (no TTL — permanent)."""
    r = _get_redis()
    if r is None:
        return False
    try:
        key = f"{_REDIS_PREFIX}{slug}"
        r.set(key, json.dumps(item, ensure_ascii=False))
        meta = {
            "slug": item.get("slug", ""),
            "condition": item.get("condition", ""),
            "status": item.get("status", ""),
            "overall": item.get("overall", ""),
            "updated_at": item.get("updated_at", ""),
            "data_source": item.get("data_source", ""),
        }
        r.hset(f"{_REDIS_PREFIX}_index", slug, json.dumps(meta))
        logger.debug("Redis: saved run '%s'", slug)
        return True
    except Exception as exc:
        logger.debug("Redis: save_run failed for '%s': %s", slug, exc)
        return False


def _redis_list_runs() -> Optional[list[dict]]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.hgetall(f"{_REDIS_PREFIX}_index")
        if not raw:
            return None
        runs = []
        for slug, meta_json in raw.items():
            meta = json.loads(meta_json)
            meta.setdefault("file", f"{meta.get('slug', slug)}_final_report.md")
            runs.append(meta)
        runs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return runs
    except Exception as exc:
        logger.debug("Redis: list_runs failed: %s", exc)
        return None


def _redis_get_run_files(slug: str) -> Optional[dict]:
    r = _get_redis()
    if r is None:
        return None
    try:
        raw = r.get(f"{_REDIS_PREFIX}{slug}")
        if raw is None:
            return None
        item = json.loads(raw)
        files: dict[str, str] = {}
        for key in _ARTIFACT_KEYS:
            val = item.get(key, "")
            if val:
                files[key] = val
        return files if files else None
    except Exception as exc:
        logger.debug("Redis: get_run_files failed for '%s': %s", slug, exc)
        return None


def _redis_delete_run(slug: str) -> bool:
    r = _get_redis()
    if r is None:
        return False
    try:
        r.delete(f"{_REDIS_PREFIX}{slug}")
        r.hdel(f"{_REDIS_PREFIX}_index", slug)
        return True
    except Exception:
        return False


# ── CRUD ──────────────────────────────────────────────────────────────────────

def save_run(
    slug: str,
    condition: str,
    files_dict: dict,
    status: str = "done",
    overall: str = "",
    logs: Optional[list[str]] = None,
    data_source: str = "",
    config: Optional[dict] = None,
) -> bool:
    """Upsert a complete run into DynamoDB + Redis.  Returns True if at least one succeeds."""
    existing = _get_raw_item(slug)
    created_at = (existing or {}).get("created_at", _now_iso())

    item: dict[str, Any] = {
        "slug": slug,
        "condition": condition,
        "status": status,
        "overall": overall,
        "data_source": data_source,
        "created_at": created_at,
        "updated_at": _now_iso(),
    }
    for key in _ARTIFACT_KEYS:
        val = files_dict.get(key, "")
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False, indent=2)
        item[key] = val or ""

    if logs is not None:
        item["logs"] = json.dumps(logs, ensure_ascii=False)

    redis_ok = _redis_save_run(slug, item)

    ddb_ok = False
    tbl = _get_table(config)
    if tbl is not None:
        try:
            ddb_item = _truncate_item(dict(item))
            tbl.put_item(Item=ddb_item)
            ddb_ok = True
            logger.info("ResultsDB: saved run '%s' to DynamoDB", slug)
        except Exception as exc:
            logger.warning("ResultsDB: DynamoDB save failed for '%s': %s", slug, exc)

    if redis_ok:
        logger.info("ResultsDB: saved run '%s' to Redis", slug)

    return ddb_ok or redis_ok


def list_runs(config: Optional[dict] = None) -> Optional[list[dict]]:
    """Return lightweight metadata for all runs.  Tries Redis → DynamoDB → None."""
    redis_runs = _redis_list_runs()
    if redis_runs:
        return redis_runs

    tbl = _get_table(config)
    if tbl is None:
        return None
    try:
        resp = tbl.scan(
            ProjectionExpression="slug, #cond, #st, overall, updated_at, data_source",
            ExpressionAttributeNames={
                "#cond": "condition",
                "#st": "status",
            },
        )
        items = resp.get("Items", [])
        while resp.get("LastEvaluatedKey"):
            resp = tbl.scan(
                ProjectionExpression="slug, #cond, #st, overall, updated_at, data_source",
                ExpressionAttributeNames={"#cond": "condition", "#st": "status"},
                ExclusiveStartKey=resp["LastEvaluatedKey"],
            )
            items.extend(resp.get("Items", []))

        runs = []
        for it in sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True):
            runs.append({
                "slug": it.get("slug", ""),
                "condition": it.get("condition", ""),
                "status": it.get("status", ""),
                "overall": it.get("overall", ""),
                "mtime": it.get("updated_at", ""),
                "data_source": it.get("data_source", ""),
            })
        return runs
    except Exception as exc:
        logger.warning("ResultsDB: list_runs failed: %s", exc)
        return None


def _get_raw_item(slug: str) -> Optional[dict]:
    tbl = _get_table()
    if tbl is None:
        return None
    try:
        resp = tbl.get_item(Key={"slug": slug})
        return resp.get("Item")
    except Exception:
        return None


def get_run_files(slug: str, config: Optional[dict] = None) -> Optional[dict]:
    """Return all artifacts for a slug.  Tries Redis → DynamoDB → None."""
    redis_files = _redis_get_run_files(slug)
    if redis_files:
        return redis_files

    tbl = _get_table(config)
    if tbl is None:
        return None
    try:
        resp = tbl.get_item(Key={"slug": slug})
        item = resp.get("Item")
        if not item:
            return None

        files: dict[str, str] = {}
        for key in _ARTIFACT_KEYS:
            val = item.get(key, "")
            if val:
                files[key] = val
        return files if files else None
    except Exception as exc:
        logger.warning("ResultsDB: get_run_files failed for '%s': %s", slug, exc)
        return None


def save_notes(slug: str, text: str, config: Optional[dict] = None) -> bool:
    tbl = _get_table(config)
    if tbl is None:
        return False
    try:
        tbl.update_item(
            Key={"slug": slug},
            UpdateExpression="SET notes = :n, updated_at = :u",
            ExpressionAttributeValues={":n": text, ":u": _now_iso()},
        )
        return True
    except Exception as exc:
        logger.warning("ResultsDB: save_notes failed for '%s': %s", slug, exc)
        return False


def get_notes(slug: str, config: Optional[dict] = None) -> Optional[str]:
    tbl = _get_table(config)
    if tbl is None:
        return None
    try:
        resp = tbl.get_item(Key={"slug": slug}, ProjectionExpression="notes")
        item = resp.get("Item")
        return item.get("notes", "") if item else None
    except Exception as exc:
        logger.warning("ResultsDB: get_notes failed for '%s': %s", slug, exc)
        return None


def save_logs(slug: str, logs: list[str], config: Optional[dict] = None) -> bool:
    tbl = _get_table(config)
    if tbl is None:
        return False
    try:
        tbl.update_item(
            Key={"slug": slug},
            UpdateExpression="SET logs = :l, updated_at = :u",
            ExpressionAttributeValues={
                ":l": json.dumps(logs, ensure_ascii=False),
                ":u": _now_iso(),
            },
        )
        return True
    except Exception as exc:
        logger.warning("ResultsDB: save_logs failed for '%s': %s", slug, exc)
        return False


def get_logs(slug: str, config: Optional[dict] = None) -> Optional[list[str]]:
    tbl = _get_table(config)
    if tbl is None:
        return None
    try:
        resp = tbl.get_item(Key={"slug": slug}, ProjectionExpression="logs")
        item = resp.get("Item")
        if item and item.get("logs"):
            return json.loads(item["logs"])
        return None
    except Exception as exc:
        logger.warning("ResultsDB: get_logs failed for '%s': %s", slug, exc)
        return None


def delete_run(slug: str, config: Optional[dict] = None) -> bool:
    _redis_delete_run(slug)
    tbl = _get_table(config)
    if tbl is None:
        return False
    try:
        tbl.delete_item(Key={"slug": slug})
        logger.info("ResultsDB: deleted run '%s'", slug)
        return True
    except Exception as exc:
        logger.warning("ResultsDB: delete_run failed for '%s': %s", slug, exc)
        return False


# ── Bulk import from disk ─────────────────────────────────────────────────────

def import_from_disk(
    results_dir: Path,
    data_dir: Optional[Path] = None,
    logs_dir: Optional[Path] = None,
    notes_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> dict:
    """Scan results/ on disk and upsert every condition into DynamoDB + Redis.

    Returns {"imported": N, "skipped": N, "errors": [...]}.
    Works with Redis alone if DynamoDB is unavailable.
    """
    redis_available = _get_redis() is not None
    ddb_available = _get_table(config) is not None
    if not redis_available and not ddb_available:
        return {"imported": 0, "skipped": 0, "errors": ["Both DynamoDB and Redis unavailable"]}

    stats = {"imported": 0, "skipped": 0, "errors": []}

    seen_slugs: set[str] = set()
    for p in sorted(results_dir.glob("*_final_report.md")):
        slug = p.stem.replace("_final_report", "")
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        condition = slug.replace("_", " ")

        files_dict: dict[str, str] = {}
        file_map = {
            "patterns": f"{slug}_patterns.json",
            "regex_patterns": f"{slug}_regex_patterns.json",
            "validation_report": f"{slug}_validation_report.json",
            "source_of_truth": f"{slug}_source_of_truth.json",
            "final_report": f"{slug}_final_report.md",
        }
        for key, fname in file_map.items():
            fp = results_dir / fname
            if fp.exists():
                files_dict[key] = fp.read_text(errors="replace")

        for candidate in results_dir.glob("*[Cc]ompare*"):
            cname = candidate.stem.lower()
            if slug.lower() in cname:
                files_dict["compare"] = candidate.read_text(errors="replace")
                break

        if logs_dir and logs_dir.exists():
            debug_path = logs_dir / f"debug_{slug}.log"
            if debug_path.exists():
                files_dict["debug_log"] = debug_path.read_text(errors="replace")

        if notes_dir and notes_dir.exists():
            note_path = notes_dir / f"{slug}.txt"
            if note_path.exists():
                files_dict["notes"] = note_path.read_text(errors="replace")

        overall = ""
        data_source = ""
        vr_raw = files_dict.get("validation_report", "")
        if vr_raw:
            try:
                vr = json.loads(vr_raw)
                overall = vr.get("overall_status", "")
                data_source = vr.get("data_source", "")
            except Exception:
                pass
        if not data_source:
            sot_raw = files_dict.get("source_of_truth", "")
            if sot_raw:
                try:
                    sot = json.loads(sot_raw)
                    data_source = sot.get("data_source", "")
                except Exception:
                    pass

        ok = save_run(
            slug=slug,
            condition=condition,
            files_dict=files_dict,
            status="done",
            overall=overall,
            data_source=data_source,
            config=config,
        )
        if ok:
            stats["imported"] += 1
        else:
            stats["errors"].append(slug)

    return stats
