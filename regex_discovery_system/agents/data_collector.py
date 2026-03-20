"""Agent 1 — Data Collector (DB + web scrape only, no LLM synthesis).

Sourcing priority:
  1. db lookup    — checks db/db_index.json for a matching reference file
  2. scrape_tool  — web scraper (adapters → DuckDuckGo search + LLM hints)

If neither source returns data the collector returns an empty result and the
pipeline proceeds with policy-only regex generation.  LLM synthesis is
intentionally removed to avoid the synthetic-data bias problems it causes.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)

SEED = 42
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_lines(path: str, lines: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _split_and_write(unique: list[str], source: str, paths: dict[str, str]) -> dict:
    rng = random.Random(SEED)
    rng.shuffle(unique)

    raw_path = paths.get("raw", "data/raw_examples.txt")
    train_path = paths.get("train", "data/train.txt")
    test_path = paths.get("test", "data/test.txt")

    train = list(unique)
    test = list(unique)
    logger.info(
        "Split (ground truth): %d unique → both train & test get all %d values",
        len(unique), len(unique),
    )

    _write_lines(raw_path, unique)
    _write_lines(train_path, train)
    _write_lines(test_path, test)

    return {"total": len(unique), "train": len(train), "test": len(test)}


# ── source 2: scrape tool (adapters → Tavily+Playwright → DDG+httpx) ────────

def _try_scrape_tool(condition: str, config: dict) -> list[str] | None:
    """Call the general-purpose scraper tool.

    Tries in order:
      a) Registered site adapters (e.g. SWIFT code sites)
      b) Tavily search + Playwright/Camoufox stealth scrape
      c) DuckDuckGo search + httpx fallback
    Returns None if nothing useful is found.

    When data is successfully scraped, it is also persisted as an Excel file
    under ``data/<slug>_scraped.xlsx`` for downstream agents.
    """
    try:
        from tools.scraper import scrape
    except ImportError as exc:
        logger.info("Agent 1 [scrape_tool]: scraper not available (%s) — skipping", exc)
        return None

    logger.info("Agent 1 [scrape_tool]: trying adapters + Tavily/Playwright for '%s'", condition)
    values = scrape(topic=condition, config=config)

    if values:
        logger.info("Agent 1 [scrape_tool]: scraped %d values for '%s'", len(values), condition)
        _save_scraped_excel(values, condition)
        return values

    logger.info("Agent 1 [scrape_tool]: no data found for '%s' — skipping", condition)
    return None


def _save_scraped_excel(values: list[str], condition: str) -> None:
    """Best-effort persistence of scraped values as xlsx."""
    try:
        from tools.scraper.excel_writer import write_to_excel
        slug = condition.lower().replace(" ", "_")
        pairs = [(v, "scraped") for v in values]
        write_to_excel(pairs, slug, "data")
    except Exception as exc:
        logger.warning("Agent 1 [scrape_tool]: failed to write excel: %s", exc)


# ── source 1: db lookup ──────────────────────────────────────────────────────

def _match_score(condition: str, keywords: list[str]) -> int:
    """Count how many keywords appear (case-insensitive) in the condition."""
    cond_lower = condition.lower()
    return sum(1 for kw in keywords if kw.lower() in cond_lower)


def _try_db_lookup(condition: str) -> tuple[list[str] | None, str | None]:
    """Search db/db_index.json for a file whose keywords match the condition.

    Returns (values_list, description) or (None, None).
    """
    index_path = os.path.join(DB_DIR, "db_index.json")
    if not os.path.isfile(index_path):
        logger.info("Agent 1 [db]: no db_index.json found — skipping")
        return None, None

    with open(index_path) as f:
        index = json.load(f)

    best_entry = None
    best_score = 0
    for entry in index.get("entries", []):
        score = _match_score(condition, entry.get("keywords", []))
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is None or best_score == 0:
        logger.info("Agent 1 [db]: no matching entry for '%s'", condition)
        return None, None

    file_path = os.path.join(DB_DIR, best_entry["file"])
    if not os.path.isfile(file_path):
        logger.warning("Agent 1 [db]: file '%s' referenced but not found", file_path)
        return None, None

    fmt = best_entry.get("format", "").lower()
    col = best_entry.get("column", None)
    desc = best_entry.get("description", "")

    values: list[str] = []

    if fmt == "xlsx":
        try:
            import openpyxl
        except ImportError:
            logger.warning("Agent 1 [db]: openpyxl not installed — cannot read xlsx")
            return None, None

        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.active
        headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        col_idx = 0
        if col and col in headers:
            col_idx = headers.index(col)

        for row in ws.iter_rows(min_row=2, max_col=col_idx + 1):
            cell = row[col_idx]
            if cell.value is not None:
                values.append(str(cell.value).strip())
        wb.close()

    elif fmt in ("txt", "csv"):
        with open(file_path) as f:
            for line in f:
                v = line.strip()
                if v:
                    values.append(v)
    else:
        logger.warning("Agent 1 [db]: unsupported format '%s'", fmt)
        return None, None

    seen: set[str] = set()
    unique: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    logger.info(
        "Agent 1 [db]: matched '%s' → %s (%d values) — %s",
        condition, best_entry["file"], len(unique), desc,
    )
    return unique, desc


# ── public API ────────────────────────────────────────────────────────────────

def collect_data(condition: str, config: dict, paths: dict[str, str] | None = None) -> dict:
    """Collect real-world data from DB or web scraping.  No LLM synthesis.

    Returns a stats dict with keys: total, train, test, source.
    If no data is found, source will be "none" and counts will be 0.
    """
    paths = paths or {}

    # 1. DB lookup (fastest, ground truth if available)
    values, db_desc = _try_db_lookup(condition)
    source = "db"

    # 2. Scrape tool (adapters → Tavily/Playwright → DuckDuckGo + httpx)
    if values is None:
        values = _try_scrape_tool(condition, config)
        source = "scrape_tool"

    # No LLM fallback — if neither source provides data we return empty
    if not values:
        logger.warning(
            "Agent 1: no data from DB or scraper for '%s' — "
            "pipeline will proceed with policy-only regex generation",
            condition,
        )
        return {"total": 0, "train": 0, "test": 0, "source": "none"}

    stats = _split_and_write(values, source, paths)
    stats["source"] = source
    logger.info("Agent 1 done: source=%s, %d unique values", source, stats["total"])
    return stats
