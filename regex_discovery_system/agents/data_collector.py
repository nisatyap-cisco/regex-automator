"""Agent 1 — Data Collector (multi-source).

Sourcing priority:
  1. db lookup    — checks db/db_index.json for a matching reference file
  2. scrape_tool  — web scraper (adapters → DuckDuckGo search + LLM hints)
  3. LLM fallback — synthesises data via Bedrock Claude

Whichever source provides data, the output is normalised into:
  - raw_examples.txt   (all unique values)
  - train.txt          (ground-truth: 100% to both; LLM: 60/40 split)
  - test.txt

The collector also returns metadata so downstream agents know the source.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from pathlib import Path

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

SEED = 42
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")

PROMPT_TEMPLATE = """You are a data-generation assistant.
Generate {batch_size} UNIQUE, REALISTIC examples of: {condition}.
Rules:
- One value per line, no numbering, no bullet points, no extra text.
- Values must resemble real-world data actually in use.
- Do not repeat values from previous batches: {previous_values_sample}
Output ONLY the values."""


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

    if source in ("db", "scrape_tool"):
        train = list(unique)
        test = list(unique)
        logger.info("Split (ground truth): %d unique → both Agent 2 & Agent 4 get all %d values", len(unique), len(unique))
    else:
        split_idx = int(len(unique) * 0.60)
        train = unique[:split_idx]
        test = unique[split_idx:]
        logger.info("Split (LLM): %d unique → %d train (60%% → Agent 2) / %d test (40%% → Agent 4)",
                     len(unique), len(train), len(test))

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


# ── source 3: LLM synthesis (fallback) ────────────────────────────────────────

def _llm_synthesize(condition: str, config: dict) -> list[str]:
    dataset_size = config.get("dataset_size", 1000)
    batch_size = config.get("batch_size", 200)
    num_batches = math.ceil(dataset_size / batch_size)

    logger.info(
        "Agent 1 [llm]: synthesising %d examples in %d batches of %d",
        dataset_size, num_batches, batch_size,
    )

    all_values: list[str] = []
    for batch_idx in range(num_batches):
        sample_prev = ", ".join(all_values[-50:]) if all_values else "N/A (first batch)"
        prompt = PROMPT_TEMPLATE.format(
            batch_size=batch_size,
            condition=condition,
            previous_values_sample=sample_prev,
        )
        try:
            response = invoke_claude(prompt, config)
        except Exception as exc:
            logger.warning("Batch %d/%d failed: %s", batch_idx + 1, num_batches, exc)
            continue

        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        all_values.extend(lines)
        logger.info("Batch %d/%d: got %d (total raw: %d)", batch_idx + 1, num_batches, len(lines), len(all_values))

    seen: set[str] = set()
    unique: list[str] = []
    for v in all_values:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    if len(unique) < dataset_size:
        logger.warning("Only collected %d unique values (target %d)", len(unique), dataset_size)

    return unique


# ── public API ────────────────────────────────────────────────────────────────

def collect_data(condition: str, config: dict, paths: dict[str, str] | None = None) -> dict:
    paths = paths or {}

    # 1. DB lookup (fastest, ground truth if available)
    values, db_desc = _try_db_lookup(condition)
    source = "db"

    # 2. Scrape tool (adapters → DuckDuckGo + LLM-guided extraction)
    if values is None:
        values = _try_scrape_tool(condition, config)
        source = "scrape_tool"

    # 3. LLM synthesis (fallback)
    if values is None:
        values = _llm_synthesize(condition, config)
        source = "llm"

    if not values:
        logger.error("Agent 1: no data collected from any source")
        return {"total": 0, "train": 0, "test": 0, "source": "none"}

    stats = _split_and_write(values, source, paths)
    stats["source"] = source
    logger.info("Agent 1 done: source=%s, %d unique values", source, stats["total"])
    return stats
