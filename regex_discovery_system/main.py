#!/usr/bin/env python3
"""Universal Regex Discovery System — CLI entrypoint.

Usage:
    python main.py --condition "California ZIP code"
    python main.py --condition "US Medicare number"

Pipeline (data + policy, policy-weighted):
    1. Data Collector + Policy Researcher  (parallel)
    2. Pattern Analyzer                    (only when real data is available)
    3. Regex Generator                     (policy-weighted: policies are primary,
                                            data patterns are supplementary)

Data is fetched from DB or web scraping only — no LLM synthesis.
If no data is found, the pipeline falls back to policy-only generation
(same as the feature/improvements branch).
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.bedrock_client import load_config
from utils.paths import build_paths

import logging.handlers as _lh

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(_log_formatter)

_file_handler = _lh.RotatingFileHandler(
    os.path.join(_LOG_DIR, "pipeline.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_console_handler, _file_handler],
)
logger = logging.getLogger("main")


def _install_run_log_handler(log_path: str) -> logging.FileHandler:
    """Add a per-run DEBUG file handler to the root logger."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_log_formatter)
    logging.getLogger().addHandler(handler)
    return handler


def _remove_run_log_handler(handler: logging.FileHandler) -> None:
    handler.flush()
    handler.close()
    logging.getLogger().removeHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover regex patterns for any identifier type",
    )
    parser.add_argument(
        "--condition",
        required=True,
        help='Identifier to analyze, e.g. "California ZIP code"',
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    paths = build_paths(args.condition)

    run_log_handler = _install_run_log_handler(paths["debug_log"])
    logger.info("Per-run debug log → %s", paths["debug_log"])
    for key, path in paths.items():
        logger.info("  %s → %s", key, path)

    start = time.time()

    # ── Step 1: Data Collector + Policy Researcher (parallel) ──
    logger.info("=" * 60)
    logger.info("STEP 1 — Data Collector + Policy Researcher (parallel)")
    logger.info("=" * 60)
    from concurrent.futures import ThreadPoolExecutor
    from agents.data_collector import collect_data
    from agents.policy_researcher import research_policies

    with ThreadPoolExecutor(max_workers=2) as pool:
        data_future = pool.submit(collect_data, args.condition, config, paths)
        policy_future = pool.submit(research_policies, args.condition, config)

    stats = data_future.result()
    policies = policy_future.result()
    data_source = stats.get("source", "none")
    has_data = stats.get("total", 0) > 0
    policy_count = len(policies.get("policies", []))

    logger.info(
        "Data: source=%s, %d values | Policies: %d rules (authority: %s)",
        data_source, stats.get("total", 0),
        policy_count, policies.get("numbering_authority", "unknown"),
    )

    from utils.paths import slugify
    slug = slugify(args.condition)
    sot = {
        "condition": args.condition,
        "data_source": data_source,
        "total_values": stats.get("total", 0),
        "policies": policies.get("policies", []),
        "numbering_authority": policies.get("numbering_authority", ""),
        "vendor_patterns": policies.get("vendor_patterns", []),
        "sources": policies.get("sources", []),
        "vendor_keywords": policies.get("vendor_keywords", []),
    }
    sot_path = f"results/{slug}_source_of_truth.json"
    with open(sot_path, "w") as f:
        json.dump(sot, f, indent=2)
    logger.info("Source-of-truth saved → %s", sot_path)

    # ── Step 2: Pattern Analyzer (only when real data exists) ──
    data_patterns = None
    if has_data:
        logger.info("=" * 60)
        logger.info("STEP 2 — Pattern Analyzer (real data available)")
        logger.info("=" * 60)
        from agents.pattern_analyzer import analyze_patterns
        data_patterns = analyze_patterns(
            args.condition, config, paths, policies=policies,
        )
        logger.info(
            "Discovered %d format rules from data",
            len(data_patterns.get("format_rules", [])),
        )
    else:
        logger.info("=" * 60)
        logger.info("STEP 2 — Pattern Analyzer SKIPPED (no real data collected)")
        logger.info("=" * 60)

    # ── Step 3: Regex Generator (policy-weighted) ──
    logger.info("=" * 60)
    if has_data:
        logger.info("STEP 3 — Regex Generator (policy-weighted: data + policy)")
    else:
        logger.info("STEP 3 — Regex Generator (policy-only, no data available)")
    logger.info("=" * 60)
    from agents.regex_generator import generate_regex
    regex_result = generate_regex(
        args.condition,
        config,
        paths,
        policies=policies,
        data_patterns=data_patterns,
    )
    logger.info("Generated %d regex patterns", len(regex_result.get("patterns", [])))

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(
        "DONE in %.1fs | Data: %s (%d values) | Policies: %d rules",
        elapsed, data_source, stats.get("total", 0), policy_count,
    )
    logger.info("Regex patterns saved → %s", paths["regex_patterns"])
    logger.info("=" * 60)

    _remove_run_log_handler(run_log_handler)


if __name__ == "__main__":
    main()
