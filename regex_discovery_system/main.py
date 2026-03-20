#!/usr/bin/env python3
"""Universal Regex Discovery System — CLI entrypoint.

Usage:
    python main.py --condition "California ZIP code"
    python main.py --condition "US Medicare number"

Simplified pipeline (no data collection or validation):
    1. Policy Researcher — finds format rules from web
    2. Regex Generator — creates regex from policies
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

    os.makedirs("results", exist_ok=True)

    paths = build_paths(args.condition)

    # Install per-run debug log — every DEBUG message for this run goes here.
    run_log_handler = _install_run_log_handler(paths["debug_log"])
    logger.info("Per-run debug log → %s", paths["debug_log"])
    for key, path in paths.items():
        logger.info("  %s → %s", key, path)

    start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1/2 — Policy Researcher")
    logger.info("=" * 60)
    from agents.policy_researcher import research_policies
    policies = research_policies(args.condition, config)
    policy_count = len(policies.get("policies", []))
    logger.info("Policy Researcher: %d rules found (authority: %s)",
                policy_count, policies.get("numbering_authority", "unknown"))

    # Save source of truth
    from utils.paths import slugify
    slug = slugify(args.condition)
    sot = {
        "condition": args.condition,
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

    logger.info("=" * 60)
    logger.info("STEP 2/2 — Regex Generator")
    logger.info("=" * 60)
    from agents.regex_generator import generate_regex
    regex_result = generate_regex(
        args.condition,
        config,
        paths,
        policies=policies,
    )
    logger.info("Generated %d regex patterns", len(regex_result.get("patterns", [])))

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE in %.1fs", elapsed)
    logger.info("Regex patterns saved → %s", paths["regex_patterns"])
    logger.info("=" * 60)

    _remove_run_log_handler(run_log_handler)


if __name__ == "__main__":
    main()
