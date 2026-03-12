#!/usr/bin/env python3
"""Universal Regex Discovery System — CLI entrypoint.

Usage:
    python main.py --condition "California ZIP code"
    python main.py --condition "US Medicare number" --dataset-size 1500
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.bedrock_client import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


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
        "--dataset-size",
        type=int,
        default=None,
        help="Number of examples to generate (default: from config.yaml)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_size is not None:
        config["dataset_size"] = args.dataset_size

    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1/4 — Data Collector")
    logger.info("=" * 60)
    from agents.data_collector import collect_data
    stats = collect_data(args.condition, config)
    logger.info("Collected %d train / %d test examples", stats["train"], stats["test"])

    logger.info("=" * 60)
    logger.info("STEP 2/4 — Pattern Analyzer")
    logger.info("=" * 60)
    from agents.pattern_analyzer import analyze_patterns
    patterns = analyze_patterns(args.condition, config)
    logger.info("Discovered %d rules", len(patterns.get("format_rules", [])))

    logger.info("=" * 60)
    logger.info("STEP 3/4 — Regex Generator")
    logger.info("=" * 60)
    from agents.regex_generator import generate_regex
    regex_result = generate_regex(config)
    logger.info("Generated %d regex patterns", len(regex_result.get("patterns", [])))

    logger.info("=" * 60)
    logger.info("STEP 4/4 — Validator")
    logger.info("=" * 60)
    from agents.validator import validate
    report = validate(config)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE in %.1fs — Overall: %s", elapsed, report["overall_status"])
    logger.info("See results/final_report.md for details.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
