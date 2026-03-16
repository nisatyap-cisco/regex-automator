#!/usr/bin/env python3
"""Universal Regex Discovery System — CLI entrypoint.

Usage:
    python main.py --condition "California ZIP code"
    python main.py --condition "US Medicare number" --dataset-size 1500
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

    paths = build_paths(args.condition)
    logger.info("Output files will use prefix: %s", list(paths.values())[0].split("/")[1].rsplit("_", 1)[0])
    for key, path in paths.items():
        logger.info("  %s → %s", key, path)

    start = time.time()

    logger.info("=" * 60)
    logger.info("STEP 1/4 — Data Collector + Policy Researcher (parallel)")
    logger.info("=" * 60)
    from concurrent.futures import ThreadPoolExecutor
    from agents.data_collector import collect_data
    from agents.policy_researcher import research_policies

    with ThreadPoolExecutor(max_workers=2) as pool:
        data_future = pool.submit(collect_data, args.condition, config, paths)
        policy_future = pool.submit(research_policies, args.condition, config)

    stats = data_future.result()
    policies = policy_future.result()
    data_source = stats.get("source", "unknown")
    policy_count = len(policies.get("policies", []))
    logger.info("Source: %s | Collected %d train / %d test examples", data_source, stats["train"], stats["test"])
    logger.info("Policy Researcher: %d rules found (authority: %s)",
                policy_count, policies.get("numbering_authority", "unknown"))

    sot = {
        "data_source": data_source,
        "total_values": stats.get("total", 0),
        "train_count": stats.get("train", 0),
        "test_count": stats.get("test", 0),
        "policies": policies.get("policies", []),
        "numbering_authority": policies.get("numbering_authority", ""),
        "vendor_patterns": policies.get("vendor_patterns", []),
        "sources": policies.get("sources", []),
        "vendor_keywords": policies.get("vendor_keywords", []),
    }
    slug = paths["raw"].split("/")[1].rsplit("_raw", 1)[0]
    sot_path = f"results/{slug}_source_of_truth.json"
    with open(sot_path, "w") as f:
        json.dump(sot, f, indent=2)
    logger.info("Source-of-truth saved → %s", sot_path)

    logger.info("=" * 60)
    logger.info("STEP 2/4 — Pattern Analyzer")
    logger.info("=" * 60)
    from agents.pattern_analyzer import analyze_patterns
    patterns = analyze_patterns(args.condition, config, paths, policies=policies)
    logger.info("Discovered %d rules", len(patterns.get("format_rules", [])))

    logger.info("=" * 60)
    logger.info("STEP 3/4 — Regex Generator")
    logger.info("=" * 60)
    from agents.regex_generator import generate_regex
    regex_result = generate_regex(config, paths)
    logger.info("Generated %d regex patterns", len(regex_result.get("patterns", [])))

    logger.info("=" * 60)
    logger.info("STEP 4/4 — Validator (synth 1000 candidates + classify + cross-check)")
    logger.info("=" * 60)
    from agents.validator import validate
    report = validate(config, paths, data_source=data_source, policies=policies)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("DONE in %.1fs — Overall: %s | Data source: %s", elapsed, report["overall_status"], data_source)
    logger.info("See %s for details.", paths["final_report"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
