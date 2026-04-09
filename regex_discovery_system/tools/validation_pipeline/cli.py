#!/usr/bin/env python3
"""Validation Pipeline CLI — validate regex patterns with generated test cases.

Usage examples:

  # Validate the combined-value rule (default) for argentina passport number
  python -m tools.validation_pipeline.cli \\
      --patterns results/argentina_passport_number_regex_patterns.json

  # Validate a specific rule_id
  python -m tools.validation_pipeline.cli \\
      --patterns results/california_zip_code_regex_patterns.json \\
      --rule-id starts_with_9

  # Run all value rules in one go
  python -m tools.validation_pipeline.cli \\
      --patterns results/california_zip_code_regex_patterns.json \\
      --all

  # Use deterministic test-gen (no LLM calls)
  python -m tools.validation_pipeline.cli \\
      --patterns results/argentina_passport_number_regex_patterns.json \\
      --no-llm

  # Control case counts and verbosity
  python -m tools.validation_pipeline.cli \\
      --patterns results/ipv6_regex_patterns.json \\
      --positive-count 200 --negative-count 200 --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.bedrock_client import load_config
from tools.validation_pipeline.test_generator import generate_test_cases
from tools.validation_pipeline.runner import run_validation, print_report, save_report

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def _discover_rule_ids(patterns_path: str) -> list[str]:
    """Return all rule_ids in the patterns file (value type only)."""
    with open(patterns_path) as f:
        data = json.load(f)
    return [p["rule_id"] for p in data.get("patterns", []) if p.get("type") != "context"]


def _run_single(
    patterns_path: str,
    rule_id: str,
    config: dict,
    positive_count: int,
    negative_count: int,
    use_llm: bool,
    verbose: bool,
) -> dict:
    """Generate test cases + validate for a single rule. Returns the report dict."""
    logger.info("── Generating test cases for rule '%s' ──", rule_id)
    test_cases = generate_test_cases(
        regex_patterns_path=patterns_path,
        rule_id=rule_id,
        config=config,
        positive_count=positive_count,
        negative_count=negative_count,
        use_llm=use_llm,
    )

    logger.info("── Running validation for rule '%s' ──", rule_id)
    report = run_validation(test_cases)
    print_report(report, verbose=verbose)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate regex patterns by generating positive/negative test cases",
    )
    parser.add_argument(
        "--patterns",
        required=True,
        help="Path to the *_regex_patterns.json file",
    )
    parser.add_argument(
        "--rule-id",
        default=None,
        help='Rule ID to validate (default: all value rules)',
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Validate ALL value rules in the file (same as default when --rule-id is omitted)",
    )
    parser.add_argument(
        "--positive-count",
        type=int,
        default=40,
        help="Number of positive (should-match) test cases (default: 40)",
    )
    parser.add_argument(
        "--negative-count",
        type=int,
        default=40,
        help="Number of negative (should-not-match) test cases (default: 40)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM — use deterministic test-case generation only",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save JSON reports (default: results/validation_pipeline/)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-case failures in console output",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    config = load_config(args.config)

    patterns_path = os.path.abspath(args.patterns)
    if not os.path.exists(patterns_path):
        logger.error("Patterns file not found: %s", patterns_path)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(patterns_path), "validation_pipeline")
    os.makedirs(output_dir, exist_ok=True)

    use_llm = not args.no_llm

    if args.run_all or args.rule_id is None:
        rule_ids = _discover_rule_ids(patterns_path)
        logger.info("Validating ALL %d value rules: %s", len(rule_ids), rule_ids)
    else:
        rule_ids = [args.rule_id]

    all_reports: list[dict] = []
    for rid in rule_ids:
        report = _run_single(
            patterns_path=patterns_path,
            rule_id=rid,
            config=config,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            use_llm=use_llm,
            verbose=args.verbose,
        )
        all_reports.append(report)

    # ── summary (printed for multi-rule runs) ──
    if len(all_reports) > 1:
        print("\n" + "=" * 70)
        print("  BATCH SUMMARY")
        print("=" * 70)
        for r in all_reports:
            emoji = "✓" if r["overall_verdict"] == "PASS" else ("⚠" if r["overall_verdict"] == "NEEDS_REVIEW" else "✗")
            s = r.get("summary", {})
            print(f"  {emoji}  {r['rule_id']:<30} {r['overall_verdict']:<15} "
                  f"P={s.get('precision',0):.2f}  R={s.get('recall',0):.2f}  F1={s.get('f1',0):.2f}")

    # ── compute overall verdict ──
    verdicts = [r["overall_verdict"] for r in all_reports]
    if all(v == "PASS" for v in verdicts):
        overall_verdict = "PASS"
    elif any(v == "FAIL" for v in verdicts):
        overall_verdict = "FAIL"
    else:
        overall_verdict = "NEEDS_REVIEW"

    if len(all_reports) > 1:
        print(f"\n  Overall batch verdict: {overall_verdict}\n")

    # ── save single combined JSON ──
    slug = os.path.basename(patterns_path).replace("_regex_patterns.json", "")
    out_path = os.path.join(output_dir, f"{slug}_validation_pipeline.json")
    combined = {
        "condition": all_reports[0].get("condition", "") if all_reports else "",
        "patterns_file": patterns_path,
        "rules_validated": len(all_reports),
        "overall_verdict": overall_verdict,
        "results": all_reports,
    }
    save_report(combined, out_path)


if __name__ == "__main__":
    main()
