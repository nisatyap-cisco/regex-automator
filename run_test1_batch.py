#!/usr/bin/env python3
"""Run the regex discovery pipeline for every identifier in test1.json
and aggregate results into results1.json."""

import argparse
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RDS_DIR = os.path.join(SCRIPT_DIR, "regex_discovery_system")
MAIN_PY = os.path.join(RDS_DIR, "main.py")
RESULTS_DIR = os.path.join(RDS_DIR, "results")
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python3")

TIMEOUT_SECONDS = 600

DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "test1.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "results1.json")


def slugify(condition: str) -> str:
    slug = condition.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def load_json_safe(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect_results(identifier: str) -> dict:
    """Read source_of_truth and regex_patterns for one identifier."""
    slug = slugify(identifier)
    sot_path = os.path.join(RESULTS_DIR, f"{slug}_source_of_truth.json")
    regex_path = os.path.join(RESULTS_DIR, f"{slug}_regex_patterns.json")

    sot_raw = load_json_safe(sot_path)
    regex_raw = load_json_safe(regex_path)

    source_of_truth = None
    if sot_raw:
        source_of_truth = {
            "sources": sot_raw.get("sources", []),
            "policies": sot_raw.get("policies", []),
            "vendor_patterns": sot_raw.get("vendor_patterns", []),
            "vendor_keywords": sot_raw.get("vendor_keywords", []),
            "numbering_authority": sot_raw.get("numbering_authority", ""),
            "policy_confidence": sot_raw.get("policy_confidence", ""),
            "policy_confidence_reason": sot_raw.get("policy_confidence_reason", ""),
        }

    regex_patterns = None
    if regex_raw:
        individual = []
        combined_value = None
        keyword_proximity = None
        for p in regex_raw.get("patterns", []):
            rid = p.get("rule_id", "")
            if rid == "combined-value":
                combined_value = {
                    "rule_id": rid,
                    "regex": p.get("regex", ""),
                    "description": p.get("description", ""),
                }
            elif rid == "keyword-proximity":
                keyword_proximity = {
                    "regex": p.get("regex", ""),
                    "keywords_readable": p.get("keywords_readable", ""),
                }
            elif p.get("type") == "value":
                individual.append({
                    "rule_id": rid,
                    "regex": p.get("regex", ""),
                    "description": p.get("description", ""),
                })
        regex_patterns = {
            "individual": individual,
            "combined_value": combined_value,
            "keyword_proximity": keyword_proximity,
        }

    return {
        "source_of_truth": source_of_truth,
        "regex_patterns": regex_patterns,
    }


def run_one(identifier: str, idx: int, total: int) -> bool:
    print(f"\n{'=' * 70}")
    print(f"[{idx}/{total}] Running: {identifier}")
    print(f"{'=' * 70}\n")
    start = time.time()
    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    result = subprocess.run(
        [python, MAIN_PY, "--condition", identifier],
        cwd=RDS_DIR,
        timeout=TIMEOUT_SECONDS,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0
    tag = "OK" if ok else f"FAIL (exit {result.returncode})"
    print(f"\n>>> [{idx}/{total}] {identifier}: {tag} in {elapsed:.1f}s\n")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run regex discovery pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="JSON file with list of identifier strings (default: test1.json)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSON file for aggregated results (default: results1.json)")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)

    with open(input_file) as f:
        identifiers = json.load(f)

    total = len(identifiers)
    print(f"Running pipeline for {total} identifiers from {input_file}\n")

    output: list[dict] = []
    overall_start = time.time()

    for i, identifier in enumerate(identifiers, 1):
        status = "OK"
        try:
            ok = run_one(identifier, i, total)
            if not ok:
                status = "FAIL"
        except subprocess.TimeoutExpired:
            print(f"\n>>> [{i}/{total}] {identifier}: TIMEOUT ({TIMEOUT_SECONDS}s)\n")
            status = "TIMEOUT"
        except Exception as e:
            print(f"\n>>> [{i}/{total}] {identifier}: ERROR — {e}\n")
            status = "ERROR"

        results = collect_results(identifier)
        output.append({
            "identifier": identifier,
            "status": status,
            "source_of_truth": results["source_of_truth"],
            "regex_patterns": results["regex_patterns"],
        })

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"BATCH COMPLETE — {total} runs in {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")
    print(f"Output saved to {output_file}")
    print(f"{'=' * 70}")
    for item in output:
        has_regex = "YES" if item["regex_patterns"] else "NO"
        print(f"  [{item['status']:>7}] {item['identifier']}  (regex: {has_regex})")


if __name__ == "__main__":
    main()
