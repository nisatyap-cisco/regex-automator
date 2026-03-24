#!/usr/bin/env python3
"""Batch runner — executes the full pipeline for every condition in regex_comparison.json."""

import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_FILE = os.path.join(SCRIPT_DIR, "results", "regex_comparison.json")
MAIN_PY = os.path.join(SCRIPT_DIR, "main.py")
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "bin", "python3")


def load_conditions() -> list[dict]:
    with open(COMPARISON_FILE) as f:
        return json.load(f)


def run_one(condition: str, idx: int, total: int) -> tuple[str, bool, float]:
    print(f"\n{'='*70}")
    print(f"[{idx}/{total}] Running: {condition}")
    print(f"{'='*70}\n")
    start = time.time()
    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    result = subprocess.run(
        [python, MAIN_PY, "--condition", condition],
        cwd=SCRIPT_DIR,
        timeout=600,
    )
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else f"FAIL (exit {result.returncode})"
    print(f"\n>>> [{idx}/{total}] {condition}: {status} in {elapsed:.1f}s\n")
    return condition, ok, elapsed


def main() -> None:
    entries = load_conditions()
    total = len(entries)
    print(f"Loaded {total} conditions from {COMPARISON_FILE}")

    results: list[tuple[str, bool, float]] = []
    overall_start = time.time()

    for i, entry in enumerate(entries, 1):
        cond = entry["condition"]
        try:
            cond, ok, elapsed = run_one(cond, i, total)
            results.append((cond, ok, elapsed))
        except subprocess.TimeoutExpired:
            print(f"\n>>> [{i}/{total}] {cond}: TIMEOUT (600s)\n")
            results.append((cond, False, 600.0))
        except Exception as e:
            print(f"\n>>> [{i}/{total}] {cond}: ERROR — {e}\n")
            results.append((cond, False, 0.0))

    total_elapsed = time.time() - overall_start

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE — {total} runs in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")
    ok_count = sum(1 for _, ok, _ in results if ok)
    fail_count = total - ok_count
    print(f"  Passed: {ok_count}  |  Failed: {fail_count}")
    for cond, ok, elapsed in results:
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag:>4}] {elapsed:6.1f}s  {cond}")


if __name__ == "__main__":
    main()
