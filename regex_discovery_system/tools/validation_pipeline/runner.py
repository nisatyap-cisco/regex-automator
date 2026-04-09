"""Runner — execute the regex against generated test cases and produce verdicts.

Runs each positive/negative test case through the regex, reports PASS/FAIL per
case, computes aggregate metrics, and returns an overall verdict.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _strip_anchors(regex_str: str) -> str:
    """Strip \\b / ^ / $ anchors so the pattern can be used with fullmatch."""
    s = regex_str
    # Remove leading/trailing \b
    while s.startswith(r"\b"):
        s = s[2:]
    while s.endswith(r"\b"):
        s = s[:-2]
    # Remove leading ^ and trailing $
    if s.startswith("^"):
        s = s[1:]
    if s.endswith("$") and not s.endswith(r"\$"):
        s = s[:-1]
    return s


# ── single-case evaluation ────────────────────────────────────────────────────

def _evaluate_case(value: str, compiled_regex: re.Pattern, expected_match: bool) -> dict:
    """Test a single value and return a per-case verdict.

    Parameters
    ----------
    value : str
        The test input.
    compiled_regex : re.Pattern
        Pre-compiled regex pattern.
    expected_match : bool
        True if this value *should* match (positive case).

    Returns
    -------
    dict with keys: value, expected, actual, verdict
    """
    # Use fullmatch — the regex has anchors stripped so it tests the whole value
    actual_match = bool(compiled_regex.fullmatch(value))

    if expected_match and actual_match:
        verdict = "PASS"           # True positive
    elif not expected_match and not actual_match:
        verdict = "PASS"           # True negative
    elif expected_match and not actual_match:
        verdict = "FAIL"           # False negative — should have matched
    else:
        verdict = "FAIL"           # False positive — should NOT have matched

    return {
        "value": value,
        "expected": "MATCH" if expected_match else "NO_MATCH",
        "actual": "MATCH" if actual_match else "NO_MATCH",
        "verdict": verdict,
    }


# ── aggregate runner ──────────────────────────────────────────────────────────

def run_validation(test_cases: dict) -> dict:
    """Run regex validation on the generated test cases.

    Parameters
    ----------
    test_cases : dict
        Output of ``test_generator.generate_test_cases`` containing
        ``regex``, ``positive_cases``, ``negative_cases``, etc.

    Returns
    -------
    dict — full report with per-case verdicts and overall status.
    """
    regex_str = test_cases["regex"]
    rule_id = test_cases["rule_id"]
    condition = test_cases["condition"]
    description = test_cases.get("description", "")

    # Strip \b / ^ / $ anchors so fullmatch tests the whole value cleanly
    stripped_regex = _strip_anchors(regex_str)
    try:
        compiled = re.compile(stripped_regex)
    except re.error as exc:
        logger.error("Regex compile failed for '%s': %s", rule_id, exc)
        return {
            "condition": condition,
            "rule_id": rule_id,
            "regex": regex_str,
            "description": description,
            "error": f"Regex compile error: {exc}",
            "overall_verdict": "ERROR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    positive_cases = test_cases.get("positive_cases", [])
    negative_cases = test_cases.get("negative_cases", [])

    # ── evaluate positive cases (should MATCH) ──
    positive_results: list[dict] = []
    for val in positive_cases:
        result = _evaluate_case(val, compiled, expected_match=True)
        positive_results.append(result)

    # ── evaluate negative cases (should NOT match) ──
    negative_results: list[dict] = []
    for val in negative_cases:
        result = _evaluate_case(val, compiled, expected_match=False)
        negative_results.append(result)

    # ── aggregate metrics ──
    total_positive = len(positive_results)
    total_negative = len(negative_results)
    total = total_positive + total_negative

    positive_passed = sum(1 for r in positive_results if r["verdict"] == "PASS")
    negative_passed = sum(1 for r in negative_results if r["verdict"] == "PASS")
    total_passed = positive_passed + negative_passed

    positive_failed = total_positive - positive_passed
    negative_failed = total_negative - negative_passed
    total_failed = positive_failed + negative_failed

    # Classic metrics
    tp = positive_passed                       # correctly matched positives
    fn = positive_failed                       # missed positives
    tn = negative_passed                       # correctly rejected negatives
    fp = negative_failed                       # wrongly matched negatives

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    pass_rate = total_passed / max(total, 1)

    # ── overall verdict ──
    if pass_rate >= 0.95:
        overall = "PASS"
    elif pass_rate >= 0.85:
        overall = "NEEDS_REVIEW"
    else:
        overall = "FAIL"

    return {
        "condition": condition,
        "rule_id": rule_id,
        "regex": regex_str,
        "description": description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_cases": total,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": round(pass_rate, 4),
            "positive_total": total_positive,
            "positive_passed": positive_passed,
            "positive_failed": positive_failed,
            "negative_total": total_negative,
            "negative_passed": negative_passed,
            "negative_failed": negative_failed,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
        },
        "overall_verdict": overall,
        "positive_results": positive_results,
        "negative_results": negative_results,
    }


# ── pretty console output ────────────────────────────────────────────────────

def print_report(report: dict, verbose: bool = False) -> None:
    """Print a human-readable summary to stdout."""
    cond = report["condition"]
    rid = report["rule_id"]
    overall = report["overall_verdict"]
    s = report.get("summary", {})

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  VALIDATION REPORT — {cond}")
    print(f"  Rule: {rid}")
    print(f"  Regex: {report['regex'][:80]}{'…' if len(report['regex']) > 80 else ''}")
    print(f"  Description: {report.get('description', '')}")
    print(f"  Timestamp: {report.get('timestamp', '')}")
    print(sep)

    print(f"\n  {'METRIC':<25} {'VALUE':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'Total cases':<25} {s.get('total_cases', 0):>10}")
    print(f"  {'Total PASSED':<25} {s.get('total_passed', 0):>10}")
    print(f"  {'Total FAILED':<25} {s.get('total_failed', 0):>10}")
    print(f"  {'Pass rate':<25} {s.get('pass_rate', 0):>10.2%}")
    print()
    print(f"  {'Positive (should match)':<25} {s.get('positive_total', 0):>10}")
    print(f"    {'Passed':<23} {s.get('positive_passed', 0):>10}")
    print(f"    {'Failed':<23} {s.get('positive_failed', 0):>10}")
    print(f"  {'Negative (should not)':<25} {s.get('negative_total', 0):>10}")
    print(f"    {'Passed':<23} {s.get('negative_passed', 0):>10}")
    print(f"    {'Failed':<23} {s.get('negative_failed', 0):>10}")
    print()
    print(f"  {'Precision':<25} {s.get('precision', 0):>10.4f}")
    print(f"  {'Recall':<25} {s.get('recall', 0):>10.4f}")
    print(f"  {'Specificity':<25} {s.get('specificity', 0):>10.4f}")
    print(f"  {'Accuracy':<25} {s.get('accuracy', 0):>10.4f}")
    print(f"  {'F1 Score':<25} {s.get('f1', 0):>10.4f}")

    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  OVERALL VERDICT:  {overall:<18} │")
    print(f"  └──────────────────────────────────────┘\n")

    if verbose:
        # Show failed positive cases
        pos_fails = [r for r in report.get("positive_results", []) if r["verdict"] == "FAIL"]
        if pos_fails:
            print(f"  FAILED POSITIVE CASES (should have matched but didn't):")
            for r in pos_fails[:30]:
                print(f"    ✗  {r['value']}")
            if len(pos_fails) > 30:
                print(f"    ... and {len(pos_fails) - 30} more")
            print()

        # Show failed negative cases
        neg_fails = [r for r in report.get("negative_results", []) if r["verdict"] == "FAIL"]
        if neg_fails:
            print(f"  FAILED NEGATIVE CASES (should NOT have matched but did):")
            for r in neg_fails[:30]:
                print(f"    ✗  {r['value']}")
            if len(neg_fails) > 30:
                print(f"    ... and {len(neg_fails) - 30} more")
            print()


# ── save report to disk ──────────────────────────────────────────────────────

def save_report(report: dict, output_path: str) -> str:
    """Persist the full report as JSON.  Returns the path written."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved → %s", output_path)
    return output_path
