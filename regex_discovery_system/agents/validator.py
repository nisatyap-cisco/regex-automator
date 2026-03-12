from __future__ import annotations

import json
import logging
import random
import re
import string
from datetime import datetime, timezone

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

MAX_REMEDIATION_ROUNDS = 2
NEGATIVE_SAMPLE_SIZE = 100

REMEDIATION_TEMPLATE = """You are a regex engineering expert.

The following regex was intended to match values for: {condition}

REGEX: {regex}
RULE: {description}

It FAILED to match these known-valid values:
{mismatches}

Fix the regex so it matches ALL the values above while still being precise.
Return ONLY the corrected regex string, no explanation."""


def _generate_negative_samples(values: list[str], count: int = NEGATIVE_SAMPLE_SIZE) -> list[str]:
    rng = random.Random(99)
    negatives: list[str] = []

    if values and values[0].isdigit():
        max_len = max(len(v) for v in values)
        for _ in range(count // 2):
            negatives.append("".join(rng.choices(string.digits, k=max_len)))
        for _ in range(count - count // 2):
            negatives.append("".join(rng.choices(string.ascii_letters, k=max_len)))
    else:
        avg_len = sum(len(v) for v in values) // max(len(values), 1)
        for _ in range(count):
            negatives.append("".join(rng.choices(string.ascii_letters + string.digits, k=avg_len)))

    return [n for n in negatives if n not in set(values)]


def _validate_value_pattern(
    pattern_entry: dict,
    test_values: list[str],
    negatives: list[str],
    config: dict,
) -> dict:
    rule_id = pattern_entry["rule_id"]
    regex_str = pattern_entry["regex"]
    description = pattern_entry.get("description", "")
    condition = config.get("_condition", "unknown")

    current_regex = regex_str
    remediation_rounds = 0

    for _ in range(MAX_REMEDIATION_ROUNDS + 1):
        try:
            compiled = re.compile(current_regex)
        except re.error as exc:
            logger.error("Regex compile failed for '%s': %s", rule_id, exc)
            return {
                "rule_id": rule_id,
                "regex": current_regex,
                "match_count": 0,
                "total": len(test_values),
                "match_rate": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "mismatches": test_values[:20],
                "false_positives": [],
                "status": "COMPILE_ERROR",
                "remediation_rounds": remediation_rounds,
            }

        matches = [v for v in test_values if compiled.fullmatch(v)]
        mismatches = [v for v in test_values if not compiled.fullmatch(v)]
        false_positives = [n for n in negatives if compiled.fullmatch(n)]

        tp = len(matches)
        fn = len(mismatches)
        fp = len(false_positives)
        tn = len(negatives) - fp
        total_eval = tp + fn + fp + tn

        match_rate = tp / max(len(test_values), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(total_eval, 1)

        if match_rate >= 0.80 or remediation_rounds >= MAX_REMEDIATION_ROUNDS:
            break

        logger.info(
            "Agent 4: '%s' scored %.2f — auto-remediating (round %d)",
            rule_id,
            match_rate,
            remediation_rounds + 1,
        )
        remediation_rounds += 1

        sample_mismatches = mismatches[:30]
        prompt = REMEDIATION_TEMPLATE.format(
            condition=condition,
            regex=current_regex,
            description=description,
            mismatches="\n".join(sample_mismatches),
        )
        try:
            response = invoke_claude(prompt, config)
            candidate = response.strip().strip("'\"` ")
            re.compile(candidate)
            current_regex = candidate
        except Exception as exc:
            logger.warning("Remediation failed for '%s': %s", rule_id, exc)
            break

    if match_rate >= 0.95:
        status = "PASS"
    elif match_rate >= 0.90:
        status = "NEEDS_REVIEW"
    else:
        status = "FAIL"

    return {
        "rule_id": rule_id,
        "regex": current_regex,
        "match_count": tp,
        "total": len(test_values),
        "match_rate": round(match_rate, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "mismatches": mismatches[:20],
        "false_positives": false_positives[:20],
        "status": status,
        "remediation_rounds": remediation_rounds,
    }


def _validate_context_pattern(pattern_entry: dict, test_values: list[str]) -> dict:
    rule_id = pattern_entry["rule_id"]
    regex_str = pattern_entry["regex"]

    rng = random.Random(77)
    keywords_in_regex = re.findall(r"(?<=\|)[^|)]+|(?<=\:)[^|)]+", regex_str)
    if not keywords_in_regex:
        keywords_in_regex = ["address", "zip code", "postal code"]

    sentences: list[str] = []
    sample = rng.sample(test_values, min(50, len(test_values)))
    for val in sample:
        kw = rng.choice(keywords_in_regex)
        filler_count = rng.randint(0, 8)
        filler = " ".join(rng.choices(["the", "is", "at", "for", "of", "in", "a", "my"], k=filler_count))
        sentences.append(f"{kw} {filler} {val}")

    try:
        compiled = re.compile(regex_str)
    except re.error as exc:
        logger.error("Context regex compile failed: %s", exc)
        return {
            "rule_id": rule_id,
            "regex": regex_str,
            "sentences_tested": len(sentences),
            "sentences_matched": 0,
            "match_rate": 0.0,
            "status": "COMPILE_ERROR",
        }

    matched = sum(1 for s in sentences if compiled.search(s))
    rate = matched / max(len(sentences), 1)

    return {
        "rule_id": rule_id,
        "regex": regex_str,
        "sentences_tested": len(sentences),
        "sentences_matched": matched,
        "match_rate": round(rate, 4),
        "status": "PASS" if rate >= 0.80 else "NEEDS_REVIEW",
    }


def _generate_final_report(condition: str, report: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# Regex Discovery Report — {condition}",
        f"",
        f"Generated: {ts}",
        f"",
        f"## Summary",
        f"",
        f"- **Condition**: {condition}",
        f"- **Test set size**: {report['test_set_size']}",
        f"- **Overall status**: {report['overall_status']}",
        f"",
        f"## Pattern Results",
        f"",
    ]

    for r in report["results"]:
        lines.append(f"### {r['rule_id']}")
        lines.append(f"")
        lines.append(f"- **Regex**: `{r['regex']}`")
        lines.append(f"- **Status**: {r['status']}")

        if "precision" in r:
            lines.append(f"- **Match rate**: {r['match_rate']:.1%}")
            lines.append(f"- **Precision**: {r['precision']:.1%}")
            lines.append(f"- **Recall**: {r['recall']:.1%}")
            lines.append(f"- **Accuracy**: {r['accuracy']:.1%}")
            if r.get("mismatches"):
                lines.append(f"- **Sample mismatches**: {', '.join(r['mismatches'][:5])}")
            if r.get("false_positives"):
                lines.append(f"- **Sample false positives**: {', '.join(r['false_positives'][:5])}")
            lines.append(f"- **Remediation rounds**: {r.get('remediation_rounds', 0)}")
        else:
            lines.append(f"- **Sentences tested**: {r.get('sentences_tested', 0)}")
            lines.append(f"- **Sentences matched**: {r.get('sentences_matched', 0)}")
            lines.append(f"- **Match rate**: {r['match_rate']:.1%}")

        lines.append("")

    return "\n".join(lines)


def validate(config: dict) -> dict:
    with open("data/test.txt") as f:
        test_values = [line.strip() for line in f if line.strip()]

    with open("results/regex_patterns.json") as f:
        regex_data = json.load(f)

    condition = regex_data.get("condition", "unknown")
    config["_condition"] = condition
    all_patterns = regex_data.get("patterns", [])

    logger.info("Agent 4: validating %d patterns against %d test values", len(all_patterns), len(test_values))

    negatives = _generate_negative_samples(test_values)
    results: list[dict] = []

    for p in all_patterns:
        if p.get("type") == "context":
            r = _validate_context_pattern(p, test_values)
        else:
            r = _validate_value_pattern(p, test_values, negatives, config)
        results.append(r)
        logger.info("Agent 4: [%s] → %s (%.1f%%)", r["rule_id"], r["status"], r["match_rate"] * 100)

    statuses = [r["status"] for r in results]
    if all(s == "PASS" for s in statuses):
        overall = "PASS"
    elif any(s in ("FAIL", "COMPILE_ERROR") for s in statuses):
        overall = "FAIL"
    else:
        overall = "NEEDS_REVIEW"

    report = {
        "condition": condition,
        "test_set_size": len(test_values),
        "results": results,
        "overall_status": overall,
    }

    with open("results/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    final_md = _generate_final_report(condition, report)
    with open("results/final_report.md", "w") as f:
        f.write(final_md)

    logger.info("Agent 4 done: overall %s", overall)
    return report
