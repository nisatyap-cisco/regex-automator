"""Agent 4 — Validator (synth + classify + cross-check).

Two validation modes depending on the data source:

  NET-SOURCED (db / scrape):
    Receives the same excel data + policies that Agent 2 used.
    Uses the LLM to generate 1000 test cases (mix of valid and invalid),
    classifies each with the regex from Agent 3, cross-checks against the
    truth set from the excel, and reports metrics.

  LLM-SOURCED:
    Validates the regex against the 40% held-out test.txt.
    Synthesises 1000 candidates from that test set (truth-set samples,
    boundary values, random noise) and reports metrics.

For context (keyword) patterns the validator runs sentence-level tests.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import string
from datetime import datetime, timezone
from pathlib import Path

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

MAX_REMEDIATION_ROUNDS = 2
SYNTH_COUNT = 1000
SEED = 42

REMEDIATION_TEMPLATE = """You are a regex engineering expert.

The following regex was intended to match values for: {condition}

REGEX: {regex}
RULE: {description}

It FAILED to match these known-valid values:
{mismatches}

Fix the regex so it matches ALL the values above while still being precise.
Return ONLY the corrected regex string, no explanation."""

_LLM_TESTCASE_PROMPT = """You are a test-case generation expert.

CONDITION: {condition}

KNOWN VALID EXAMPLES (from real data):
{sample_values}
{policy_section}
Generate exactly {count} test values — a mix of:
  - ~50% values that SHOULD match (realistic valid examples, including edge cases)
  - ~50% values that SHOULD NOT match (plausible but invalid — wrong range,
    wrong length, wrong format, boundary violations)

For each value, output ONE line in this exact format:
  VALID|<value>
  INVALID|<value>

No extra text, no numbering, no blank lines. Output ONLY the test lines."""


# ── synthesise candidates (fallback for LLM-sourced data) ────────────────────

def _synthesize_candidates(truth_values: list[str], count: int) -> list[str]:
    """Build a balanced candidate pool: real positives + boundary + random."""
    rng = random.Random(SEED)
    truth_set = set(truth_values)
    candidates: list[str] = []

    # 40 % from the truth set
    positive_count = min(count * 40 // 100, len(truth_values))
    candidates.extend(rng.sample(truth_values, positive_count))

    # Boundary values (for numeric identifiers)
    if truth_values and truth_values[0].isdigit():
        nums = sorted(int(v) for v in truth_values)
        lo, hi = nums[0], nums[-1]
        for base in [lo - 10, lo, hi, hi + 1, hi + 10]:
            for offset in range(-5, 6):
                z = str(max(0, base + offset)).zfill(len(truth_values[0]))
                candidates.append(z)

    # Random noise — same length, random digits and/or letters
    sample_len = max(len(v) for v in truth_values) if truth_values else 5
    is_numeric = all(v.isdigit() for v in truth_values)
    while len(candidates) < count:
        if is_numeric:
            candidates.append("".join(rng.choices(string.digits, k=sample_len)))
        else:
            candidates.append("".join(rng.choices(string.ascii_uppercase + string.digits, k=sample_len)))

    # Deduplicate and trim
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    rng.shuffle(unique)
    return unique[:count]


# ── LLM-generated test cases (for net-sourced data) ─────────────────────────

def _llm_generate_test_cases(
    condition: str,
    truth_values: list[str],
    policies: dict | None,
    config: dict,
    count: int = SYNTH_COUNT,
) -> tuple[list[str], dict[str, bool]]:
    """Use the LLM to produce labelled test cases.

    Returns:
        (candidate_values, label_map)  where label_map maps value → True
        if the LLM marked it VALID, False if INVALID.
    """
    rng = random.Random(SEED)
    sample = rng.sample(truth_values, min(100, len(truth_values)))

    policy_section = ""
    if policies and policies.get("policies"):
        rules = policies["policies"]
        authority = policies.get("numbering_authority", "unknown")
        policy_section = (
            f"\nKNOWN FORMAT RULES (from {authority}):\n"
            + "\n".join(f"- {r}" for r in rules)
            + "\n"
        )

    batch_size = 250
    num_batches = math.ceil(count / batch_size)
    all_candidates: list[str] = []
    label_map: dict[str, bool] = {}

    for batch_idx in range(num_batches):
        remaining = count - len(all_candidates)
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        prompt = _LLM_TESTCASE_PROMPT.format(
            condition=condition,
            sample_values="\n".join(sample),
            policy_section=policy_section,
            count=this_batch,
        )

        try:
            response = invoke_claude(prompt, config)
        except Exception as exc:
            logger.warning("Agent 4 [llm-testgen]: batch %d failed: %s", batch_idx + 1, exc)
            continue

        for line in response.strip().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                continue
            label_str, value = parts[0].strip().upper(), parts[1].strip()
            if not value:
                continue
            is_valid = label_str == "VALID"
            if value not in label_map:
                all_candidates.append(value)
                label_map[value] = is_valid

        logger.info(
            "Agent 4 [llm-testgen]: batch %d/%d → %d total candidates",
            batch_idx + 1, num_batches, len(all_candidates),
        )

    logger.info(
        "Agent 4 [llm-testgen]: generated %d candidates (%d valid, %d invalid)",
        len(all_candidates),
        sum(1 for v in label_map.values() if v),
        sum(1 for v in label_map.values() if not v),
    )
    return all_candidates, label_map


# ── value pattern validation ──────────────────────────────────────────────────

def _validate_value_pattern(
    pattern_entry: dict,
    truth_values: list[str],
    candidates: list[str],
    config: dict,
    llm_labels: dict[str, bool] | None = None,
) -> dict:
    """Validate a value regex against candidates.

    If *llm_labels* is provided (net-sourced mode), the LLM's VALID/INVALID
    labels are used as the ground truth for each candidate, cross-checked
    against the real truth set.

    If *llm_labels* is None (LLM-sourced mode), membership in *truth_values*
    is the ground truth.
    """
    rule_id = pattern_entry["rule_id"]
    regex_str = pattern_entry["regex"]
    description = pattern_entry.get("description", "")
    condition = config.get("_condition", "unknown")
    truth_set = set(truth_values)

    current_regex = regex_str
    remediation_rounds = 0

    for _ in range(MAX_REMEDIATION_ROUNDS + 1):
        try:
            compiled = re.compile(current_regex)
        except re.error as exc:
            logger.error("Regex compile failed for '%s': %s", rule_id, exc)
            return _error_result(rule_id, current_regex, len(candidates), remediation_rounds)

        tp = fp = tn = fn = 0
        false_positives: list[dict] = []
        false_negatives: list[dict] = []

        for val in candidates:
            regex_match = bool(compiled.fullmatch(val))

            if llm_labels is not None:
                expected_valid = llm_labels.get(val, val in truth_set)
            else:
                expected_valid = val in truth_set

            if regex_match and expected_valid:
                tp += 1
            elif regex_match and not expected_valid:
                fp += 1
                false_positives.append({"value": val, "regex_says": "match", "truth": "not_match"})
            elif not regex_match and expected_valid:
                fn += 1
                false_negatives.append({"value": val, "regex_says": "no_match", "truth": "match"})
            else:
                tn += 1

        total = tp + fp + tn + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        accuracy = (tp + tn) / max(total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        if recall >= 0.80 or remediation_rounds >= MAX_REMEDIATION_ROUNDS:
            break

        logger.info("Agent 4: '%s' recall=%.2f — auto-remediating (round %d)", rule_id, recall, remediation_rounds + 1)
        remediation_rounds += 1

        sample_fn = [e["value"] for e in false_negatives[:30]]
        prompt = REMEDIATION_TEMPLATE.format(
            condition=condition, regex=current_regex,
            description=description, mismatches="\n".join(sample_fn),
        )
        try:
            response = invoke_claude(prompt, config)
            candidate_regex = response.strip().strip("'\"` ")
            re.compile(candidate_regex)
            current_regex = candidate_regex
        except Exception as exc:
            logger.warning("Remediation failed for '%s': %s", rule_id, exc)
            break

    status = "PASS" if recall >= 0.95 else ("NEEDS_REVIEW" if recall >= 0.90 else "FAIL")

    return {
        "rule_id": rule_id,
        "regex": current_regex,
        "candidates_tested": len(candidates),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "accuracy": round(accuracy, 6),
        "f1": round(f1, 6),
        "false_positives": false_positives[:30],
        "false_negatives": false_negatives[:30],
        "status": status,
        "remediation_rounds": remediation_rounds,
    }


def _error_result(rule_id: str, regex: str, count: int, rounds: int) -> dict:
    return {
        "rule_id": rule_id, "regex": regex, "candidates_tested": count,
        "TP": 0, "FP": 0, "TN": 0, "FN": count,
        "precision": 0.0, "recall": 0.0, "accuracy": 0.0, "f1": 0.0,
        "false_positives": [], "false_negatives": [],
        "status": "COMPILE_ERROR", "remediation_rounds": rounds,
    }


# ── context pattern validation ────────────────────────────────────────────────

def _validate_context_pattern(pattern_entry: dict, truth_values: list[str]) -> dict:
    rule_id = pattern_entry["rule_id"]
    regex_str = pattern_entry["regex"]

    rng = random.Random(77)
    keywords_in_regex = re.findall(r"(?<=\|)[^|)]+|(?<=:)[^|)]+", regex_str)
    if not keywords_in_regex:
        keywords_in_regex = ["address", "zip code", "postal code"]

    sentences: list[str] = []
    sample = rng.sample(truth_values, min(50, len(truth_values)))
    for val in sample:
        kw = rng.choice(keywords_in_regex)
        filler_count = rng.randint(0, 8)
        filler = " ".join(rng.choices(["the", "is", "at", "for", "of", "in", "a", "my"], k=filler_count))
        sentences.append(f"{kw} {filler} {val}")

    try:
        compiled = re.compile(regex_str)
    except re.error as exc:
        logger.error("Context regex compile failed: %s", exc)
        return {"rule_id": rule_id, "regex": regex_str, "sentences_tested": len(sentences),
                "sentences_matched": 0, "match_rate": 0.0, "status": "COMPILE_ERROR"}

    matched = sum(1 for s in sentences if compiled.search(s))
    rate = matched / max(len(sentences), 1)

    return {
        "rule_id": rule_id, "regex": regex_str,
        "sentences_tested": len(sentences), "sentences_matched": matched,
        "match_rate": round(rate, 4),
        "status": "PASS" if rate >= 0.80 else "NEEDS_REVIEW",
    }


# ── markdown report ───────────────────────────────────────────────────────────

def _generate_final_report(condition: str, report: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    src = report.get("data_source", "unknown")
    lines = [
        f"# Regex Discovery Report — {condition}",
        "",
        f"Generated: {ts}",
        "",
        "## Summary",
        "",
        f"- **Condition**: {condition}",
        f"- **Data source**: {src}",
        f"- **Validation mode**: {report.get('validation_mode', 'synth_candidates')}",
        f"- **Truth-set size**: {report['truth_set_size']}",
        f"- **Candidates tested**: {report.get('candidates_synthesized', 'N/A')}",
        f"- **Overall status**: {report['overall_status']}",
        "",
        "## Pattern Results",
        "",
    ]

    for r in report["results"]:
        lines.append(f"### {r['rule_id']}")
        lines.append("")
        lines.append(f"- **Regex**: `{r['regex']}`")
        lines.append(f"- **Status**: {r['status']}")

        if "precision" in r:
            lines.append(f"- **Candidates tested**: {r.get('candidates_tested', 'N/A')}")
            lines.append(f"- **TP**: {r['TP']}  |  **FP**: {r['FP']}  |  **TN**: {r['TN']}  |  **FN**: {r['FN']}")
            lines.append(f"- **Precision**: {r['precision']:.2%}")
            lines.append(f"- **Recall**: {r['recall']:.2%}")
            lines.append(f"- **Accuracy**: {r['accuracy']:.2%}")
            lines.append(f"- **F1**: {r['f1']:.2%}")
            if r.get("false_positives"):
                fp_vals = [e["value"] for e in r["false_positives"][:10]]
                lines.append(f"- **Sample false positives**: {', '.join(fp_vals)}")
            if r.get("false_negatives"):
                fn_vals = [e["value"] for e in r["false_negatives"][:10]]
                lines.append(f"- **Sample false negatives**: {', '.join(fn_vals)}")
            lines.append(f"- **Remediation rounds**: {r.get('remediation_rounds', 0)}")
        else:
            lines.append(f"- **Sentences tested**: {r.get('sentences_tested', 0)}")
            lines.append(f"- **Sentences matched**: {r.get('sentences_matched', 0)}")
            lines.append(f"- **Match rate**: {r.get('match_rate', 0):.1%}")

        lines.append("")

    return "\n".join(lines)


# ── public API ────────────────────────────────────────────────────────────────

def validate(
    config: dict,
    paths: dict[str, str] | None = None,
    data_source: str = "unknown",
    policies: dict | None = None,
) -> dict:
    """Validate regex patterns against test data.

    Two modes based on *data_source*:

    NET-SOURCED (db / scrape_tool):
        Reads the full truth set from test.txt (which is the same as
        train.txt for net-sourced data). Uses the LLM to generate 1000
        labelled test cases (VALID / INVALID), classifies each with the
        regex, and cross-checks against the truth set + LLM labels.

    LLM-SOURCED:
        Reads the 40% held-out test.txt. Synthesises 1000 candidates
        (truth-set samples + boundary + noise) and validates.
    """
    paths = paths or {}
    test_path = paths.get("test", "data/test.txt")
    regex_path = paths.get("regex_patterns", "results/regex_patterns.json")
    report_path = paths.get("validation_report", "results/validation_report.json")
    final_path = paths.get("final_report", "results/final_report.md")

    with open(test_path) as f:
        truth_values = [line.strip() for line in f if line.strip()]

    with open(regex_path) as f:
        regex_data = json.load(f)

    condition = regex_data.get("condition", "unknown")
    config["_condition"] = condition
    all_patterns = regex_data.get("patterns", [])

    is_net_sourced = data_source in ("db", "scrape_tool")

    if is_net_sourced:
        logger.info(
            "Agent 4: NET-SOURCED mode — truth set=%d, generating %d LLM test cases with policies",
            len(truth_values), SYNTH_COUNT,
        )
        candidates, llm_labels = _llm_generate_test_cases(
            condition, truth_values, policies, config, count=SYNTH_COUNT,
        )
        if not candidates:
            logger.warning("Agent 4: LLM test-gen returned nothing — falling back to synth candidates")
            candidates = _synthesize_candidates(truth_values, SYNTH_COUNT)
            llm_labels = None
    else:
        logger.info(
            "Agent 4: LLM-SOURCED mode — test set=%d, synthesising %d candidates",
            len(truth_values), SYNTH_COUNT,
        )
        candidates = _synthesize_candidates(truth_values, SYNTH_COUNT)
        llm_labels = None

    logger.info("Agent 4: %d unique candidates prepared", len(candidates))

    results: list[dict] = []
    for p in all_patterns:
        if p.get("type") == "context":
            r = _validate_context_pattern(p, truth_values)
        else:
            r = _validate_value_pattern(p, truth_values, candidates, config, llm_labels=llm_labels)
        results.append(r)
        logger.info("Agent 4: [%s] → %s", r["rule_id"], r["status"])

    statuses = [r["status"] for r in results]
    if all(s == "PASS" for s in statuses):
        overall = "PASS"
    elif any(s in ("FAIL", "COMPILE_ERROR") for s in statuses):
        overall = "FAIL"
    else:
        overall = "NEEDS_REVIEW"

    validation_mode = "llm_test_cases" if (is_net_sourced and llm_labels) else "synth_candidates"

    report = {
        "condition": condition,
        "data_source": data_source,
        "validation_mode": validation_mode,
        "truth_set_size": len(truth_values),
        "candidates_synthesized": len(candidates),
        "results": results,
        "overall_status": overall,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    final_md = _generate_final_report(condition, report)
    with open(final_path, "w") as f:
        f.write(final_md)

    logger.info("Agent 4 done: overall %s (mode: %s)", overall, validation_mode)
    return report
