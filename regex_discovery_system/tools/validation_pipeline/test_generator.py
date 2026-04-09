"""Test-case generator — produces positive and negative test cases for a regex rule.

Uses the LLM (via Bedrock) to craft realistic test values based on the
condition, its policies/rules, and known-good examples from the training data.
Falls back to a deterministic generator when the LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import string
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import utils/agents
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

SEED = 42

# ── LLM prompt templates ─────────────────────────────────────────────────────

_POSITIVE_PROMPT = """\
You are a test-data generation expert.

CONDITION: {condition}
RULE BEING TESTED: {rule_id} — {description}
REGEX UNDER TEST: {regex}
{policy_section}
KNOWN VALID EXAMPLES (from real data):
{sample_values}

Your job is to verify whether the regex correctly implements the RULE above.
Generate exactly {count} values that SATISFY the rule "{description}".
These values should match the regex if the regex is correct.

Focus ONLY on what the rule says — for example if the rule says "length is
10 or 14 digits", generate values that are exactly 10 or 14 digits long.
Do NOT worry about whether these values are real-world valid identifiers.
Include a variety: typical cases, edge cases, and boundary values that
still satisfy the rule.

Output ONE value per line.  No numbering, no labels, no blank lines, no
extra text — ONLY the raw values."""

_NEGATIVE_PROMPT = """\
You are a test-data generation expert specialising in *negative* cases.

CONDITION: {condition}
RULE BEING TESTED: {rule_id} — {description}
REGEX UNDER TEST: {regex}
{policy_section}
KNOWN VALID EXAMPLES (for reference — do NOT reproduce these):
{sample_values}

Your job is to verify whether the regex correctly REJECTS values that
VIOLATE the rule "{description}".
Generate exactly {count} values that BREAK / DO NOT SATISFY the rule above.
The regex should NOT match any of these if it is correct.

Focus ONLY on what the rule says — for example if the rule says "length is
10 or 14 digits", generate values with wrong lengths (9, 11, 12, 13, 15,
16 digits), values with letters mixed in, values with special characters,
etc.  Every value must violate the rule in some way.

Output ONE value per line.  No numbering, no labels, no blank lines, no
extra text — ONLY the raw values."""


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_train_values(train_path: str, max_samples: int = 200) -> list[str]:
    """Read training examples from a text file (one value per line)."""
    values: list[str] = []
    if not os.path.exists(train_path):
        logger.warning("Training file not found: %s", train_path)
        return values
    with open(train_path) as f:
        for line in f:
            v = line.strip()
            if v:
                values.append(v)
    rng = random.Random(SEED)
    if len(values) > max_samples:
        values = rng.sample(values, max_samples)
    return values


def _build_policy_section(source_of_truth: dict | None) -> str:
    """Format policies for inclusion in the LLM prompt."""
    if not source_of_truth:
        return ""
    policies = source_of_truth.get("policies", [])
    authority = source_of_truth.get("numbering_authority", "unknown")
    if not policies:
        return ""
    return (
        f"\nKNOWN FORMAT RULES (authority: {authority}):\n"
        + "\n".join(f"  - {r}" for r in policies)
        + "\n"
    )


# ── deterministic fallback generator ─────────────────────────────────────────

def _deterministic_positive(regex_str: str, truth_values: list[str], count: int) -> list[str]:
    """Return known-good values that actually match the regex (fallback)."""
    try:
        compiled = re.compile(regex_str)
    except re.error:
        return truth_values[:count]
    matches = [v for v in truth_values if compiled.fullmatch(v) or compiled.search(v)]
    rng = random.Random(SEED)
    rng.shuffle(matches)
    return matches[:count]


def _deterministic_negative(regex_str: str, truth_values: list[str], count: int) -> list[str]:
    """Generate values that should NOT match (fallback)."""
    rng = random.Random(SEED)
    negatives: list[str] = []

    try:
        compiled = re.compile(regex_str)
    except re.error:
        compiled = None

    sample_len = max((len(v) for v in truth_values), default=5)
    is_numeric = all(v.replace(".", "").replace("-", "").replace(" ", "").isdigit() for v in truth_values) if truth_values else False

    # Strategy 1: mutate valid values
    for v in truth_values[:50]:
        # wrong length
        negatives.append(v + rng.choice(string.digits))
        if len(v) > 1:
            negatives.append(v[:-1])
        # wrong characters
        if is_numeric:
            negatives.append(rng.choice(string.ascii_uppercase) + v[1:])
        else:
            negatives.append(v[0] + "###" + v[3:] if len(v) > 3 else "###")

    # Strategy 2: random noise
    while len(negatives) < count * 2:
        charset = string.digits if is_numeric else string.ascii_uppercase + string.digits
        negatives.append("".join(rng.choices(charset, k=rng.randint(sample_len - 3, sample_len + 3))))

    # Filter: keep only those that do NOT match the regex
    if compiled:
        negatives = [v for v in negatives if not compiled.fullmatch(v) and not compiled.search(v)]

    seen: set[str] = set()
    unique: list[str] = []
    for v in negatives:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    rng.shuffle(unique)
    return unique[:count]


# ── LLM-based generator ──────────────────────────────────────────────────────

def _llm_generate(
    prompt_template: str,
    condition: str,
    rule_id: str,
    description: str,
    regex: str,
    policy_section: str,
    sample_values: list[str],
    config: dict,
    count: int,
) -> list[str]:
    """Call the LLM with the given prompt template and parse one-value-per-line output."""
    batch_size = 250
    num_batches = math.ceil(count / batch_size)
    all_values: list[str] = []

    rng = random.Random(SEED)
    display_sample = rng.sample(sample_values, min(80, len(sample_values))) if sample_values else []

    for batch_idx in range(num_batches):
        remaining = count - len(all_values)
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        prompt = prompt_template.format(
            condition=condition,
            rule_id=rule_id,
            description=description,
            regex=regex,
            policy_section=policy_section,
            sample_values="\n".join(display_sample),
            count=this_batch,
        )

        try:
            response = invoke_claude(prompt, config)
        except Exception as exc:
            logger.warning("LLM test-gen batch %d failed: %s", batch_idx + 1, exc)
            continue

        for line in response.strip().splitlines():
            val = line.strip()
            if val:
                all_values.append(val)

        logger.info("Test-gen batch %d/%d → %d total values", batch_idx + 1, num_batches, len(all_values))

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for v in all_values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique[:count]


# ── public API ────────────────────────────────────────────────────────────────

def generate_test_cases(
    regex_patterns_path: str,
    rule_id: str,
    config: dict,
    positive_count: int = 100,
    negative_count: int = 100,
    source_of_truth_path: str | None = None,
    train_path: str | None = None,
    use_llm: bool = True,
) -> dict:
    """Generate positive and negative test cases for a specific rule_id.

    Parameters
    ----------
    regex_patterns_path : str
        Path to the ``*_regex_patterns.json`` file.
    rule_id : str
        Which rule to test (e.g. ``"length_9"``, ``"combined-value"``).
    config : dict
        Pipeline config (for LLM calls).
    positive_count / negative_count : int
        How many test cases of each kind to generate.
    source_of_truth_path : str | None
        Path to the ``*_source_of_truth.json`` (auto-discovered if None).
    train_path : str | None
        Path to the ``*_train.txt`` (auto-discovered if None).
    use_llm : bool
        If True, use LLM to generate; else use deterministic fallback.

    Returns
    -------
    dict with keys: condition, rule_id, regex, positive_cases, negative_cases
    """
    # ── load regex patterns file ──
    with open(regex_patterns_path) as f:
        regex_data = json.load(f)

    condition = regex_data.get("condition", "unknown")
    patterns = regex_data.get("patterns", [])

    # Find the requested rule
    target_pattern = None
    for p in patterns:
        if p["rule_id"] == rule_id:
            target_pattern = p
            break

    if target_pattern is None:
        available = [p["rule_id"] for p in patterns]
        raise ValueError(
            f"rule_id '{rule_id}' not found. Available: {available}"
        )

    regex_str = target_pattern["regex"]
    description = target_pattern.get("description", "")

    # For combined-value, enrich description with all individual value rules
    # so the LLM knows exactly what constraints the combined regex must satisfy.
    if rule_id == "combined-value":
        individual_rules = [
            p for p in patterns
            if p.get("type") == "value" and p["rule_id"] != "combined-value"
        ]
        if individual_rules:
            rule_lines = [f"  - {p['rule_id']}: {p.get('description', '')}" for p in individual_rules]
            description = (
                "Unified regex that must satisfy ALL of these rules simultaneously:\n"
                + "\n".join(rule_lines)
            )

    # ── auto-discover sibling files ──
    base_dir = os.path.dirname(regex_patterns_path)
    slug = os.path.basename(regex_patterns_path).replace("_regex_patterns.json", "")

    if source_of_truth_path is None:
        candidate = os.path.join(base_dir, f"{slug}_source_of_truth.json")
        if os.path.exists(candidate):
            source_of_truth_path = candidate

    if train_path is None:
        candidate = os.path.join(base_dir, "..", "data", f"{slug}_train.txt")
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            train_path = candidate

    # Load supporting data
    sot: dict | None = None
    if source_of_truth_path and os.path.exists(source_of_truth_path):
        with open(source_of_truth_path) as f:
            sot = json.load(f)

    train_values = _load_train_values(train_path) if train_path else []

    policy_section = _build_policy_section(sot)

    # ── generate test cases ──
    if use_llm and train_values:
        logger.info("Generating %d positive + %d negative cases via LLM for rule '%s'",
                     positive_count, negative_count, rule_id)
        positive_cases = _llm_generate(
            _POSITIVE_PROMPT, condition, rule_id, description,
            regex_str, policy_section, train_values, config, positive_count,
        )
        negative_cases = _llm_generate(
            _NEGATIVE_PROMPT, condition, rule_id, description,
            regex_str, policy_section, train_values, config, negative_count,
        )
    else:
        logger.info("Generating %d positive + %d negative cases via deterministic fallback for rule '%s'",
                     positive_count, negative_count, rule_id)
        positive_cases = _deterministic_positive(regex_str, train_values, positive_count)
        negative_cases = _deterministic_negative(regex_str, train_values, negative_count)

    # If LLM produced too few, top up with deterministic
    if len(positive_cases) < positive_count:
        logger.info("Topping up positive cases: LLM gave %d, need %d", len(positive_cases), positive_count)
        extra = _deterministic_positive(regex_str, train_values, positive_count - len(positive_cases))
        existing = set(positive_cases)
        positive_cases.extend(v for v in extra if v not in existing)

    if len(negative_cases) < negative_count:
        logger.info("Topping up negative cases: LLM gave %d, need %d", len(negative_cases), negative_count)
        extra = _deterministic_negative(regex_str, train_values, negative_count - len(negative_cases))
        existing = set(negative_cases)
        negative_cases.extend(v for v in extra if v not in existing)

    return {
        "condition": condition,
        "rule_id": rule_id,
        "regex": regex_str,
        "description": description,
        "positive_cases": positive_cases[:positive_count],
        "negative_cases": negative_cases[:negative_count],
    }
