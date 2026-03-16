from __future__ import annotations

import json
import logging
import random
from collections import Counter
from statistics import mode as stats_mode

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a pattern-analysis expert.

CONDITION: {condition}

STATISTICAL SUMMARY:
{stats_json}

SAMPLE VALUES (200 of {total}):
{sample_values}
{policies}
TASKS — return a JSON object with these keys:

1. "format_rules": array of objects, each with:
   - "rule_id": short slug
   - "description": plain-English description of the structural rule
   - "applies_to": "all" | "subset"
   - "example_matches": [3+ examples from the sample]

2. "range_restrictions": array of objects, each with:
   - "position": which digit(s) or segment
   - "allowed_values": explicit set or range description
   - "source": cite the real-world policy or numbering authority if known

3. "contextual_keywords": array of strings — words/phrases that commonly
   appear near this identifier in documents (case-insensitive).
   IMPORTANT: Include as MANY relevant keywords as possible — aim for 8-15.
   Cast a wide net but do NOT include generic words that would cause false
   positives (e.g. avoid "number", "code", "ID" alone — too ambiguous).
   Every keyword must be specific enough that its presence near a value
   is a strong signal that the value is this identifier type.
   Think broadly across these categories:
     - Form field labels (e.g. "ZIP code", "postal code", "mailing address")
     - Column headers in spreadsheets/databases
     - Surrounding text in official documents, forms, invoices
     - Abbreviations and acronyms used in the industry
     - Multilingual variants if the identifier is used internationally
     - Related field names in software systems and APIs
     - Labels used by DLP/security vendors (Microsoft, Netskope, Broadcom,
       Zscaler, Skyhigh) for this same identifier type

4. "keyword_proximity": integer — how many terms away a keyword can be
   from the identifier and still indicate a match (default 10).

5. "numbering_policy": free-text summary of the real-world authority,
   allocation rules, check-digit algorithms, or geographic mapping that
   governs this identifier.

Return ONLY valid JSON, no markdown fences."""


def _compute_stats(values: list[str]) -> dict:
    lengths = [len(v) for v in values]
    length_counts = Counter(lengths)

    char_classes = {
        "all_digit": all(v.isdigit() for v in values),
        "all_alpha": all(v.isalpha() for v in values),
        "alphanumeric": all(v.isalnum() for v in values),
        "has_hyphens": any("-" in v for v in values),
        "has_spaces": any(" " in v for v in values),
        "has_slashes": any("/" in v for v in values),
    }

    prefix_1 = Counter(v[:1] for v in values if len(v) >= 1).most_common(10)
    prefix_2 = Counter(v[:2] for v in values if len(v) >= 2).most_common(10)
    prefix_3 = Counter(v[:3] for v in values if len(v) >= 3).most_common(10)

    suffix_1 = Counter(v[-1:] for v in values if len(v) >= 1).most_common(10)
    suffix_2 = Counter(v[-2:] for v in values if len(v) >= 2).most_common(10)

    delimiters = [c for c in "-/ ." if any(c in v for v in values)]
    segment_structures: list[str] = []
    if delimiters:
        for delim in delimiters:
            patterns: Counter = Counter()
            for v in values:
                parts = v.split(delim)
                sig = delim.join("X" * len(p) for p in parts)
                patterns[sig] += 1
            segment_structures.extend(
                f"{sig} ({cnt})" for sig, cnt in patterns.most_common(5)
            )

    try:
        length_mode = stats_mode(lengths)
    except Exception:
        length_mode = lengths[0] if lengths else 0

    return {
        "count": len(values),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "length_mode": length_mode,
        "length_distribution": dict(length_counts.most_common(10)),
        "character_classes": char_classes,
        "prefix_1_char": prefix_1,
        "prefix_2_char": prefix_2,
        "prefix_3_char": prefix_3,
        "suffix_1_char": suffix_1,
        "suffix_2_char": suffix_2,
        "delimiters_found": delimiters,
        "segment_structures": segment_structures,
    }


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def analyze_patterns(
    condition: str,
    config: dict,
    paths: dict[str, str] | None = None,
    policies: dict | None = None,
) -> dict:
    train_path = (paths or {}).get("train", "data/train.txt")
    patterns_path = (paths or {}).get("patterns", "results/patterns.json")

    with open(train_path) as f:
        values = [line.strip() for line in f if line.strip()]

    logger.info("Agent 2: analyzing %d training values for '%s'", len(values), condition)

    stats = _compute_stats(values)
    logger.info("Agent 2: statistical pre-analysis complete")

    sample_size = min(200, len(values))
    rng = random.Random(42)
    sample = rng.sample(values, sample_size)

    policy_section = ""
    if policies:
        parts: list[str] = []

        if policies.get("policies"):
            rules = policies["policies"]
            authority = policies.get("numbering_authority", "unknown")
            parts.append(
                f"\nKNOWN POLICIES/RULES (from {authority}):\n"
                + "\n".join(f"- {r}" for r in rules)
            )
            logger.info("Agent 2: injecting %d policy rules from Agent 1P", len(rules))

        if policies.get("vendor_patterns"):
            vendor_lines: list[str] = []
            for vp in policies["vendor_patterns"]:
                vendor = vp.get("vendor", "Unknown")
                vr = vp.get("regex")
                vk = vp.get("keywords", [])
                vrl = vp.get("rules", [])
                line = f"  {vendor}:"
                if vr:
                    line += f" regex={vr}"
                if vk:
                    line += f" keywords={vk}"
                if vrl:
                    line += f" rules={vrl}"
                vendor_lines.append(line)
            parts.append(
                "\nVENDOR/COMPETITOR DLP PATTERNS (for reference — use to improve keyword coverage):\n"
                + "\n".join(vendor_lines)
            )
            logger.info("Agent 2: injecting %d vendor patterns from Agent 1P", len(policies["vendor_patterns"]))

        if policies.get("vendor_keywords"):
            parts.append(
                "\nADDITIONAL KEYWORDS FROM VENDOR PRODUCTS:\n"
                + ", ".join(policies["vendor_keywords"])
                + "\n(Include any of these that are genuinely relevant to the identifier.)"
            )
            logger.info("Agent 2: injecting %d vendor keywords from Agent 1P", len(policies["vendor_keywords"]))

        policy_section = "\n".join(parts) + "\n" if parts else ""

    prompt = PROMPT_TEMPLATE.format(
        condition=condition,
        stats_json=json.dumps(stats, indent=2),
        total=len(values),
        sample_values="\n".join(sample),
        policies=policy_section,
    )

    response = invoke_claude(prompt, config)

    try:
        patterns = _parse_json_response(response)
    except json.JSONDecodeError:
        logger.warning("Agent 2: invalid JSON from LLM, retrying with correction prompt")
        retry_prompt = (
            "Your previous response was not valid JSON. Please fix it.\n"
            "Previous response:\n" + response + "\n\n"
            "Return ONLY valid JSON, no markdown fences."
        )
        response = invoke_claude(retry_prompt, config)
        patterns = _parse_json_response(response)

    patterns["condition"] = condition

    with open(patterns_path, "w") as f:
        json.dump(patterns, f, indent=2)

    rule_count = len(patterns.get("format_rules", []))
    kw_count = len(patterns.get("contextual_keywords", []))
    logger.info("Agent 2 done: %d format rules, %d keywords discovered", rule_count, kw_count)

    return patterns
