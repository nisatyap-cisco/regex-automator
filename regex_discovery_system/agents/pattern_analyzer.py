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

NOTE ON STATISTICAL OBSERVATIONS: The "statistical_observations" field in the
summary above describes what characters APPEAR at key positions across the sample
data.  Use these observations to understand the structural FORMAT of the identifier
(e.g. length, whether it starts with letters vs digits, fixed prefixes, delimiters).
Do NOT convert per-position character observations into hard range_restrictions —
only officially documented rules belong there.  Absence of a character in sample
data does NOT mean it is invalid.

SAMPLE VALUES (200 of {total}):
{sample_values}
{policies}
TASKS — return a JSON object with these keys:

1. "format_rules": array of objects, each with:
   - "rule_id": short slug
   - "description": plain-English description of the structural rule
   - "applies_to": "all" | "subset"
   - "example_matches": [3+ examples from the sample]
   Use statistical_observations to discover format rules like overall length,
   character types (all-digit, alphanumeric), fixed prefixes, delimiter patterns,
   and segment structures.  These are ADDITIVE patterns (what is present across
   all values) — not per-position character restrictions.

2. "range_restrictions": array of objects, each with:
   - "position": which digit(s) or segment (use 0-based index)
   - "allowed_values": explicit set or range description (e.g. "[2-9]", "[A-Z]", "1-12")
   - "source": cite the real-world policy or numbering authority
   CRITICAL — range_restrictions must ONLY come from these authoritative sources:
   a) Every per-position character class in any vendor regex provided above
      (e.g. [2-9] at position 0 of a vendor regex → range_restriction for position 0).
   b) Every rule in KNOWN POLICIES that mentions a digit or character restriction
      (e.g. "first digit cannot be 0 or 1" → position 0, allowed_values "[2-9]").
   c) Officially documented enumerated values (e.g. state codes, country codes).
   NEVER create a range_restriction from statistical observations alone.
   Absence of a character in sample data is NOT evidence of a real restriction.
   If no policy or vendor source documents a per-position restriction, use the
   full character class for that position type (e.g. [0-9] for digits, [A-Z] for letters).

3. "contextual_keywords": array of strings — words/phrases that commonly
   appear near this identifier in documents (case-insensitive).
   IMPORTANT: Include as MANY relevant keywords as possible — aim for 8-15.
   Cast a wide net but do NOT include generic words that would cause false
   positives (e.g. avoid "number", "code", "ID" alone — too ambiguous).
   Every keyword must be specific enough that its presence near a value
   is a strong signal that the value is this identifier type.

   CRITICAL — MULTILINGUAL KEYWORDS: You MUST include keywords in the
   LOCAL LANGUAGE of the country this identifier belongs to. For example:
     - Latvia → include Latvian terms: "adrese", "pasta indekss",
       "dzīvesvieta", "pilsēta", "pasta indeksa sektors"
     - Czech Republic → include Czech terms: "poštovní směrovací číslo",
       "adresa", "město"
     - Mexico → include Spanish terms: "dirección", "código postal",
       "lugar de residencia", "ciudad"
     - Israel → include Hebrew terms
     - India → include Hindi/regional terms if applicable
     - Sweden → include Swedish terms: "personnummer", "postnummer"
   Use the actual Unicode characters (ā, ē, ī, č, š, ñ, ö, etc.) in the
   keywords — not ASCII approximations. For each local-language keyword
   that has diacritics, the downstream regex builder will handle matching.

   CRITICAL — LANGUAGE-EXCLUSIVITY RULE:
   Every keyword you list MUST be inherently exclusive to the local language
   of the identifier's country. A keyword is language-exclusive if a document
   written in another language (e.g. English, German, Chinese) CANNOT contain
   that exact string for an unrelated reason.

   The two reliable tests for language-exclusivity:

   TEST 1 — Contains at least one diacritic / non-ASCII character:
     "Número"       ✅  contains ú — impossible in plain English ASCII text
     "Identificación" ✅  contains ó and ó — impossible in plain English
     "personnummer"  ✅  Swedish compound word, not an English word
     "poštovní"      ✅  contains š — not ASCII
     FAIL: "number"  ❌  pure ASCII English word
     FAIL: "KT"      ❌  pure ASCII, appears in any language

   TEST 2 — Multi-word LOCAL LANGUAGE phrase that cannot appear verbatim in English:
     "pasaporte argentino"        ✅  Spanish phrase, not English
     "Documento Nacional de Identidad" ✅  Spanish phrase, not English
     "Registro Nacional de las Personas" ✅  Spanish phrase, not English
     FAIL: "passport number"      ❌  plain English phrase

   STRICT RULE: Do NOT include any keyword that fails BOTH tests.
   - Pure ASCII single words (e.g. "KT", "ID", "NO", "DOB") → EXCLUDED
     (these are Latin-script ASCII and can appear in any language)
   - Pure ASCII abbreviations with no diacritics (e.g. "RENAPER" if the full
     name is in Spanish but the abbreviation itself is ASCII) → include ONLY
     if the abbreviation is exclusively used in that country's system and has
     zero common meaning in English (e.g. "RENAPER" is fine;
     "DOC", "REG", "NO", "KT" are NOT fine)
   - English-language labels for the identifier (e.g. "Argentina passport",
     "Argentine national ID") → ALLOWED because they explicitly name the
     country/language in English, making them selective

   NON-LATIN SCRIPTS — SPECIAL RULE (Cyrillic, Arabic, Hebrew, Devanagari,
   Greek, CJK, Georgian, Armenian, Thai, etc.):
   Any keyword written in a non-Latin script is AUTOMATICALLY language-exclusive
   because its Unicode code points physically cannot appear in Latin/English text.
   This applies even to short abbreviations:
     "КТ"   ✅  Cyrillic К+Т — regex КТ will NEVER match Latin KT
     "МРТ"  ✅  Cyrillic М+Р+Т — regex МРТ will NEVER match Latin MRT
     "ЦНС"  ✅  Cyrillic — impossible in English text
     "מס׳"  ✅  Hebrew — impossible in English text
     "رقم"  ✅  Arabic — impossible in English text
   INCLUDE all such abbreviations freely — they are inherently script-exclusive.

   Think broadly across these categories:
     - Form field labels (e.g. "ZIP code", "postal code", "mailing address")
     - Column headers in spreadsheets/databases
     - Surrounding text in official documents, forms, invoices
     - Abbreviations and acronyms used in the industry
     - LOCAL LANGUAGE terms used in official forms and documents of that country
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

    # Per-position character observations (additive — what IS present).
    # These are statistical observations for the LLM to learn structural
    # patterns (e.g. "all values start with a letter", "position 4 is
    # always a digit").  They are NOT hard constraints — only official
    # policy or vendor sources can produce hard range_restrictions.
    statistical_observations: list[dict] = []
    all_digits = set("0123456789")
    all_upper = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    dominant_length = length_mode
    check_positions = list(range(min(dominant_length, 6))) if dominant_length >= 2 else [0]
    if dominant_length >= 2 and (dominant_length - 1) not in check_positions:
        check_positions.append(dominant_length - 1)

    for pos in check_positions:
        if pos < 0:
            continue
        chars_at_pos = Counter(v[pos] for v in values if len(v) > pos)
        observed = set(chars_at_pos.keys())
        sample_count = sum(chars_at_pos.values())

        if observed.issubset(all_digits):
            char_type = "digits"
        elif observed.issubset(all_upper):
            char_type = "letters"
        elif observed.issubset(all_digits | all_upper):
            char_type = "alphanumeric"
        else:
            char_type = "mixed"

        # Check for universally fixed characters (e.g. position 4 is ALWAYS '0')
        if len(observed) == 1:
            fixed_char = next(iter(observed))
            statistical_observations.append({
                "position": pos,
                "char_type": char_type,
                "observed_chars": sorted(observed),
                "sample_count": sample_count,
                "note": f"Position {pos} is always '{fixed_char}' across all {sample_count} samples.",
                "confidence": "high" if sample_count >= 50 else "medium",
            })
        else:
            statistical_observations.append({
                "position": pos,
                "char_type": char_type,
                "observed_chars": sorted(observed),
                "sample_count": sample_count,
                "note": (
                    f"Position {pos} contains {char_type}: "
                    f"{sorted(observed)} across {sample_count} samples."
                ),
                "confidence": "high" if sample_count >= 200 else "low",
            })

    # Detect universal prefix patterns (e.g. "LV-", "+91 ", "784-")
    common_prefixes: list[dict] = []
    for plen in (2, 3, 4):
        pfx_counter = Counter(v[:plen] for v in values if len(v) >= plen)
        total = sum(pfx_counter.values())
        for pfx, cnt in pfx_counter.most_common(3):
            pct = cnt / total if total else 0
            if pct >= 0.95:
                common_prefixes.append({
                    "prefix": pfx,
                    "frequency": f"{pct:.1%}",
                    "count": cnt,
                    "note": f"'{pfx}' appears as prefix in {pct:.0%} of values — likely a fixed prefix.",
                })

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
        "statistical_observations": statistical_observations,
        "common_prefixes": common_prefixes,
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
                "\nVENDOR/COMPETITOR DLP PATTERNS — mine these for format rules "
                "AND range restrictions.\n"
                "For every vendor regex provided:\n"
                "  1. Decode each character class and quantifier into a format_rule.\n"
                "  2. If a position uses a restricted class (e.g. [2-9], [A-Z], [013]),\n"
                "     add it as a range_restriction citing the vendor as source.\n"
                "  3. Add any vendor-listed rules as additional format_rules.\n"
                "  EXAMPLE: regex=([2-9]\\d{3}\\s?\\d{4}\\s?\\d{4}) means position 0\n"
                "  is restricted to [2-9] — that should appear in range_restrictions\n"
                "  with source citing the vendor.\n\n"
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

    # Attach statistical observations from pre-analysis so downstream agents
    # can use them as soft context (but never as hard constraints).
    patterns["statistical_observations"] = stats.get("statistical_observations", [])
    patterns["common_prefixes"] = stats.get("common_prefixes", [])

    # Cross-check: mine numbering_policy free text for per-position constraints
    # that the LLM may have written correctly in prose but missed in range_restrictions.
    patterns = _backfill_range_restrictions_from_policy(patterns)

    with open(patterns_path, "w") as f:
        json.dump(patterns, f, indent=2)

    rule_count = len(patterns.get("format_rules", []))
    kw_count = len(patterns.get("contextual_keywords", []))
    rr_count = len(patterns.get("range_restrictions", []))
    obs_count = len(patterns.get("statistical_observations", []))
    logger.info(
        "Agent 2 done: %d format rules, %d range restrictions (policy-only), "
        "%d statistical observations, %d keywords discovered",
        rule_count, rr_count, obs_count, kw_count,
    )

    return patterns


# Patterns that indicate a first-digit restriction mentioned in free text.
import re as _re

_FIRST_DIGIT_CANNOT = _re.compile(
    r"first\s+digit[^.]*cannot\s+be\s+([\d\s,and]+)",
    _re.IGNORECASE,
)
_FIRST_DIGIT_RANGE = _re.compile(
    r"first\s+digit[^.]*(?:must\s+be|is\s+always|range[s]?\s+from?|between)\s*"
    r"[\[\(]?(\d)\s*[-–to]+\s*(\d)[\]\)]?",
    _re.IGNORECASE,
)
_FIRST_DIGIT_FIXED = _re.compile(
    r"first\s+digit[^.]*(?:is\s+always|must\s+be|is\s+fixed\s+at|always\s+starts?\s+with)\s*(\d)",
    _re.IGNORECASE,
)


def _backfill_range_restrictions_from_policy(patterns: dict) -> dict:
    """Parse numbering_policy prose for per-position constraints and ensure
    they appear in range_restrictions so the regex generator sees them."""
    policy = patterns.get("numbering_policy", "")
    if not policy:
        return patterns

    existing_rr: list[dict] = patterns.setdefault("range_restrictions", [])
    existing_positions = {str(r.get("position", "")).strip() for r in existing_rr}

    inferred: list[dict] = []

    # "first digit cannot be 0 or 1" → [2-9]
    m = _FIRST_DIGIT_CANNOT.search(policy)
    if m and "0" not in existing_positions:
        forbidden_digits = set(_re.findall(r"\d", m.group(1)))
        all_digits = set("0123456789")
        allowed = sorted(all_digits - forbidden_digits)
        if allowed:
            allowed_str = f"[{allowed[0]}-{allowed[-1]}]" if len(allowed) >= 2 else f"[{''.join(allowed)}]"
            inferred.append({
                "position": "0",
                "allowed_values": allowed_str,
                "source": f"Inferred from numbering_policy: '{m.group(0).strip()}'",
            })
            logger.info(
                "Agent 2 [policy-backfill]: position 0 → %s (forbidden: %s)",
                allowed_str, sorted(forbidden_digits),
            )

    # "first digit must be between 2 and 9" or "range 2-9"
    m = _FIRST_DIGIT_RANGE.search(policy)
    if m and "0" not in existing_positions and not inferred:
        lo, hi = m.group(1), m.group(2)
        allowed_str = f"[{lo}-{hi}]"
        inferred.append({
            "position": "0",
            "allowed_values": allowed_str,
            "source": f"Inferred from numbering_policy: '{m.group(0).strip()}'",
        })
        logger.info("Agent 2 [policy-backfill]: position 0 → %s (range)", allowed_str)

    # "first digit is always 6"
    m = _FIRST_DIGIT_FIXED.search(policy)
    if m and "0" not in existing_positions and not inferred:
        digit = m.group(1)
        inferred.append({
            "position": "0",
            "allowed_values": digit,
            "source": f"Inferred from numbering_policy: '{m.group(0).strip()}'",
        })
        logger.info("Agent 2 [policy-backfill]: position 0 → %s (fixed)", digit)

    if inferred:
        # Remove any overly broad position-0 entry (e.g. "[0-9]") and replace.
        existing_rr[:] = [
            r for r in existing_rr
            if not (str(r.get("position", "")).strip() in ("0", "0-13", "1")
                    and r.get("allowed_values") in ("[0-9]", "0-9", "\\d"))
        ]
        existing_rr[:0] = inferred  # prepend so position 0 appears first

    return patterns
