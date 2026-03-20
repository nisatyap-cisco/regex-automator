from __future__ import annotations

import json
import logging
import re

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a regex engineering expert.

Write a single Python-compatible regex that matches values satisfying this rule:

RULE: {rule_description}
RANGE CONSTRAINTS: {range_restrictions}
EXAMPLE MATCHES: {example_matches}

=== AUTHORITATIVE NUMBERING POLICY (HIGHEST PRIORITY — these override data patterns) ===
{numbering_policy}

=== DATA-DERIVED OBSERVATIONS (supplementary — use ONLY when they don't conflict with policy) ===
{data_observations}

PRIORITY RULES (you MUST follow these):
1. POLICY ALWAYS WINS: If an official policy rule contradicts a data-derived
   observation, follow the POLICY. Data may be incomplete or biased.
2. Policy-stated restrictions on character positions (e.g. "first digit is 2-9",
   "checksum can be digit or letter") MUST be encoded exactly as stated.
3. Data observations are useful for format hints (delimiters, segment structure)
   but NEVER override policy on allowed values or lengths.
4. If policy says nothing about a position but data shows a restriction, you may
   use the data observation — but prefer the broader range if the sample is small.

Requirements:
- Use raw string notation.
- Do NOT use ^ or $ anchors. The regex will be used for search within text.
- Use \\b (word boundary) at the start and end of the pattern to prevent
  partial matches within longer strings.
- IMPORTANT: \\b before a non-word character like + or ( will NOT match.
  For patterns starting with a non-word character (e.g. +91), use a
  lookbehind (?<=\\s|^) or omit the leading \\b and rely on the trailing \\b.
- Avoid catastrophic backtracking.
- Prefer character classes and quantifiers over alternation where possible.
- CRITICAL: If the numbering policy states a per-position restriction
  (e.g. "first digit cannot be 0 or 1", "first digit is always 6"),
  you MUST encode it as an explicit character class at that position
  (e.g. [2-9] instead of \\d, or [6] instead of \\d).
- NEVER include literal placeholder strings like "XXX" or "NNN" in the regex.
  If an optional part can be any characters, use the appropriate character
  class (e.g. [A-Z0-9]{{3}} not XXX).
- In character classes, do NOT use commas to separate ranges. Write [0-68-9]
  not [0-6,8-9] — a comma inside [] is a literal character in regex.
- If the rule involves check digits, encode the structural pattern only
  (check-digit validation is done in code, not regex).

Return ONLY the regex string, no explanation."""

COMBINED_PROMPT_TEMPLATE = """You are a regex engineering expert.

Write a SINGLE Python-compatible regex that matches values satisfying ALL of the following rules SIMULTANEOUSLY (AND logic, not OR):

RULES:
{rules_list}

RANGE CONSTRAINTS: {range_restrictions}
EXAMPLE MATCHES: {example_matches}

=== AUTHORITATIVE NUMBERING POLICY (HIGHEST PRIORITY — these override data patterns) ===
{numbering_policy}

=== DATA-DERIVED OBSERVATIONS (supplementary — use ONLY when they don't conflict with policy) ===
{data_observations}

PRIORITY RULES (you MUST follow these):
1. POLICY ALWAYS WINS: If an official policy rule contradicts a data-derived
   observation, follow the POLICY. Data may be incomplete or biased.
2. Policy-stated restrictions MUST be encoded exactly as stated.
3. Data observations are supplementary context only.

Requirements:
- The regex must enforce ALL rules at the same time — a value must satisfy every rule to match.
- Use lookahead assertions ((?=...)) to layer simultaneous constraints where needed.
- Do NOT use ^ or $ anchors inside lookaheads or the main pattern.
  The regex will be used for search within text.
- Use \\b (word boundary) at the start and end of the pattern to prevent partial matches.
- IMPORTANT: \\b before a non-word character like + or ( will NOT match.
  For patterns starting with a non-word character, use a lookbehind or omit leading \\b.
- Avoid catastrophic backtracking.
- Prefer the most specific structural pattern that naturally satisfies all constraints over a
  chain of lookaheads when possible.
- CRITICAL: If the numbering policy states a per-position restriction
  (e.g. "first digit cannot be 0 or 1", "second character must be a letter"),
  you MUST encode it as an explicit character class at that position
  (e.g. [2-9]\\d{{3}} \\d{{4}} \\d{{4}} instead of \\d{{4}} \\d{{4}} \\d{{4}}).
  Do NOT substitute \\d where a restricted character class applies.
- NEVER use alternation (|) to join the individual rule regexes — that would be OR logic.
- NEVER include literal placeholder strings like "XXX" or "NNN".
- In character classes, do NOT use commas to separate ranges. Write [0-68-9]
  not [0-6,8-9] — a comma inside [] is a literal character in regex.
- If a rule involves check digits, encode the structural pattern only.

Return ONLY the regex string, no explanation."""

REFINEMENT_TEMPLATE = """The regex you provided failed to compile in Python.

ORIGINAL REGEX: {regex}
COMPILE ERROR: {error}

Fix the regex so it compiles with Python's re module.
Return ONLY the corrected regex string, no explanation."""


# Mapping of accented/diacritic characters to their ASCII base equivalents.
# Used to generate [base|accented] character classes so keywords match both
# the proper Unicode form (e.g. "Número") and the plain ASCII fallback a user
# might type when the character is not on their keyboard (e.g. "Numero").
_DIACRITIC_MAP: dict[str, str] = {
    # --- lowercase ---
    # a-variants
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
    'ā': 'a', 'ă': 'a', 'ą': 'a',
    # e-variants
    'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
    'ē': 'e', 'ĕ': 'e', 'ę': 'e', 'ě': 'e',
    # i-variants
    'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
    'ī': 'i', 'ĭ': 'i', 'į': 'i',
    # o-variants
    'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
    'ō': 'o', 'ŏ': 'o', 'ő': 'o',
    # u-variants
    'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
    'ū': 'u', 'ŭ': 'u', 'ů': 'u', 'ű': 'u',
    # y-variants
    'ý': 'y', 'ÿ': 'y',
    # n-variants
    'ñ': 'n', 'ń': 'n', 'ň': 'n',
    # c-variants
    'ç': 'c', 'ć': 'c', 'č': 'c',
    # z-variants
    'ž': 'z', 'ź': 'z', 'ż': 'z',
    # s-variants
    'š': 's', 'ś': 's',
    # r-variants
    'ř': 'r', 'ŗ': 'r',
    # l-variants
    'ļ': 'l', 'ł': 'l', 'ľ': 'l',
    # k-variants
    'ķ': 'k',
    # g-variants
    'ģ': 'g', 'ğ': 'g',
    # d-variants
    'ď': 'd', 'đ': 'd',
    # t-variants
    'ț': 't', 'ţ': 't', 'ť': 't',
    # --- uppercase ---
    'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A',
    'Ā': 'A', 'Ă': 'A', 'Ą': 'A',
    'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
    'Ē': 'E', 'Ĕ': 'E', 'Ę': 'E', 'Ě': 'E',
    'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I',
    'Ī': 'I', 'Ĭ': 'I', 'Į': 'I',
    'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
    'Ō': 'O', 'Ŏ': 'O', 'Ő': 'O',
    'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
    'Ū': 'U', 'Ŭ': 'U', 'Ů': 'U', 'Ű': 'U',
    'Ý': 'Y',
    'Ñ': 'N', 'Ń': 'N', 'Ň': 'N',
    'Ç': 'C', 'Ć': 'C', 'Č': 'C',
    'Ž': 'Z', 'Ź': 'Z', 'Ż': 'Z',
    'Š': 'S', 'Ś': 'S',
    'Ř': 'R', 'Ŗ': 'R',
    'Ļ': 'L', 'Ł': 'L', 'Ľ': 'L',
    'Ķ': 'K',
    'Ģ': 'G', 'Ğ': 'G',
    'Ď': 'D', 'Đ': 'D',
    'Ț': 'T', 'Ţ': 'T', 'Ť': 'T',
}


def _escape_keyword(kw: str) -> str:
    """Escape regex metacharacters, preserve spaces and Unicode, and expand
    diacritics into [base_char|accented_char] character classes.

    This handles the common case where a user types a keyword without special
    characters because the key is not on their keyboard — e.g. typing "Numero"
    instead of "Número".  For every accented character the output character
    class accepts both the plain ASCII base and the accented form:
        'ú' → [uú],  'é' → [eé],  'ó' → [oó],  'ñ' → [nñ], …

    Example:  "N[uú]mero de pasaporte"  matches both "Número de pasaporte"
              and "Numero de pasaporte".
    """
    specials = set(r'\.^$*+?{}[]|()')
    result: list[str] = []
    for ch in kw:
        base = _DIACRITIC_MAP.get(ch)
        if base is not None:
            # Build [base_char accented_char] — no escaping needed inside []
            result.append(f'[{base}{ch}]')
        elif ch in specials:
            result.append(f'\\{ch}')
        else:
            result.append(ch)
    return ''.join(result)


def _is_language_exclusive(kw: str) -> bool:
    """Return True when a keyword is inherently tied to the local language
    of the identifier and therefore CANNOT appear in unrelated foreign-language
    documents.

    A keyword is language-exclusive when it satisfies at least ONE of:
      1. Contains a non-ASCII character (diacritic, CJK, Hebrew, etc.) —
         such characters are structurally absent from plain English/ASCII text,
         so the keyword can only match in documents of the correct script.
         Examples: "Número" (ú), "Dirección" (ó), "pasta indekss" (pure ASCII →
         FAILS this test), "personnummer" (FAILS — pure ASCII)
      2. Is a multi-word phrase where at least ONE word is NOT a common English
         word.  This test is approximated by checking that the phrase contains
         a space AND that at least one token is not pure ASCII OR the phrase
         as a whole could not plausibly appear in an English document.
         Examples: "pasaporte argentino" ✅ (Spanish words), "ZIP code" ❌
         (English phrase).

    NOTE: This function is intentionally conservative.  It flags a keyword
    as potentially non-exclusive if it is pure ASCII and short (≤ 4 chars),
    but does NOT drop it — the calling code emits a warning so the human
    operator can review and correct the keyword list.
    """
    stripped = kw.strip()
    # Test 1: contains any non-ASCII character → definitely language-exclusive
    if not stripped.isascii():
        return True
    # Heuristic for Test 2: multi-word AND not a pair of plain English short words
    tokens = stripped.split()
    if len(tokens) >= 2:
        # If the phrase contains the country/language name it's selective enough
        # (e.g. "Argentina passport", "Argentine national ID")
        return True
    # Single-token, pure ASCII → NOT language-exclusive
    return False


def build_keyword_regex(keywords: list[str], proximity: int = 10) -> str:
    """Build keyword proximity regex: (?i)(?:kw1|kw2|...) — no proximity suffix.

    Each keyword is processed by ``_escape_keyword`` which expands diacritics
    into [base|accented] character classes.

    A passive language-exclusivity check is run via ``_is_language_exclusive``:
    keywords that fail (pure-ASCII single-token strings like "KT" or "DNI")
    are logged as warnings so the operator can decide whether to keep them.
    They are NOT removed — removal would require changing Agent 2's output.
    The intent is that Agent 2's prompt already prevents such keywords from
    being generated; this warning surfaces any that slipped through.
    """
    for kw in keywords:
        if not _is_language_exclusive(kw):
            logger.warning(
                "Agent 3 [lang-exclusive]: keyword %r is a pure-ASCII single token "
                "and may appear in unrelated non-%s-language documents, causing "
                "false-positive keyword matches.  Consider replacing it with the "
                "full local-language phrase or an accented form.",
                kw,
                "local",
            )
    kw_pattern = "|".join(_escape_keyword(k) for k in keywords)
    return f"(?i)(?:{kw_pattern})"


def _build_keywords_readable(kw_regex: str) -> str:
    """Convert a keyword proximity regex back to a human-readable comma-separated list.

    Transformation rules (applied in order):
    1. Strip the ``(?i)(?:`` prefix and trailing ``)``
    2. Replace ``\\s+`` → single space
    3. Replace ``[xY]`` character classes → accented char (last char of class),
       e.g. ``[uú]`` → ``ú``,  ``[oó]`` → ``ó``,  ``[eé]`` → ``é``
    4. Split on unescaped ``|``
    5. Strip and join with ``", "``
    """
    inner = kw_regex.strip()

    # Strip (?i)(?:...) wrapper
    m = re.match(r'^\(\?i\)\(\?:(.*)\)$', inner, re.DOTALL)
    if m:
        inner = m.group(1)

    # \s+ → single space
    inner = re.sub(r'\\s\+', ' ', inner)

    # [xY] → last character in class (the accented/diacritic form)
    # Matches classes of exactly 2 characters, e.g. [uú], [oó], [Uu]
    inner = re.sub(r'\[(.)(.)?\]', lambda mo: mo.group(2) if mo.group(2) else mo.group(1), inner)

    # Split on | and clean up
    parts = [p.strip() for p in inner.split('|') if p.strip()]
    return ', '.join(parts)


def _try_compile(pattern: str) -> re.Pattern | None:
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _clean_regex_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    for prefix in ("r'", 'r"', "r'''", 'r"""'):
        if text.startswith(prefix):
            quote = prefix[1:]
            if text.endswith(quote):
                text = text[len(prefix):-len(quote)]
                break
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
    return text.strip()


def _strip_anchors_add_boundaries(regex_str: str) -> str:
    """Strip ^ and $ anchors, add \\b word boundaries."""
    r = regex_str.strip()
    if r.startswith("^"):
        r = r[1:]
    if r.endswith("$"):
        r = r[:-1]
    r = r.strip()
    if "XXX" in r:
        r = r.replace("XXX", "[A-Z0-9]{3}")
    if not r.startswith("\\b"):
        r = "\\b" + r
    if not r.endswith("\\b"):
        r = r + "\\b"
    return r


def generate_regex(
    condition: str,
    config: dict,
    paths: dict[str, str] | None = None,
    policies: dict | None = None,
    data_patterns: dict | None = None,
) -> dict:
    """Generate regex patterns from policies (primary) and data patterns (supplementary).

    When both policies and data_patterns are available, policies take priority.
    Data-derived format rules and range restrictions are passed as supplementary
    observations that the LLM may use when they don't conflict with policy.

    Args:
        condition: The identifier type (e.g. "India Bank Account Number")
        config: Configuration dict
        paths: Path dict for output files
        policies: Policy Researcher output containing policies, vendor_patterns, etc.
        data_patterns: Pattern Analyzer output (None when no real data was collected)
    """
    regex_path = (paths or {}).get("regex_patterns", "results/regex_patterns.json")

    # ── Build format_rules: policy rules are PRIMARY ──
    format_rules: list[dict] = []
    if policies and policies.get("policies"):
        for i, rule in enumerate(policies["policies"]):
            format_rules.append({
                "rule_id": f"policy-rule-{i+1}",
                "description": rule,
                "applies_to": "all",
                "example_matches": [],
            })

    # Extract vendor regex as additional format rules
    if policies and policies.get("vendor_patterns"):
        for vp in policies["vendor_patterns"]:
            vendor = vp.get("vendor", "Unknown")
            vendor_rules = vp.get("rules", [])
            if vendor_rules:
                for j, vrule in enumerate(vendor_rules):
                    format_rules.append({
                        "rule_id": f"vendor-{vendor.lower().replace(' ', '-')}-{j+1}",
                        "description": f"[{vendor}] {vrule}",
                        "applies_to": "all",
                        "example_matches": [],
                    })

    # Merge example_matches from data_patterns into policy rules where applicable
    if data_patterns and data_patterns.get("format_rules"):
        all_data_examples: list[str] = []
        for dr in data_patterns["format_rules"]:
            all_data_examples.extend(dr.get("example_matches", []))
        unique_examples = list(dict.fromkeys(all_data_examples))
        for fr in format_rules:
            if not fr.get("example_matches"):
                fr["example_matches"] = unique_examples[:5]

    # ── Range restrictions: prefer policy, supplement with data ──
    range_restrictions: list[dict] = []
    if data_patterns and data_patterns.get("range_restrictions"):
        range_restrictions = list(data_patterns["range_restrictions"])

    # ── Keywords: prefer data_patterns (richer), fallback to vendor ──
    keywords: list[str] = []
    if data_patterns and data_patterns.get("contextual_keywords"):
        keywords = data_patterns["contextual_keywords"]
    if not keywords and policies and policies.get("vendor_keywords"):
        keywords = policies["vendor_keywords"]
    elif not keywords and policies and policies.get("vendor_patterns"):
        for vp in policies["vendor_patterns"]:
            keywords.extend(vp.get("keywords", []))
        keywords = list(dict.fromkeys(keywords))

    proximity = 10
    if data_patterns and data_patterns.get("keyword_proximity"):
        proximity = data_patterns["keyword_proximity"]

    # ── Build the AUTHORITATIVE policy string (highest priority) ──
    enriched_policy = ""
    if policies:
        raw_rules = policies.get("policies", [])
        vendor_patterns = policies.get("vendor_patterns", [])

        if raw_rules:
            rules_block = "\n".join(f"- {r}" for r in raw_rules)
            enriched_policy += (
                f"AUTHORITATIVE FORMAT RULES (from official sources — "
                f"encode ALL of these in the regex):\n{rules_block}"
            )

        if vendor_patterns:
            vp_lines: list[str] = []
            for vp in vendor_patterns:
                vendor = vp.get("vendor", "Unknown")
                vr     = vp.get("regex", "")
                vrules = vp.get("rules", [])
                block  = f"\n  {vendor}:"
                if vr:
                    block += f"\n    Reference regex: {vr}"
                if vrules:
                    block += "\n    Structural rules:\n" + "\n".join(f"      - {r}" for r in vrules)
                vp_lines.append(block)
            enriched_policy += (
                "\n\nVENDOR DLP PATTERN REFERENCE "
                "(use for structural validation, not as the final regex):\n"
                + "".join(vp_lines)
            )

    # ── Build the DATA OBSERVATIONS string (supplementary, lower priority) ──
    data_observations = "(no real-world data was collected — rely on policy rules above)"
    if data_patterns:
        obs_parts: list[str] = []
        if data_patterns.get("format_rules"):
            obs_parts.append("Data-derived format observations:")
            for dr in data_patterns["format_rules"]:
                obs_parts.append(f"  - {dr.get('description', '')} (examples: {dr.get('example_matches', [])[:3]})")
        if data_patterns.get("range_restrictions"):
            obs_parts.append("Data-derived position constraints:")
            for rr in data_patterns["range_restrictions"]:
                obs_parts.append(
                    f"  - Position {rr.get('position')}: allowed={rr.get('allowed_values')} "
                    f"(source: {rr.get('source', 'data')})"
                )
        np_text = data_patterns.get("numbering_policy", "")
        if np_text:
            obs_parts.append(f"Data-derived numbering policy summary: {np_text}")
        data_observations = "\n".join(obs_parts) if obs_parts else "(no data patterns)"

    has_data = data_patterns is not None
    logger.info(
        "Agent 3: policy injected (%d chars) | data observations %s (%d chars)",
        len(enriched_policy),
        "available" if has_data else "NONE",
        len(data_observations),
    )
    logger.info("Agent 3: generating regex for %d rules + keyword pattern", len(format_rules))

    if not format_rules:
        logger.info("Agent 3: no format_rules from policies, creating direct condition rule")
        format_rules = [{
            "rule_id": "direct-condition",
            "description": f"Match valid {condition} values",
            "applies_to": "all",
            "example_matches": [],
        }]

    range_lookup: dict[str, dict] = {}
    for rr in range_restrictions:
        pos = rr.get("position", "")
        if isinstance(pos, list):
            pos = str(pos)
        range_lookup[pos] = rr

    output_patterns: list[dict] = []

    for rule in format_rules:
        rule_id = rule.get("rule_id", "unnamed")
        description = rule.get("description", "")
        examples = rule.get("example_matches", [])

        range_text = json.dumps(
            [rr for rr in range_restrictions],
            indent=2,
        )

        prompt = PROMPT_TEMPLATE.format(
            rule_description=description,
            range_restrictions=range_text,
            example_matches=", ".join(str(e) for e in examples),
            numbering_policy=enriched_policy or "(none provided)",
            data_observations=data_observations,
        )

        response = invoke_claude(prompt, config)
        regex_str = _clean_regex_response(response)
        regex_str = _strip_anchors_add_boundaries(regex_str)

        compiled = _try_compile(regex_str)
        if compiled is None:
            logger.warning("Agent 3: regex for '%s' failed compile, re-prompting", rule_id)
            try:
                re.compile(regex_str)
            except re.error as e:
                error_msg = str(e)

            refinement = REFINEMENT_TEMPLATE.format(regex=regex_str, error=error_msg)
            response2 = invoke_claude(refinement, config)
            regex_str = _clean_regex_response(response2)
            regex_str = _strip_anchors_add_boundaries(regex_str)

            compiled = _try_compile(regex_str)
            if compiled is None:
                logger.error("Agent 3: regex for '%s' still invalid after refinement, keeping as-is", rule_id)

        output_patterns.append({
            "rule_id": rule_id,
            "regex": regex_str,
            "description": description,
            "type": "value",
        })
        logger.info("Agent 3: [%s] → %s", rule_id, regex_str)

    # Build combined/unified value regex satisfying ALL individual rules simultaneously (AND logic)
    value_patterns = [p for p in output_patterns if p["type"] == "value"]
    all_examples = []
    for rule in format_rules:
        all_examples.extend(rule.get("example_matches", []))
    unique_examples = list(dict.fromkeys(all_examples))

    if len(value_patterns) > 1:
        rules_list = "\n".join(
            f"{i + 1}. [{vp['rule_id']}] {vp['description']}"
            for i, vp in enumerate(value_patterns)
        )
        combined_prompt = COMBINED_PROMPT_TEMPLATE.format(
            rules_list=rules_list,
            range_restrictions=json.dumps(range_restrictions, indent=2),
            example_matches=", ".join(str(e) for e in unique_examples),
            numbering_policy=enriched_policy or "(none provided)",
            data_observations=data_observations,
        )
        combined_response = invoke_claude(combined_prompt, config)
        combined_regex = _clean_regex_response(combined_response)
        combined_regex = _strip_anchors_add_boundaries(combined_regex)

        compiled = _try_compile(combined_regex)
        if compiled is None:
            logger.warning("Agent 3: combined-value regex failed compile, re-prompting")
            try:
                re.compile(combined_regex)
            except re.error as e:
                error_msg = str(e)
            refinement = REFINEMENT_TEMPLATE.format(regex=combined_regex, error=error_msg)
            combined_response2 = invoke_claude(refinement, config)
            combined_regex = _clean_regex_response(combined_response2)
            combined_regex = _strip_anchors_add_boundaries(combined_regex)
            compiled = _try_compile(combined_regex)

        if compiled is not None:
            output_patterns.append({
                "rule_id": "combined-value",
                "regex": combined_regex,
                "description": "Unified value regex satisfying all individual rules simultaneously",
                "type": "value",
            })
            logger.info("Agent 3: [combined-value] → %s", combined_regex)
        else:
            logger.error("Agent 3: combined-value regex still invalid after refinement, skipping")
    elif len(value_patterns) == 1:
        output_patterns.append({
            "rule_id": "combined-value",
            "regex": value_patterns[0]["regex"],
            "description": "Unified value regex (same as single rule)",
            "type": "value",
        })

    if keywords:
        kw_regex = build_keyword_regex(keywords, proximity)
        compiled = _try_compile(kw_regex)
        if compiled is None:
            logger.error("Agent 3: keyword proximity regex failed compile")
        else:
            output_patterns.append({
                "rule_id": "keyword-proximity",
                "regex": kw_regex,
                "keywords_readable": _build_keywords_readable(kw_regex),
                "description": f"Contextual keywords within {proximity}-term proximity",
                "type": "context",
            })
            logger.info("Agent 3: [keyword-proximity] → %s", kw_regex)

    result = {
        "condition": condition,
        "patterns": output_patterns,
    }

    with open(regex_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Agent 3 done: %d patterns written", len(output_patterns))
    return result
