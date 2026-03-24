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

NUMBERING POLICY (authoritative real-world rules — encode ALL constraints here):
{numbering_policy}

Requirements:
- Do NOT use ^ or $ anchors. The regex will be used for search within text.
- Use \\b (word boundary) at the start and end of the pattern.
- Avoid catastrophic backtracking.
- Prefer character classes and quantifiers over alternation where possible.
- CRITICAL per-position restrictions: use explicit character classes, never \\d or .
  where a stricter class applies (e.g. [2-9] not \\d when first digit cannot be 0 or 1).
- REDUNDANCY: if the pattern has a fixed length through explicit quantifiers,
  do NOT add a length lookahead — it is redundant.
- NEVER include literal placeholder strings like "XXX" or "NNN".
- If the rule involves check digits, encode the structural pattern only.
- SEPARATOR HANDLING: If the numbering policy or vendor patterns indicate this
  identifier is commonly displayed with separators (spaces, hyphens, periods)
  between digit groups, include optional separator groups like [\s-]? between
  each digit group so both the separated and contiguous forms match.

Return ONLY the regex string, no explanation, no markdown."""

COMBINED_PROMPT_TEMPLATE = """You are a regex engineering expert tasked with producing the mathematically minimal unified regex.

Given the individual rules and their regexes below, produce ONE unified Python-compatible regex with ZERO duplicated constraints.

INDIVIDUAL RULES:
{rules_list}

RANGE CONSTRAINTS: {range_restrictions}
EXAMPLE MATCHES: {example_matches}

NUMBERING POLICY (authoritative — encode ALL structural constraints here):
{numbering_policy}

MINIMALITY RULES — follow strictly:

1. MERGE first: encode every constraint directly into the positional pattern where possible.
   A constraint that fits into the character class at a specific position must go there —
   NOT in a lookahead. Example: "first char is a letter" → [A-Z] at position 0, not (?=[A-Z]).

2. LOOKAHEADS only for cross-position constraints that CANNOT be expressed positionally
   (e.g. "total digit count across non-contiguous segments must equal N").
   If a lookahead only re-asserts something already guaranteed by the main pattern, DELETE it.

3. REMOVE redundant length assertions: if the final pattern has a fixed length through
   explicit quantifiers (e.g. [A-Z]{{2}}\\d{{4}}[A-Z]{{2}} = exactly 8 chars), do NOT also
   add (?=.{{8}}) or (?=\\S{{8}}) — that is redundant.

4. REMOVE general patterns subsumed by stricter ones: if one rule says \\d{{10}} and another
   says [2-9]\\d{{9}}, use only [2-9]\\d{{9}} — the general form is made redundant by the strict one.

5. NEVER use alternation (|) across the individual rule regexes — that would be OR logic.

6. NEVER use ^ or $ anchors. Use \\b at start and end for word boundary.

7. NEVER include literal placeholders like XXX or NNN.

8. If a rule involves check digits, encode the structural pattern only (not the check algorithm).

9. CRITICAL per-position restrictions: always use explicit character classes, never \\d or .
   where a stricter class applies (e.g. [2-9] not \\d when first digit cannot be 0 or 1).

10. SEPARATOR HANDLING: If the numbering policy or vendor patterns indicate this
    identifier is commonly displayed with separators (spaces, hyphens, periods)
    between digit groups, include optional separator groups like [\s-]? between
    each digit group. The contiguous (no-separator) form MUST also match.
    Do NOT drop separators during merging — they are NOT redundancy.

Before writing the final regex, mentally verify:
- Is every constraint from the rules encoded exactly once?
- Does any lookahead duplicate what the main pattern already guarantees? If yes, remove it.
- Does the pattern have a fixed length through quantifiers? If yes, remove any length lookahead.

Return ONLY the final unified regex string. No explanation, no markdown, no comments."""

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


# ── Separator injection ───────────────────────────────────────────────────────

# Regex that matches digit-consuming tokens in a regex string.
_DIGIT_TOKEN_PATTERN = re.compile(
    r'(?:'
    r'\\d(?:\{\d+\})?'              # \d or \d{N}
    r'|\[[0-9][0-9\-]*\](?:\{\d+\})?'  # [2-9], [0-9]{3}, etc.
    r')'
)

# Regex that matches existing rigid separator tokens between digit groups.
_RIGID_SEP_PATTERN = re.compile(
    r'(?:'
    r'\\[.]'                        # \.
    r'|\\s[?]?'                     # \s or \s?
    r'|\[(?:[^\]]*(?:\\s|\\\-|\s|\-|\.)[^\]]*)\][?]?'  # [\s-]?, etc.
    r'|(?<!\\)-'                    # literal hyphen (not escaped \-)
    r')'
)


def _digit_token_width(token: str) -> int:
    """Return how many digits a single regex token consumes.

    Examples: \\d{4} → 4, \\d → 1, [2-9] → 1, [0-9]{3} → 3.
    """
    m = re.search(r'\{(\d+)\}$', token)
    if m:
        return int(m.group(1))
    return 1


def _build_separator_class(observed_separators: list[str]) -> str:
    """Build an optional separator character class from observed separator chars.

    Examples:
        [" ", "-"]   → "[\\s\\-]?"
        ["."]        → "[.]?"
        [" "]        → "\\s?"
        [".", " ", "-"] → "[.\\s\\-]?"
    """
    if not observed_separators:
        return "[\\s\\-]?"  # safe default

    parts: list[str] = []
    for sep in observed_separators:
        if sep == " ":
            parts.append("\\s")
        elif sep == "-":
            parts.append("\\-")
        elif sep == ".":
            parts.append(".")
        else:
            parts.append(re.escape(sep))

    if len(parts) == 1 and parts[0] == "\\s":
        return "\\s?"
    return "[" + "".join(parts) + "]?"


def _inject_optional_separators(regex_str: str, separator_spec: dict) -> str:
    """Deterministic post-processing: inject optional separators between digit groups.

    Uses the separator_spec (from vendor/policy analysis) to know:
      - group_lengths: e.g. [4, 4, 4] for Aadhaar
      - observed_separators: e.g. [" ", "-"]

    Walks the regex, accumulates digit-token widths into "buckets" defined
    by group_lengths, and inserts separator character classes at each
    bucket boundary.  Existing rigid separators (like \\.) are replaced
    with flexible ones (like [.\\s\\-]?).

    If the regex already has correct flexible separators, it is returned
    unchanged.  If injection breaks compilation, the original is returned.
    """
    if not separator_spec.get("has_separators"):
        return regex_str

    group_lengths = separator_spec.get("group_lengths", [])
    observed_seps = separator_spec.get("observed_separators", [" ", "-"])

    if not group_lengths or len(group_lengths) < 2:
        return regex_str

    sep_class = _build_separator_class(observed_seps)

    # Strip \b boundaries for processing, re-add later.
    core = regex_str
    prefix = ""
    suffix = ""
    if core.startswith("\\b"):
        prefix = "\\b"
        core = core[2:]
    if core.endswith("\\b"):
        suffix = "\\b"
        core = core[:-2]

    # Tokenize the core regex into digit tokens, existing separators, and "other" chunks.
    segments: list[dict] = []  # {"type": "digit"|"sep"|"other", "text": str, "width": int}
    pos = 0
    while pos < len(core):
        # Try digit token
        md = _DIGIT_TOKEN_PATTERN.match(core, pos)
        if md:
            token_text = md.group(0)
            segments.append({
                "type": "digit",
                "text": token_text,
                "width": _digit_token_width(token_text),
            })
            pos = md.end()
            continue

        # Try existing separator token (between digit groups)
        ms = _RIGID_SEP_PATTERN.match(core, pos)
        if ms:
            segments.append({"type": "sep", "text": ms.group(0), "width": 0})
            pos = ms.end()
            continue

        # Other character (lookahead, anchor, group marker, etc.) — keep as-is
        segments.append({"type": "other", "text": core[pos], "width": 0})
        pos += 1

    # Count total digit width in the regex
    total_digit_width = sum(s["width"] for s in segments if s["type"] == "digit")
    total_group_width = sum(group_lengths)

    # If total digits in regex don't match group_lengths sum, we can't reliably
    # align buckets — bail out to avoid breaking the regex.
    if total_digit_width != total_group_width:
        logger.warning(
            "Agent 3 [separator]: digit width mismatch (regex=%d vs groups=%d), skipping injection",
            total_digit_width, total_group_width,
        )
        return regex_str

    # Walk segments, fill buckets, and insert separators at boundaries.
    result_parts: list[str] = []
    bucket_idx = 0
    filled = 0

    for seg in segments:
        if seg["type"] != "digit":
            # Existing separator between digit groups — replace with flexible class
            if seg["type"] == "sep" and bucket_idx < len(group_lengths) and filled == 0:
                # This separator is at a bucket boundary — replace it
                result_parts.append(sep_class)
            elif seg["type"] == "sep":
                # Separator mid-bucket or at boundary — replace regardless
                result_parts.append(sep_class)
            else:
                result_parts.append(seg["text"])
            continue

        # Digit token — fill the current bucket
        remaining_width = seg["width"]

        while remaining_width > 0 and bucket_idx < len(group_lengths):
            space_in_bucket = group_lengths[bucket_idx] - filled

            if remaining_width <= space_in_bucket:
                # Token fits entirely in current bucket
                if remaining_width == seg["width"]:
                    # Emit the original token text (preserves [2-9] etc.)
                    result_parts.append(seg["text"])
                else:
                    # We already emitted part of a split token; emit remainder
                    result_parts.append(f"\\d{{{remaining_width}}}")
                filled += remaining_width
                remaining_width = 0

                # Check if bucket is now full
                if filled == group_lengths[bucket_idx]:
                    bucket_idx += 1
                    filled = 0
                    # Insert separator if not the last bucket, and next segment
                    # is not already a separator
                    if bucket_idx < len(group_lengths):
                        # Peek ahead to see if next segment is already a separator
                        seg_idx = segments.index(seg)
                        next_is_sep = (
                            seg_idx + 1 < len(segments)
                            and segments[seg_idx + 1]["type"] == "sep"
                        )
                        if not next_is_sep:
                            result_parts.append(sep_class)
            else:
                # Token overflows current bucket — split it
                take = space_in_bucket
                if take == seg["width"] and remaining_width == seg["width"]:
                    # First chunk — preserve original token if possible
                    # But we need to split, so emit \d{take}
                    result_parts.append(f"\\d{{{take}}}")
                else:
                    result_parts.append(f"\\d{{{take}}}")
                filled += take
                remaining_width -= take

                # Bucket is full
                bucket_idx += 1
                filled = 0
                if bucket_idx < len(group_lengths):
                    result_parts.append(sep_class)

    injected = prefix + "".join(result_parts) + suffix

    # Safety: verify the injected regex still compiles
    if _try_compile(injected) is None:
        logger.warning("Agent 3 [separator]: injection broke regex, reverting to original")
        return regex_str

    logger.info("Agent 3 [separator]: %s → %s", regex_str, injected)
    return injected


def generate_regex(
    config: dict,
    paths: dict[str, str] | None = None,
    policies: dict | None = None,
) -> dict:
    patterns_path = (paths or {}).get("patterns", "results/patterns.json")
    regex_path = (paths or {}).get("regex_patterns", "results/regex_patterns.json")

    with open(patterns_path) as f:
        patterns = json.load(f)

    condition = patterns.get("condition", "unknown")
    format_rules = patterns.get("format_rules", [])
    range_restrictions = patterns.get("range_restrictions", [])
    keywords = patterns.get("contextual_keywords", [])
    proximity = patterns.get("keyword_proximity", 10)
    numbering_policy = patterns.get("numbering_policy", "").strip()

    # Build an enriched policy string that also includes the raw policy rules and
    # vendor DLP patterns discovered by Agent 1P.  Agent 2 only captures what it
    # *sees in data samples* — so if the condition spans multiple identifier types
    # (e.g. "india taxpayer id" covers PAN *and* GSTIN) the vendor structural rules
    # are the most authoritative source for the combined regex.
    enriched_policy = numbering_policy
    if policies:
        raw_rules = policies.get("policies", [])
        vendor_patterns = policies.get("vendor_patterns", [])

        if raw_rules:
            rules_block = "\n".join(f"- {r}" for r in raw_rules)
            enriched_policy += (
                f"\n\nAUTHORITATIVE FORMAT RULES (from official sources — "
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

    if enriched_policy:
        logger.info(
            "Agent 3: numbering_policy injected (%d chars, includes Agent 1P rules=%s vendor=%s)",
            len(enriched_policy),
            bool(policies and policies.get("policies")),
            bool(policies and policies.get("vendor_patterns")),
        )
    else:
        logger.warning("Agent 3: no numbering_policy found in patterns — per-position constraints may be missed")

    # Extract separator spec from policies (vendor regex / policy text).
    separator_spec = (policies or {}).get("separator_spec", {})
    if separator_spec.get("has_separators"):
        logger.info(
            "Agent 3: separator_spec detected — groups=%s seps=%s (source: %s)",
            separator_spec.get("group_lengths"),
            separator_spec.get("observed_separators"),
            separator_spec.get("source", "unknown"),
        )

    logger.info("Agent 3: generating regex for %d rules + keyword pattern", len(format_rules))

    range_lookup: dict[str, dict] = {}
    for rr in range_restrictions:
        pos = rr.get("position", "")
        # position can be a list (e.g. [0, 1]) when the LLM returns a range;
        # convert to a stable string key so it's hashable.
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
            example_matches=", ".join(examples),
            numbering_policy=enriched_policy or "(none provided)",
        )

        response = invoke_claude(prompt, config)
        regex_str = _clean_regex_response(response)
        regex_str = _strip_anchors_add_boundaries(regex_str)
        regex_str = _inject_optional_separators(regex_str, separator_spec)

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
            regex_str = _inject_optional_separators(regex_str, separator_spec)

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
            example_matches=", ".join(unique_examples),
            numbering_policy=enriched_policy or "(none provided)",
        )
        combined_response = invoke_claude(combined_prompt, config)
        combined_regex = _clean_regex_response(combined_response)
        combined_regex = _strip_anchors_add_boundaries(combined_regex)
        combined_regex = _inject_optional_separators(combined_regex, separator_spec)

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
            combined_regex = _inject_optional_separators(combined_regex, separator_spec)
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
