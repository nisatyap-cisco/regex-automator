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

Requirements:
- Use raw string notation.
- Anchor with ^ and $ for the value regex.
- Avoid catastrophic backtracking.
- Prefer character classes and quantifiers over alternation where possible.
- If the rule involves check digits, encode the structural pattern only
  (check-digit validation is done in code, not regex).

Return ONLY the regex string, no explanation."""

REFINEMENT_TEMPLATE = """The regex you provided failed to compile in Python.

ORIGINAL REGEX: {regex}
COMPILE ERROR: {error}

Fix the regex so it compiles with Python's re module.
Return ONLY the corrected regex string, no explanation."""


def build_keyword_regex(keywords: list[str], proximity: int = 10) -> str:
    kw_pattern = "|".join(re.escape(k) for k in keywords)
    return rf"(?i)(?:{kw_pattern})(?:\W+\w+){{0,{proximity}}}\W+"


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


def generate_regex(config: dict, paths: dict[str, str] | None = None) -> dict:
    patterns_path = (paths or {}).get("patterns", "results/patterns.json")
    regex_path = (paths or {}).get("regex_patterns", "results/regex_patterns.json")

    with open(patterns_path) as f:
        patterns = json.load(f)

    condition = patterns.get("condition", "unknown")
    format_rules = patterns.get("format_rules", [])
    range_restrictions = patterns.get("range_restrictions", [])
    keywords = patterns.get("contextual_keywords", [])
    proximity = patterns.get("keyword_proximity", 10)

    logger.info("Agent 3: generating regex for %d rules + keyword pattern", len(format_rules))

    range_lookup: dict[str, dict] = {}
    for rr in range_restrictions:
        pos = rr.get("position", "")
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
        )

        response = invoke_claude(prompt, config)
        regex_str = _clean_regex_response(response)

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

    if keywords:
        kw_regex = build_keyword_regex(keywords, proximity)
        compiled = _try_compile(kw_regex)
        if compiled is None:
            logger.error("Agent 3: keyword proximity regex failed compile")
        else:
            output_patterns.append({
                "rule_id": "keyword-proximity",
                "regex": kw_regex,
                "description": f"Contextual keywords within {proximity}-term proximity",
                "type": "context",
            })
            logger.info("Agent 3: [keyword-proximity] → %s", kw_regex)

    result = {
        "condition": condition,
        "patterns": output_patterns,
    }

    with open(regex_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Agent 3 done: %d patterns written", len(output_patterns))
    return result
