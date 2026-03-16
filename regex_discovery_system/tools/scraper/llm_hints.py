"""LLM-guided hints for the scraper.

Asks the LLM to suggest:
  1. Search queries to find lists of data for the condition
  2. A regex pattern that matches individual values of the condition
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_HINT_PROMPT = """You are helping a web scraper find data for: "{condition}"

Return a JSON object with exactly these keys:
{{
  "country": "the country this data belongs to (e.g. Czech Republic, India, USA)",
  "gov_domains": ["gov.xx", "official-authority.xx"],
  "search_queries": ["query1", "query2", "query3", "query4", "query5"],
  "value_regex": "regex_pattern_here",
  "description": "brief description of the data format"
}}

Rules for country and gov_domains:
- Identify WHICH COUNTRY this data type belongs to
- List 2-3 government or official authority domains for THAT country:
  * The country's government domain (e.g. gov.cz, gov.in, gov.uk, gov for USA)
  * The official authority/agency that manages this data type
    (e.g. ceskaposta.cz for Czech postcodes, indiapost.gov.in for India PIN)
- These domains are used to filter search results — only pages from these
  domains (or containing the country name) will be trusted

Rules for search_queries (5 queries, ASCII only):
- Queries should find web pages that LIST many real examples of this data
- Include terms like "list", "all", "complete", "database", "directory"
- Prioritize government/official sites first but DO NOT limit to them:
  * 2 queries SHOULD use "site:<domain>" targeting the gov_domains above
  * 2-3 queries MUST target third-party data aggregators, open-data portals,
    Wikipedia, or well-known reference sites (e.g. worldpostalcode.com,
    geonames.org, numberingplans.com, zipcodebase.com, etc.)
- The goal is to gather data even if government sites have none — always
  include broad queries that will find data on any reputable site
- Use ASCII characters only (no accented characters, no Unicode)

Rules for value_regex:
- A Python regex that matches a SINGLE value of this data type WITHIN text
- NEVER use anchors (^ or $) — the regex is used with re.findall on HTML pages
- Use \\b word boundaries instead if you need to delimit the match
- Must be specific enough to avoid false positives from random text
- Example: for "california zip code" → "\\b9[0-6]\\d{{3}}\\b"
- Example: for "czech postcode" → "\\b\\d{{3}}\\s?\\d{{2}}\\b"
- Example: for "usa swift code" → "\\b[A-Z]{{4}}US[A-Z0-9]{{2}}(?:[A-Z0-9]{{3}})?\\b"
- Example: for "usa medicaid number" → "\\b\\d{{8,13}}\\b"

Return ONLY valid JSON, no markdown fences, no explanation."""


def _extract_json(raw: str) -> Optional[dict]:
    """Best-effort extraction of a JSON object from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def get_scraper_hints(condition: str, config: dict) -> Optional[dict]:
    """Ask the LLM for scraping guidance.

    Returns:
        dict with keys: search_queries (list[str]), value_regex (str),
        description (str).  Or None on failure.
    """
    try:
        from utils.bedrock_client import invoke_claude
    except ImportError:
        logger.warning("bedrock_client not available — no LLM hints")
        return None

    prompt = _HINT_PROMPT.format(condition=condition)

    try:
        raw = invoke_claude(prompt, config)
    except Exception as exc:
        logger.warning("LLM hint call failed: %s", exc)
        return None

    hints = _extract_json(raw)
    if hints is None:
        logger.warning("LLM returned invalid JSON for hints — retrying with stricter prompt")
        try:
            retry_prompt = (
                "Your previous response was not valid JSON. "
                "Return ONLY a JSON object with keys: country, gov_domains, "
                "search_queries, value_regex, description. "
                "Use ASCII only — no accented or Unicode characters anywhere. "
                "No explanation, no markdown.\n"
                f"Topic: {condition}"
            )
            raw2 = invoke_claude(retry_prompt, config)
            hints = _extract_json(raw2)
        except Exception:
            pass
    if hints is None:
        return None

    if "value_regex" in hints and hints["value_regex"]:
        vr = hints["value_regex"]
        vr = re.sub(r"^\^", "", vr)
        vr = re.sub(r"\$$", "", vr)
        if vr != hints["value_regex"]:
            logger.info("Stripped anchors from LLM regex: '%s' → '%s'", hints["value_regex"], vr)
        hints["value_regex"] = vr
        try:
            re.compile(vr)
        except re.error as exc:
            logger.warning("LLM returned invalid regex '%s': %s", vr, exc)
            hints["value_regex"] = None

    logger.info(
        "LLM hints for '%s': %d queries, regex=%s",
        condition,
        len(hints.get("search_queries", [])),
        hints.get("value_regex", "none"),
    )
    return hints
