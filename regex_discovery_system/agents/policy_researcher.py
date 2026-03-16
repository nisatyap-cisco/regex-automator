"""Agent 1P — Policy Researcher (runs parallel to Agent 1).

Two-phase research:
  Phase 1 — Official policies: searches government/.gov sites for format
            rules, allocation policies, and structural specs.
  Phase 2 — Vendor/competitor regex: searches for how major DLP and
            security vendors (Microsoft, Netskope, Broadcom, Zscaler,
            Skyhigh, etc.) identify this same data type — their regex
            patterns, keywords, and classification rules.

Output:
    {
        "condition": str,
        "policies": [str, ...],           # pointwise rules
        "numbering_authority": str,        # who governs this format
        "vendor_patterns": [str, ...],     # vendor regex / classification info
        "sources": [str, ...]              # URLs consulted
    }
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

_POLICY_PROMPT = """\
You are a numbering-policy expert.

CONDITION: {condition}

I have scraped the following web pages that may describe the format rules,
allocation policies, or structural constraints for the identifier above.
These pages were sourced primarily from government (.gov) and official
regulatory body websites.

--- START OF SCRAPED TEXT ---
{texts}
--- END OF SCRAPED TEXT ---

TASKS — return a JSON object with these keys:

1. "policies": an array of concise, factual strings.  Each string is ONE
   structural rule or constraint (e.g. "First digit is always 9",
   "Length is exactly 5 digits", "Positions 1-2 encode the state").
   Include at least 5 rules if the text supports it.
   CRITICAL: Only include rules that are verifiable from the scraped text
   or that you know to be established by the official governing body.
   Cite the government or regulatory source where possible.

2. "numbering_authority": the name of the official government body or
   regulatory authority that governs this numbering system
   (e.g. "USPS", "CMS/HHS", "ITU", "India Post", "Ofcom").

Return ONLY valid JSON, no markdown fences."""

_VENDOR_PROMPT = """\
You are a DLP/data classification expert.

CONDITION: {condition}

I have scraped pages from major security and networking vendors that may
describe how they detect or classify this identifier type in their DLP,
CASB, or data classification products.

--- START OF SCRAPED TEXT ---
{texts}
--- END OF SCRAPED TEXT ---

TASKS — return a JSON object with these keys:

1. "vendor_patterns": an array of objects, each with:
   - "vendor": vendor name (e.g. "Microsoft Purview", "Netskope DLP")
   - "regex": the regex pattern they use (if found in the text), or null
   - "keywords": array of keywords/phrases they associate with this data type
   - "rules": array of format rules or validation logic they describe

2. "common_keywords": array of ALL unique keywords/phrases across vendors
   that are used to identify or label this data type. Include form field
   labels, DLP policy names, classification labels, etc.

Return ONLY valid JSON, no markdown fences."""

_VENDORS = [
    "Microsoft Purview",
    "Netskope",
    "Broadcom Symantec DLP",
    "Zscaler",
    "Skyhigh Security",
    "Forcepoint DLP",
    "Digital Guardian",
    "Palo Alto",
]


def research_policies(condition: str, config: dict) -> dict:
    """Search for format rules and vendor patterns, distil via LLM.

    Returns a dict with keys ``condition``, ``policies``, ``numbering_authority``,
    ``vendor_patterns``, and ``sources``.
    """
    result: dict = {
        "condition": condition,
        "policies": [],
        "numbering_authority": "unknown",
        "vendor_patterns": [],
        "sources": [],
    }

    # Phase 1: Official policies
    policy_urls = _find_policy_urls(condition, config)
    if policy_urls:
        result["sources"].extend(policy_urls)
        logger.info("Agent 1P [policy]: found %d policy URLs for '%s'", len(policy_urls), condition)
        raw_texts = _scrape_policy_pages(policy_urls, config)
        if raw_texts:
            combined = "\n\n---\n\n".join(raw_texts[:3])
            if len(combined) > 30_000:
                combined = combined[:30_000] + "\n... (truncated)"
            try:
                response = invoke_claude(
                    _POLICY_PROMPT.format(condition=condition, texts=combined), config,
                )
                parsed = _parse_json(response)
                result["policies"] = parsed.get("policies", [])
                result["numbering_authority"] = parsed.get("numbering_authority", "unknown")
                logger.info(
                    "Agent 1P [policy]: %d rules (authority: %s)",
                    len(result["policies"]), result["numbering_authority"],
                )
            except Exception as exc:
                logger.warning("Agent 1P [policy]: LLM extraction failed: %s", exc)
    else:
        logger.info("Agent 1P [policy]: no policy URLs found for '%s'", condition)

    # Phase 2: Vendor/competitor regex and classification patterns
    vendor_urls = _find_vendor_urls(condition, config)
    if vendor_urls:
        result["sources"].extend(vendor_urls)
        logger.info("Agent 1P [vendor]: found %d vendor URLs for '%s'", len(vendor_urls), condition)
        vendor_texts = _scrape_policy_pages(vendor_urls, config)
        if vendor_texts:
            combined = "\n\n---\n\n".join(vendor_texts[:3])
            if len(combined) > 30_000:
                combined = combined[:30_000] + "\n... (truncated)"
            try:
                response = invoke_claude(
                    _VENDOR_PROMPT.format(condition=condition, texts=combined), config,
                )
                parsed = _parse_json(response)
                result["vendor_patterns"] = parsed.get("vendor_patterns", [])
                common_kw = parsed.get("common_keywords", [])
                if common_kw:
                    result["vendor_keywords"] = common_kw
                logger.info(
                    "Agent 1P [vendor]: %d vendor patterns, %d common keywords",
                    len(result["vendor_patterns"]), len(common_kw),
                )
            except Exception as exc:
                logger.warning("Agent 1P [vendor]: LLM extraction failed: %s", exc)
    else:
        logger.info("Agent 1P [vendor]: no vendor URLs found for '%s'", condition)

    # Phase 3: If no web results at all, ask the LLM directly from its knowledge
    if not result["policies"] and not result["vendor_patterns"]:
        logger.info("Agent 1P: no web results — asking LLM from its own knowledge")
        result = _llm_fallback(condition, config, result)

    return result


def _find_policy_urls(condition: str, config: dict) -> list[str]:
    try:
        from tools.search_tool import search_policy_urls
        return search_policy_urls(
            condition,
            tavily_api_key=config.get("tavily_api_key"),
            max_results=3,
        )
    except Exception as exc:
        logger.warning("Agent 1P [policy]: URL search failed: %s", exc)
        return []


def _find_vendor_urls(condition: str, config: dict) -> list[str]:
    """Search for how DLP/security vendors identify this data type."""
    vendor_queries = [
        f"{condition} regex DLP sensitive data type Microsoft Purview Netskope",
        f"{condition} data identifier pattern Broadcom Symantec Zscaler Skyhigh",
        f"{condition} classification regex pattern DLP policy",
    ]
    api_key = config.get("tavily_api_key")
    try:
        if api_key:
            from tools.search_tool import _tavily_search_multi
            urls = _tavily_search_multi(vendor_queries, api_key, max_results=3)
            if urls:
                return urls
        from tools.scraper.search import search_urls
        return search_urls(vendor_queries, max_total=3)
    except Exception as exc:
        logger.warning("Agent 1P [vendor]: URL search failed: %s", exc)
        return []


_LLM_FALLBACK_PROMPT = """\
You are an expert on data classification and numbering systems.

CONDITION: {condition}

I could not find web pages with format rules for this identifier.
Using your training knowledge, provide:

1. "policies": array of concise structural rules/constraints for this
   identifier type. Include format, length, character types, ranges,
   check digits, and any allocation rules you know.

2. "numbering_authority": the official governing body.

3. "vendor_patterns": array of objects describing how major DLP vendors
   (Microsoft Purview, Netskope, Broadcom/Symantec, Zscaler, Skyhigh,
   Forcepoint, Palo Alto) classify this data type. For each vendor you
   know about, include:
   - "vendor": name
   - "regex": the regex they use (if known), or null
   - "keywords": keywords they associate with this data
   - "rules": validation rules they apply

4. "common_keywords": array of ALL keywords/phrases that any vendor or
   official source uses to label or detect this identifier. Be exhaustive.

Return ONLY valid JSON, no markdown fences."""


def _llm_fallback(condition: str, config: dict, result: dict) -> dict:
    """Ask the LLM directly when no web results are available."""
    try:
        response = invoke_claude(
            _LLM_FALLBACK_PROMPT.format(condition=condition), config,
        )
        parsed = _parse_json(response)
        result["policies"] = parsed.get("policies", [])
        result["numbering_authority"] = parsed.get("numbering_authority", "unknown")
        result["vendor_patterns"] = parsed.get("vendor_patterns", [])
        if parsed.get("common_keywords"):
            result["vendor_keywords"] = parsed["common_keywords"]
        logger.info(
            "Agent 1P [llm-fallback]: %d policies, %d vendor patterns",
            len(result["policies"]), len(result["vendor_patterns"]),
        )
    except Exception as exc:
        logger.warning("Agent 1P [llm-fallback]: failed: %s", exc)
    return result


def _scrape_policy_pages(urls: list[str], config: dict) -> list[str]:
    """Scrape policy pages via scrape_pages (Camoufox → httpx fallback)."""
    from tools.scraper.browser import scrape_pages

    proxies = config.get("proxies", [])
    page_results = scrape_pages(urls[:3], proxies=proxies or None, sleep_s=1.5)

    texts: list[str] = []
    for url, html in page_results:
        text = _html_to_text(html)
        if text:
            texts.append(text)
            logger.info("Agent 1P [policy]: scraped %s (%d chars text)", url, len(text))
    return texts


def _html_to_text(html: str) -> str:
    """Extract visible text from HTML, stripping tags."""
    try:
        from selectolax.parser import HTMLParser
        tree = HTMLParser(html)
        for tag in tree.css("script, style, nav, footer, header"):
            tag.decompose()
        text = tree.body.text(separator="\n") if tree.body else ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)
    except Exception:
        return ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)
