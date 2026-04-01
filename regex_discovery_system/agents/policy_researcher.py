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
import re as _re
from urllib.parse import urlparse

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

# ── Domain authority scoring tiers ────────────────────────────────────────────

_AUTHORITY_TIERS: list[tuple[list[str], int]] = [
    ([".gov", ".gov."], 10),
    (["iso.org", "itu.int", "ietf.org", "iana.org", "w3.org"], 9),
    (["wikipedia.org", "wikidata.org"], 8),
    ([".edu", ".ac."], 7),
    ([
        "geeksforgeeks.org", "stackoverflow.com", "numbering.org",
        "geonames.org", "worldpostalcode.com", "postcodebase.com",
        "geopostcodes.com", "zipcodebase.com",
    ], 6),
    ([
        "microsoft.com", "learn.microsoft.com", "netskope.com",
        "broadcom.com", "zscaler.com", "skyhighsecurity.com",
        "success.skyhighsecurity.com", "docs.trellix.com", "trellix.com",
        "forcepoint.com", "digitalguardian.com", "paloaltonetworks.com",
    ], 5),
]

_PATH_BOOST_KEYWORDS = {
    "format", "specification", "structure", "rules", "validation",
    "numbering", "allocation", "standard", "definition", "policy",
    "regex", "pattern", "check-digit", "checkdigit",
}


def _score_url(url: str) -> int:
    """Score a URL by domain authority + path keyword boost."""
    domain = urlparse(url).netloc.lower()
    path = urlparse(url).path.lower()

    base = 3
    for patterns, score in _AUTHORITY_TIERS:
        if any(p in domain for p in patterns):
            base = score
            break

    path_parts = set(_re.split(r"[/\-_.]", path))
    boost = min(3, len(path_parts & _PATH_BOOST_KEYWORDS))

    return base + boost


def _rank_and_dedupe(urls: list[str], max_per_domain: int = 2, top_n: int = 15) -> list[str]:
    """Score, deduplicate, and return the top-N URLs by domain authority."""
    domain_counts: dict[str, int] = {}
    scored: list[tuple[int, str]] = []

    for url in urls:
        domain = urlparse(url).netloc.lower().replace("www.", "")
        if domain_counts.get(domain, 0) >= max_per_domain:
            continue
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        scored.append((_score_url(url), url))

    scored.sort(key=lambda x: -x[0])
    ranked = [url for _, url in scored[:top_n]]

    if ranked:
        logger.info(
            "Agent 1P [rank]: %d → %d URLs after dedup+rank (top score: %d, bottom: %d)",
            len(urls), len(ranked), scored[0][0] if scored else 0,
            scored[min(top_n - 1, len(scored) - 1)][0] if scored else 0,
        )
        for i, (sc, u) in enumerate(scored[:top_n]):
            logger.debug("Agent 1P [rank] #%d (score=%d): %s", i + 1, sc, u)
    return ranked

_POLICY_PROMPT = """\
You are a numbering-policy expert.

CONDITION: {condition}

I have scraped the following web pages that may describe the format rules,
allocation policies, or structural constraints for the identifier above.
These pages may include government (.gov) sites, official regulatory body
websites, reputable third-party references (Wikipedia, open-data portals,
educational institutions), or well-known industry sites.

IMPORTANT: Accept and extract rules from ANY reputable, legitimate source —
not just government sites. If no government data is available but a credible
third-party site provides format rules, use that data confidently.

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
   Cite the source where possible (government, Wikipedia, industry ref).

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
    "Trellix DLP",
    "Zscaler",
    "Skyhigh Security",
    "Forcepoint DLP",
    "Digital Guardian",
    "Palo Alto",
]

# Known vendor documentation hubs that list data identifier definitions.
# These are always searched alongside dynamically discovered URLs.
_VENDOR_REFERENCE_BASES = [
    "https://success.skyhighsecurity.com/Skyhigh_Data_Loss_Prevention/Data_Identifiers",
    "https://docs.trellix.com/bundle/data-loss-prevention-11.10.x-classification-definitions-reference-guide/page/GUID-3CFCC6AE-1709-43B7-B790-34E2D141ADB7.html",
    https://learn.microsoft.com/en-us/purview/sit-sensitive-information-type-entity-definitions?view=o365-worldwide
]


def research_policies(condition: str, config: dict) -> dict:
    """Search → rank → scrape → extract → LLM for format rules and vendor patterns.

    Pipeline:
      1. Search for policy URLs via Tavily (with DDG fallback).
      2. Score, rank, and deduplicate → top 15 URLs.
      3. Scrape those 15 pages with structured rule-block extraction.
      4. Feed focused text to LLM for policy extraction.
      5. (Secondary) Search vendor pages for corroboration.
      6. Fallback to LLM knowledge if web yields nothing.
    """
    result: dict = {
        "condition": condition,
        "policies": [],
        "numbering_authority": "unknown",
        "vendor_patterns": [],
        "sources": [],
        "_url_discovery": {},
    }

    # ── Phase 1 (PRIMARY): Official policies ──────────────────────────────────
    url_items = _find_policy_urls(condition, config)
    logger.info(
        "Agent 1P [search]: found %d raw policy URLs for '%s'",
        len(url_items), condition,
    )

    if url_items:
        ranked_urls = _rank_and_dedupe(url_items, max_per_domain=2, top_n=15)
        result["_url_discovery"] = {
            "total_raw": len(url_items),
            "ranked_count": len(ranked_urls),
        }

        result["sources"].extend(ranked_urls)
        logger.info(
            "Agent 1P [policy]: ranked %d → %d URLs to scrape",
            len(url_items), len(ranked_urls),
        )

        scraped = _scrape_and_extract(ranked_urls, config)
        logger.info(
            "Agent 1P [policy]: extracted rule blocks from %d/%d pages",
            len(scraped), len(ranked_urls),
        )

        if scraped:
            pages_to_use = scraped[:15]
            capped = [text[:15_000] for _, text in pages_to_use]
            combined = "\n\n--- PAGE BREAK ---\n\n".join(capped)
            if len(combined) > 60_000:
                combined = combined[:60_000] + "\n... (truncated)"
            logger.info(
                "Agent 1P [policy]: sending %d chars to LLM (%d pages)",
                len(combined), len(pages_to_use),
            )
            try:
                response = invoke_claude(
                    _POLICY_PROMPT.format(condition=condition, texts=combined), config,
                )
                parsed = _parse_json(response)
                result["policies"] = parsed.get("policies", [])
                result["numbering_authority"] = parsed.get("numbering_authority", "unknown")
                logger.info(
                    "Agent 1P [policy]: %d rules extracted (authority: %s)",
                    len(result["policies"]), result["numbering_authority"],
                )
                for i, rule in enumerate(result["policies"]):
                    logger.debug("Agent 1P [policy] rule[%d]: %s", i, rule)
            except Exception as exc:
                logger.warning("Agent 1P [policy]: LLM extraction failed: %s", exc)
    else:
        logger.info("Agent 1P [policy]: no policy URLs found for '%s'", condition)

    # ── Phase 2 (SECONDARY): Vendor/competitor regex — corroboration ──────────
    vendor_urls = _find_vendor_urls(condition, config)
    if vendor_urls:
        result["sources"].extend(vendor_urls)
        logger.info("Agent 1P [vendor]: found %d vendor URLs for '%s'", len(vendor_urls), condition)
        scraped = _scrape_and_extract(vendor_urls, config)
        logger.info("Agent 1P [vendor]: extracted from %d/%d pages", len(scraped), len(vendor_urls))
        if scraped:
            pages_to_use = scraped[:5]
            capped = [text[:12_000] for _, text in pages_to_use]
            combined = "\n\n--- PAGE BREAK ---\n\n".join(capped)
            if len(combined) > 50_000:
                combined = combined[:50_000] + "\n... (truncated)"
            logger.info(
                "Agent 1P [vendor]: sending %d chars to LLM (%d pages)",
                len(combined), len(pages_to_use),
            )
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

    # ── Phase 3: LLM fallback if web yielded nothing ──────────────────────────
    if not result["policies"] and not result["vendor_patterns"]:
        logger.info("Agent 1P: no web results — asking LLM from its own knowledge")
        result = _llm_fallback(condition, config, result)

    return result


def _find_policy_urls(condition: str, config: dict) -> list[str]:
    """Discover policy URLs via Tavily (with DDG fallback)."""
    tavily_key = config.get("tavily_api_key", "")

    try:
        from tools.search_tool import search_policy_urls
        return search_policy_urls(
            condition,
            tavily_api_key=tavily_key,
            max_results=20,
        )
    except Exception as exc:
        logger.warning("Agent 1P [policy]: URL search failed: %s", exc)
        return []


def _find_vendor_urls(condition: str, config: dict) -> list[str]:
    """Search for how DLP/security vendors identify this data type.

    Combines dynamically searched URLs with known vendor documentation
    hubs (Skyhigh, Trellix) that are always included.
    """
    vendor_queries = [
        f"{condition} Microsoft Purview sensitive information type regex",
        f"{condition} regex DLP detection rules Netskope Broadcom Symantec",
        f"{condition} Zscaler Skyhigh data classification pattern",
        f"{condition} Trellix DLP data identifier definition regex",
        f"{condition} DLP policy regex pattern identification rules",
        f"{condition} sensitive data type regex format validation",
        f"site:success.skyhighsecurity.com {condition} data identifier",
        f"site:docs.trellix.com {condition} classification definition",
    ]

    seen: set[str] = set()
    urls: list[str] = list(_VENDOR_REFERENCE_BASES)
    seen.update(_VENDOR_REFERENCE_BASES)

    api_key = config.get("tavily_api_key")
    try:
        if api_key:
            from tools.search_tool import _tavily_search_multi
            searched = _tavily_search_multi(vendor_queries, api_key, max_results=8)
            for u in searched:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        else:
            from tools.scraper.search import search_urls
            searched = search_urls(vendor_queries, max_total=6)
            for u in searched:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
    except Exception as exc:
        logger.warning("Agent 1P [vendor]: URL search failed: %s", exc)

    logger.info("Agent 1P [vendor]: %d total vendor URLs (%d known + searched)", len(urls), len(_VENDOR_REFERENCE_BASES))
    return urls


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


def _scrape_and_extract(urls: list[str], config: dict) -> list[tuple[str, str]]:
    """Scrape URLs and extract rule-heavy blocks from each page.

    Returns ``(url, extracted_text)`` pairs.  Uses ``_extract_rule_blocks``
    for structured extraction, falling back to ``_html_to_text`` if the
    structured pass yields very little.
    """
    from tools.scraper.browser import scrape_pages

    proxies = config.get("proxies", [])
    page_results = scrape_pages(urls[:15], proxies=proxies or None, sleep_s=1.5)

    extracted: list[tuple[str, str]] = []
    for url, html in page_results:
        blocks = _extract_rule_blocks(html)
        if len(blocks) < 200:
            blocks = _html_to_text(html)
        if blocks:
            extracted.append((url, blocks))
            logger.info("Agent 1P: extracted %s (%d chars)", url, len(blocks))
        else:
            logger.warning("Agent 1P: scraped %s but extracted no useful text", url)
    return extracted


# ── Rule-heavy block extraction ───────────────────────────────────────────────

_HEADING_RULE_KEYWORDS = _re.compile(
    r"format|structur|rule|specif|valid|number|alloc|standard|defin|"
    r"pattern|policy|check.?digit|length|example|overview|description|"
    r"breakdown|segment|component|syntax|encod|prefix|suffix",
    _re.IGNORECASE,
)

_REGEX_PATTERN = _re.compile(r"[\[\(][0-9A-Za-z\-\\\{\}]+[\]\)]")


def _extract_rule_blocks(html: str) -> str:
    """Extract rule-rich sections from HTML: headings with format/rule
    keywords, tables, ordered/unordered lists, and code blocks.

    Produces a focused, high-signal text block for LLM consumption.
    """
    try:
        from selectolax.parser import HTMLParser
    except ImportError:
        return _html_to_text(html)

    tree = HTMLParser(html)
    for tag in tree.css("script, style, nav, footer, header, aside, "
                        ".cookie, .banner, .sidebar, .ad, .advertisement"):
        tag.decompose()

    if not tree.body:
        return ""

    blocks: list[str] = []
    seen_text: set[str] = set()

    def _add(text: str) -> None:
        text = text.strip()
        if not text or text in seen_text:
            return
        seen_text.add(text)
        blocks.append(text)

    # 1) Tables — almost always contain structured format specs
    for table in tree.body.css("table"):
        table_text = table.text(separator=" | ").strip()
        if table_text and len(table_text) > 20:
            _add("[TABLE]\n" + table_text)

    # 2) Headings + their content — keep sections with rule-related keywords
    for heading in tree.body.css("h1, h2, h3, h4, h5, h6"):
        heading_text = heading.text(separator=" ").strip()
        if not _HEADING_RULE_KEYWORDS.search(heading_text):
            continue

        section_text_parts = [heading_text]
        sibling = heading.next
        while sibling is not None:
            tag_name = getattr(sibling, "tag", None)
            if tag_name and tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                break
            text = sibling.text(separator="\n").strip() if hasattr(sibling, "text") else ""
            if text:
                section_text_parts.append(text)
            sibling = sibling.next

        section_text = "\n".join(section_text_parts)
        if len(section_text) > 30:
            _add(section_text)

    # 3) All lists (ol, ul) — rules are often listed
    for lst in tree.body.css("ol, ul"):
        list_text = lst.text(separator="\n").strip()
        if list_text and len(list_text) > 20:
            _add(list_text)

    # 4) Code blocks — may contain regex patterns or format examples
    for code in tree.body.css("pre, code"):
        code_text = code.text().strip()
        if code_text and (_REGEX_PATTERN.search(code_text) or len(code_text) > 10):
            _add("[CODE] " + code_text)

    # 5) Paragraphs with regex or digit-rule keywords (catch-all)
    for p in tree.body.css("p"):
        p_text = p.text(separator=" ").strip()
        if p_text and (
            _REGEX_PATTERN.search(p_text)
            or _HEADING_RULE_KEYWORDS.search(p_text)
        ):
            _add(p_text)

    return "\n\n".join(blocks)


def _html_to_text(html: str) -> str:
    """Fallback: extract visible text from HTML, stripping tags."""
    try:
        from selectolax.parser import HTMLParser
        tree = HTMLParser(html)
        for tag in tree.css("script, style, nav, footer, header"):
            tag.decompose()
        text = tree.body.text(separator="\n") if tree.body else ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
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
