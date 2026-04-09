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
from typing import Optional
from urllib.parse import urlparse

from utils.bedrock_client import invoke_claude

logger = logging.getLogger(__name__)

# ── Domain authority scoring tiers ────────────────────────────────────────────

_AUTHORITY_TIERS: list[tuple[list[str], int]] = [
    # Government / official regulatory
    ([".gov", ".gov."], 10),
    # Standards bodies
    (["iso.org", "itu.int", "ietf.org", "iana.org", "w3.org"], 9),
    # Wikipedia / major reference
    (["wikipedia.org", "wikidata.org"], 8),
    # Educational
    ([".edu", ".ac."], 7),
    # Well-known data/format reference sites
    ([
        "geeksforgeeks.org", "stackoverflow.com", "numbering.org",
        "geonames.org", "worldpostalcode.com", "postcodebase.com",
        "geopostcodes.com", "zipcodebase.com",
    ], 6),
    # DLP/security vendors (useful but secondary)
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


def _rank_and_dedupe(urls: list[str], max_per_domain: int = 2, top_n: int = 8) -> list[str]:
    """Score, deduplicate, and return the top-N policy URLs.

    - Scores each URL by domain authority tier + path keyword relevance.
    - Limits to *max_per_domain* URLs per domain to avoid duplication.
    - Returns the top *top_n* by score.
    """
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
]

_PURVIEW_BASE = "https://learn.microsoft.com/en-us/purview"
_PURVIEW_SIT_INDEX = f"{_PURVIEW_BASE}/sit-sensitive-information-type-entity-definitions"

_PURVIEW_SIT_SLUGS: list[tuple[str, str]] = [
    ("aba routing number", "sit-defn-aba-routing"),
    ("argentina national identity dni number", "sit-defn-argentina-national-identity-numbers"),
    ("argentina unique tax identification key cuit cuil", "sit-defn-argentina-unique-tax-identification-key"),
    ("australia bank account number", "sit-defn-australia-bank-account-number"),
    ("australia business number", "sit-defn-australia-business-number"),
    ("australia company number", "sit-defn-australia-business-number"),
    ("australia drivers license number", "sit-defn-australia-drivers-license-number"),
    ("australia medical account number", "sit-defn-australia-medical-account-number"),
    ("australia passport number", "sit-defn-australia-passport-number"),
    ("australia tax file number", "sit-defn-australia-tax-file-number"),
    ("austria drivers license number", "sit-defn-austria-drivers-license-number"),
    ("austria identity card", "sit-defn-austria-identity-card"),
    ("austria passport number", "sit-defn-austria-passport-number"),
    ("austria social security number", "sit-defn-austria-social-security-number"),
    ("austria tax identification number", "sit-defn-austria-tax-identification-number"),
    ("austria value added tax", "sit-defn-austria-value-added-tax"),
    ("belgium drivers license number", "sit-defn-belgium-drivers-license-number"),
    ("belgium national number", "sit-defn-belgium-national-number"),
    ("belgium passport number", "sit-defn-belgium-passport-number"),
    ("belgium value added tax number", "sit-defn-belgium-value-added-tax-number"),
    ("brazil cpf number", "sit-defn-brazil-cpf-number"),
    ("brazil legal entity number cnpj", "sit-defn-brazil-legal-entity-number"),
    ("brazil national identification card rg", "sit-defn-brazil-national-identification-card"),
    ("bulgaria drivers license number", "sit-defn-bulgaria-drivers-license-number"),
    ("bulgaria passport number", "sit-defn-bulgaria-passport-number"),
    ("bulgaria uniform civil number", "sit-defn-bulgaria-uniform-civil-number"),
    ("canada bank account number", "sit-defn-canada-bank-account-number"),
    ("canada drivers license number", "sit-defn-canada-drivers-license-number"),
    ("canada health service number", "sit-defn-canada-health-service-number"),
    ("canada passport number", "sit-defn-canada-passport-number"),
    ("canada personal health identification number phin", "sit-defn-canada-personal-health-identification-number"),
    ("canada social insurance number", "sit-defn-canada-social-insurance-number"),
    ("chile identity card number", "sit-defn-chile-identity-card-number"),
    ("china resident identity card number", "sit-defn-china-resident-identity-card-number"),
    ("credit card number", "sit-defn-credit-card-number"),
    ("croatia drivers license number", "sit-defn-croatia-drivers-license-number"),
    ("croatia identity card number", "sit-defn-croatia-identity-card-number"),
    ("croatia passport number", "sit-defn-croatia-passport-number"),
    ("croatia personal identification oib number", "sit-defn-croatia-personal-identification-number"),
    ("cyprus drivers license number", "sit-defn-cyprus-drivers-license-number"),
    ("cyprus identity card", "sit-defn-cyprus-identity-card"),
    ("cyprus passport number", "sit-defn-cyprus-passport-number"),
    ("cyprus tax identification number", "sit-defn-cyprus-tax-identification-number"),
    ("czech drivers license number", "sit-defn-czech-drivers-license-number"),
    ("czech passport number", "sit-defn-czech-passport-number"),
    ("czech personal identity number", "sit-defn-czech-personal-identity-number"),
    ("denmark drivers license number", "sit-defn-denmark-drivers-license-number"),
    ("denmark passport number", "sit-defn-denmark-passport-number"),
    ("denmark personal identification number", "sit-defn-denmark-personal-identification-number"),
    ("drug enforcement agency dea number", "sit-defn-drug-enforcement-agency-number"),
    ("ecuador unique identification number", "sit-defn-ecuador-unique-identification-number"),
    ("estonia drivers license number", "sit-defn-estonia-drivers-license-number"),
    ("estonia passport number", "sit-defn-estonia-passport-number"),
    ("estonia personal identification code", "sit-defn-estonia-personal-identification-code"),
    ("eu debit card number", "sit-defn-eu-debit-card-number"),
    ("eu drivers license number", "sit-defn-eu-drivers-license-number"),
    ("eu national identification number", "sit-defn-eu-national-identification-number"),
    ("eu passport number", "sit-defn-eu-passport-number"),
    ("eu social security number", "sit-defn-eu-social-security-number-equivalent-identification"),
    ("eu tax identification number", "sit-defn-eu-tax-identification-number"),
    ("finland drivers license number", "sit-defn-finland-drivers-license-number"),
    ("finland national id", "sit-defn-finland-national-id"),
    ("finland passport number", "sit-defn-finland-passport-number"),
    ("france drivers license number", "sit-defn-france-drivers-license-number"),
    ("france health insurance number", "sit-defn-france-health-insurance-number"),
    ("france national id card cni", "sit-defn-france-national-id-card"),
    ("france passport number", "sit-defn-france-passport-number"),
    ("france social security number insee", "sit-defn-france-social-security-number"),
    ("france tax identification number", "sit-defn-france-tax-identification-number"),
    ("france value added tax number", "sit-defn-france-value-added-tax-number"),
    ("germany drivers license number", "sit-defn-germany-drivers-license-number"),
    ("germany identity card number", "sit-defn-germany-identity-card-number"),
    ("germany passport number", "sit-defn-germany-passport-number"),
    ("germany tax identification number", "sit-defn-germany-tax-identification-number"),
    ("germany value added tax number", "sit-defn-germany-value-added-tax-number"),
    ("greece drivers license number", "sit-defn-greece-drivers-license-number"),
    ("greece national id card", "sit-defn-greece-national-id-card"),
    ("greece passport number", "sit-defn-greece-passport-number"),
    ("greece social security number amka", "sit-defn-greece-social-security-number"),
    ("greece tax identification number", "sit-defn-greece-tax-identification-number"),
    ("hong kong identity card hkid number", "sit-defn-hong-kong-identity-card-number"),
    ("hungary drivers license number", "sit-defn-hungary-drivers-license-number"),
    ("hungary passport number", "sit-defn-hungary-passport-number"),
    ("hungary personal identification number", "sit-defn-hungary-personal-identification-number"),
    ("hungary social security number taj", "sit-defn-hungary-social-security-number"),
    ("hungary tax identification number", "sit-defn-hungary-tax-identification-number"),
    ("hungary value added tax number", "sit-defn-hungary-value-added-tax-number"),
    ("india drivers license number", "sit-defn-india-drivers-license-number"),
    ("india gst number", "sit-defn-india-gst-number"),
    ("india permanent account number pan", "sit-defn-india-permanent-account-number"),
    ("india unique identification aadhaar number", "sit-defn-india-unique-identification-number"),
    ("india voter id card", "sit-defn-india-voter-id-card"),
    ("indonesia drivers license number", "sit-defn-indonesia-drivers-license-number"),
    ("indonesia identity card ktp number", "sit-defn-indonesia-identity-card-number"),
    ("indonesia passport number", "sit-defn-indonesia-passport-number"),
    ("international banking account number iban", "sit-defn-international-banking-account-number"),
    ("ip address", "sit-defn-ip-address"),
    ("ip address v4", "sit-defn-ip-address-v4"),
    ("ip address v6", "sit-defn-ip-address-v6"),
    ("ipv4", "sit-defn-ip-address-v4"),
    ("ipv6", "sit-defn-ip-address-v6"),
    ("ireland drivers license number", "sit-defn-ireland-drivers-license-number"),
    ("ireland passport number", "sit-defn-ireland-passport-number"),
    ("ireland personal public service pps number", "sit-defn-ireland-personal-public-service-number"),
    ("israel bank account number", "sit-defn-israel-bank-account-number"),
    ("israel national identification number", "sit-defn-israel-national-identification-number"),
    ("italy drivers license number", "sit-defn-italy-drivers-license-number"),
    ("italy fiscal code", "sit-defn-italy-fiscal-code"),
    ("italy passport number", "sit-defn-italy-passport-number"),
    ("italy value added tax number", "sit-defn-italy-value-added-tax-number"),
    ("japan bank account number", "sit-defn-japan-bank-account-number"),
    ("japan drivers license number", "sit-defn-japan-drivers-license-number"),
    ("japan my number corporate", "sit-defn-japan-my-number-corporate"),
    ("japan my number personal", "sit-defn-japan-my-number-personal"),
    ("japan passport number", "sit-defn-japan-passport-number"),
    ("japan residence card number", "sit-defn-japan-residence-card-number"),
    ("japan resident registration number", "sit-defn-japan-resident-registration-number"),
    ("japan social insurance number sin", "sit-defn-japan-social-insurance-number"),
    ("latvia drivers license number", "sit-defn-latvia-drivers-license-number"),
    ("latvia passport number", "sit-defn-latvia-passport-number"),
    ("latvia personal code", "sit-defn-latvia-personal-code"),
    ("lithuania drivers license number", "sit-defn-lithuania-drivers-license-number"),
    ("lithuania passport number", "sit-defn-lithuania-passport-number"),
    ("lithuania personal code", "sit-defn-lithuania-personal-code"),
    ("luxemburg drivers license number", "sit-defn-luxemburg-drivers-license-number"),
    ("luxemburg national identification number", "sit-defn-luxemburg-national-identification-number-natural-persons"),
    ("luxemburg passport number", "sit-defn-luxemburg-passport-number"),
    ("malaysia identification card number", "sit-defn-malaysia-identification-card-number"),
    ("malaysia passport number", "sit-defn-malaysia-passport-number"),
    ("malta drivers license number", "sit-defn-malta-drivers-license-number"),
    ("malta identity card number", "sit-defn-malta-identity-card-number"),
    ("malta passport number", "sit-defn-malta-passport-number"),
    ("malta tax identification number", "sit-defn-malta-tax-identification-number"),
    ("mexico unique population registry code curp", "sit-defn-mexico-unique-population-registry-code"),
    ("netherlands citizens service bsn number", "sit-defn-netherlands-citizens-service-number"),
    ("netherlands drivers license number", "sit-defn-netherlands-drivers-license-number"),
    ("netherlands passport number", "sit-defn-netherlands-passport-number"),
    ("netherlands tax identification number", "sit-defn-netherlands-tax-identification-number"),
    ("netherlands value added tax number", "sit-defn-netherlands-value-added-tax-number"),
    ("new zealand bank account number", "sit-defn-new-zealand-bank-account-number"),
    ("new zealand drivers license number", "sit-defn-new-zealand-drivers-license-number"),
    ("new zealand inland revenue number", "sit-defn-new-zealand-inland-revenue-number"),
    ("new zealand ministry of health number", "sit-defn-new-zealand-ministry-of-health-number"),
    ("new zealand social welfare number", "sit-defn-new-zealand-social-welfare-number"),
    ("norway identification number", "sit-defn-norway-identification-number"),
    ("philippines national identification number", "sit-defn-philippines-national-identification-number"),
    ("philippines passport number", "sit-defn-philippines-passport-number"),
    ("philippines unified multi purpose identification number", "sit-defn-philippines-unified-multi-purpose-identification-number"),
    ("poland drivers license number", "sit-defn-poland-drivers-license-number"),
    ("poland identity card", "sit-defn-poland-identity-card"),
    ("poland national id pesel", "sit-defn-poland-national-id"),
    ("poland passport number", "sit-defn-poland-passport-number"),
    ("poland regon number", "sit-defn-poland-regon-number"),
    ("poland tax identification number", "sit-defn-poland-tax-identification-number"),
    ("portugal citizen card number", "sit-defn-portugal-citizen-card-number"),
    ("portugal drivers license number", "sit-defn-portugal-drivers-license-number"),
    ("portugal passport number", "sit-defn-portugal-passport-number"),
    ("portugal tax identification number", "sit-defn-portugal-tax-identification-number"),
    ("qatar identification card number", "sit-defn-qatari-id-card-number"),
    ("romania drivers license number", "sit-defn-romania-drivers-license-number"),
    ("romania passport number", "sit-defn-romania-passport-number"),
    ("romania personal numeric code cnp", "sit-defn-romania-personal-numeric-code"),
    ("russia passport number domestic", "sit-defn-russia-passport-number-domestic"),
    ("russia passport number international", "sit-defn-russia-passport-number-international"),
    ("saudi arabia national id", "sit-defn-saudi-arabia-national-id"),
    ("singapore passport number", "sit-defn-singapore-passport-number"),
    ("singapore national registration identity card nric number", "sit-defn-singapore-national-registration-identity-card-number"),
    ("slovakia drivers license number", "sit-defn-slovakia-drivers-license-number"),
    ("slovakia passport number", "sit-defn-slovakia-passport-number"),
    ("slovakia personal number", "sit-defn-slovakia-personal-number"),
    ("slovenia drivers license number", "sit-defn-slovenia-drivers-license-number"),
    ("slovenia passport number", "sit-defn-slovenia-passport-number"),
    ("slovenia tax identification number", "sit-defn-slovenia-tax-identification-number"),
    ("slovenia unique master citizen number", "sit-defn-slovenia-unique-master-citizen-number"),
    ("south africa identification number", "sit-defn-south-africa-identification-number"),
    ("south korea drivers license number", "sit-defn-south-korea-drivers-license-number"),
    ("south korea passport number", "sit-defn-south-korea-passport-number"),
    ("south korea resident registration number", "sit-defn-south-korea-resident-registration-number"),
    ("spain dni", "sit-defn-spain-dni"),
    ("spain drivers license number", "sit-defn-spain-drivers-license-number"),
    ("spain passport number", "sit-defn-spain-passport-number"),
    ("spain social security number ssn", "sit-defn-spain-social-security-number"),
    ("spain tax identification number", "sit-defn-spain-tax-identification-number"),
    ("sql server connection string", "sit-defn-sql-server-connection-string"),
    ("sweden drivers license number", "sit-defn-sweden-drivers-license-number"),
    ("sweden national id", "sit-defn-sweden-national-id"),
    ("sweden passport number", "sit-defn-sweden-passport-number"),
    ("sweden tax identification number", "sit-defn-sweden-tax-identification-number"),
    ("swift code", "sit-defn-swift-code"),
    ("switzerland ssn ahv number", "sit-defn-switzerland-ssn-ahv-number"),
    ("taiwan national identification number", "sit-defn-taiwan-national-identification-number"),
    ("taiwan passport number", "sit-defn-taiwan-passport-number"),
    ("taiwan resident certificate arc tarc number", "sit-defn-taiwan-resident-certificate-number"),
    ("thai population identification code", "sit-defn-thai-population-identification-code"),
    ("turkey national identification number", "sit-defn-turkey-national-identification-number"),
    ("uae identity card number", "sit-defn-uae-identity-card-number"),
    ("uae passport number", "sit-defn-uae-passport-number"),
    ("uk drivers license number", "sit-defn-uk-drivers-license-number"),
    ("uk electoral roll number", "sit-defn-uk-electoral-roll-number"),
    ("uk national health service number", "sit-defn-uk-national-health-service-number"),
    ("uk national insurance number nino", "sit-defn-uk-national-insurance-number"),
    ("uk unique taxpayer reference number", "sit-defn-uk-unique-taxpayer-reference-number"),
    ("us bank account number", "sit-defn-us-bank-account-number"),
    ("us drivers license number", "sit-defn-us-drivers-license-number"),
    ("us individual taxpayer identification number itin", "sit-defn-us-individual-taxpayer-identification-number"),
    ("us social security number ssn", "sit-defn-us-social-security-number"),
    ("us uk passport number", "sit-defn-us-uk-passport-number"),
    ("ukraine passport domestic", "sit-defn-ukraine-passport-domestic"),
    ("ukraine passport international", "sit-defn-ukraine-passport-international"),
]


def _find_purview_sit_urls(condition: str) -> list[str]:
    """Find Microsoft Purview SIT definition URLs that match the condition.

    Uses fuzzy keyword matching: a SIT entry matches if every significant
    word in the condition appears somewhere in the SIT label (or vice versa).
    Returns 0-3 URLs, best matches first.
    """
    cond_words = set(
        w for w in _re.split(r"[\s_\-']+", condition.lower()) if len(w) > 1
    )
    stopwords = {"the", "of", "and", "for", "in", "an", "a", "or", "to", "is"}
    cond_words -= stopwords

    if not cond_words:
        return []

    scored: list[tuple[float, str]] = []
    for label, slug in _PURVIEW_SIT_SLUGS:
        label_words = set(
            w for w in _re.split(r"[\s_\-']+", label.lower()) if len(w) > 1
        ) - stopwords
        if not label_words:
            continue

        overlap = cond_words & label_words
        if not overlap:
            continue

        jaccard = len(overlap) / len(cond_words | label_words)
        cond_coverage = len(overlap) / len(cond_words)
        score = 0.6 * jaccard + 0.4 * cond_coverage

        if score > 0.25:
            url = f"{_PURVIEW_BASE}/{slug}"
            scored.append((score, url))

    scored.sort(key=lambda x: -x[0])
    urls = [url for _, url in scored[:3]]

    if urls:
        logger.info(
            "Agent 1P [purview]: matched %d SIT definitions for '%s'",
            len(urls), condition,
        )
        for sc, u in scored[:3]:
            logger.debug("Agent 1P [purview]: score=%.2f → %s", sc, u)

    return urls


def research_policies(condition: str, config: dict) -> dict:
    """Search → rank → scrape → extract → LLM for format rules and vendor patterns.

    Pipeline:
      1. Search broadly for ~20 candidate policy URLs.
      2. Score, rank, and deduplicate → top 8 URLs.
      3. Scrape those 8 pages with structured rule-block extraction.
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
        "policy_confidence": "high",
    }

    policy_none_relevant = False
    vendor_none_relevant = False

    # ── Phase 1 (PRIMARY): Official policies ──────────────────────────────────
    raw_policy_urls = _find_policy_urls(condition, config)
    logger.info(
        "Agent 1P [search]: found %d raw policy URLs for '%s'",
        len(raw_policy_urls), condition,
    )

    if raw_policy_urls:
        ranked_urls = _rank_and_dedupe(raw_policy_urls, max_per_domain=2, top_n=8)
        logger.info(
            "Agent 1P [policy]: ranked %d → %d URLs to scrape",
            len(raw_policy_urls), len(ranked_urls),
        )

        scraped = _scrape_and_extract(ranked_urls, config)
        logger.info(
            "Agent 1P [policy]: extracted rule blocks from %d/%d pages",
            len(scraped), len(ranked_urls),
        )

        if scraped:
            scraped, policy_none_relevant = _filter_relevant_pages(
                condition, scraped, config, phase="policy",
            )

            if not policy_none_relevant:
                relevant_policy_urls = [url for url, _ in scraped]
                result["sources"].extend(relevant_policy_urls)

            if policy_none_relevant:
                logger.warning(
                    "Agent 1P [policy]: skipping LLM extraction — no on-topic "
                    "pages found for '%s'. This identifier may lack standardised "
                    "format documentation.", condition,
                )
            else:
                pages_to_use = scraped[:8]
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
        logger.info("Agent 1P [vendor]: found %d vendor URLs for '%s'", len(vendor_urls), condition)
        scraped = _scrape_and_extract(vendor_urls, config)
        logger.info("Agent 1P [vendor]: extracted from %d/%d pages", len(scraped), len(vendor_urls))
        if scraped:
            scraped, vendor_none_relevant = _filter_relevant_pages(
                condition, scraped, config, phase="vendor",
            )

            if not vendor_none_relevant:
                relevant_vendor_urls = [url for url, _ in scraped]
                result["sources"].extend(relevant_vendor_urls)

            if vendor_none_relevant:
                logger.warning(
                    "Agent 1P [vendor]: skipping LLM extraction — no on-topic "
                    "vendor pages found for '%s'.", condition,
                )
            else:
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

    # ── Confidence assessment ─────────────────────────────────────────────────
    if policy_none_relevant and vendor_none_relevant:
        result["policy_confidence"] = "low"
        result["policy_confidence_reason"] = (
            "No scraped pages were found to be specifically about this "
            "identifier. The format may not be standardised or publicly "
            "documented."
        )
        logger.warning(
            "Agent 1P: policy_confidence=low — neither policy nor vendor "
            "pages were on-topic for '%s'", condition,
        )
    elif policy_none_relevant or vendor_none_relevant:
        result["policy_confidence"] = "medium"
        failed_phase = "policy" if policy_none_relevant else "vendor"
        result["policy_confidence_reason"] = (
            f"The {failed_phase} research phase found no on-topic pages. "
            "Extracted rules may be incomplete or from a related identifier."
        )
        logger.info(
            "Agent 1P: policy_confidence=medium — %s phase had no on-topic pages",
            failed_phase,
        )

    # ── Phase 3: LLM fallback if web yielded nothing ──────────────────────────
    if not result["policies"] and not result["vendor_patterns"]:
        logger.info("Agent 1P: no web results — asking LLM from its own knowledge")
        result = _llm_fallback(condition, config, result)

    return result


def _find_policy_urls(condition: str, config: dict) -> list[str]:
    """Cast a wide net: aim for ~20 candidate policy URLs."""
    try:
        from tools.search_tool import search_policy_urls
        return search_policy_urls(
            condition,
            tavily_api_key=config.get("tavily_api_key"),
            max_results=20,
        )
    except Exception as exc:
        logger.warning("Agent 1P [policy]: URL search failed: %s", exc)
        return []


def _find_vendor_urls(condition: str, config: dict) -> list[str]:
    """Search for how DLP/security vendors identify this data type.

    Combines dynamically searched URLs with known vendor documentation
    hubs (Skyhigh, Trellix) and Microsoft Purview SIT definitions.
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

    purview_urls = _find_purview_sit_urls(condition)
    for u in purview_urls:
        if u not in seen:
            seen.add(u)
            urls.append(u)

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

    logger.info(
        "Agent 1P [vendor]: %d total vendor URLs (%d known + %d purview + searched)",
        len(urls), len(_VENDOR_REFERENCE_BASES), len(purview_urls),
    )
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
    page_results = scrape_pages(urls[:8], proxies=proxies or None, sleep_s=1.5)

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


# ── Content-relevance gate ────────────────────────────────────────────────────

_RELEVANCE_PROMPT = """\
CONDITION: {condition}

I scraped the following pages while researching format rules for the
identifier above.  For each page, decide whether it SPECIFICALLY describes
format rules, structural constraints, or policies for "{condition}" — NOT
for a different (but related) identifier type, a different country's version,
or a generic international standard unless it contains a section specifically
about this condition.

{page_summaries}

Return ONLY a JSON object: {{"relevant": [1, 3, ...]}} containing the
page numbers (1-based) that are relevant.  If none are relevant return
{{"relevant": []}}.  No explanation, no markdown fences."""


def _filter_relevant_pages(
    condition: str,
    scraped: list[tuple[str, str]],
    config: dict,
    phase: str = "policy",
) -> tuple[list[tuple[str, str]], bool]:
    """Drop scraped pages that are not actually about *condition*.

    Uses a lightweight LLM call: sends the URL + a short text preview
    of each page and asks which are relevant.  Falls through gracefully
    (returns all pages) if the LLM call fails.

    Returns:
        (filtered_pages, none_relevant) — *none_relevant* is True when the
        LLM determined that zero scraped pages are on-topic.  Callers
        should treat this as a low-confidence signal and avoid extracting
        rules from the (irrelevant) text.
    """
    if not scraped:
        return scraped, False

    summaries: list[str] = []
    for idx, (url, text) in enumerate(scraped, 1):
        preview = text[:300].replace("\n", " ").strip()
        summaries.append(f"Page {idx}: {url}\n  Preview: {preview}")

    prompt = _RELEVANCE_PROMPT.format(
        condition=condition,
        page_summaries="\n\n".join(summaries),
    )

    try:
        response = invoke_claude(prompt, config)
        parsed = _parse_json(response)
        relevant_ids: list[int] = parsed.get("relevant", [])

        if not relevant_ids:
            logger.warning(
                "Agent 1P [%s-relevance]: LLM says NONE of %d pages are relevant "
                "— flagging low confidence (no on-topic documentation found)",
                phase, len(scraped),
            )
            return scraped, True

        kept = [scraped[i - 1] for i in relevant_ids if 1 <= i <= len(scraped)]
        dropped = len(scraped) - len(kept)
        if dropped > 0:
            dropped_urls = [
                url for idx, (url, _) in enumerate(scraped, 1)
                if idx not in relevant_ids
            ]
            for u in dropped_urls:
                logger.info("Agent 1P [%s-relevance]: DROPPED irrelevant page: %s", phase, u)
            logger.info(
                "Agent 1P [%s-relevance]: kept %d/%d pages (dropped %d irrelevant)",
                phase, len(kept), len(scraped), dropped,
            )
        else:
            logger.info("Agent 1P [%s-relevance]: all %d pages are relevant", phase, len(scraped))

        return kept, False

    except Exception as exc:
        logger.warning(
            "Agent 1P [%s-relevance]: LLM check failed (%s) — keeping all %d pages",
            phase, exc, len(scraped),
        )
        return scraped, False


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
