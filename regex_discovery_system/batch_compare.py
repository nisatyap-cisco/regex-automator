#!/usr/bin/env python3
"""Batch comparison: Researcher regex vs Agent-generated regex.

Reads a JSON file of identifiers (with researcher-provided regex),
runs the agent pipeline for each, then scores both against the
agent's truth set and reports which is better.

Usage:
    # Single entry test (first entry only):
    python batch_compare.py --input /path/to/country_identifiers.json --limit 1

    # Full batch:
    python batch_compare.py --input /path/to/country_identifiers.json

    # Skip LLM analysis (faster, just scores):
    python batch_compare.py --input /path/to/country_identifiers.json --no-llm

    # Skip pipeline if results already exist:
    python batch_compare.py --input /path/to/country_identifiers.json --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.bedrock_client import load_config, invoke_claude
from utils.paths import build_paths, slugify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("batch_compare")

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ── Scoring (reused from web/app.py) ─────────────────────────────

def score_regex(regex_str: str, label: str, truth: list[str]) -> dict:
    """Test a single regex against the truth set and return metrics."""
    entry = {"label": label, "regex": regex_str}
    try:
        compiled = re.compile(regex_str)
        matches = [v for v in truth if compiled.fullmatch(v)]
        search_matches = [v for v in truth if compiled.search(v)]
        entry["fullmatch"] = len(matches)
        entry["search_match"] = len(search_matches)
        entry["total"] = len(truth)
        entry["fullmatch_pct"] = round(100 * len(matches) / len(truth), 2) if truth else 0
        entry["search_pct"] = round(100 * len(search_matches) / len(truth), 2) if truth else 0
        missed = [v for v in truth if not compiled.search(v)]
        entry["missed_samples"] = missed[:5]
        entry["error"] = None
    except Exception as e:
        entry["fullmatch"] = 0
        entry["search_match"] = 0
        entry["total"] = len(truth)
        entry["fullmatch_pct"] = 0
        entry["search_pct"] = 0
        entry["missed_samples"] = []
        entry["error"] = str(e)
    return entry


# ── Structural comparison (no test data needed) ───────────────────

_STRUCTURAL_ANALYSIS_PROMPT = """\
You are a regex analysis expert. Analyze the following regex pattern and extract ALL the structural conditions/rules it enforces.

CONDITION NAME: {condition}
REGEX: {regex}

List every structural rule this regex enforces. Be specific and exhaustive. Examples:
- "Length is exactly 9 characters"
- "First 2 characters must be uppercase letters A-Z"
- "Positions 3-4 are digits 0-9"
- "Optional hyphen or space separator allowed"
- "Last character is a check digit 0-9"
- "Starts with specific prefixes: AN, AP, AR, etc."

Return a JSON object with:
{{
  "rules": ["rule 1", "rule 2", ...],
  "length": "fixed 9" or "variable 8-12" or "unknown",
  "format_summary": "2 letters + 7 digits" (brief format description)
}}

Return ONLY valid JSON, no markdown fences."""


_STRUCTURAL_COMPARE_PROMPT = """\
You are a regex comparison expert. Compare these two regex patterns for detecting **{condition}**.

RESEARCHER REGEX: {researcher_regex}
RESEARCHER RULES:
{researcher_rules}

AGENT REGEX: {agent_regex}
AGENT RULES:
{agent_rules}

Compare and analyze:

1. **Rule Coverage**: What percentage of the Researcher's rules are also covered by the Agent's regex?
   List each Researcher rule and whether Agent covers it (✓) or not (✗).

2. **Extra Agent Rules**: What rules does the Agent enforce that the Researcher doesn't?

3. **Missing Agent Rules**: What Researcher rules are NOT covered by the Agent?

4. **Strictness Comparison**: Which regex is more strict/specific? Which might have more false positives?

5. **Winner**: Based on rule coverage and quality, which is better for production use?

6. **Coverage Score**: Give a percentage (0-100%) of how much the Agent covers the Researcher's rules.

Return a JSON object:
{{
  "coverage_pct": <number 0-100>,
  "researcher_rules_matched": ["rule that agent covers", ...],
  "researcher_rules_missed": ["rule that agent misses", ...],
  "agent_extra_rules": ["extra rule agent has", ...],
  "strictness": "Agent more strict" | "Researcher more strict" | "Similar",
  "winner": "Agent" | "Researcher" | "Tie",
  "winner_reason": "brief explanation",
  "analysis": "2-3 sentence summary"
}}

Return ONLY valid JSON, no markdown fences."""


def analyze_regex_structure(regex: str, condition: str, config: dict) -> dict:
    """Use LLM to extract structural rules from a regex pattern."""
    try:
        response = invoke_claude(
            _STRUCTURAL_ANALYSIS_PROMPT.format(condition=condition, regex=regex),
            config,
            system="You are a regex analysis expert. Return only valid JSON."
        )
        return _parse_json_safe(response)
    except Exception as e:
        logger.warning("Failed to analyze regex structure: %s", e)
        return {"rules": [], "length": "unknown", "format_summary": "unknown", "error": str(e)}


def compare_structural(condition: str, researcher_regex: str, agent_regex: str, config: dict) -> dict:
    """Compare two regex patterns structurally using LLM analysis."""
    logger.info("  📊 Analyzing Researcher regex structure...")
    researcher_struct = analyze_regex_structure(researcher_regex, condition, config)
    
    logger.info("  📊 Analyzing Agent regex structure...")
    agent_struct = analyze_regex_structure(agent_regex, condition, config)
    
    researcher_rules_text = "\n".join(f"  - {r}" for r in researcher_struct.get("rules", []))
    agent_rules_text = "\n".join(f"  - {r}" for r in agent_struct.get("rules", []))
    
    logger.info("  📊 Comparing structures...")
    try:
        response = invoke_claude(
            _STRUCTURAL_COMPARE_PROMPT.format(
                condition=condition,
                researcher_regex=researcher_regex,
                agent_regex=agent_regex,
                researcher_rules=researcher_rules_text or "  (no rules extracted)",
                agent_rules=agent_rules_text or "  (no rules extracted)",
            ),
            config,
            system="You are a regex comparison expert. Return only valid JSON."
        )
        comparison = _parse_json_safe(response)
    except Exception as e:
        logger.warning("Failed to compare structures: %s", e)
        comparison = {"coverage_pct": 0, "winner": "Unknown", "error": str(e)}
    
    return {
        "researcher_structure": researcher_struct,
        "agent_structure": agent_struct,
        "comparison": comparison,
    }


def _parse_json_safe(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        import re as re_mod
        match = re_mod.search(r'\{[\s\S]*\}', text)
        if match:
            return json.loads(match.group())
        return {"error": "Failed to parse JSON", "raw": text[:500]}


# ── Parse researcher regex ────────────────────────────────────────

def parse_researcher_regex(raw: str) -> dict:
    """Split the researcher regex string into value regex(es) and keyword regex.

    The JSON format uses ' , ' to separate value regex from keyword regex.
    Some entries have multiple parts: value_regex , keyword_regex , extra_regex
    """
    parts = [p.strip() for p in raw.split(" , ")]
    result = {"value_regexes": [], "keyword_regex": ""}

    if len(parts) == 1:
        # Only a value regex (or only a keyword regex if it starts with (?i))
        if parts[0].startswith("(?i)"):
            result["keyword_regex"] = parts[0]
        else:
            result["value_regexes"].append(parts[0])
    elif len(parts) == 2:
        # value_regex , keyword_regex
        result["value_regexes"].append(parts[0])
        result["keyword_regex"] = parts[1]
    elif len(parts) >= 3:
        # value_regex , keyword_regex , extra_value_regex (e.g. Colombia Driver's License)
        result["value_regexes"].append(parts[0])
        result["keyword_regex"] = parts[1]
        for extra in parts[2:]:
            result["value_regexes"].append(extra)

    return result


# ── Run the agent pipeline ────────────────────────────────────────

def run_agent_pipeline(condition: str, skip_existing: bool = False) -> bool:
    """Run main.py for a condition. Returns True if results are available."""
    slug = slugify(condition)
    regex_path = RESULTS_DIR / f"{slug}_regex_patterns.json"
    test_path = DATA_DIR / f"{slug}_test.txt"

    if skip_existing and regex_path.exists() and test_path.exists():
        logger.info("⏭  Skipping pipeline for '%s' (results already exist)", condition)
        return True

    logger.info("🚀 Running agent pipeline for '%s' ...", condition)
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--condition", condition],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
        if result.returncode != 0:
            logger.error("❌ Pipeline failed for '%s':\n%s", condition, result.stderr[-500:] if result.stderr else "no stderr")
            return False
        logger.info("✅ Pipeline completed for '%s'", condition)
        return True
    except subprocess.TimeoutExpired:
        logger.error("⏰ Pipeline timed out for '%s'", condition)
        return False
    except Exception as e:
        logger.error("❌ Pipeline error for '%s': %s", condition, e)
        return False


# ── Compare one entry ─────────────────────────────────────────────

def compare_one(condition: str, researcher_regex_raw: str, config: dict, use_llm: bool = True) -> dict | None:
    """Compare researcher regex vs agent regex for a single identifier."""
    slug = slugify(condition)

    # Load agent's regex patterns
    regex_path = RESULTS_DIR / f"{slug}_regex_patterns.json"
    if not regex_path.exists():
        logger.error("No agent regex found for '%s'", condition)
        return None

    with open(regex_path) as f:
        sys_data = json.load(f)

    sys_patterns = sys_data.get("patterns", [])
    sys_value_patterns = [p for p in sys_patterns if p.get("type") == "value"]
    sys_keyword = next((p["regex"] for p in sys_patterns if p.get("type") == "context"), "")

    # Parse researcher regex
    researcher = parse_researcher_regex(researcher_regex_raw)

    # Load truth set (optional now)
    test_path = DATA_DIR / f"{slug}_test.txt"
    truth = []
    if test_path.exists():
        truth = [line.strip() for line in test_path.read_text().splitlines() if line.strip()]

    # Get the best agent value regex (prefer combined-value, else first)
    agent_combined = next((p for p in sys_value_patterns if p.get("rule_id") == "combined-value"), None)
    agent_best_regex = agent_combined["regex"] if agent_combined else (sys_value_patterns[0]["regex"] if sys_value_patterns else "")
    researcher_best_regex = researcher["value_regexes"][0] if researcher["value_regexes"] else ""

    result = {
        "condition": condition,
        "slug": slug,
        "agent_regex": agent_best_regex,
        "researcher_regex": researcher_best_regex,
        "agent_keyword": sys_keyword,
        "researcher_keyword": researcher.get("keyword_regex", ""),
        "mode": "structural",  # Always use structural comparison
    }

    # ── Structural comparison (always) ──
    logger.info("  📊 Using structural comparison (rule-based)")
    result["truth_size"] = 0

    if not agent_best_regex or not researcher_best_regex:
        logger.warning("Missing regex for structural comparison")
        result["winner"] = "None"
        result["coverage_pct"] = 0
        return result

    if use_llm:
        struct_result = compare_structural(condition, researcher_best_regex, agent_best_regex, config)
        result["structural"] = struct_result

        comparison = struct_result.get("comparison", {})
        result["coverage_pct"] = comparison.get("coverage_pct", 0)
        result["winner"] = comparison.get("winner", "Unknown")
        result["winner_reason"] = comparison.get("winner_reason", "")
        result["analysis"] = comparison.get("analysis", "")
        result["rules_matched"] = comparison.get("researcher_rules_matched", [])
        result["rules_missed"] = comparison.get("researcher_rules_missed", [])
        result["agent_extra_rules"] = comparison.get("agent_extra_rules", [])
    else:
        result["winner"] = "Unknown"
        result["coverage_pct"] = 0

    return result


def _llm_compare(slug: str, scored: list[dict], sys_kw: str, user_kw: str, config: dict) -> str:
    condition = slug.replace("_", " ")
    table_rows = []
    for s in scored:
        table_rows.append(
            f"- {s['label']}: `{s['regex']}` → fullmatch {s['fullmatch_pct']}%, "
            f"search {s['search_pct']}%, missed samples: {s.get('missed_samples', [])}"
        )

    prompt = f"""\
You are a regex accuracy analyst. Compare the following regex patterns for
detecting **{condition}** identifiers.

Truth set: {scored[0]['total'] if scored else 0} known-valid values.

Regex results:
{chr(10).join(table_rows)}

System keyword regex: `{sys_kw or 'N/A'}`
Researcher keyword regex: `{user_kw or 'N/A'}`

Provide a concise analysis in Markdown:
1. **Winner** — which regex is most accurate and why (consider both precision and recall).
2. **Strengths & weaknesses** of each regex (1-2 sentences each).
3. **False positive risk** — which regex is more likely to match non-{condition} values and why.
4. **Recommendation** — the ideal regex for production use.
5. If keyword regexes are provided, briefly compare their coverage.

Be specific and reference the actual patterns. Keep it under 300 words.
"""
    return invoke_claude(prompt, config, system="You are a regex accuracy analyst.")


# ── Pretty-print one result ───────────────────────────────────────

def print_result(r: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  {r['condition']}")
    print("=" * 70)
    
    mode = r.get("mode", "data")
    print(f"  Mode: {'📊 Data-based' if mode == 'data' else '🔍 Structural'} comparison")
    
    if mode == "data":
        print(f"  Truth set: {r['truth_size']} values")
        print()

        # Agent scores
        if r.get("best_agent"):
            a = r["best_agent"]
            print(f"  🤖 Agent best:      fullmatch {a['fullmatch_pct']}%  |  search {a['search_pct']}%")
            print(f"     Regex: {a['regex'][:80]}{'...' if len(a['regex']) > 80 else ''}")
            if a.get("error"):
                print(f"     ⚠ Error: {a['error']}")
        else:
            print("  🤖 Agent: no value regex generated")

        # Researcher scores
        if r.get("best_researcher"):
            rs = r["best_researcher"]
            print(f"  📝 Researcher best: fullmatch {rs['fullmatch_pct']}%  |  search {rs['search_pct']}%")
            print(f"     Regex: {rs['regex'][:80]}{'...' if len(rs['regex']) > 80 else ''}")
            if rs.get("error"):
                print(f"     ⚠ Error: {rs['error']}")
        else:
            print("  📝 Researcher: no value regex provided")

        # Winner
        w = r["winner"]
        emoji = "🤖" if w == "Agent" else "📝" if w == "Researcher" else "🤝"
        print(f"\n  {emoji} Winner: {w}", end="")
        if r.get("margin"):
            print(f"  (by {r['margin']}%)")
        else:
            print()

        # LLM analysis
        if r.get("llm_analysis"):
            print("\n  --- LLM Analysis ---")
            for line in r["llm_analysis"].split("\n"):
                print(f"  {line}")

    else:
        # Structural comparison output
        print()
        print(f"  📝 Researcher regex: {r.get('researcher_regex', 'N/A')[:70]}...")
        print(f"  🤖 Agent regex:      {r.get('agent_regex', 'N/A')[:70]}...")
        
        coverage = r.get("coverage_pct", 0)
        print(f"\n  📈 Rule Coverage: {coverage}%")
        
        # Show matched rules
        matched = r.get("rules_matched", [])
        if matched:
            print(f"\n  ✅ Rules Agent covers ({len(matched)}):")
            for rule in matched[:5]:
                print(f"     • {rule}")
            if len(matched) > 5:
                print(f"     ... and {len(matched) - 5} more")
        
        # Show missed rules
        missed = r.get("rules_missed", [])
        if missed:
            print(f"\n  ❌ Rules Agent misses ({len(missed)}):")
            for rule in missed[:5]:
                print(f"     • {rule}")
            if len(missed) > 5:
                print(f"     ... and {len(missed) - 5} more")
        
        # Show extra agent rules
        extra = r.get("agent_extra_rules", [])
        if extra:
            print(f"\n  ➕ Extra rules Agent has ({len(extra)}):")
            for rule in extra[:3]:
                print(f"     • {rule}")
            if len(extra) > 3:
                print(f"     ... and {len(extra) - 3} more")
        
        # Winner
        w = r.get("winner", "Unknown")
        emoji = "🤖" if w == "Agent" else "📝" if w == "Researcher" else "🤝"
        print(f"\n  {emoji} Winner: {w}")
        if r.get("winner_reason"):
            print(f"     Reason: {r['winner_reason']}")
        
        if r.get("analysis"):
            print(f"\n  📋 Analysis: {r['analysis']}")

    # Keywords (both modes)
    if r.get("agent_keyword"):
        print(f"\n  🤖 Agent keyword:      {r['agent_keyword'][:60]}...")
    if r.get("researcher_keyword"):
        print(f"  📝 Researcher keyword: {r['researcher_keyword'][:60]}...")

    print()


# ── Aggregate summary ─────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    total = len(results)
    agent_wins = sum(1 for r in results if r.get("winner") == "Agent")
    researcher_wins = sum(1 for r in results if r.get("winner") == "Researcher")
    ties = sum(1 for r in results if r.get("winner") == "Tie")
    unknown = sum(1 for r in results if r.get("winner") in ("None", "Unknown"))

    data_mode = sum(1 for r in results if r.get("mode") == "data")
    struct_mode = sum(1 for r in results if r.get("mode") == "structural")

    print("\n" + "=" * 70)
    print("  BATCH SUMMARY")
    print("=" * 70)
    print(f"  Total identifiers tested: {total}")
    print(f"  📊 Data-based comparisons: {data_mode}")
    print(f"  🔍 Structural comparisons: {struct_mode}")
    print()
    print(f"  🤖 Agent wins:      {agent_wins:3d}  ({round(100*agent_wins/total, 1) if total else 0}%)")
    print(f"  📝 Researcher wins: {researcher_wins:3d}  ({round(100*researcher_wins/total, 1) if total else 0}%)")
    print(f"  🤝 Ties:            {ties:3d}  ({round(100*ties/total, 1) if total else 0}%)")
    if unknown:
        print(f"  ❓ Unknown:         {unknown:3d}  ({round(100*unknown/total, 1) if total else 0}%)")

    # Average coverage (for structural mode)
    struct_results = [r for r in results if r.get("mode") == "structural"]
    if struct_results:
        coverages = [r.get("coverage_pct", 0) for r in struct_results]
        avg_coverage = round(sum(coverages) / len(coverages), 1) if coverages else 0
        print(f"\n  📈 Avg structural coverage: {avg_coverage}%")

    # Average scores (for data mode)
    data_results = [r for r in results if r.get("mode") == "data"]
    if data_results:
        agent_search = [r["best_agent"]["search_pct"] for r in data_results if r.get("best_agent")]
        researcher_search = [r["best_researcher"]["search_pct"] for r in data_results if r.get("best_researcher")]
        if agent_search:
            print(f"\n  🤖 Agent avg search:      {round(sum(agent_search)/len(agent_search), 2)}%")
        if researcher_search:
            print(f"  📝 Researcher avg search: {round(sum(researcher_search)/len(researcher_search), 2)}%")

    # Per-entry table
    print(f"\n  {'Identifier':<40} {'Mode':>10} {'Coverage/Score':>15} {'Winner':>12}")
    print("  " + "-" * 80)
    for r in results:
        mode = r.get("mode", "?")[:5]
        if r.get("mode") == "data":
            score = f"{r['best_agent']['search_pct']}%" if r.get("best_agent") else "N/A"
        else:
            score = f"{r.get('coverage_pct', 0)}%"
        w = r.get("winner", "?")
        print(f"  {r['condition'][:39]:<40} {mode:>10} {score:>15} {w:>12}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch compare researcher vs agent regex")
    parser.add_argument("--input", required=True, help="Path to JSON file with identifier + regex entries")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N entries")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis (faster)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip pipeline if agent results already exist")
    parser.add_argument("--no-pipeline", action="store_true", help="Only compare (don't run agent pipeline)")
    args = parser.parse_args()

    with open(args.input) as f:
        entries = json.load(f)

    if args.limit:
        entries = entries[:args.limit]

    logger.info("Loaded %d entries from %s", len(entries), args.input)

    config = load_config("config.yaml")
    results: list[dict] = []
    failed: list[str] = []

    for i, entry in enumerate(entries, 1):
        condition = entry["value"]
        researcher_regex = entry["regex"]

        logger.info("━" * 60)
        logger.info("[%d/%d] %s", i, len(entries), condition)
        logger.info("━" * 60)

        # Step 1: Run agent pipeline (unless --no-pipeline)
        if not args.no_pipeline:
            ok = run_agent_pipeline(condition, skip_existing=args.skip_existing)
            if not ok:
                logger.error("Skipping '%s' — pipeline failed", condition)
                failed.append(condition)
                continue

        # Step 2: Compare
        result = compare_one(condition, researcher_regex, config, use_llm=not args.no_llm)
        if result:
            results.append(result)
            print_result(result)
        else:
            logger.warning("No comparison result for '%s'", condition)
            failed.append(condition)

    # Save results
    if results:
        print_summary(results)

        out_path = RESULTS_DIR / "batch_compare_results.json"
        with open(out_path, "w") as f:
            # Strip non-serializable bits
            clean = []
            for r in results:
                c = {k: v for k, v in r.items() if k != "all_scores"}
                c["all_scores_summary"] = [
                    {"label": s["label"], "fullmatch_pct": s["fullmatch_pct"],
                     "search_pct": s["search_pct"], "error": s["error"]}
                    for s in r["all_scores"]
                ]
                clean.append(c)
            json.dump(clean, f, indent=2)
        logger.info("Results saved → %s", out_path)

    if failed:
        logger.warning("Failed entries (%d): %s", len(failed), ", ".join(failed))


if __name__ == "__main__":
    main()
