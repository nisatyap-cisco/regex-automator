"""
update_reports.py
-----------------
Updates the combined-value regex in *_final_report.md and
*_validation_report.json to match whatever is now stored in
*_regex_patterns.json (after rerun_combined_regex.py ran).

Run from regex_discovery_system/:
    source .venv/bin/activate && python3 update_reports.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _get_new_combined(slug: str) -> str | None:
    rp = RESULTS_DIR / f"{slug}_regex_patterns.json"
    if not rp.exists():
        return None
    data = json.loads(rp.read_text())
    for p in data.get("patterns", []):
        if p.get("rule_id") == "combined-value":
            return p.get("regex")
    return None


def update_validation_report(slug: str, new_regex: str) -> bool:
    path = RESULTS_DIR / f"{slug}_validation_report.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    changed = False
    for result in data.get("results", []):
        if result.get("rule_id") == "combined-value":
            old = result.get("regex", "")
            if old != new_regex:
                result["regex"] = new_regex
                changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return changed


def update_final_report(slug: str, new_regex: str) -> bool:
    path = RESULTS_DIR / f"{slug}_final_report.md"
    if not path.exists():
        return False
    text = path.read_text()

    # Find the combined-value section and replace the Regex line inside it
    # Section starts with "### combined-value" and the regex is on the next
    # "- **Regex**:" line.
    pattern = re.compile(
        r'(### combined-value\n(?:.*\n)*?- \*\*Regex\*\*: `)([^`]+)(`)',
    )
    m = pattern.search(text)
    if not m:
        return False
    old_regex = m.group(2)
    if old_regex == new_regex:
        return False
    new_text = pattern.sub(
        lambda mo: mo.group(1) + new_regex + mo.group(3),
        text,
        count=1,
    )
    path.write_text(new_text)
    return True


def main() -> None:
    slugs = sorted(
        p.stem.replace("_regex_patterns", "")
        for p in RESULTS_DIR.glob("*_regex_patterns.json")
    )

    print(f"{'SLUG':<50}  {'MD':>4}  {'JSON':>4}  NEW REGEX")
    print("-" * 120)

    for slug in slugs:
        new_regex = _get_new_combined(slug)
        if not new_regex:
            print(f"{slug:<50}  {'–':>4}  {'–':>4}  (no combined-value found)")
            continue

        md_changed   = update_final_report(slug, new_regex)
        json_changed = update_validation_report(slug, new_regex)

        md_tag   = "✅" if md_changed   else "–"
        json_tag = "✅" if json_changed else "–"
        print(f"{slug:<50}  {md_tag:>4}  {json_tag:>4}  {new_regex}")

    print("\nDone.")


if __name__ == "__main__":
    main()
