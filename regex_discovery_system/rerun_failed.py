#!/usr/bin/env python3
"""Re-run only the FAILED identifiers and update the output JSON."""

import json, os, re, subprocess, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
MAIN_PY    = os.path.join(SCRIPT_DIR, "main.py")
RESULTS_DIR= os.path.join(SCRIPT_DIR, "results")
OUTPUT_FILE= os.path.join(ROOT_DIR, "batch2_first10_agent_regex.json")
VENV_PYTHON= os.path.join(ROOT_DIR, ".venv", "bin", "python3")

FAILED = [
    "Denmark Routing number",
    "Sweden Social Security Number",
]

def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def extract_agent_regex(identifier):
    slug = slugify(identifier)
    path = os.path.join(RESULTS_DIR, f"{slug}_regex_patterns.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    for p in data.get("patterns", []):
        if p.get("rule_id") == "combined-value" and p.get("type") == "value":
            return p["regex"]
    for p in data.get("patterns", []):
        if p.get("type") == "value":
            return p["regex"]
    return None

def main():
    python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable
    total = len(FAILED)

    for i, ident in enumerate(FAILED, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{total}] Re-running: {ident}")
        print(f"{'='*70}\n")
        start = time.time()
        result = subprocess.run(
            [python, MAIN_PY, "--condition", ident],
            cwd=SCRIPT_DIR, timeout=600,
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        regex = extract_agent_regex(ident) if ok else None
        status = f"OK → {regex}" if regex else f"FAIL (exit {result.returncode})"
        print(f"\n>>> [{i}/{total}] {ident}: {status} in {elapsed:.1f}s\n")

    # Patch the output JSON
    with open(OUTPUT_FILE) as f:
        entries = json.load(f)
    for entry in entries:
        if entry["agent_regex"] == "FAILED":
            regex = extract_agent_regex(entry["identifier"])
            if regex:
                entry["agent_regex"] = regex
                print(f"  PATCHED: {entry['identifier']} → {regex}")
            else:
                print(f"  STILL FAILED: {entry['identifier']}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nUpdated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
