"""Flask UI for the Regex Discovery System."""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from utils.bedrock_client import load_config
from utils.paths import build_paths

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
)
app.secret_key = os.urandom(32)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Pipeline loggers that should stream at DEBUG level to the terminal.
_PIPELINE_PREFIXES = (
    "agents.", "tools.", "utils.", "web", "main",
    "botocore.credentials",
)

class _PipelineConsoleFilter(logging.Filter):
    """Pass DEBUG+ for pipeline loggers; suppress DEBUG for everything else
    (werkzeug, urllib3, boto3 internals, etc.) so HTTP noise stays quiet."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if any(name == p.rstrip(".") or name.startswith(p) for p in _PIPELINE_PREFIXES):
            return True                      # full DEBUG detail for our code
        return record.levelno >= logging.WARNING  # silence 3rd-party spam


_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)   # let the filter decide, not the level
_console_handler.setFormatter(_log_formatter)
_console_handler.addFilter(_PipelineConsoleFilter())

_file_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "pipeline.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8",
)
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_console_handler, _file_handler],
)

# Keep werkzeug access logs at WARNING in the terminal (they still go to file).
logging.getLogger("werkzeug").setLevel(logging.WARNING)

logger = logging.getLogger("web")

_runs: dict[str, dict] = {}
_logs: dict[str, list[str]] = {}
_resume_events: dict[str, threading.Event] = {}

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
NOTES_DIR = ROOT / "notes"


def _list_past_runs() -> list[dict]:
    """Scan results/ for *_final_report.md to build run history."""
    runs = []
    for p in sorted(RESULTS_DIR.glob("*_final_report.md"), reverse=True):
        slug = p.stem.replace("_final_report", "")
        condition = slug.replace("_", " ")
        overall = ""
        vr_path = RESULTS_DIR / f"{slug}_validation_report.json"
        if vr_path.exists():
            try:
                vr = json.loads(vr_path.read_text())
                overall = vr.get("overall_status", "")
            except Exception:
                pass
        runs.append({
            "slug": slug,
            "condition": condition,
            "file": p.name,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
            "overall": overall,
        })
    return runs


def _get_run_files(slug: str) -> dict:
    """Gather all output files for a given run slug."""
    files = {}
    mapping = {
        "patterns": f"{slug}_patterns.json",
        "regex_patterns": f"{slug}_regex_patterns.json",
        "validation_report": f"{slug}_validation_report.json",
        "final_report": f"{slug}_final_report.md",
        "source_of_truth": f"{slug}_source_of_truth.json",
    }
    for key, fname in mapping.items():
        path = RESULTS_DIR / fname
        if path.exists():
            files[key] = path.read_text()

    debug_log_path = LOG_DIR / f"debug_{slug}.log"
    if debug_log_path.exists():
        files["debug_log"] = debug_log_path.read_text(errors="replace")

    compare_candidates: list[tuple[int, Path]] = []
    for candidate in RESULTS_DIR.glob("*[Cc]ompare*"):
        cname = candidate.stem.lower().replace("_compare", "").replace("compare", "")
        slug_lower = slug.lower()
        slug_words = [w for w in slug_lower.split("_") if len(w) > 2]

        if slug_lower in cname or cname in slug_lower:
            compare_candidates.append((100, candidate))
        elif slug_words and slug_words[0] in cname:
            overlap = sum(1 for w in slug_words if w in cname)
            compare_candidates.append((overlap * 10, candidate))

    if compare_candidates:
        compare_candidates.sort(key=lambda x: (x[0], x[1].stat().st_size), reverse=True)
        files["compare"] = compare_candidates[0][1].read_text()

    return files


def _run_pipeline(run_id: str, condition: str) -> None:
    """Execute the 4-agent pipeline in a background thread."""
    run_log_handler = None
    try:
        _runs[run_id]["status"] = "running"
        _runs[run_id]["step"] = "Loading config..."
        _log(run_id, "Starting pipeline for: " + condition)

        config = load_config("config.yaml")
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        paths = build_paths(condition)
        slug = list(paths.values())[0].split("/")[1].rsplit("_raw", 1)[0]
        _runs[run_id]["slug"] = slug

        # Per-run debug log — path comes from build_paths so naming is consistent
        # with the CLI. Written in overwrite mode so each run starts fresh.
        run_log_path = ROOT / paths["debug_log"]
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_handler = logging.FileHandler(run_log_path, mode="w", encoding="utf-8")
        run_log_handler.setLevel(logging.DEBUG)
        run_log_handler.setFormatter(_log_formatter)
        logging.getLogger().addHandler(run_log_handler)
        _runs[run_id]["debug_log"] = str(run_log_path.relative_to(ROOT))
        logger.info("Per-run debug log → %s", run_log_path)

        try:
            import tools.search_tool as _st
            _st.tavily_exhausted = False
            _st.tavily_exhausted_msg = ""
        except ImportError:
            pass

        try:
            from tools.scraper.browser import reset_circuit_breaker
            reset_circuit_breaker()
        except ImportError:
            pass

        _runs[run_id]["step"] = "Step 1/4 — Data Collector + Policy Researcher"
        _log(run_id, "Step 1/4: Data Collector + Policy Researcher (parallel)")
        from concurrent.futures import ThreadPoolExecutor
        from agents.data_collector import collect_data
        from agents.policy_researcher import research_policies

        with ThreadPoolExecutor(max_workers=2) as pool:
            data_future = pool.submit(collect_data, condition, config, paths)
            policy_future = pool.submit(research_policies, condition, config)

        stats = data_future.result()
        policies = policy_future.result()
        source = stats.get("source", "unknown")
        policy_count = len(policies.get("policies", []))
        _log(run_id, f"  Source: {source} | {stats['total']} values collected")
        _log(run_id, f"  Policy Researcher: {policy_count} rules (authority: {policies.get('numbering_authority', 'unknown')})")

        try:
            from tools.search_tool import tavily_exhausted as _tav_flag, tavily_exhausted_msg as _tav_msg
            if _tav_flag:
                _log(run_id, "  *** TAVILY QUOTA EXHAUSTED — Step 1 used DuckDuckGo fallback ***")
                _log(run_id, "  Pipeline paused. Click 'Continue' to proceed with collected data, or stop and top up your Tavily credits.")
                _runs[run_id]["status"] = "paused_tavily"
                _runs[run_id]["step"] = "⏸ Paused — Tavily quota exhausted after Step 1"
                _runs[run_id]["warning"] = (
                    "Tavily quota exhausted — Step 1 used DuckDuckGo fallback. "
                    "Click Continue to proceed with collected data, or stop to top up credits first."
                )
                evt = threading.Event()
                _resume_events[run_id] = evt
                evt.wait()          # blocks until /api/resume is called
                del _resume_events[run_id]
                _runs[run_id]["status"] = "running"
                _runs[run_id]["warning"] = ""
                _log(run_id, "  Resuming pipeline (Steps 2–4)...")
        except ImportError:
            pass

        policy_confidence = policies.get("policy_confidence", "high")
        sot = {
            "data_source": source,
            "total_values": stats.get("total", 0),
            "train_count": stats.get("train", 0),
            "test_count": stats.get("test", 0),
            "policy_confidence": policy_confidence,
            "policies": policies.get("policies", []),
            "numbering_authority": policies.get("numbering_authority", ""),
            "vendor_patterns": policies.get("vendor_patterns", []),
            "sources": policies.get("sources", []),
            "vendor_keywords": policies.get("vendor_keywords", []),
        }
        if policy_confidence != "high":
            sot["policy_confidence_reason"] = policies.get(
                "policy_confidence_reason",
                "Relevance gate flagged one or both research phases.",
            )
            _log(run_id, f"  ⚠ Policy confidence: {policy_confidence} — "
                         f"{sot['policy_confidence_reason']}")
        sot_path = RESULTS_DIR / f"{slug}_source_of_truth.json"
        sot_path.write_text(json.dumps(sot, indent=2))
        _log(run_id, f"  Source-of-truth saved → {sot_path.name}")

        if stats.get("total", 0) == 0:
            _log(run_id, f"  ⚠ No example data in DB (source: {source}). "
                         "Continuing with policies only.")

        _runs[run_id]["step"] = "Step 2/4 — Pattern Analyzer"
        _log(run_id, "Step 2/4: Pattern Analyzer")
        from agents.pattern_analyzer import analyze_patterns
        patterns = analyze_patterns(condition, config, paths, policies=policies)
        _log(run_id, f"  Found {len(patterns.get('format_rules', []))} rules")

        _runs[run_id]["step"] = "Step 3/4 — Regex Generator"
        _log(run_id, "Step 3/4: Regex Generator")
        from agents.regex_generator import generate_regex
        regex_result = generate_regex(config, paths, policies=policies)
        _log(run_id, f"  Generated {len(regex_result.get('patterns', []))} patterns")

        _runs[run_id]["step"] = "Step 4/4 — Validator"
        _log(run_id, "Step 4/4: Validator")
        from agents.validator import validate
        report = validate(config, paths, data_source=source, policies=policies)
        _log(run_id, f"  Overall: {report['overall_status']} (mode: {report.get('validation_mode', 'unknown')})")

        _runs[run_id]["status"] = "done"
        _runs[run_id]["step"] = f"Done — {report['overall_status']}"
        _runs[run_id]["overall"] = report["overall_status"]
        _log(run_id, "Pipeline complete.")

    except Exception as exc:
        _runs[run_id]["status"] = "error"
        _runs[run_id]["step"] = f"Error: {exc}"
        _log(run_id, f"ERROR: {exc}")
        logger.exception("Pipeline failed for run %s", run_id)
    finally:
        if run_log_handler:
            logging.getLogger().removeHandler(run_log_handler)
            run_log_handler.close()


def _log(run_id: str, msg: str) -> None:
    _logs.setdefault(run_id, []).append(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    runs = _list_past_runs()
    return render_template("index.html", runs=runs)


@app.route("/api/run", methods=["POST"])
def api_run():
    condition = (request.json or {}).get("condition", "").strip()
    if not condition:
        return jsonify({"error": "condition is required"}), 400

    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {"condition": condition, "status": "starting", "step": "Queued", "slug": ""}
    _logs[run_id] = []
    t = threading.Thread(target=_run_pipeline, args=(run_id, condition), daemon=True)
    t.start()
    return jsonify({"run_id": run_id})


@app.route("/api/status/<run_id>")
def api_status(run_id):
    run = _runs.get(run_id)
    if not run:
        return jsonify({"error": "not found"}), 404
    return jsonify(run)


@app.route("/api/resume/<run_id>", methods=["POST"])
def api_resume(run_id):
    evt = _resume_events.get(run_id)
    if not evt:
        return jsonify({"error": "run not paused or not found"}), 404
    evt.set()
    return jsonify({"ok": True})


@app.route("/api/logs/<run_id>")
def api_logs(run_id):
    logs = _logs.get(run_id, [])
    return jsonify({"logs": logs})


@app.route("/api/results/<slug>")
def api_results(slug):
    files = _get_run_files(slug)
    return jsonify(files)


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_past_runs())


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Multi-regex comparison with LLM-powered analysis."""
    import re
    data = request.json or {}
    slug = data.get("slug", "")
    user_value_regexes: list[str] = data.get("value_regexes", [])
    user_keyword_regex: str = data.get("keyword_regex", "")

    if data.get("value_regex"):
        user_value_regexes.append(data["value_regex"])

    if not slug:
        return jsonify({"error": "slug is required"}), 400

    regex_path = RESULTS_DIR / f"{slug}_regex_patterns.json"
    if not regex_path.exists():
        return jsonify({"error": f"No regex results found for {slug}"}), 404

    with open(regex_path) as f:
        sys_data = json.load(f)

    sys_patterns = sys_data.get("patterns", [])
    sys_value_patterns = [p for p in sys_patterns if p.get("type") == "value"]
    sys_keyword = next((p["regex"] for p in sys_patterns if p.get("type") == "context"), "")

    test_path = DATA_DIR / f"{slug}_test.txt"
    truth = []
    if test_path.exists():
        truth = [l.strip() for l in test_path.read_text().splitlines() if l.strip()]

    scored: list[dict] = []

    for sp in sys_value_patterns:
        scored.append(_score_regex(sp["regex"], f"System ({sp['rule_id']})", truth, re))

    for i, ur in enumerate(user_value_regexes):
        label = f"User #{i+1}" if len(user_value_regexes) > 1 else "User"
        scored.append(_score_regex(ur, label, truth, re))

    md = _build_multi_compare_md(slug, scored, sys_keyword, user_keyword_regex, truth)

    try:
        config = load_config("config.yaml")
        from utils.bedrock_client import invoke_claude
        llm_analysis = _llm_compare(slug, scored, sys_keyword, user_keyword_regex, config)
        if llm_analysis:
            md += "\n\n---\n\n## LLM Analysis (Claude Sonnet)\n\n" + llm_analysis
    except Exception as exc:
        logger.warning("LLM compare failed: %s", exc)
        md += f"\n\n> LLM analysis unavailable: {exc}\n"

    compare_path = RESULTS_DIR / f"{slug}_compare.md"
    compare_path.write_text(md)

    return jsonify({"compare_md": md, "scored": scored})


def _score_regex(regex_str: str, label: str, truth: list[str], re_mod) -> dict:
    """Test a single regex against the truth set and return metrics."""
    entry = {"label": label, "regex": regex_str}
    try:
        compiled = re_mod.compile(regex_str)
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


def _build_multi_compare_md(slug: str, scored: list[dict], sys_kw: str, user_kw: str, truth: list[str]) -> str:
    condition = slug.replace("_", " ")
    lines = [
        f"# Regex Comparison — {condition}",
        "",
        f"**Truth-set size:** {len(truth)} values",
        "",
        "## Value Regex Scores",
        "",
        "| # | Label | fullmatch | search | Errors |",
        "|---|-------|-----------|--------|--------|",
    ]
    for i, s in enumerate(scored, 1):
        err = s["error"] or ""
        lines.append(
            f"| {i} | **{s['label']}** | "
            f"{s['fullmatch']}/{s['total']} ({s['fullmatch_pct']}%) | "
            f"{s['search_match']}/{s['total']} ({s['search_pct']}%) | "
            f"{err} |"
        )
    lines.append("")

    lines.append("## Value Regex Patterns")
    lines.append("")
    for i, s in enumerate(scored, 1):
        lines.append(f"**{i}. {s['label']}**")
        lines.append("")
        lines.append("~~~~")
        lines.append(s["regex"])
        lines.append("~~~~")
        lines.append("")

    for s in scored:
        if s.get("missed_samples"):
            lines.append(f"**{s['label']}** missed (sample): {', '.join(f'`{v}`' for v in s['missed_samples'])}")
    lines.append("")

    if sys_kw or user_kw:
        lines.append("## Keyword / Proximity Regex")
        lines.append("")
        if sys_kw:
            lines.append("**System:**")
            lines.append("")
            lines.append("~~~~")
            lines.append(sys_kw)
            lines.append("~~~~")
            lines.append("")
        if user_kw:
            lines.append("**User:**")
            lines.append("")
            lines.append("~~~~")
            lines.append(user_kw)
            lines.append("~~~~")
            lines.append("")

    return "\n".join(lines)


def _llm_compare(slug: str, scored: list[dict], sys_kw: str, user_kw: str, config: dict) -> str:
    """Ask Claude to analyse the scored regexes and produce a condition-by-condition
    comparison table plus a final winner verdict."""
    from utils.bedrock_client import invoke_claude

    condition = slug.replace("_", " ")

    # Load the discovered format rules so the LLM knows exactly what conditions
    # the identifier is supposed to satisfy.
    known_rules: list[str] = []
    range_restrictions: list[str] = []
    patterns_path = RESULTS_DIR / f"{slug}_patterns.json"
    if patterns_path.exists():
        try:
            pd = json.loads(patterns_path.read_text())
            for r in pd.get("format_rules", []):
                known_rules.append(f"[{r.get('rule_id','')}] {r.get('description','')}")
            for rr in pd.get("range_restrictions", []):
                range_restrictions.append(
                    f"position {rr.get('position','')} → {rr.get('allowed_values','')} ({rr.get('source','')})"
                )
        except Exception:
            pass

    rules_block = "\n".join(f"  - {r}" for r in known_rules) if known_rules else "  (not available)"
    ranges_block = "\n".join(f"  - {r}" for r in range_restrictions) if range_restrictions else "  (none)"

    regex_block_lines = []
    for s in scored:
        regex_block_lines.append(
            f"  • {s['label']}\n"
            f"    regex  : `{s['regex']}`\n"
            f"    results: fullmatch {s['fullmatch_pct']}% ({s['fullmatch']}/{s['total']}), "
            f"search {s['search_pct']}% ({s['search_match']}/{s['total']})\n"
            f"    missed : {s.get('missed_samples') or 'none'}"
        )
    regex_block = "\n\n".join(regex_block_lines)

    labels = [s["label"] for s in scored]
    col_header = " | ".join(labels)
    col_sep    = " | ".join(["---"] * len(labels))

    prompt = f"""\
You are a senior regex accuracy analyst.

IDENTIFIER TYPE: {condition}
TRUTH-SET SIZE : {scored[0]['total'] if scored else 0} known-valid values

─── KNOWN FORMAT RULES (discovered by the system) ───────────────────────────
{rules_block}

─── RANGE / POSITION RESTRICTIONS ──────────────────────────────────────────
{ranges_block}

─── REGEXES UNDER COMPARISON ────────────────────────────────────────────────
{regex_block}

System keyword regex : `{sys_kw or 'N/A'}`
User keyword regex   : `{user_kw or 'N/A'}`

─── YOUR TASK ────────────────────────────────────────────────────────────────
Produce a concise Markdown analysis with EXACTLY the following sections.
Keep the TOTAL response under 350 words. Be precise, no filler sentences.

---

## Condition Comparison

For EVERY known format rule and range restriction above (plus any you can
infer from the regexes), one row per condition:

| Condition | {col_header} | Winner |
|-----------|{col_sep}|--------|
| <condition> | ✅ / ❌ / ⚠️ | ... | **Label** or Tie |

Symbols: ✅ fully enforced · ❌ not enforced · ⚠️ partially enforced

Cover at minimum (where applicable):
length · character type · segment structure · per-position restrictions ·
optional parts · word boundaries · check-digit encoding

---

## Scores

| Metric | {col_header} |
|--------|{col_sep}|
| fullmatch % | ... |
| search % | ... |
| Conditions ✅ | X/{len(known_rules) or '?'} | ... |

---

## Key Differences

3 bullet points maximum. Cite actual character classes / quantifiers.
For keyword regexes, note language coverage and missing/extra terms in one bullet.

---

## Winner

One sentence: which regex wins and the single most important reason why.
If neither is ideal, give an improved regex in a fenced code block (no explanation needed).
"""
    return invoke_claude(prompt, config, system="You are a regex accuracy analyst.")





@app.route("/api/delete/<slug>", methods=["DELETE"])
def api_delete(slug):
    """Delete all files associated with a run slug."""
    if not slug or "/" in slug or ".." in slug:
        return jsonify({"error": "invalid slug"}), 400

    deleted: list[str] = []
    for directory in (RESULTS_DIR, DATA_DIR, NOTES_DIR):
        if not directory.exists():
            continue
        for p in directory.glob(f"{slug}*"):
            try:
                p.unlink()
                deleted.append(str(p.relative_to(ROOT)))
            except Exception as exc:
                logger.warning("Failed to delete %s: %s", p, exc)

    logger.info("Deleted %d files for slug '%s'", len(deleted), slug)
    return jsonify({"deleted": deleted, "count": len(deleted)})


@app.route("/api/notes", methods=["GET", "POST"])
def api_notes():
    """Save/load user notes per slug."""
    NOTES_DIR.mkdir(exist_ok=True)
    if request.method == "POST":
        data = request.json or {}
        slug = data.get("slug", "default")
        text = data.get("text", "")
        (NOTES_DIR / f"{slug}.txt").write_text(text)
        return jsonify({"saved": True})
    else:
        slug = request.args.get("slug", "default")
        path = NOTES_DIR / f"{slug}.txt"
        text = path.read_text() if path.exists() else ""
        return jsonify({"text": text})


# ── Batch Re-run ──────────────────────────────────────────────────────────────

_batch_state: dict = {
    "running": False,
    "total": 0,
    "done": 0,
    "failed": 0,
    "current": "",
    "results": [],   # list of {slug, condition, status, error}
}


def _batch_rerun_worker(conditions: list[tuple[str, str]]) -> None:
    """Run all conditions sequentially in a background thread."""
    _batch_state["running"] = True
    _batch_state["total"] = len(conditions)
    _batch_state["done"] = 0
    _batch_state["failed"] = 0
    _batch_state["results"] = []

    for slug, condition in conditions:
        _batch_state["current"] = condition
        logger.info("[batch-rerun] starting '%s'", condition)
        entry: dict = {"slug": slug, "condition": condition, "status": "running", "error": None}
        _batch_state["results"].append(entry)

        # Reuse the existing _run_pipeline logic by fabricating a throw-away run_id.
        run_id = f"batch_{slug}"
        _runs[run_id] = {"condition": condition, "status": "starting", "step": "Queued", "slug": ""}
        _logs[run_id] = []
        try:
            _run_pipeline(run_id, condition)
            final_status = _runs[run_id].get("status", "unknown")
            if final_status == "done":
                entry["status"] = "done"
                logger.info("[batch-rerun] '%s' → DONE", condition)
            else:
                entry["status"] = "error"
                entry["error"] = _runs[run_id].get("step", "unknown error")
                _batch_state["failed"] += 1
                logger.warning("[batch-rerun] '%s' → ERROR: %s", condition, entry["error"])
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            _batch_state["failed"] += 1
            logger.error("[batch-rerun] '%s' crashed: %s", condition, exc)
        finally:
            _batch_state["done"] += 1
            # Clean up ephemeral run state to keep memory tidy.
            _runs.pop(run_id, None)
            _logs.pop(run_id, None)

    _batch_state["running"] = False
    _batch_state["current"] = ""
    logger.info(
        "[batch-rerun] all done — %d/%d succeeded, %d failed",
        _batch_state["done"] - _batch_state["failed"],
        _batch_state["total"],
        _batch_state["failed"],
    )


@app.route("/api/rerun-all", methods=["POST"])
def api_rerun_all():
    """Start a sequential batch re-run of all past runs."""
    if _batch_state["running"]:
        return jsonify({"error": "A batch re-run is already in progress"}), 409

    # Collect all existing conditions from final_report files.
    conditions: list[tuple[str, str]] = []
    for p in sorted(RESULTS_DIR.glob("*_final_report.md")):
        slug = p.stem.replace("_final_report", "")
        condition = slug.replace("_", " ")
        conditions.append((slug, condition))

    if not conditions:
        return jsonify({"error": "No past runs found"}), 404

    t = threading.Thread(target=_batch_rerun_worker, args=(conditions,), daemon=True)
    t.start()

    logger.info("[batch-rerun] kicked off %d conditions", len(conditions))
    return jsonify({"started": True, "total": len(conditions),
                    "conditions": [c for _, c in conditions]})


@app.route("/api/rerun-all/status")
def api_rerun_all_status():
    return jsonify(dict(_batch_state))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5001)))
    p.add_argument("--debug", action="store_true", default=False)
    args = p.parse_args()
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")
