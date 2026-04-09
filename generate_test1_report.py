#!/usr/bin/env python3
"""Read results1.json and produce test1_report.html — a comprehensive
dark-themed report showing sources, policies, competitor patterns,
individual/combined regexes, and keyword proximity for each identifier."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT, "results1.json")
DEFAULT_OUTPUT = os.path.join(ROOT, "test1_report.html")


def esc(s):
    return html.escape(str(s))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _render_summary_table(entries: list[dict]) -> str:
    rows = ""
    for i, e in enumerate(entries, 1):
        ident = e["identifier"]
        status = e["status"]
        sot = e.get("source_of_truth") or {}
        rp = e.get("regex_patterns") or {}

        policy_count = len(sot.get("policies", []))
        individual_count = len(rp.get("individual", []))
        has_combined = "Yes" if rp.get("combined_value") else "No"
        has_kw = "Yes" if rp.get("keyword_proximity") else "No"
        confidence = sot.get("policy_confidence", "—")

        if status == "OK":
            status_cls = "status-ok"
        elif status == "FAIL":
            status_cls = "status-fail"
        else:
            status_cls = "status-warn"

        rows += f"""
        <tr>
          <td>{i}</td>
          <td class="ident"><a href="#detail-{i}">{esc(ident)}</a></td>
          <td><span class="{status_cls}">{esc(status)}</span></td>
          <td>{esc(confidence)}</td>
          <td>{policy_count}</td>
          <td>{individual_count}</td>
          <td>{esc(has_combined)}</td>
          <td>{esc(has_kw)}</td>
        </tr>"""

    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>#</th><th>Identifier</th><th>Status</th><th>Confidence</th>
          <th>Policies</th><th>Individual Regex</th><th>Combined</th><th>Keywords</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _render_sources(sources: list) -> str:
    if not sources:
        return '<p class="empty">No sources found.</p>'
    items = ""
    for url in sources:
        items += f'<li><a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(url)}</a></li>\n'
    return f"<ul class='source-list'>{items}</ul>"


def _render_policies(policies: list) -> str:
    if not policies:
        return '<p class="empty">No policies found.</p>'
    items = "".join(f"<li>{esc(p)}</li>\n" for p in policies)
    return f"<ul class='policy-list'>{items}</ul>"


def _render_vendor_patterns(vendor_patterns: list) -> str:
    if not vendor_patterns:
        return '<p class="empty">No competitor / vendor patterns found.</p>'
    rows = ""
    for vp in vendor_patterns:
        if isinstance(vp, dict):
            name = vp.get("name") or vp.get("vendor") or vp.get("source", "—")
            pattern = vp.get("pattern") or vp.get("regex") or json.dumps(vp, ensure_ascii=False)
            rows += f"""
            <tr>
              <td>{esc(name)}</td>
              <td><code>{esc(pattern)}</code></td>
            </tr>"""
        else:
            rows += f"""
            <tr>
              <td colspan="2"><code>{esc(str(vp))}</code></td>
            </tr>"""
    return f"""
    <table class="vendor-table">
      <thead><tr><th>Source / Vendor</th><th>Pattern</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _render_individual_regex(individual: list[dict]) -> str:
    if not individual:
        return '<p class="empty">No individual regex patterns.</p>'
    blocks = ""
    for p in individual:
        blocks += f"""
        <div class="regex-card">
          <div class="rule-id">{esc(p['rule_id'])}</div>
          <div class="rule-desc">{esc(p.get('description', ''))}</div>
          <code class="regex-code">{esc(p['regex'])}</code>
        </div>"""
    return blocks


def _render_combined(combined: dict | None) -> str:
    if not combined:
        return '<p class="empty">No combined-value regex.</p>'
    return f"""
    <div class="regex-card combined">
      <div class="rule-id">{esc(combined['rule_id'])}</div>
      <div class="rule-desc">{esc(combined.get('description', ''))}</div>
      <code class="regex-code">{esc(combined['regex'])}</code>
    </div>"""


def _render_keywords(kw: dict | None) -> str:
    if not kw:
        return '<p class="empty">No keyword proximity pattern.</p>'
    keywords_str = kw.get("keywords_readable", "")
    regex = kw.get("regex", "")
    kw_items = ""
    if keywords_str:
        for k in keywords_str.split(", "):
            kw_items += f'<span class="kw-tag">{esc(k.strip())}</span> '
    return f"""
    <div class="kw-block">
      <div class="kw-tags">{kw_items}</div>
      <code class="regex-code">{esc(regex)}</code>
    </div>"""


def _render_detail(i: int, entry: dict) -> str:
    ident = entry["identifier"]
    status = entry["status"]
    sot = entry.get("source_of_truth") or {}
    rp = entry.get("regex_patterns") or {}

    confidence = sot.get("policy_confidence", "—")
    conf_reason = sot.get("policy_confidence_reason", "")

    conf_badge = ""
    if confidence == "high":
        conf_badge = '<span class="badge badge-high">high</span>'
    elif confidence == "none":
        conf_badge = '<span class="badge badge-none">none</span>'
    else:
        conf_badge = f'<span class="badge badge-med">{esc(confidence)}</span>'

    conf_reason_html = ""
    if conf_reason:
        conf_reason_html = f'<p class="conf-reason">{esc(conf_reason)}</p>'

    return f"""
    <div class="detail-block" id="detail-{i}">
      <h3>
        <span class="idx">{i}.</span> {esc(ident)}
        {conf_badge}
      </h3>
      {conf_reason_html}

      <details open>
        <summary>Sources ({len(sot.get('sources', []))})</summary>
        {_render_sources(sot.get('sources', []))}
      </details>

      <details open>
        <summary>Policies ({len(sot.get('policies', []))})</summary>
        {_render_policies(sot.get('policies', []))}
      </details>

      <details>
        <summary>Competitor / Vendor Patterns ({len(sot.get('vendor_patterns', []))})</summary>
        {_render_vendor_patterns(sot.get('vendor_patterns', []))}
      </details>

      <details open>
        <summary>Individual Regex Patterns ({len(rp.get('individual', []))})</summary>
        {_render_individual_regex(rp.get('individual', []))}
      </details>

      <details open>
        <summary>Combined-Value Regex</summary>
        {_render_combined(rp.get('combined_value'))}
      </details>

      <details>
        <summary>Keyword Proximity</summary>
        {_render_keywords(rp.get('keyword_proximity'))}
      </details>
    </div>"""


def render_html(entries: list[dict]) -> str:
    ok_count = sum(1 for e in entries if e["status"] == "OK")
    fail_count = len(entries) - ok_count
    total_policies = sum(len((e.get("source_of_truth") or {}).get("policies", [])) for e in entries)
    total_regex = sum(len((e.get("regex_patterns") or {}).get("individual", [])) for e in entries)

    detail_html = ""
    for i, e in enumerate(entries, 1):
        detail_html += _render_detail(i, e)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Regex Discovery Report — test1 batch</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9;
    --accent: #58a6ff; --green: #27ae60; --red: #e74c3c; --orange: #f39c12;
    --yellow: #f1c40f; --muted: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.6;
  }}
  h1 {{ color: #fff; margin-bottom: 4px; }}
  h2 {{ color: var(--accent); margin: 32px 0 12px; }}
  h3 {{ color: #fff; margin-bottom: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  .summary-cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .scard {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px 24px; min-width: 160px;
  }}
  .scard .num {{ font-size: 2em; font-weight: 700; }}
  .scard .lbl {{ color: var(--muted); font-size: .85em; }}

  .summary-table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  .summary-table th, .summary-table td {{
    padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border);
  }}
  .summary-table th {{ background: var(--card); color: #fff; font-weight: 600; }}
  .ident {{ font-weight: 500; }}
  .status-ok {{ color: var(--green); font-weight: 700; }}
  .status-fail {{ color: var(--red); font-weight: 700; }}
  .status-warn {{ color: var(--orange); font-weight: 700; }}

  .detail-block {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 20px;
  }}
  .idx {{ color: var(--muted); font-weight: 400; }}

  .badge {{
    font-size: .7em; padding: 2px 8px; border-radius: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .5px;
  }}
  .badge-high {{ background: var(--green); color: #fff; }}
  .badge-med {{ background: var(--orange); color: #fff; }}
  .badge-none {{ background: var(--red); color: #fff; }}
  .conf-reason {{ color: var(--muted); font-size: .85em; margin-bottom: 12px; font-style: italic; }}

  details {{ margin: 10px 0; }}
  details summary {{
    cursor: pointer; color: var(--accent); font-weight: 600; font-size: .95em;
    padding: 4px 0;
  }}
  details summary:hover {{ text-decoration: underline; }}

  .empty {{ color: var(--muted); font-style: italic; font-size: .85em; margin: 6px 0; }}

  .source-list, .policy-list {{ margin: 8px 0 8px 20px; }}
  .source-list li, .policy-list li {{ margin-bottom: 4px; font-size: .9em; }}
  .source-list a {{ word-break: break-all; }}

  .vendor-table {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
  .vendor-table th, .vendor-table td {{
    padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); font-size: .85em;
  }}
  .vendor-table th {{ background: #0d1117; color: #fff; }}
  .vendor-table code {{ font-size: .82em; word-break: break-all; white-space: pre-wrap; }}

  .regex-card {{
    background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; margin: 8px 0; overflow-x: auto;
  }}
  .regex-card.combined {{ border-color: var(--accent); }}
  .rule-id {{ font-weight: 700; color: var(--accent); font-size: .85em; margin-bottom: 2px; }}
  .rule-desc {{ color: var(--muted); font-size: .82em; margin-bottom: 6px; }}
  .regex-code {{
    display: block; font-size: .82em; word-break: break-all; white-space: pre-wrap;
    color: #e6edf3; background: transparent; padding: 0;
  }}

  .kw-block {{ margin: 8px 0; }}
  .kw-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
  .kw-tag {{
    background: #21262d; border: 1px solid var(--border); border-radius: 12px;
    padding: 2px 10px; font-size: .8em; color: var(--text);
  }}

  footer {{
    margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
    color: #484f58; font-size: .8em;
  }}
</style>
</head>
<body>

<h1>Regex Discovery Report</h1>
<p class="subtitle">Batch results for {len(entries)} identifiers from test1.json</p>

<div class="summary-cards">
  <div class="scard"><div class="num">{len(entries)}</div><div class="lbl">Identifiers</div></div>
  <div class="scard"><div class="num" style="color:var(--green)">{ok_count}</div><div class="lbl">Succeeded</div></div>
  <div class="scard"><div class="num" style="color:var(--red)">{fail_count}</div><div class="lbl">Failed</div></div>
  <div class="scard"><div class="num">{total_policies}</div><div class="lbl">Total Policies</div></div>
  <div class="scard"><div class="num">{total_regex}</div><div class="lbl">Individual Regexes</div></div>
</div>

<h2>Summary</h2>
{_render_summary_table(entries)}

<h2>Detailed Results</h2>
{detail_html}

<footer>
  Generated by regex-automator &bull; {len(entries)} identifiers analysed
</footer>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from batch results JSON")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Results JSON file (default: results1.json)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output HTML file (default: test1_report.html)")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    output_html = os.path.abspath(args.output)

    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found. Run run_test1_batch.py first.")
        sys.exit(1)

    entries = load_json(input_file)
    html_out = render_html(entries)

    with open(output_html, "w") as f:
        f.write(html_out)

    print(f"Report saved to {output_html}  ({len(entries)} identifiers)")


if __name__ == "__main__":
    main()
