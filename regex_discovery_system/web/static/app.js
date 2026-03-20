/* Regex Discovery System — Frontend */

let currentSlug = null;
let currentCondition = null;
let pollTimer = null;

/* ── Run Pipeline ─────────────────────────────────────────────── */

function reRunCurrent() {
  if (!currentCondition) return;
  reRun(currentCondition);
}

function reRun(condition) {
  // Scroll to top, fill the input, and trigger the pipeline.
  window.scrollTo({ top: 0, behavior: "smooth" });
  const input = document.getElementById("condition-input");
  input.value = condition;
  startRun();
}

async function startRun() {
  const input = document.getElementById("condition-input");
  const condition = input.value.trim();
  if (!condition) { input.focus(); return; }

  // Disable the Re-run button for the results section while running.
  const rerunBtn = document.getElementById("rerun-btn");
  if (rerunBtn) {
    rerunBtn.disabled = true;
    document.getElementById("rerun-btn-text").textContent = "Running...";
    document.getElementById("rerun-btn-spinner").classList.remove("hidden");
  }

  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  document.getElementById("run-btn-text").textContent = "Running...";
  document.getElementById("run-btn-spinner").classList.remove("hidden");

  showStatus();
  setStep("Starting pipeline...");
  setProgress(5);
  clearLogs();
  showWarning("");
  hideContinueBtn();

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ condition }),
    });
    const data = await resp.json();
    if (data.error) { setStep("Error: " + data.error); return; }
    pollStatus(data.run_id);
  } catch (e) {
    setStep("Network error: " + e.message);
    resetRunBtn();
  }
}

function pollStatus(runId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const [statusResp, logsResp] = await Promise.all([
        fetch(`/api/status/${runId}`),
        fetch(`/api/logs/${runId}`),
      ]);
      const status = await statusResp.json();
      const logs = await logsResp.json();

      setStep(status.step || "");
      updateLogs(logs.logs || []);
      showWarning(status.warning || "");

      const stepMap = {
        "Step 1/4": 25, "Step 2/4": 50, "Step 3/4": 75, "Step 4/4": 90,
      };
      for (const [k, v] of Object.entries(stepMap)) {
        if ((status.step || "").includes(k)) setProgress(v);
      }

      if (status.status === "done") {
        clearInterval(pollTimer);
        setProgress(100);
        setStep("Done — " + (status.overall || ""));
        hideContinueBtn();
        resetRunBtn();
        if (status.slug) loadRun(status.slug);
        refreshHistory();
      } else if (status.status === "error") {
        clearInterval(pollTimer);
        setProgress(100);
        hideContinueBtn();
        resetRunBtn();
      } else if (status.status === "paused_tavily") {
        clearInterval(pollTimer);
        setProgress(25);
        showContinueBtn(runId);
      }
    } catch (e) {
      console.error("Poll error:", e);
    }
  }, 2000);
}

/* ── Load a past run ──────────────────────────────────────────── */

async function loadRun(slug) {
  currentSlug = slug;
  currentCondition = slug.replace(/_/g, " ");

  document.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".history-item").forEach(el => {
    if (el.onclick && el.onclick.toString().includes(slug)) el.classList.add("active");
  });

  const resp = await fetch(`/api/results/${slug}`);
  const files = await resp.json();

  const resultsSection = document.getElementById("results-section");
  const compareSection = document.getElementById("compare-section");
  const notesSection = document.getElementById("notes-section");

  resultsSection.classList.remove("hidden");
  compareSection.classList.remove("hidden");
  notesSection.classList.remove("hidden");

  document.getElementById("results-title").textContent =
    "Results — " + slug.replace(/_/g, " ");

  // Determine badge from validation report
  let overall = "";
  if (files.validation_report) {
    try {
      const vr = JSON.parse(files.validation_report);
      overall = vr.overall_status || "";
    } catch (_) {}
  }
  setBadge(overall);

  // Fill tabs
  fillTab("final_report", files.final_report || "", true);
  fillSourceOfTruth(files.source_of_truth || "");
  fillTab("patterns", files.patterns || "");
  fillRegexPatterns(files.regex_patterns || "");
  fillTab("validation_report", files.validation_report || "");

  if (files.compare) {
    fillTab("compare", files.compare, true);
  } else {
    document.getElementById("content-compare").innerHTML =
      '<p class="muted">No comparison yet. Use the Compare section below.</p>';
  }

  // Load notes
  const notesResp = await fetch(`/api/notes?slug=${slug}`);
  const notesData = await notesResp.json();
  document.getElementById("notes-textarea").value = notesData.text || "";
  document.getElementById("notes-saved").classList.add("hidden");

  // Switch to the best available tab: regex_patterns if no final_report
  if (files.final_report) {
    switchTab("final_report");
  } else if (files.regex_patterns) {
    switchTab("regex_patterns");
  } else if (files.source_of_truth) {
    switchTab("source_of_truth");
  } else {
    switchTab("final_report");
  }
}

function fillTab(name, content, isMarkdown = false) {
  const el = document.getElementById(`content-${name}`);
  if (!el) return;
  if (isMarkdown) {
    el.innerHTML = renderMarkdown(content);
  } else {
    if (content) {
      try {
        el.textContent = JSON.stringify(JSON.parse(content), null, 2);
      } catch (_) {
        el.textContent = content;
      }
    } else {
      el.textContent = "No data available.";
    }
  }
}

/* ── Source of Truth renderer ─────────────────────────────────── */

function fillSourceOfTruth(raw) {
  const el = document.getElementById("content-source_of_truth");
  if (!el) return;
  if (!raw) { el.innerHTML = '<p class="muted">No source-of-truth data available.</p>'; return; }

  let sot;
  try { sot = JSON.parse(raw); } catch (_) {
    el.textContent = raw;
    return;
  }

  const lines = [];
  lines.push("# Source of Truth");
  lines.push("");

  lines.push("## Data Collection");
  lines.push("");
  lines.push("| Property | Value |");
  lines.push("|----------|-------|");
  lines.push(`| **Data Source** | \`${sot.data_source || "unknown"}\` |`);
  lines.push(`| **Total Values** | ${sot.total_values || 0} |`);
  lines.push(`| **Train Set** | ${sot.train_count || 0} |`);
  lines.push(`| **Test Set** | ${sot.test_count || 0} |`);
  lines.push(`| **Numbering Authority** | ${sot.numbering_authority || "N/A"} |`);
  lines.push("");

  const sources = sot.sources || [];
  if (sources.length) {
    lines.push("## Scraped URLs");
    lines.push("");
    sources.forEach((url, i) => {
      lines.push(`${i + 1}. [${url}](${url})`);
    });
    lines.push("");
  }

  const policies = sot.policies || [];
  if (policies.length) {
    lines.push("## Policies / Rules Discovered");
    lines.push("");
    policies.forEach(p => {
      lines.push(`- ${p}`);
    });
    lines.push("");
  }

  const vp = sot.vendor_patterns || [];
  if (vp.length) {
    lines.push("## Vendor / Competitor DLP Patterns");
    lines.push("");
    vp.forEach(v => {
      if (typeof v === "object") {
        lines.push(`### ${v.vendor || "Unknown Vendor"}`);
        if (v.regex) lines.push(`- **Regex:** \`${v.regex}\``);
        if (v.keywords && v.keywords.length) lines.push(`- **Keywords:** ${v.keywords.join(", ")}`);
        if (v.rules && v.rules.length) {
          lines.push("- **Rules:**");
          v.rules.forEach(r => lines.push(`  - ${r}`));
        }
        lines.push("");
      } else {
        lines.push(`- ${v}`);
      }
    });
    lines.push("");
  }

  const vk = sot.vendor_keywords || [];
  if (vk.length) {
    lines.push("## Vendor Keywords");
    lines.push("");
    lines.push(vk.map(k => `\`${k}\``).join(", "));
    lines.push("");
  }

  el.innerHTML = renderMarkdown(lines.join("\n"));
}

/* ── Regex Patterns renderer ──────────────────────────────────── */

function fillRegexPatterns(raw) {
  const el = document.getElementById("content-regex_patterns");
  if (!el) return;
  if (!raw) { el.innerHTML = '<p class="muted">No regex patterns available.</p>'; return; }

  let data;
  try { data = JSON.parse(raw); } catch (_) {
    el.innerHTML = "<pre>" + raw.replace(/</g, "&lt;") + "</pre>";
    return;
  }

  const patterns = data.patterns || [];
  if (!patterns.length) {
    el.innerHTML = '<p class="muted">No patterns found.</p>';
    return;
  }

  const lines = [];
  lines.push(`# Regex Patterns — ${data.condition || ""}`);
  lines.push("");

  patterns.forEach((p, i) => {
    const typeLabel = p.type === "context" ? "🔑 context" : "🔍 value";
    lines.push(`## ${i + 1}. \`${p.rule_id}\` — ${typeLabel}`);
    lines.push("");
    lines.push(`**Description:** ${p.description}`);
    lines.push("");
    lines.push("~~~~");
    lines.push(p.regex);
    lines.push("~~~~");

    if (p.keywords_readable) {
      lines.push("");
      lines.push(`**Keywords (readable):** ${p.keywords_readable}`);
    }

    lines.push("");
  });

  el.innerHTML = renderMarkdown(lines.join("\n"));
}

/* ── Tabs ─────────────────────────────────────────────────────── */

document.addEventListener("click", (e) => {
  if (e.target.classList.contains("tab")) {
    switchTab(e.target.dataset.tab);
  }
});

function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `panel-${name}`));
}

/* ── Compare (multi-regex + LLM) ─────────────────────────────── */

function addRegexInput() {
  const container = document.getElementById("regex-inputs");
  const row = document.createElement("div");
  row.className = "regex-input-row";
  row.innerHTML = `
    <input type="text" class="input-field mono regex-value-input"
           placeholder="e.g. ^LV-\\d{4}$">
    <button class="btn-remove-regex" onclick="removeRegexInput(this)" title="Remove">&times;</button>
  `;
  container.appendChild(row);
}

function removeRegexInput(btn) {
  const container = document.getElementById("regex-inputs");
  if (container.children.length > 1) {
    btn.parentElement.remove();
  }
}

async function runCompare() {
  if (!currentSlug) return;

  const inputs = document.querySelectorAll(".regex-value-input");
  const valueRegexes = [];
  inputs.forEach(el => {
    const v = el.value.trim();
    if (v) valueRegexes.push(v);
  });
  const keywordRegex = document.getElementById("user-keyword-regex").value.trim();

  if (!valueRegexes.length && !keywordRegex) return;

  const btn = document.getElementById("compare-btn");
  btn.disabled = true;
  document.getElementById("compare-btn-text").textContent = "Analyzing...";
  document.getElementById("compare-btn-spinner").classList.remove("hidden");

  try {
    const resp = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slug: currentSlug,
        value_regexes: valueRegexes,
        keyword_regex: keywordRegex,
      }),
    });
    const data = await resp.json();
    const resultDiv = document.getElementById("compare-result");
    resultDiv.classList.remove("hidden");

    if (data.compare_md) {
      document.getElementById("compare-result-content").innerHTML = renderMarkdown(data.compare_md);
      fillTab("compare", data.compare_md, true);
    } else if (data.error) {
      document.getElementById("compare-result-content").innerHTML =
        `<p style="color:var(--red)">Error: ${data.error}</p>`;
    }
  } catch (e) {
    document.getElementById("compare-result-content").innerHTML =
      `<p style="color:var(--red)">Network error: ${e.message}</p>`;
  } finally {
    btn.disabled = false;
    document.getElementById("compare-btn-text").textContent = "Compare with LLM Analysis";
    document.getElementById("compare-btn-spinner").classList.add("hidden");
  }
}

/* ── Batch Re-run All ─────────────────────────────────────────── */

let batchPollTimer = null;

async function startRerunAll() {
  if (!confirm(`Re-run the pipeline for ALL ${document.querySelectorAll("#history-list li").length} past conditions?\n\nThis will overwrite existing results with the latest logic. It may take a while.`)) return;

  const btn = document.getElementById("rerun-all-btn");
  btn.disabled = true;
  btn.textContent = "Running…";

  try {
    const resp = await fetch("/api/rerun-all", { method: "POST" });
    const data = await resp.json();
    if (data.error) {
      alert("Could not start batch re-run: " + data.error);
      btn.disabled = false;
      btn.textContent = "↻ Re-run All";
      return;
    }
    showBatchPanel(data.total);
    pollBatchStatus();
  } catch (e) {
    alert("Network error: " + e.message);
    btn.disabled = false;
    btn.textContent = "↻ Re-run All";
  }
}

function showBatchPanel(total) {
  const panel = document.getElementById("batch-panel");
  panel.classList.remove("hidden");
  document.getElementById("batch-counter").textContent = `0 / ${total}`;
  document.getElementById("batch-progress-fill").style.width = "0%";
  document.getElementById("batch-results").innerHTML = "";
  document.getElementById("batch-current").textContent = "";
}

function pollBatchStatus() {
  if (batchPollTimer) clearInterval(batchPollTimer);
  batchPollTimer = setInterval(async () => {
    try {
      const resp = await fetch("/api/rerun-all/status");
      const s = await resp.json();

      const pct = s.total > 0 ? Math.round((s.done / s.total) * 100) : 0;
      document.getElementById("batch-progress-fill").style.width = pct + "%";
      document.getElementById("batch-counter").textContent = `${s.done} / ${s.total}`;
      document.getElementById("batch-current").textContent =
        s.current ? `Running: ${s.current}` : "";

      // Render per-item results
      const ul = document.getElementById("batch-results");
      ul.innerHTML = (s.results || []).map(r => {
        const icon = r.status === "done" ? "✅" : r.status === "error" ? "❌" : "⏳";
        const err = r.error ? `<span class="batch-err" title="${r.error}"> — ${r.error}</span>` : "";
        return `<li class="batch-result-row">${icon} ${r.condition}${err}</li>`;
      }).join("");

      if (!s.running) {
        clearInterval(batchPollTimer);
        document.getElementById("batch-label").textContent =
          `Done — ${s.total - s.failed} succeeded, ${s.failed} failed`;
        document.getElementById("batch-current").textContent = "";
        const btn = document.getElementById("rerun-all-btn");
        btn.disabled = false;
        btn.textContent = "↻ Re-run All";
        // Refresh the history list so mtimes are updated
        refreshHistory();
      }
    } catch (e) {
      console.error("Batch poll error:", e);
    }
  }, 3000);
}

/* ── Notes ────────────────────────────────────────────────────── */

async function saveNotes() {
  if (!currentSlug) return;
  const text = document.getElementById("notes-textarea").value;
  await fetch("/api/notes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug: currentSlug, text }),
  });
  const saved = document.getElementById("notes-saved");
  saved.classList.remove("hidden");
  setTimeout(() => saved.classList.add("hidden"), 2000);
}

/* ── History ──────────────────────────────────────────────────── */

function _histBadge(overall) {
  if (!overall) return "";
  if (overall === "PASS")
    return `<span class="hist-badge hist-badge-pass">✅ PASS</span>`;
  if (overall === "FAIL")
    return `<span class="hist-badge hist-badge-fail">❌ FAIL</span>`;
  return `<span class="hist-badge hist-badge-review">⚠️ ${overall}</span>`;
}

async function refreshHistory() {
  const resp = await fetch("/api/runs");
  const runs = await resp.json();
  const list = document.getElementById("history-list");
  list.innerHTML = runs.map(r => `
    <li class="history-row">
      <button class="history-item ${r.slug === currentSlug ? 'active' : ''}"
              onclick="loadRun('${r.slug}')">
        <span class="history-name">${_histBadge(r.overall)}${r.condition}</span>
        <span class="history-date">${r.mtime}</span>
      </button>
      <div class="history-actions">
        <button class="btn-rerun-sm" onclick="reRun('${r.condition}')" title="Re-run">↻</button>
        <button class="btn-delete" onclick="deleteRun('${r.slug}')" title="Delete run">&times;</button>
      </div>
    </li>
  `).join("");
}

async function deleteRun(slug) {
  if (!confirm(`Delete all files for "${slug.replace(/_/g, ' ')}"?`)) return;
  try {
    const resp = await fetch(`/api/delete/${slug}`, { method: "DELETE" });
    const data = await resp.json();
    if (data.error) { alert("Delete failed: " + data.error); return; }

    if (currentSlug === slug) {
      currentSlug = null;
      document.getElementById("results-section").classList.add("hidden");
      document.getElementById("compare-section").classList.add("hidden");
      document.getElementById("notes-section").classList.add("hidden");
    }
    refreshHistory();
  } catch (e) {
    alert("Delete failed: " + e.message);
  }
}

/* ── Helpers ──────────────────────────────────────────────────── */

function showContinueBtn(runId) {
  let btn = document.getElementById("continue-btn");
  if (!btn) {
    btn = document.createElement("button");
    btn.id = "continue-btn";
    btn.className = "btn btn-primary";
    btn.style.cssText = "margin-top:10px;width:100%;";
    const statusBar = document.getElementById("status-bar");
    statusBar.appendChild(btn);
  }
  btn.textContent = "▶ Continue Pipeline (Steps 2–4)";
  btn.onclick = () => resumeRun(runId);
  btn.style.display = "block";
}

function hideContinueBtn() {
  const btn = document.getElementById("continue-btn");
  if (btn) btn.style.display = "none";
}

async function resumeRun(runId) {
  const btn = document.getElementById("continue-btn");
  if (btn) { btn.disabled = true; btn.textContent = "Resuming…"; }
  showWarning("");
  try {
    await fetch(`/api/resume/${runId}`, { method: "POST" });
    pollStatus(runId);
  } catch (e) {
    setStep("Resume error: " + e.message);
    if (btn) { btn.disabled = false; btn.textContent = "▶ Continue Pipeline (Steps 2–4)"; }
  }
}

function showWarning(msg) {
  let banner = document.getElementById("warning-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "warning-banner";
    banner.style.cssText = "background:#b91c1c;color:#fff;padding:10px 16px;border-radius:6px;margin-bottom:10px;font-weight:600;display:none;text-align:center;";
    const statusBar = document.getElementById("status-bar");
    statusBar.parentElement.insertBefore(banner, statusBar);
  }
  if (msg) {
    banner.textContent = msg;
    banner.style.display = "block";
  } else {
    banner.style.display = "none";
  }
}

function showStatus() { document.getElementById("status-bar").classList.remove("hidden"); }
function setStep(s) { document.getElementById("status-step").textContent = s; }
function setProgress(pct) { document.getElementById("progress-fill").style.width = pct + "%"; }
function clearLogs() { document.getElementById("log-box").textContent = ""; }
function updateLogs(logs) {
  const box = document.getElementById("log-box");
  box.textContent = logs.join("\n");
  box.scrollTop = box.scrollHeight;
}

function resetRunBtn() {
  const btn = document.getElementById("run-btn");
  btn.disabled = false;
  document.getElementById("run-btn-text").textContent = "Run Pipeline";
  document.getElementById("run-btn-spinner").classList.add("hidden");

  const rerunBtn = document.getElementById("rerun-btn");
  if (rerunBtn) {
    rerunBtn.disabled = false;
    document.getElementById("rerun-btn-text").textContent = "↻ Re-run";
    document.getElementById("rerun-btn-spinner").classList.add("hidden");
  }
}

function setBadge(status) {
  const badge = document.getElementById("results-badge");
  badge.textContent = status;
  badge.className = "badge";
  if (status === "PASS") badge.classList.add("badge-pass");
  else if (status === "FAIL") badge.classList.add("badge-fail");
  else badge.classList.add("badge-review");
}

/* Markdown → HTML using marked.js library */
function renderMarkdown(md) {
  if (!md) return "";
  if (typeof marked !== "undefined") {
    marked.setOptions({
      gfm: true,
      breaks: false,
      tables: true,
    });
    return marked.parse(md);
  }
  return "<pre>" + md.replace(/</g, "&lt;") + "</pre>";
}

/* Enter key triggers run */
document.getElementById("condition-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") startRun();
});
