/* Regex Discovery System — Frontend */

let currentSlug = null;
let pollTimer = null;

/* ── Run Pipeline ─────────────────────────────────────────────── */

async function startRun() {
  const input = document.getElementById("condition-input");
  const condition = input.value.trim();
  if (!condition) { input.focus(); return; }

  const btn = document.getElementById("run-btn");
  btn.disabled = true;
  document.getElementById("run-btn-text").textContent = "Running...";
  document.getElementById("run-btn-spinner").classList.remove("hidden");

  showStatus();
  setStep("Starting pipeline...");
  setProgress(5);
  clearLogs();

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
        resetRunBtn();
        if (status.slug) loadRun(status.slug);
        refreshHistory();
      } else if (status.status === "error") {
        clearInterval(pollTimer);
        setProgress(100);
        resetRunBtn();
      }
    } catch (e) {
      console.error("Poll error:", e);
    }
  }, 2000);
}

/* ── Load a past run ──────────────────────────────────────────── */

async function loadRun(slug) {
  currentSlug = slug;

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
  fillTab("regex_patterns", files.regex_patterns || "");
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

  // Switch to first tab
  switchTab("final_report");
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

async function refreshHistory() {
  const resp = await fetch("/api/runs");
  const runs = await resp.json();
  const list = document.getElementById("history-list");
  list.innerHTML = runs.map(r => `
    <li class="history-row">
      <button class="history-item ${r.slug === currentSlug ? 'active' : ''}"
              onclick="loadRun('${r.slug}')">
        <span class="history-name">${r.condition}</span>
        <span class="history-date">${r.mtime}</span>
      </button>
      <button class="btn-delete" onclick="deleteRun('${r.slug}')" title="Delete run">&times;</button>
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
