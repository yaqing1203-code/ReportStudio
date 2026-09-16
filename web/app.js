/* Report Studio frontend.
 *
 * Talks to the local FastAPI server (same origin — the server serves this file).
 * Three responsibilities:
 *   1. Chat: submit a topic, render the user + agent bubbles.
 *   2. Live status: open an SSE stream and show each pipeline stage as it happens
 *      ('Scraping…', 'Verifying…', 'Generating LaTeX…', 'Compiling…').
 *   3. Settings: load/save the API protocol, base URL, and keys via /api/settings.
 *
 * No framework, no build step — one file the pywebview shell loads directly.
 */
"use strict";

const $ = (id) => document.getElementById(id);
// Per-launch loopback session token, injected into the window URL by the shell
// (?token=...). The server rejects tokenless requests when a token is configured,
// so append it to every API call (fetch, SSE, download links all route via api()).
const SESSION_TOKEN = new URLSearchParams(location.search).get("token");
const api = (path) => {
  if (!SESSION_TOKEN) return path; // same origin
  return path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(SESSION_TOKEN);
};

let running = false;

// Files staged via the composer paperclip BEFORE (or during) a report's document
// session exists. Each: { key, file, status: "staged"|"uploading"|"uploaded"|"error" }.
// When a document panel mounts (or is already open), these are flushed to that
// session's upload endpoint and cleared from the tray.
let _stagedFiles = [];
// The session currently accepting uploads. Set when a document panel is mounted;
// staged files flush to it (and future picks upload straight through).
let _activeUploadSession = null;

// Live progress is now driven entirely by backend events (appended sequentially in
// appendStatusEvent), not a fixed client-side stage list — so no STAGES constant.

// ============================ history store ================================
// History is now SERVER-BACKED: the backend persists it to a JSON file on disk
// (user_data_dir()/history.json) so it survives app restarts. This module keeps an
// in-memory mirror, loaded from GET /api/history on startup, and re-fetches after
// changes. PDFs are stored on disk keyed by job id, so a completed history item's
// Download works across restarts via /api/report/{job_id}/pdf.
//
// The backend owns entry creation/updates for reports (it knows job + session ids);
// the frontend just reads the list and can Clear it.
let _historyCache = [];

async function fetchHistory() {
  try {
    const r = await fetch(api("/api/history"));
    if (!r.ok) return _historyCache;
    const body = await r.json();
    _historyCache = Array.isArray(body.items) ? body.items : [];
  } catch { /* server may not be up yet; keep what we have */ }
  return _historyCache;
}

async function refreshHistory() {
  await fetchHistory();
  renderHistory();
}

async function clearHistory() {
  try {
    await fetch(api("/api/history"), { method: "DELETE" });
  } catch { /* ignore */ }
  _historyCache = [];
  renderHistory();
}

function renderHistory() {
  const list = $("history-list");
  if (!list) return;
  const items = _historyCache;
  if (!items.length) {
    list.innerHTML = `<div class="history-empty">No history yet.</div>`;
    return;
  }
  list.innerHTML = "";
  for (const it of items) {
    const el = document.createElement("button");
    el.className = "history-item";
    el.dataset.id = it.id;
    const icon = it.kind === "report" ? "▤" : "💬";
    const when = new Date(it.ts).toLocaleString();

    // Reports carry a "kind" badge: "Brief" when no documents were uploaded (a quick
    // web-only summary), or a documents count when the user added their own files.
    let kindBadge = "";
    if (it.kind === "report") {
      const files = Array.isArray(it.files) ? it.files : [];
      const n = files.length || it.file_count || 0;
      if (n === 0) {
        kindBadge = `<span class="history-tag brief">Brief</span>`;
      } else {
        kindBadge = `<span class="history-tag docs">${n} file${n > 1 ? "s" : ""}</span>`;
      }
    }
    const statusTag = it.kind === "report"
      ? `<span class="history-status ${it.status || ""}">${historyStatusLabel(it)}</span>`
      : "";
    el.innerHTML = `
      <span class="history-icon">${icon}</span>
      <span class="history-body">
        <span class="history-title">${escapeHtml(it.title || "(untitled)")}</span>
        <span class="history-meta">${escapeHtml(when)} ${kindBadge} ${statusTag}</span>
      </span>`;
    el.addEventListener("click", () => openHistoryItem(it.id));

    // Hover popover of uploaded file names for document-backed reports. Attached to
    // the whole row; positioned next to it so the scrolling sidebar never clips it.
    const files = Array.isArray(it.files) ? it.files : [];
    if (it.kind === "report" && files.length) {
      el.addEventListener("mouseenter", () => showFileTooltip(el, it.title, files));
      el.addEventListener("mouseleave", hideFileTooltip);
      el.addEventListener("focus", () => showFileTooltip(el, it.title, files));
      el.addEventListener("blur", hideFileTooltip);
    }

    list.appendChild(el);
  }
}

// ---- hover popover listing a report's uploaded documents -------------------
function showFileTooltip(anchorEl, title, files) {
  const tip = $("file-tooltip");
  if (!tip) return;
  const rows = files.map((f) =>
    `<li><span class="ft-doc-icon">▤</span><span class="ft-name">${escapeHtml(f)}</span></li>`
  ).join("");
  tip.innerHTML =
    `<div class="ft-head">${files.length} document${files.length > 1 ? "s" : ""} in this report</div>` +
    `<ul class="ft-list">${rows}</ul>`;
  tip.classList.remove("hidden");
  tip.setAttribute("aria-hidden", "false");

  // Position to the right of the sidebar row, vertically centered on it, clamped to
  // the viewport. Falls back to the left side if there isn't room on the right.
  const r = anchorEl.getBoundingClientRect();
  const margin = 10;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = r.right + margin;
  if (left + tw > window.innerWidth - 8) left = r.left - tw - margin;
  if (left < 8) left = 8;
  let top = r.top + r.height / 2 - th / 2;
  top = Math.max(8, Math.min(top, window.innerHeight - th - 8));
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
}

function hideFileTooltip() {
  const tip = $("file-tooltip");
  if (!tip) return;
  tip.classList.add("hidden");
  tip.setAttribute("aria-hidden", "true");
}

function historyStatusLabel(it) {
  return { completed: "PDF", out_of_scope: "no data",
           error: "failed", running: "…" }[it.status] || "";
}

function openHistoryItem(id) {
  const it = _historyCache.find((x) => x.id === id);
  if (!it) return;
  clearWelcome();

  // A completed report keeps its PDF on disk keyed by job_id — offer Download directly.
  // It survives restarts (served from disk), so this works in a fresh session too.
  if (it.kind === "report" && it.status === "completed" && it.pdf_ready && it.job_id) {
    const bubble = addAgentMessage();
    bubble.classList.remove("agent-bubble");
    bubble.classList.add("report-card");
    const box = document.createElement("div");
    box.className = "result ok";
    box.innerHTML = `
      <div class="result-title">▤ ${escapeHtml(it.title)}</div>
      <p>Generated ${escapeHtml(new Date(it.ts).toLocaleString())}.</p>
      <a class="download-btn" href="${api(`/api/report/${it.job_id}/pdf`)}" download>⬇ Download PDF</a>
      <button class="rerun-btn" onclick="rerunTopic('${escapeAttr(it.title)}')">Re-run report</button>`;
    bubble.appendChild(box);
    scrollDown();
  } else {
    // Chat entry, or a report whose PDF is unavailable -> reload the prompt into the
    // composer so the user can resend / regenerate.
    $("topic").value = it.title || "";
    autogrow($("topic"));
    $("topic").focus();
  }
}

function rerunTopic(topic) {
  $("topic").value = topic;
  autogrow($("topic"));
  sendMessage(topic);
  $("topic").value = "";
  autogrow($("topic"));
}

function escapeAttr(s) {
  return String(s).replace(/'/g, "\\'").replace(/"/g, "&quot;");
}

// ============================ chat + run ===================================
async function sendMessage(message) {
  if (running) return;
  running = true;
  $("send").disabled = true;

  clearWelcome();
  addUserMessage(message);

  try {
    // Call the dual-mode endpoint. Chat stays chat; a report request starts the
    // unified brief-first flow (the backend creates a database session for it).
    const resp = await fetch(api("/api/message"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      throw new Error(extractErrorMessage(body, resp.status));
    }
    const result = await resp.json();

    if (result.type === "chat") {
      // Immediate chat response. The backend records chat turns in history itself.
      handleChatResponse(result.content);
      refreshHistory();
    } else if (result.type === "report") {
      // Unified flow: the brief runs now; when it finishes we mount the document
      // panel so the user can add material and get a comprehensive report.
      handleBriefReport(message, result.session_id, result.job_id);
    } else {
      throw new Error("Unknown response type from server");
    }
  } catch (e) {
    const errMsg = e.message || e.toString() || JSON.stringify(e);
    const card = addAgentMessage();
    card.innerHTML = `<div class="error-message">Could not process message: ${escapeHtml(errMsg)}</div>`;
    console.error("Message error:", e);
    running = false;
    $("send").disabled = false;
  }
}

// Robustly turn any error body into a readable string. Handles: FastAPI string
// `detail`, FastAPI validation `detail` (an ARRAY of {loc,msg,...} objects — the
// source of the "[object Object]" bug), a plain {error} field, or nothing at all.
function extractErrorMessage(body, status) {
  if (body && typeof body === "object") {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // Pydantic validation errors: join each into "loc: msg".
      const parts = detail.map((d) => {
        if (d && typeof d === "object") {
          const loc = Array.isArray(d.loc) ? d.loc.join(".") : (d.loc || "");
          return loc ? `${loc}: ${d.msg || ""}` : (d.msg || JSON.stringify(d));
        }
        return String(d);
      });
      return parts.join("; ");
    }
    if (typeof body.error === "string") return body.error;
    if (detail) return JSON.stringify(detail);
  }
  return `Server error ${status}`;
}

function handleChatResponse(content) {
  // Display chat response inside the agent bubble as rendered markdown.
  const card = addAgentMessage();
  card.classList.add("chat-response");  // add, don't overwrite bubble classes
  card.innerHTML = renderMarkdown(content);

  running = false;
  $("send").disabled = false;
  $("topic").focus();
}

// Self-contained Markdown -> HTML renderer.
//
// The app is a fully offline desktop bundle, so we deliberately avoid an external
// library (marked) that would need a CDN or a separate bundled file that must load
// before app.js. This covers the constructs an LLM chat reply actually uses:
// fenced/inline code, bold, italic, headings, links, and ordered/unordered lists.
//
// SECURITY: the input is escaped FIRST, so LLM output cannot inject live HTML/JS
// (marked.parse on raw text did not do this). Only our own generated tags survive.
function renderMarkdown(src) {
  if (src == null) return "";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));

  // 1. Pull fenced code blocks out first so their contents aren't touched.
  const codeBlocks = [];
  let text = String(src).replace(/```[ \t]*([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const i = codeBlocks.length;
    codeBlocks.push(`<pre class="code-block"><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `@@CODE${i}@@`;
  });

  // 2. Escape everything else.
  text = esc(text);

  // 3. Inline spans (operate on already-escaped text).
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // 4. Block-level pass, line by line, grouping list items and paragraphs.
  const lines = text.split("\n");
  const html = [];
  let listType = null;   // "ul" | "ol" | null
  const closeList = () => { if (listType) { html.push(`</${listType}>`); listType = null; } };

  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {            // heading
      closeList();
      const level = m[1].length;
      html.push(`<h${level}>${inline(m[2])}</h${level}>`);
    } else if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {      // unordered item
      if (listType !== "ul") { closeList(); html.push("<ul>"); listType = "ul"; }
      html.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {      // ordered item
      if (listType !== "ol") { closeList(); html.push("<ol>"); listType = "ol"; }
      html.push(`<li>${inline(m[1])}</li>`);
    } else if (line.trim() === "") {                         // blank -> paragraph break
      closeList();
    } else if (line.startsWith("@@CODE")) {              // code-block placeholder
      closeList();
      html.push(line);
    } else {                                                 // paragraph text
      closeList();
      html.push(`<p>${inline(line)}</p>`);
    }
  }
  closeList();

  // 5. Restore code blocks.
  let result = html.join("\n");
  result = result.replace(/@@CODE(\d+)@@/g, (_, i) => codeBlocks[Number(i)] || "");
  return result;
}

// ============================ composer attach (paperclip) ==================
// The paperclip lets users add documents right from the input area. Uploads attach
// to a report's document SESSION, which may not exist yet (before a report is run) —
// so files are STAGED as chips and flushed to the session as soon as one is active.
// If a document panel is already open, picks upload straight through to it.
function addStagedFiles(files) {
  for (const file of files) {
    _stagedFiles.push({
      key: `f${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      file,
      status: "staged",
    });
  }
  renderAttachTray();
  // If a session is already accepting uploads (a panel is open), flush immediately.
  if (_activeUploadSession) flushStagedFiles(_activeUploadSession);
}

function removeStagedFile(key) {
  _stagedFiles = _stagedFiles.filter((s) => s.key !== key);
  renderAttachTray();
}

function renderAttachTray() {
  const tray = $("attach-tray");
  const btn = $("attach-btn");
  if (!tray) return;
  if (!_stagedFiles.length) {
    tray.classList.add("hidden");
    tray.innerHTML = "";
    if (btn) btn.classList.remove("has-files");
    return;
  }
  tray.classList.remove("hidden");
  if (btn) btn.classList.add("has-files");
  tray.innerHTML = "";
  for (const s of _stagedFiles) {
    const chip = document.createElement("span");
    chip.className = "attach-chip" +
      (s.status === "uploaded" ? " uploaded" : s.status === "error" ? " errored" : "");
    const label = s.status === "uploading" ? "↑ " : (s.status === "error" ? "⚠ " : "📎 ");
    chip.innerHTML =
      `<span class="chip-name" title="${escapeHtml(s.file.name)}">${label}${escapeHtml(s.file.name)}</span>` +
      `<button class="chip-remove" title="Remove" data-key="${escapeHtml(s.key)}">✕</button>`;
    chip.querySelector(".chip-remove").addEventListener("click", () => removeStagedFile(s.key));
    tray.appendChild(chip);
  }
}

// Upload every still-staged file to the given session, then drop the succeeded ones
// from the tray. Called when a document panel mounts and after each new pick while a
// panel is open. Reuses the same endpoint the panel uses, so the panel's list and
// auto-comprehensive trigger stay authoritative.
async function flushStagedFiles(sessionId, topic) {
  const pending = _stagedFiles.filter((s) => s.status === "staged" || s.status === "error");
  if (!pending.length) return 0;
  let uploaded = 0;
  for (const s of pending) {
    s.status = "uploading";
    renderAttachTray();
    const form = new FormData();
    form.append("file", s.file, s.file.name);
    try {
      const resp = await fetch(api(`/api/database/${sessionId}/articles`), {
        method: "POST", body: form,
      });
      const body = await resp.json().catch(() => null);
      if (!resp.ok) {
        s.status = "error";
        s.error = extractErrorMessage(body, resp.status);
        renderAttachTray();
        continue;
      }
      s.status = "uploaded";
      uploaded += 1;
      // Reflect it in the open panel's list if present.
      if (body && body.articles) renderArticleList(sessionId, topic, body.articles);
    } catch (e) {
      s.status = "error";
      s.error = e.message || String(e);
      renderAttachTray();
    }
  }
  // Clear successfully-uploaded chips; leave errors visible for the user to retry/remove.
  _stagedFiles = _stagedFiles.filter((s) => s.status !== "uploaded");
  renderAttachTray();
  // If files landed and a session is active, (re)arm the auto-comprehensive debounce.
  if (uploaded > 0 && !running) scheduleAutoComprehensive(sessionId, topic);
  return uploaded;
}

// ============================ unified report flow ==========================
// Every report request runs the SAME flow, no mode toggle:
//   1. Brief report runs autonomously (web search + verification).
//   2. When it finishes, a document panel appears.
//   3. If the user adds documents, the comprehensive report is generated
//      AUTOMATICALLY (after a short debounce) — or immediately via the button.

const DB_ACCEPT = ".pdf,.docx,.xlsx,.csv,.txt,.md,.markdown";
const AUTO_COMPREHENSIVE_MS = 3000;   // debounce after the last upload before auto-run

// Step 1: stream the brief. On completion, mount the document panel.
function handleBriefReport(topic, sessionId, jobId) {
  // A new report is starting: staged files should wait for THIS report's panel, not
  // flush into a previous session. The panel mount below re-activates uploads.
  _activeUploadSession = null;
  const card = addAgentMessage();
  card.classList.remove("agent-bubble");
  card.classList.add("report-card");
  const intro = document.createElement("div");
  intro.className = "db-phase-label";
  intro.textContent = "Step 1 · Brief report";
  card.appendChild(intro);
  const statusEl = renderStatusCard(card);

  // History is server-owned; reflect the new 'running' entry in the sidebar.
  refreshHistory();

  streamEvents(jobId, statusEl, card, null, {
    onComplete: (status) => {
      refreshHistory();
      if (status === "completed" || status === "out_of_scope") {
        mountDocumentPanel(sessionId, topic, status);
      }
    },
  });
}

// Step 2: the always-present document panel — upload/list/remove, with automatic
// comprehensive generation once documents are added.
function mountDocumentPanel(sessionId, topic, briefStatus) {
  const card = addAgentMessage();
  card.classList.remove("agent-bubble");
  card.classList.add("report-card", "db-panel");
  card.dataset.session = sessionId;

  const briefNote = briefStatus === "out_of_scope"
    ? `<p class="db-hint">The brief found little authoritative material on its own —
       adding your own documents below is the best way to get a useful report.</p>`
    : `<p class="db-hint">Add your own articles or documents to deepen this report.
       As soon as you add one, a comprehensive report is generated automatically —
       combining your material with the brief. Or leave it as-is.</p>`;

  card.innerHTML = `
    <div class="db-phase-label">Step 2 · Add your documents (optional)</div>
    ${briefNote}
    <div class="db-drop" id="db-drop-${sessionId}">
      <input type="file" id="db-file-${sessionId}" multiple accept="${DB_ACCEPT}" hidden />
      <button class="db-upload-btn" id="db-upload-${sessionId}">Choose files…</button>
      <span class="db-drop-hint">or drag &amp; drop — PDF, Word, Excel, CSV, TXT, Markdown</span>
    </div>
    <div class="db-article-list" id="db-list-${sessionId}">
      <div class="db-empty">No documents added yet.</div>
    </div>
    <div class="db-actions">
      <span class="db-status" id="db-status-${sessionId}"></span>
      <button class="primary-btn db-generate" id="db-generate-${sessionId}" disabled>
        Generate comprehensive report now
      </button>
    </div>`;
  scrollDown();

  const fileInput = $(`db-file-${sessionId}`);
  const drop = $(`db-drop-${sessionId}`);
  $(`db-upload-${sessionId}`).addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length) {
      uploadArticles(sessionId, topic, Array.from(fileInput.files));
      fileInput.value = "";
    }
  });
  // Drag & drop.
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (e) => {
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) uploadArticles(sessionId, topic, Array.from(files));
  });

  $(`db-generate-${sessionId}`).addEventListener("click", () => {
    cancelAutoComprehensive(sessionId);
    generateComprehensive(sessionId, topic);
  });

  // This session now accepts uploads: route composer-paperclip picks straight to it,
  // and flush anything the user staged before/while the brief was running.
  _activeUploadSession = { sessionId, topic };
  flushStagedFiles(sessionId, topic);

  // Free the composer; the panel drives the rest.
  running = false;
  $("send").disabled = false;
}

// Per-session debounce timers for auto-generation.
const _autoTimers = {};

function cancelAutoComprehensive(sessionId) {
  if (_autoTimers[sessionId]) {
    clearTimeout(_autoTimers[sessionId]);
    delete _autoTimers[sessionId];
  }
}

function scheduleAutoComprehensive(sessionId, topic) {
  cancelAutoComprehensive(sessionId);
  const statusEl = $(`db-status-${sessionId}`);
  let secs = Math.round(AUTO_COMPREHENSIVE_MS / 1000);
  if (statusEl) statusEl.textContent = `Auto-generating comprehensive report in ${secs}s…`;
  const tick = setInterval(() => {
    secs -= 1;
    if (statusEl && secs > 0) statusEl.textContent =
      `Auto-generating comprehensive report in ${secs}s…`;
  }, 1000);
  _autoTimers[sessionId] = setTimeout(() => {
    clearInterval(tick);
    delete _autoTimers[sessionId];
    if (!running) generateComprehensive(sessionId, topic);
  }, AUTO_COMPREHENSIVE_MS);
}

async function uploadArticles(sessionId, topic, files) {
  const statusEl = $(`db-status-${sessionId}`);
  cancelAutoComprehensive(sessionId);   // pause auto-run while more files arrive
  let added = 0;
  for (const file of files) {
    if (statusEl) statusEl.textContent = `Uploading ${file.name}…`;
    const form = new FormData();
    form.append("file", file, file.name);
    try {
      const resp = await fetch(api(`/api/database/${sessionId}/articles`), {
        method: "POST", body: form,
      });
      const body = await resp.json().catch(() => null);
      if (!resp.ok) {
        if (statusEl) statusEl.textContent = `${file.name}: ${extractErrorMessage(body, resp.status)}`;
        continue;
      }
      renderArticleList(sessionId, topic, body.articles);
      added += 1;
      if (statusEl) statusEl.textContent = "";
    } catch (e) {
      if (statusEl) statusEl.textContent = `Upload failed: ${e.message || e}`;
    }
  }
  // Auto-trigger the comprehensive report once at least one document was added.
  if (added > 0 && !running) scheduleAutoComprehensive(sessionId, topic);
}

function renderArticleList(sessionId, topic, articles) {
  const list = $(`db-list-${sessionId}`);
  if (!list) return;
  if (!articles || !articles.length) {
    list.innerHTML = `<div class="db-empty">No documents added yet.</div>`;
  } else {
    list.innerHTML = "";
    for (const a of articles) {
      const row = document.createElement("div");
      row.className = "db-article";
      const kb = Math.max(1, Math.round((a.chars || 0) / 1024));
      row.innerHTML = `
        <span class="db-article-icon">▤</span>
        <span class="db-article-body">
          <span class="db-article-title">${escapeHtml(a.title || a.filename)}</span>
          <span class="db-article-meta">${escapeHtml(a.filename)} · ~${kb}k chars</span>
        </span>
        <button class="db-article-remove" title="Remove" data-id="${escapeHtml(a.id)}">✕</button>`;
      row.querySelector(".db-article-remove").addEventListener("click", () =>
        removeArticle(sessionId, topic, a.id));
      list.appendChild(row);
    }
  }
  // Enable the manual generate button only when there is at least one document.
  const gen = $(`db-generate-${sessionId}`);
  if (gen) gen.disabled = !(articles && articles.length);
  scrollDown();
}

async function removeArticle(sessionId, topic, articleId) {
  try {
    const resp = await fetch(api(`/api/database/${sessionId}/articles/${articleId}`), {
      method: "DELETE",
    });
    const body = await resp.json().catch(() => null);
    if (resp.ok) {
      renderArticleList(sessionId, topic, body.articles);
      // If no documents remain, cancel any pending auto-run.
      if (!body.articles || !body.articles.length) cancelAutoComprehensive(sessionId);
    }
  } catch (e) {
    console.error("Remove article failed:", e);
  }
}

// Step 3: run the comprehensive report over brief sources + user documents.
async function generateComprehensive(sessionId, topic) {
  if (running) return;
  cancelAutoComprehensive(sessionId);
  const gen = $(`db-generate-${sessionId}`);
  const statusEl = $(`db-status-${sessionId}`);
  running = true;
  if (gen) gen.disabled = true;
  $("send").disabled = true;
  if (statusEl) statusEl.textContent = "Starting comprehensive report…";

  try {
    const resp = await fetch(api(`/api/database/${sessionId}/comprehensive`), {
      method: "POST",
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      throw new Error(extractErrorMessage(body, resp.status));
    }
    const { job_id } = await resp.json();
    if (statusEl) statusEl.textContent = "";

    const card = addAgentMessage();
    card.classList.remove("agent-bubble");
    card.classList.add("report-card");
    const intro = document.createElement("div");
    intro.className = "db-phase-label";
    intro.textContent = "Step 3 · Comprehensive report";
    card.appendChild(intro);
    const csEl = renderStatusCard(card);
    streamEvents(job_id, csEl, card, null, {
      onComplete: () => refreshHistory(),
    });
  } catch (e) {
    if (statusEl) statusEl.textContent = `Failed: ${e.message || e}`;
    if (gen) gen.disabled = false;
    running = false;
    $("send").disabled = false;
  }
}

function streamEvents(jobId, statusEl, card, _unused, opts) {
  const es = new EventSource(api(`/api/report/${jobId}/events`));
  const onComplete = opts && opts.onComplete;
  let done = false;

  // Inactivity watchdog: if NO event arrives for this long, assume the stream stalled
  // and clear the loading state rather than spinning forever. Reset on every event.
  const IDLE_MS = 90000;
  let watchdog = null;
  const armWatchdog = () => {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => {
      if (!done) {
        finalize("error",
          "No progress received for a while — the engine may be stuck or a provider "
          + "is not responding. Stopping. Check Settings (API keys) and try again.");
      }
    }, IDLE_MS);
  };

  const finalize = (status, detail, data) => {
    if (done) return;               // guarantee we only clear the UI once
    done = true;
    clearTimeout(watchdog);
    try { es.close(); } catch {}
    removeConfirmationCard(card);   // a pending HITL card is moot once the run ends
    running = false;
    $("send").disabled = false;
    if (status === "completed" && data && data.has_pdf) {
      finishCompleted(statusEl, card, jobId, data);
    } else if (status === "out_of_scope") {
      finishOutOfScope(statusEl, detail);
    } else {
      finishError(statusEl, detail || "The run did not complete.");
    }
    // History is server-owned (persisted on disk by the backend). Refresh the sidebar
    // to reflect the final status. The onComplete hook advances the unified flow
    // (mount the document panel after the brief, refresh history after comprehensive).
    refreshHistory();
    if (onComplete) {
      try { onComplete(status, data); } catch (e) { console.error(e); }
    }
  };

  armWatchdog();

  es.onmessage = (ev) => {
    armWatchdog();                  // fresh activity — reset the idle timer
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }

    // Heartbeat: the server sends these during long silent phases purely to keep the
    // connection warm. armWatchdog() above already counted it as activity — render
    // nothing and wait for the next real event.
    if (data.heartbeat) return;

    if (!data.final && data.stage) {
      // HITL checkpoint (workflow F): the pipeline is paused awaiting a human decision
      // on isolated claims. Show the review card; the run resumes (or aborts) once the
      // user clicks and POSTs /api/jobs/{id}/confirm.
      if (data.stage === "awaiting_confirmation") {
        appendStatusEvent(statusEl, data.stage, "Waiting for human confirmation…");
        showConfirmationCard(card, jobId, data.detail);
        return;
      }
      if (data.stage === "confirmation_resolved") {
        removeConfirmationCard(card);
      }
      appendStatusEvent(statusEl, data.stage, data.detail);
      return;
    }
    if (data.final) {
      finalize(data.status, data.detail, data);
    }
  };

  es.onerror = () => {
    // EventSource auto-retries transient drops; only surface an error if we never
    // finished AND the connection is truly closed.
    if (!done && es.readyState === EventSource.CLOSED) {
      finalize("error", "Lost connection to the local engine.");
    }
  };
}

// ============================ status card (sequential) =====================
// The card starts EMPTY. Each backend event is appended as its own line as it
// arrives — steps are NOT shown all at once. A "<stage>_start" event adds an
// in-progress line (spinner); the matching "<stage>_done" flips it to complete and
// shows the result detail. Unpaired progress events append as their own lines too.
function renderStatusCard(card) {
  const wrap = document.createElement("div");
  wrap.className = "status-card";
  card.appendChild(wrap);
  scrollDown();
  return wrap;
}

// Map a raw stage key to (base, phase). "scraping_start" -> ("scraping","start").
function _stagePhase(stage) {
  if (stage.endsWith("_start")) return [stage.slice(0, -6), "start"];
  if (stage.endsWith("_done")) return [stage.slice(0, -5), "done"];
  if (stage.endsWith("_progress")) return [stage.slice(0, -9), "progress"];
  return [stage, "info"];
}

function appendStatusEvent(statusEl, stage, detail) {
  const [base, phase] = _stagePhase(stage);
  // For a *_done / *_progress, update the existing line for this base stage if present.
  let row = statusEl.querySelector(`.stage[data-base="${base}"]`);
  if (!row) {
    row = document.createElement("div");
    row.className = "stage active";
    row.dataset.base = base;
    row.innerHTML = `<span class="dot"></span>` +
                    `<span class="stage-label"></span>` +
                    `<span class="stage-detail"></span>`;
    statusEl.appendChild(row);
  }
  const label = row.querySelector(".stage-label");
  const det = row.querySelector(".stage-detail");
  if (phase === "done") {
    row.classList.remove("active");
    row.classList.add("done");
    if (detail) det.textContent = detail;
    if (!label.textContent) label.textContent = _prettyStage(base);
  } else if (phase === "start") {
    row.classList.add("active");
    label.textContent = detail || _prettyStage(base);
  } else {
    // progress / info — keep active, update the detail line.
    row.classList.add("active");
    if (!label.textContent) label.textContent = _prettyStage(base);
    if (detail) det.textContent = detail;
  }
  scrollDown();
}

function _prettyStage(base) {
  return ({
    scraping: "Scraping sources",
    extracting: "Extracting claims",
    verifying: "Cross-verifying claims",
    analyzing: "Synthesizing analysis",
    generating_latex: "Generating LaTeX",
    compiling: "Compiling PDF",
    awaiting_confirmation: "Awaiting human confirmation",
    confirmation_resolved: "Isolated-claim review",
  })[base] || base.replace(/_/g, " ");
}

// ============================ HITL confirmation card =======================
// Rendered when the backend emits stage="awaiting_confirmation" (workflow F, gated
// by the hitl_on_isolated_core_claim setting). `detail` is a JSON string:
//   {"claims": [{"text": ..., "domain": ..., "credibility": ...}], "timeout_s": N}
// The card blocks nothing client-side — the pipeline waits server-side; clicking a
// button POSTs the decision and the SSE stream carries the run to its final event.
function showConfirmationCard(card, jobId, detail) {
  removeConfirmationCard(card);
  let summary = null;
  try { summary = JSON.parse(detail); } catch { summary = null; }
  const claims = summary && Array.isArray(summary.claims) ? summary.claims : [];
  const items = claims.map((c) => `
    <li class="hitl-claim">
      <span class="hitl-claim-text">${escapeHtml(c.text || "")}</span>
      <span class="hitl-claim-meta">source: ${escapeHtml(c.domain || "unknown")} ·
        credibility: ${escapeHtml(c.credibility || "ungraded")}</span>
    </li>`).join("");

  const box = document.createElement("div");
  box.className = "hitl-card";
  box.innerHTML = `
    <div class="hitl-title">⚠ Human review required</div>
    <p class="hitl-hint">The following claim(s) rest on a single source and have not
      been independently corroborated. Decide whether the report should continue.</p>
    <ul class="hitl-list">${items}</ul>
    <div class="hitl-actions">
      <button class="primary-btn hitl-continue">Continue generation</button>
      <button class="hitl-abort">Abort report</button>
      <span class="hitl-status"></span>
    </div>`;

  const status = box.querySelector(".hitl-status");
  const setButtons = (disabled) =>
    box.querySelectorAll("button").forEach((b) => { b.disabled = disabled; });
  const send = async (decision) => {
    setButtons(true);
    status.textContent = decision === "continue" ? "Continuing…" : "Aborting…";
    try {
      const resp = await fetch(api(`/api/jobs/${jobId}/confirm`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        status.textContent = extractErrorMessage(body, resp.status);
        setButtons(false);
      }
      // On success the pipeline resumes; the card closes on confirmation_resolved
      // or the final event.
    } catch (e) {
      status.textContent = `Failed: ${e.message || e}`;
      setButtons(false);
    }
  };
  box.querySelector(".hitl-continue").addEventListener("click", () => send("continue"));
  box.querySelector(".hitl-abort").addEventListener("click", () => send("abort"));
  card.appendChild(box);
  scrollDown();
}

function removeConfirmationCard(card) {
  const el = card && card.querySelector(".hitl-card");
  if (el) el.remove();
}

function markAllDone(statusEl) {
  statusEl.querySelectorAll(".stage").forEach((r) => {
    r.classList.remove("active");
    r.classList.add("done");
  });
}

function finishCompleted(statusEl, card, jobId, data) {
  markAllDone(statusEl);
  const box = document.createElement("div");
  box.className = "result ok";
  // Show the exact absolute path the PDF was saved to, so the user can locate the file
  // on disk without hunting — with a one-click "copy path" affordance.
  const savedPath = data.pdf_path || "";
  const pathBlock = savedPath
    ? `<div class="saved-path">
         <span class="saved-path-label">Report saved to:</span>
         <code class="saved-path-value">${escapeHtml(savedPath)}</code>
         <button class="copy-path-btn" data-path="${escapeHtml(savedPath)}"
                 title="Copy full path">Copy path</button>
       </div>`
    : "";
  box.innerHTML = `
    <div class="result-title">✓ Report verified and compiled</div>
    <p>${escapeHtml(data.detail || "")}</p>
    ${pathBlock}
    <a class="download-btn" href="${api(`/api/report/${jobId}/pdf`)}" download>
      ⭳ Download PDF
    </a>`;
  const copyBtn = box.querySelector(".copy-path-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(copyBtn.dataset.path);
        const prev = copyBtn.textContent;
        copyBtn.textContent = "Copied";
        setTimeout(() => { copyBtn.textContent = prev; }, 1500);
      } catch { /* clipboard may be unavailable; the path is still visible to copy manually */ }
    });
  }
  card.appendChild(box);
  scrollDown();
}

function finishOutOfScope(statusEl, detail) {
  const active = statusEl.querySelector(".stage.active");
  if (active) active.classList.remove("active");
  const box = document.createElement("div");
  box.className = "result scope";
  // The mandated, user-facing out-of-scope message comes from the server.
  box.innerHTML = `<div class="result-title">Out of scope</div>
                   <p>This topic is outside my current business scope.</p>`;
  statusEl.parentElement.appendChild(box);
  scrollDown();
}

function finishError(statusEl, detail) {
  const box = document.createElement("div");
  box.className = "result error";
  box.innerHTML = `<div class="result-title">Could not complete</div>
                   <p>${escapeHtml(detail || "")}</p>`;
  (statusEl.parentElement || statusEl).appendChild(box);
  scrollDown();
}

// ============================ message DOM ==================================
function clearWelcome() {
  const w = document.querySelector(".welcome");
  if (w) w.remove();
}
function addUserMessage(text) {
  // User turn: right-aligned blue bubble.
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble user-bubble">${escapeHtml(text)}</div>`;
  $("messages").appendChild(el);
  scrollDown();
}
function addAgentMessage() {
  // Agent turn: left-aligned contrasting bubble. Returns the inner content node the
  // caller fills (chat markdown, status card, or result card).
  const el = document.createElement("div");
  el.className = "msg agent";
  const bubble = document.createElement("div");
  bubble.className = "bubble agent-bubble";
  el.appendChild(bubble);
  $("messages").appendChild(el);
  scrollDown();
  return bubble;
}
function scrollDown() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// ============================ settings =====================================
async function loadSettings() {
  try {
    const s = await (await fetch(api("/api/settings"))).json();
    $("llm_provider").value = s.llm_provider || "fake";
    $("llm_model").value = s.llm_model || "";
    $("llm_base_url").value = s.llm_base_url || "";
    $("search_provider").value = s.search_provider || "duckduckgo";
    $("verify_mode").value = s.verify_mode || "traceable";
    $("latex_engine").value = s.latex_engine || "auto";
    setKeyState("llm_key_state", s.llm_api_key_set);
    setKeyState("anthropic_key_state", s.anthropic_api_key_set);
    setKeyState("search_key_state", s.search_api_key_set);
    syncSettingsVisibility();
  } catch { /* server may not be up yet; ignore */ }
}

function setKeyState(id, isSet) {
  const el = $(id);
  if (!el) return;
  el.textContent = isSet ? "key configured" : "not set";
  el.className = "key-state " + (isSet ? "set" : "unset");
}

async function saveSettings() {
  const payload = {
    llm_provider: $("llm_provider").value,
    llm_model: $("llm_model").value,
    llm_base_url: $("llm_base_url").value,
    search_provider: $("search_provider").value,
    verify_mode: $("verify_mode").value,
    latex_engine: $("latex_engine").value,
  };
  // Only send secrets that were actually typed (blank = keep current).
  for (const [id, key] of [["llm_api_key", "llm_api_key"],
                           ["anthropic_api_key", "anthropic_api_key"],
                           ["search_api_key", "search_api_key"]]) {
    const v = $(id).value.trim();
    if (v) payload[key] = v;
  }
  $("settings-status").textContent = "Saving…";
  try {
    const resp = await fetch(api("/api/settings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      const msg = extractErrorMessage(body, resp.status);
      $("settings-status").textContent = `Failed: ${msg}`;
      console.error("Settings save failed:", resp.status, body);
      return;
    }
    $("settings-status").textContent = "Saved";
    // Clear only password/key fields for security
    ["llm_api_key", "anthropic_api_key", "search_api_key"].forEach((id) => ($(id).value = ""));
    // Reload settings to show the saved values in non-password fields
    await loadSettings();
    await refreshHealth();
    setTimeout(() => ($("settings-status").textContent = ""), 1500);
  } catch (e) {
    const errMsg = e.message || e.toString() || JSON.stringify(e);
    $("settings-status").textContent = `Save failed: ${errMsg}`;
    console.error("Settings save exception:", e);
  }
}

// Show only the fields relevant to the chosen protocol/provider.
function syncSettingsVisibility() {
  const provider = $("llm_provider").value;
  $("wrap_base_url").style.display = provider === "openai" ? "" : "none";
  $("wrap_llm_key").style.display = provider === "openai" ? "" : "none";
  $("wrap_anthropic_key").style.display = provider === "anthropic" ? "" : "none";
  $("wrap_search_key").style.display =
    $("search_provider").value === "tavily" ? "" : "none";
}

async function refreshHealth() {
  try {
    const h = await (await fetch(api("/api/health"))).json();
    const parts = [
      `LLM: ${h.llm_provider}`,
      `search: ${h.search_provider}`,
      `PDF: ${h.latex_available ? h.latex_engine : "unavailable"}`,
    ];
    $("engine-status").textContent = parts.join("  ·  ");
  } catch {
    $("engine-status").textContent = "engine offline";
  }
}

// ============================ wiring =======================================
function autogrow(t) {
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 200) + "px";
}

function init() {
  const topic = $("topic");
  topic.addEventListener("input", () => autogrow(topic));
  topic.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = topic.value.trim();
      if (v) { sendMessage(v); topic.value = ""; autogrow(topic); }
    }
  });
  $("send").addEventListener("click", () => {
    const v = topic.value.trim();
    if (v) { sendMessage(v); topic.value = ""; autogrow(topic); }
  });

  // Composer paperclip: browse local files and stage them for the report's session.
  const attachBtn = $("attach-btn");
  const attachInput = $("attach-input");
  if (attachBtn && attachInput) {
    attachBtn.addEventListener("click", () => attachInput.click());
    attachInput.addEventListener("change", () => {
      if (attachInput.files && attachInput.files.length) {
        addStagedFiles(Array.from(attachInput.files));
        attachInput.value = "";  // allow re-picking the same file
      }
    });
  }

  $("new-chat").addEventListener("click", () => {
    if (running) return;
    location.reload();
  });
  document.querySelectorAll("#examples button").forEach((b) => {
    b.addEventListener("click", () => {
      $("topic").value = b.textContent.trim();
      autogrow($("topic"));
      $("topic").focus();
    });
  });

  // settings modal
  $("open-settings").addEventListener("click", () => {
    $("settings-overlay").classList.remove("hidden");
    loadSettings();
  });
  $("close-settings").addEventListener("click", () =>
    $("settings-overlay").classList.add("hidden"));
  $("settings-overlay").addEventListener("click", (e) => {
    if (e.target === $("settings-overlay")) $("settings-overlay").classList.add("hidden");
  });
  $("save-settings").addEventListener("click", saveSettings);
  $("llm_provider").addEventListener("change", syncSettingsVisibility);
  $("search_provider").addEventListener("change", syncSettingsVisibility);

  // history sidebar — server-backed, loaded from disk so it survives restarts.
  const clearBtn = $("clear-history");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (confirm("Clear all history? This cannot be undone.")) clearHistory();
    });
  }
  refreshHistory();

  refreshHealth();
}

document.addEventListener("DOMContentLoaded", init);
