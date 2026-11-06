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
let interactionMode = "auto";
let activeWritingId = null;
let templateItems = [];
let templateBusy = false;
let templatesReady = Promise.resolve();
let composerOptionsOpen = true;
let generationDepth = "standard";

function normalizeDepth(depth) {
  return depth === "deep" ? "deep" : "standard";
}

function depthLabel(depth) {
  return normalizeDepth(depth) === "deep" ? "深入" : "简洁";
}

function selectGenerationDepth(depth) {
  if (running || templateBusy) return;
  generationDepth = normalizeDepth(depth);
  try { localStorage.setItem("rs-generation-depth", generationDepth); } catch {}
  syncModeControls();
}

function templateSelection() {
  const fields = {};
  document.querySelectorAll("#template-fields input").forEach(input => { fields[input.dataset.field] = input.value; });
  return { template_id: $("output-template").value || null, template_fields: fields };
}

function selectOutputTemplate(id, fields = {}) {
  const info = templateItems.find(item => item.id === id);
  $("output-template").value = info ? id : "";
  try { localStorage.setItem("rs-output-template", info ? id : ""); } catch {}
  $("template-fields").replaceChildren();
  for (const key of info?.fields || []) {
    const label = document.createElement("label"), input = document.createElement("input");
    label.textContent = key;
    input.dataset.field = key;
    input.value = fields[key] || "";
    input.maxLength = 1000;
    label.appendChild(input);
    $("template-fields").appendChild(label);
  }
  $("template-status").textContent = info ? info.hint : id ? "原模板已不可用；已生成文件仍可下载。请重新选择输出模板。" : "";
  syncModeControls();
}

async function refreshTemplates(selectedId, fields = {}) {
  const response = await fetch(api("/api/templates"));
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(extractErrorMessage(data, response.status));
  templateItems = data.items || [];
  $("output-template").replaceChildren(new Option("系统默认格式", ""));
  for (const item of templateItems) $("output-template").add(new Option(item.name, item.id));
  selectOutputTemplate(selectedId, fields);
}

async function uploadOutputTemplate(file) {
  if (running || templateBusy) return;
  templateBusy = true;
  $("send").disabled = true;
  $("template-status").textContent = "正在读取 Word 模板…";
  syncModeControls();
  try {
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(api("/api/templates"), { method: "POST", body: form });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(extractErrorMessage(data, response.status));
    await refreshTemplates(data.id);
    $("template-details").open = !!data.fields.length;
  } catch (e) { $("template-status").textContent = `模板上传失败：${e.message || e}`; }
  finally {
    templateBusy = false;
    $("send").disabled = running;
    syncModeControls();
  }
}

function selectMode(mode) {
  if (running || templateBusy) return;
  interactionMode = ["auto", "chat", "report", "writing"].includes(mode) ? mode : "auto";
  try { localStorage.setItem("rs-interaction-mode", interactionMode); } catch {}
  if (!["auto", "report"].includes(interactionMode)) {
    Object.keys(_autoTimers).forEach(cancelAutoComprehensive);
  }
  syncModeControls();
}

function syncModeControls() {
  const busy = running || templateBusy;
  const labels = { auto: "智能识别", chat: "普通聊天", report: "研究报告", writing: "成文模式" };
  const descriptions = {
    auto: "自动判断聊天、研究报告或成文需求",
    chat: "从一个问题开始，把想法聊清楚",
    report: "检索资料、梳理证据，形成可追溯的研究报告",
    writing: activeWritingId ? "输入修改要求，继续完善这篇文章" : "整理材料与思路，写成一篇完整文章",
  };
  document.body.dataset.interactionMode = interactionMode;
  $("workspace-mode").textContent = labels[interactionMode];
  $("mode-description").textContent = descriptions[interactionMode];
  $("new-chat").disabled = busy;
  $("generation-depth-controls").classList.toggle("hidden", interactionMode === "chat");
  $("generation-depth-hint").textContent = generationDepth === "deep" ? "展开比较与局限" : "重点概述";
  document.querySelectorAll(".generation-depth-buttons [data-depth]").forEach(button => {
    button.setAttribute("aria-pressed", String(button.dataset.depth === generationDepth));
    button.disabled = busy;
  });
  $("composer-options").classList.toggle("hidden", interactionMode === "chat" || !composerOptionsOpen);
  $("toggle-options").classList.toggle("hidden", interactionMode === "chat");
  $("toggle-options").setAttribute("aria-expanded", String(composerOptionsOpen));
  $("toggle-options").title = composerOptionsOpen ? "收起输出设置" : "展开输出设置";
  $("toggle-options").setAttribute("aria-label", $("toggle-options").title);
  document.querySelectorAll("[data-mode]").forEach(b => {
    b.setAttribute("aria-pressed", String(b.dataset.mode === interactionMode));
    b.disabled = busy;
  });
  $("auto-mode").checked = interactionMode === "auto";
  $("auto-mode").disabled = busy;
  $("output-template-controls").classList.toggle("hidden", interactionMode === "chat");
  document.querySelectorAll("#output-template-controls input, #output-template-controls select, #output-template-controls button")
    .forEach(el => { el.disabled = busy; });
  $("template-delete").disabled = busy || !$("output-template").value;
  document.querySelectorAll("#writing-options input, #writing-options select, #writing-options textarea, #writing-options button")
    .forEach(el => { el.disabled = running; });
  const writing = interactionMode === "writing";
  $("writing-options").classList.toggle("hidden", !writing);
  const acceptsFiles = interactionMode === "auto" || interactionMode === "report";
  $("attach-btn").disabled = !acceptsFiles || running;
  $("attach-btn").title = acceptsFiles ? "添加研究资料" : "成文材料请粘贴到上方资料区；文件上传用于研究报告";
  $("attach-tray").classList.toggle("hidden", !acceptsFiles || !_stagedFiles.length);
  $("topic").placeholder = writing ? "文章主题或修改要求，例如：写一篇关于人工智能与学习的课程文章…"
    : interactionMode === "chat" ? "输入要讨论的问题…" : "输入需求或报告主题…";
  $("send").title = writing ? "开始成文" : "发送";
  $("send").setAttribute("aria-label", $("send").title);
  $("composer-note").textContent = writing
    ? "材料默认按写作前提采用，只检查明显的内部矛盾；输出完整文章，未进行逐条事实核验。"
    : interactionMode === "chat" ? "直接问答，不启动联网报告流程。"
    : "研究报告先生成简报；补充文件后可生成综合报告。成文需求也可直接选择成文模式。";
}

function writingOptions() {
  return { length: Number($("writing-length").value), style: $("writing-style").value,
    materials: $("writing-materials").value, use_web: $("writing-web").checked,
    allow_set_phrases: $("writing-phrases").checked, previous_article_id: activeWritingId };
}

// Files staged via the composer paperclip BEFORE (or during) a report's document
// session exists. Each: { key, file, status: "staged"|"uploading"|"uploaded"|"error" }.
// When a document panel mounts (or is already open), these are flushed to that
// session's upload endpoint and cleared from the tray.
let _stagedFiles = [];
// The session currently accepting uploads. Set when a document panel is mounted;
// staged files flush to it (and future picks upload straight through).
let _activeUploadSession = null;
// Depth belongs to each report session, including its later automatic uploads.
// Changing the toolbar for another task must not change an existing report.
const _sessionDepths = new Map();

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
    list.innerHTML = `<div class="history-empty">暂无记录</div>`;
    return;
  }
  list.innerHTML = "";
  for (const it of items) {
    const el = document.createElement("button");
    el.className = "history-item";
    el.dataset.id = it.id;
    el.title = it.title || "未命名内容";
    el.setAttribute("aria-label", it.title || "未命名内容");
    const icon = { report: '<path d="M4 2h8l4 4v12H4zM12 2v5h4M7 11h6M7 14h5"/>',
      writing: '<path d="m12 3 5 5M3 17l4-1L18 5l-4-4L3 12z"/>',
      chat: '<path d="M3 3h14v10H7l-4 4z"/>' }[it.kind] || '<circle cx="10" cy="10" r="6"/>';
    const when = new Date(it.ts).toLocaleDateString();

    // Reports carry a "kind" badge: "Brief" when no documents were uploaded (a quick
    // web-only summary), or a documents count when the user added their own files.
    let kindBadge = "";
    if (it.kind === "writing") kindBadge = '<span class="history-tag writing">成文 · 未核验</span>';
    if (it.kind === "report") {
      const files = Array.isArray(it.files) ? it.files : [];
      const n = files.length || it.file_count || 0;
      if (n === 0) {
        kindBadge = `<span class="history-tag brief">简报</span>`;
      } else {
        kindBadge = `<span class="history-tag docs">${n} 份资料</span>`;
      }
    }
    const statusTag = ["report", "writing"].includes(it.kind)
      ? `<span class="history-status ${it.status || ""}">${historyStatusLabel(it)}</span>`
      : "";
    el.innerHTML = `
      <span class="history-icon ${["report", "writing", "chat"].includes(it.kind) ? it.kind : "chat"}"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" aria-hidden="true">${icon}</svg></span>
      <span class="history-body">
        <span class="history-title">${escapeHtml(it.title || "未命名内容")}</span>
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
    `<div class="ft-head">本报告包含 ${files.length} 份资料</div>` +
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
  if (it.kind === "writing") return { completed: "文章", running: "生成中", error: "失败" }[it.status] || "";
  if (it.status === "completed" && it.quality_status === "needs_review") return "待复核";
  const completedLabel = it.pdf_ready ? "PDF" : it.docx_ready ? "Word" : "已完成";
  return { completed: completedLabel, out_of_scope: "资料不足",
           error: "失败", running: "生成中" }[it.status] || "";
}

function openHistoryItem(id) {
  const it = _historyCache.find((x) => x.id === id);
  if (!it) return;
  if (!running && !templateBusy && ["report", "writing"].includes(it.kind)) selectGenerationDepth(it.depth);
  $("workspace-title").textContent = it.title || "历史内容";
  document.querySelectorAll(".history-item").forEach(el => el.setAttribute("aria-current", String(el.dataset.id === id)));
  setSidebarOpen(false);
  clearWelcome();
  if (it.kind === "writing") {
    if (!running) selectMode("writing");
    if (it.status === "completed" && it.job_id) {
      const card = addAgentMessage();
      card.classList.remove("agent-bubble");
      card.classList.add("report-card");
      loadWritingArticle(it.job_id, card);
    } else if (it.status === "running" && it.job_id && !running) {
      running = true;
      $("send").disabled = true;
      syncModeControls();
      const card = addAgentMessage();
      card.classList.remove("agent-bubble");
      card.classList.add("report-card");
      streamEvents(it.job_id, renderStatusCard(card), card, null);
    } else {
      $("topic").value = it.title || "";
      autogrow($("topic"));
    }
    return;
  }
  if (!running) selectMode(it.kind === "report" ? "report" : "chat");

  // A completed report keeps its PDF on disk keyed by job_id — offer Download directly.
  // It survives restarts (served from disk), so this works in a fresh session too.
  if (it.kind === "report" && (it.status === "completed" || it.status === "out_of_scope") && (it.pdf_ready || it.evidence_ready || it.docx_ready) && it.job_id) {
    const bubble = addAgentMessage();
    bubble.classList.remove("agent-bubble");
    bubble.classList.add("report-card");
    const box = document.createElement("div");
    box.className = "result ok";
    box.innerHTML = `
      <div class="result-title">▤ ${escapeHtml(it.title)}</div>
      <p>生成于 ${escapeHtml(new Date(it.ts).toLocaleString())}</p>
      ${it.pdf_ready ? `<a class="download-btn" href="${api(`/api/report/${it.job_id}/pdf`)}" download>下载 PDF</a>` : ""}
      ${it.docx_ready ? `<a class="download-btn" href="${api(`/api/documents/${it.job_id}/docx`)}" download>下载模板 Word</a>` : ""}
      ${it.evidence_ready ? `<a class="download-btn" href="${api(`/api/report/${it.job_id}/evidence`)}" download>下载逐句证据</a>` : ""}
      <button class="rerun-btn">重新生成</button>`;
    box.querySelector(".rerun-btn").addEventListener("click", () => rerunTopic(it.title));
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
  if (running) return;
  selectMode("report");
  $("topic").value = topic;
  autogrow($("topic"));
  sendMessage(topic);
  $("topic").value = "";
  autogrow($("topic"));
}

async function loadWritingArticle(id, card) {
  // Set revision context immediately; a later history fetch must not overwrite a
  // newer selection or the options of a request already in progress.
  if (!running) activeWritingId = id;
  const loading = document.createElement("p");
  loading.textContent = "正在读取文章…";
  card.appendChild(loading);
  try {
    const resp = await fetch(api(`/api/writing/${encodeURIComponent(id)}`));
    const article = await resp.json().catch(() => null);
    if (!resp.ok) throw new Error(extractErrorMessage(article, resp.status));
    await templatesReady;
    loading.remove();
    card.classList.add("article-card");
    card.querySelector(".writing-notice")?.remove();
    const notice = document.createElement("div");
    notice.className = "writing-notice";
    notice.textContent = article.notice;
    const title = document.createElement("h2");
    title.textContent = article.title;
    const meta = document.createElement("p");
    meta.className = "writing-meta";
    meta.textContent = `成文模式 · ${depthLabel(article.depth)} · 约 ${article.length} 字/词 · ${article.review_status === "checked" ? "行文复核完成" : "行文复核待人工检查"}`;
    if (article.conflict_review_status === "unavailable") meta.textContent += " · 材料分歧待人工检查";
    const body = document.createElement("div");
    body.className = "writing-article";
    body.innerHTML = renderMarkdown(article.content);
    const actions = document.createElement("div");
    actions.className = "writing-actions";
    const copy = document.createElement("button");
    copy.textContent = "复制正文";
    copy.addEventListener("click", async () => {
      const plain = `${article.title}\n\n${article.content.replace(/^#{1,6}\s+/gm, "")}`;
      try {
        if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(plain);
        else {
          const area = document.createElement("textarea");
          area.value = plain;
          document.body.appendChild(area);
          area.select();
          const copied = document.execCommand("copy");
          area.remove();
          if (!copied) throw new Error("clipboard unavailable");
        }
        copy.textContent = "已复制";
      } catch { copy.textContent = "复制失败，请下载 TXT"; }
    });
    const download = document.createElement("a");
    download.className = "download-btn";
    download.href = api(`/api/writing/${encodeURIComponent(id)}/download`);
    download.download = "成文文章.txt";
    download.textContent = "下载 TXT";
    const revise = document.createElement("button");
    revise.textContent = "继续修改这篇";
    revise.addEventListener("click", () => {
      if (running) return;
      selectMode("writing");
      activateWritingArticle(id, article);
      $("topic").focus();
    });
    actions.append(copy, download, revise);
    for (const extension of ["docx", "pdf"]) {
      if (!article.template?.[`has_${extension}`]) continue;
      const link = document.createElement("a");
      link.className = "download-btn";
      link.href = api(`/api/documents/${encodeURIComponent(id)}/${extension}`);
      link.download = `文章.${extension}`;
      link.textContent = extension === "docx" ? "下载模板 Word" : "下载模板 PDF";
      actions.appendChild(link);
    }
    card.append(notice, title, meta, actions, body);
    const comments = [...(article.notes || []), ...(article.template?.warnings || [])];
    if (article.template?.template_name) comments.unshift(`输出模板：${article.template.template_name}`);
    for (const c of article.conflicts || []) comments.push(`暂不采用：${c.left} / ${c.right}。${c.reason}`);
    if (comments.length || article.sources?.length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `材料处理说明（${article.conflicts?.length || 0} 组分歧；背景链接未经逐条核验）`;
      details.appendChild(summary);
      for (const text of comments) {
        const p = document.createElement("p");
        p.textContent = text;
        details.appendChild(p);
      }
      for (const source of article.sources || []) {
        try {
          const url = new URL(source.url);
          if (!["http:", "https:"].includes(url.protocol)) continue;
          const p = document.createElement("p"), link = document.createElement("a");
          link.href = url.href;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          link.textContent = source.title || url.hostname;
          p.appendChild(link);
          details.appendChild(p);
        } catch { /* malformed source URL is not a navigable link */ }
      }
      card.appendChild(details);
    }
    if (!running && activeWritingId === id) activateWritingArticle(id, article);
    scrollDown();
  } catch (e) {
    loading.textContent = `文章读取失败：${e.message || e}。可在历史记录中重新打开。`;
    if (activeWritingId === id) {
      activeWritingId = null;
      $("writing-context").textContent = "新文章";
    }
  }
}

function activateWritingArticle(id, article) {
  activeWritingId = id;
  selectGenerationDepth(article.depth);
  $("writing-context").textContent = `正在续改：${article.title}`;
  $("writing-length").value = article.target_length;
  $("writing-style").value = article.style;
  $("writing-web").checked = !!article.use_web;
  $("writing-phrases").checked = article.allow_set_phrases !== false;
  if (article.template) selectOutputTemplate(article.template.template_id, article.template.fields || {});
  syncModeControls();
}

// ============================ chat + run ===================================
async function sendMessage(message) {
  if (running || templateBusy) return;
  if (interactionMode === "writing" && !$("writing-length").checkValidity()) {
    $("writing-length").reportValidity();
    return;
  }
  const payload = { message, mode: interactionMode, ...templateSelection() };
  if (interactionMode !== "chat") payload.depth = generationDepth;
  if (interactionMode === "writing") payload.writing_options = writingOptions();
  running = true;
  $("send").disabled = true;
  syncModeControls();

  clearWelcome();
  $("workspace-title").textContent = message;
  addUserMessage(message);

  try {
    // Call the dual-mode endpoint. Chat stays chat; a report request starts the
    // unified brief-first flow (the backend creates a database session for it).
    const resp = await fetch(api("/api/message"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
      handleBriefReport(message, result.session_id, result.job_id, result.depth || payload.depth);
    } else if (result.type === "writing") {
      const card = addAgentMessage();
      card.classList.remove("agent-bubble");
      card.classList.add("report-card");
      const label = document.createElement("div");
      label.className = "writing-notice";
      label.textContent = result.notice;
      card.appendChild(label);
      streamEvents(result.job_id, renderStatusCard(card), card, null);
      refreshHistory();
    } else {
      throw new Error("Unknown response type from server");
    }
  } catch (e) {
    const errMsg = e.message || e.toString() || JSON.stringify(e);
    const card = addAgentMessage();
    card.innerHTML = `<div class="error-message">消息处理失败：${escapeHtml(errMsg)}</div>`;
    console.error("Message error:", e);
    running = false;
    $("send").disabled = false;
    syncModeControls();
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
  return `服务异常（${status}）`;
}

function handleChatResponse(content) {
  // Display chat response inside the agent bubble as rendered markdown.
  const card = addAgentMessage();
  card.classList.add("chat-response");  // add, don't overwrite bubble classes
  card.innerHTML = renderMarkdown(content);

  running = false;
  $("send").disabled = false;
  syncModeControls();
  $("topic").focus();
}

// Self-contained Markdown -> HTML renderer.
//
// The app is a fully offline desktop bundle, so we deliberately avoid an external
// library (marked) that would need a CDN or a separate bundled file that must load
// before app.js. This covers the constructs an LLM chat reply actually uses:
// fenced/inline code, bold, italic, headings, links, lists, quotes and tables.
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

  const cells = line => line.trim().replace(/^\|/, "").replace(/(?<!\\)\|$/, "")
    .split(/(?<!\\)\|/).map(cell => cell.trim().replace(/\\\|/g, "|"));
  const tableDivider = line => cells(line).every(cell => /^:?-{3,}:?$/.test(cell));
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trimEnd();
    let m;
    if (line.includes("|") && index + 1 < lines.length && tableDivider(lines[index + 1]) &&
        cells(line).length === cells(lines[index + 1]).length) {
      closeList();
      const headers = cells(line), dividers = cells(lines[++index]);
      const align = dividers.map(cell => cell.endsWith(":") ? (cell.startsWith(":") ? "center" : "right") : "left");
      const renderCells = (row, tag) => headers.map((_, col) => `<${tag} class="align-${align[col]}">${inline(row[col] || "")}</${tag}>`).join("");
      html.push(`<div class="markdown-table" role="region" aria-label="表格，可横向滚动" tabindex="0"><table><thead><tr>${renderCells(headers, "th")}</tr></thead><tbody>`);
      while (index + 1 < lines.length && lines[index + 1].includes("|") && lines[index + 1].trim()) {
        html.push(`<tr>${renderCells(cells(lines[++index]), "td")}</tr>`);
      }
      html.push("</tbody></table></div>");
    } else if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {     // heading
      closeList();
      const level = m[1].length;
      html.push(`<h${level}>${inline(m[2])}</h${level}>`);
    } else if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {      // unordered item
      if (listType !== "ul") { closeList(); html.push("<ul>"); listType = "ul"; }
      html.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*\d+\.\s+(.*)$/))) {      // ordered item
      if (listType !== "ol") { closeList(); html.push("<ol>"); listType = "ol"; }
      html.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\s*&gt;\s?(.*)$/))) {
      closeList();
      html.push(`<blockquote>${inline(m[1])}</blockquote>`);
    } else if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      closeList();
      html.push("<hr>");
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
  if (_activeUploadSession) flushStagedFiles(_activeUploadSession.sessionId, _activeUploadSession.topic);
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
      `<button class="chip-remove" title="移除附件" aria-label="移除附件" data-key="${escapeHtml(s.key)}">✕</button>`;
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
function handleBriefReport(topic, sessionId, jobId, depth) {
  // A new report is starting: staged files should wait for THIS report's panel, not
  // flush into a previous session. The panel mount below re-activates uploads.
  _activeUploadSession = null;
  _sessionDepths.set(sessionId, normalizeDepth(depth));
  const card = addAgentMessage();
  card.classList.remove("agent-bubble");
  card.classList.add("report-card");
  const intro = document.createElement("div");
  intro.className = "db-phase-label";
  intro.textContent = "01 · 检索与简报";
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
  const sessionDepth = _sessionDepths.get(sessionId) || "standard";
  card.dataset.sessionDepth = sessionDepth;

  const briefNote = briefStatus === "out_of_scope"
    ? `<p class="db-hint">目前可用的权威资料不足。你可以补充论文、笔记或其他材料，帮助进一步回答研究问题。</p>`
    : `<p class="db-hint">补充自己的论文或资料，继续深入研究。上传后将自动结合简报生成综合报告；也可以保留当前版本。</p>`;

  card.innerHTML = `
    <div class="db-phase-label">02 · 补充研究资料 <span class="phase-optional">可选</span><span class="session-depth-label">沿用${depthLabel(sessionDepth)}深度</span></div>
    ${briefNote}
    <div class="db-drop" id="db-drop-${sessionId}">
      <input type="file" id="db-file-${sessionId}" multiple accept="${DB_ACCEPT}" hidden />
      <button class="db-upload-btn" id="db-upload-${sessionId}">选择资料</button>
      <span class="db-drop-hint">也可拖入文件 · PDF / Word / Excel / CSV / TXT / Markdown</span>
    </div>
    <div class="db-article-list" id="db-list-${sessionId}">
      <div class="db-empty">尚未添加资料</div>
    </div>
    <div class="db-actions">
      <span class="db-status" id="db-status-${sessionId}"></span>
      <button class="primary-btn db-generate" id="db-generate-${sessionId}" disabled>
        生成${depthLabel(sessionDepth)}综合报告
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
  syncModeControls();
}

// Per-session debounce timers for auto-generation.
const _autoTimers = {};

function cancelAutoComprehensive(sessionId) {
  if (_autoTimers[sessionId]) {
    clearTimeout(_autoTimers[sessionId].timeout);
    clearInterval(_autoTimers[sessionId].tick);
    delete _autoTimers[sessionId];
    const statusEl = $(`db-status-${sessionId}`);
    if (statusEl) statusEl.textContent = "";
  }
}

function scheduleAutoComprehensive(sessionId, topic) {
  cancelAutoComprehensive(sessionId);
  if (!["auto", "report"].includes(interactionMode)) return;
  const statusEl = $(`db-status-${sessionId}`);
  let secs = Math.round(AUTO_COMPREHENSIVE_MS / 1000);
  if (statusEl) statusEl.textContent = `${secs} 秒后自动生成综合报告…`;
  const tick = setInterval(() => {
    secs -= 1;
    if (statusEl && secs > 0) statusEl.textContent =
      `${secs} 秒后自动生成综合报告…`;
  }, 1000);
  const timeout = setTimeout(() => {
    clearInterval(tick);
    delete _autoTimers[sessionId];
    if (!running) generateComprehensive(sessionId, topic);
  }, AUTO_COMPREHENSIVE_MS);
  _autoTimers[sessionId] = { timeout, tick };
}

async function uploadArticles(sessionId, topic, files) {
  const statusEl = $(`db-status-${sessionId}`);
  cancelAutoComprehensive(sessionId);   // pause auto-run while more files arrive
  let added = 0;
  for (const file of files) {
    if (statusEl) statusEl.textContent = `正在上传 ${file.name}…`;
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
      if (statusEl) statusEl.textContent = `上传失败：${e.message || e}`;
    }
  }
  // Auto-trigger the comprehensive report once at least one document was added.
  if (added > 0 && !running) scheduleAutoComprehensive(sessionId, topic);
}

function renderArticleList(sessionId, topic, articles) {
  const list = $(`db-list-${sessionId}`);
  if (!list) return;
  if (!articles || !articles.length) {
    list.innerHTML = `<div class="db-empty">尚未添加资料</div>`;
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
  syncModeControls();
  if (statusEl) statusEl.textContent = "正在启动综合报告…";

  try {
    const resp = await fetch(api(`/api/database/${sessionId}/comprehensive`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...templateSelection(), depth: _sessionDepths.get(sessionId) || "standard" }),
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
    intro.textContent = "03 · 综合报告";
    card.appendChild(intro);
    const csEl = renderStatusCard(card);
    streamEvents(job_id, csEl, card, null, {
      onComplete: () => refreshHistory(),
    });
  } catch (e) {
    if (statusEl) statusEl.textContent = `启动失败：${e.message || e}`;
    if (gen) gen.disabled = false;
    running = false;
    $("send").disabled = false;
    syncModeControls();
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
          "暂时没有收到新进度，服务可能无响应。请检查设置中的接口与密钥后重试。");
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
    if (status === "completed" && data && data.kind === "writing") {
      selectMode("writing");
      statusEl.remove();
      loadWritingArticle(data.article_id || jobId, card);
    } else if (status === "completed" && data) {
      finishCompleted(statusEl, card, jobId, data);
    } else if (status === "out_of_scope") {
      finishOutOfScope(statusEl, detail, jobId, data);
    } else {
      finishError(statusEl, detail || "本次生成未能完成。");
    }
    // History is server-owned (persisted on disk by the backend). Refresh the sidebar
    // to reflect the final status. The onComplete hook advances the unified flow
    // (mount the document panel after the brief, refresh history after comprehensive).
    refreshHistory();
    syncModeControls();
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
        appendStatusEvent(statusEl, data.stage, "等待你检查来源并确认…");
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
      finalize("error", "与本地服务的连接已断开。");
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
    scraping: "获取来源资料",
    extracting: "提取资料中的论点",
    verifying: "交叉检查证据",
    analyzing: "综合分析",
    generating_latex: "排版报告",
    compiling: "生成 PDF",
    awaiting_confirmation: "等待人工确认",
    confirmation_resolved: "单一来源论点复核",
    writing_sources: "补充背景材料",
    writing_search_plan: "提炼检索主题",
    writing_plan: "规划与矛盾检查",
    writing_conflicts: "检查材料分歧",
    writing_draft: "撰写完整文章",
    writing_review: "行文复核",
    writing_model: "补足文章输出",
    template_export: "按 Word 模板导出",
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
      <span class="hitl-claim-meta">来源：${escapeHtml(c.domain || "未知")} ·
        可信度：${escapeHtml(c.credibility || "未分级")}</span>
    </li>`).join("");

  const box = document.createElement("div");
  box.className = "hitl-card";
  box.innerHTML = `
    <div class="hitl-title">以下论点需要你复核</div>
    <p class="hitl-hint">这些论点依赖单一来源，尚未获得独立印证。请判断是否继续生成报告。</p>
    <ul class="hitl-list">${items}</ul>
    <div class="hitl-actions">
      <button class="primary-btn hitl-continue">继续生成</button>
      <button class="hitl-abort">停止生成</button>
      <span class="hitl-status"></span>
    </div>`;

  const status = box.querySelector(".hitl-status");
  const setButtons = (disabled) =>
    box.querySelectorAll("button").forEach((b) => { b.disabled = disabled; });
  const send = async (decision) => {
    setButtons(true);
    status.textContent = decision === "continue" ? "正在继续…" : "正在停止…";
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
  const savedPath = data.pdf_path || data.docx_path || data.tex_path || "";
  const research = data.research_audit || {};
  const questions = Array.isArray(research.questions) ? research.questions : [];
  const gapCount = questions.filter(q => q.status !== "supported").length;
  const researchBlock = questions.length
    ? `<p>研究问题 ${questions.length} 项 · 剩余证据缺口 ${gapCount} 项 · 补充检索 ${(research.searches || []).length} 次</p>`
    : "";
  const analysisBlock = data.analysis_audit && data.analysis_audit.accepted
    ? `<p>形成 ${Number(data.analysis_audit.accepted)} 项分析判断，请结合报告中的假设、替代解释与不确定性阅读。</p>`
    : "";
  const policy = data.evidence_policy || {};
  const evidenceBlock = policy.version
    ? `<p>来源陈述 ${Number(policy.source_reported || 0)} 条 · 多源印证 ${Number(policy.corroborated || 0)} 条 · 存在分歧 ${Number(policy.disputed || 0)} 条 · 待核验线索 ${Number(policy.lead_only || 0)} 条。证据等级表示来源支持程度，不代表事实必然正确。</p>`
    : "";
  const sparseBlock = data.source_quality && data.source_quality.evidence_mode === "limited_evidence"
    ? `<p>证据有限：现有陈述可追溯到来源，但缺少充分的独立印证。请留意报告中的证据限制。</p>`
    : "";
  const pathBlock = savedPath
    ? `<div class="saved-path">
         <span class="saved-path-label">已保存至</span>
         <code class="saved-path-value">${escapeHtml(savedPath)}</code>
         <button class="copy-path-btn" data-path="${escapeHtml(savedPath)}"
                 title="复制完整路径">复制路径</button>
       </div>`
    : "";
  box.innerHTML = `
    <div class="result-title">${data.quality_status === "needs_review" ? "报告草稿已生成 · 待复核" : "报告已生成"}${data.has_pdf ? " · PDF 已就绪" : ""}</div>
    <p>${escapeHtml(data.detail || "")}</p>
    ${researchBlock}
    ${analysisBlock}
    ${evidenceBlock}
    ${sparseBlock}
    ${pathBlock}
    ${data.has_pdf ? `<a class="download-btn" href="${api(`/api/report/${jobId}/pdf`)}" download>
      下载 PDF
    </a>` : ""}
    ${data.has_evidence ? `<a class="download-btn" href="${api(`/api/report/${jobId}/evidence`)}" download>下载研究与证据</a>` : ""}`;
  if (data.has_docx) {
    const name = document.createElement("p");
    name.textContent = `输出模板：${data.template?.template_name || "Word 模板"}`;
    box.appendChild(name);
    const link = document.createElement("a");
    link.className = "download-btn";
    link.href = api(`/api/documents/${jobId}/docx`);
    link.download = "报告.docx";
    link.textContent = "下载模板 Word";
    box.appendChild(link);
  }
  for (const warning of data.template?.warnings || []) {
    if ((data.detail || "").includes(warning)) continue;
    const p = document.createElement("p");
    p.textContent = warning;
    box.appendChild(p);
  }
  const copyBtn = box.querySelector(".copy-path-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(copyBtn.dataset.path);
        const prev = copyBtn.textContent;
        copyBtn.textContent = "已复制";
        setTimeout(() => { copyBtn.textContent = prev; }, 1500);
      } catch { /* clipboard may be unavailable; the path is still visible to copy manually */ }
    });
  }
  card.appendChild(box);
  scrollDown();
}

function finishOutOfScope(statusEl, detail, jobId, data) {
  const active = statusEl.querySelector(".stage.active");
  if (active) active.classList.remove("active");
  const box = document.createElement("div");
  box.className = "result scope";
  box.innerHTML = `<div class="result-title">现有证据不足</div>
    <p>${escapeHtml(detail || "现有资料还无法回答这个问题。")}</p>
    ${data && data.has_evidence ? `<a class="download-btn" href="${api(`/api/report/${jobId}/evidence`)}" download>下载证据与缺口清单</a>` : ""}`;
  statusEl.parentElement.appendChild(box);
  scrollDown();
}

function finishError(statusEl, detail) {
  const box = document.createElement("div");
  box.className = "result error";
  box.innerHTML = `<div class="result-title">本次生成未能完成</div>
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
    $("openalex_search_enabled").checked = s.openalex_search_enabled ?? true;
    $("verify_mode").value = s.verify_mode || "traceable";
    $("latex_engine").value = s.latex_engine || "auto";
    setKeyState("llm_key_state", s.llm_api_key_set);
    setKeyState("anthropic_key_state", s.anthropic_api_key_set);
    setKeyState("search_key_state", s.search_api_key_set);
    setKeyState("openalex_key_state", s.openalex_api_key_set);
    syncSettingsVisibility();
  } catch { /* server may not be up yet; ignore */ }
}

function setKeyState(id, isSet) {
  const el = $(id);
  if (!el) return;
  el.textContent = isSet ? "密钥已配置" : "尚未配置";
  el.className = "key-state " + (isSet ? "set" : "unset");
}

async function saveSettings() {
  const payload = {
    llm_provider: $("llm_provider").value,
    llm_model: $("llm_model").value,
    llm_base_url: $("llm_base_url").value,
    search_provider: $("search_provider").value,
    openalex_search_enabled: $("openalex_search_enabled").checked,
    verify_mode: $("verify_mode").value,
    latex_engine: $("latex_engine").value,
  };
  // Only send secrets that were actually typed (blank = keep current).
  for (const [id, key] of [["llm_api_key", "llm_api_key"],
                           ["anthropic_api_key", "anthropic_api_key"],
                           ["search_api_key", "search_api_key"],
                           ["openalex_api_key", "openalex_api_key"]]) {
    const v = $(id).value.trim();
    if (v) payload[key] = v;
  }
  $("settings-status").textContent = "正在保存…";
  try {
    const resp = await fetch(api("/api/settings"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => null);
      const msg = extractErrorMessage(body, resp.status);
      $("settings-status").textContent = `保存失败：${msg}`;
      console.error("Settings save failed:", resp.status, body);
      return;
    }
    $("settings-status").textContent = "设置已保存";
    // Clear only password/key fields for security
    ["llm_api_key", "anthropic_api_key", "search_api_key", "openalex_api_key"].forEach((id) => ($(id).value = ""));
    // Reload settings to show the saved values in non-password fields
    await loadSettings();
    await refreshHealth();
    setTimeout(() => ($("settings-status").textContent = ""), 1500);
  } catch (e) {
    const errMsg = e.message || e.toString() || JSON.stringify(e);
    $("settings-status").textContent = `保存失败：${errMsg}`;
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
    const response = await fetch(api("/api/health"));
    if (!response.ok) throw new Error("Service unavailable");
    const h = await response.json();
    const parts = [
      `模型：${h.llm_provider === "fake" ? "离线演示" : h.llm_provider}`,
      `检索：${h.search_provider === "fake" ? "离线演示" : h.search_provider}`,
      `PDF：${h.latex_available ? h.latex_engine : "未就绪"}`,
    ];
    $("engine-status").textContent = parts.join("  ·  ");
    $("connection-status").className = "connection-status is-online";
    $("connection-status").lastElementChild.textContent = "本地服务已连接";
  } catch {
    $("engine-status").textContent = "本地服务暂不可用";
    $("connection-status").className = "connection-status is-offline";
    $("connection-status").lastElementChild.textContent = "本地服务未连接";
  }
}

// ============================ wiring =======================================
let settingsReturnFocus = null;
function setSidebarOpen(open, restoreFocus = false) {
  document.body.classList.toggle("sidebar-open", open);
  $("sidebar-backdrop").classList.toggle("hidden", !open);
  $("sidebar-toggle").setAttribute("aria-expanded", String(open));
  document.querySelector(".chat").inert = open;
  if (open) $("sidebar-close").focus();
  else if (restoreFocus) $("sidebar-toggle").focus();
  hideFileTooltip();
}

function setSettingsOpen(open) {
  if (open) {
    settingsReturnFocus = document.activeElement;
    setSidebarOpen(false);
    $("advanced-search-settings").open = false;
  }
  $("settings-overlay").classList.toggle("hidden", !open);
  $("app").inert = open;
  if (open) {
    $("close-settings").focus();
    loadSettings();
  } else {
    // The settings launcher lives in the off-canvas menu on small screens.
    const target = matchMedia("(max-width: 760px)").matches ? $("sidebar-toggle") : settingsReturnFocus;
    target?.focus();
  }
}

function keepFocusWithin(event, container) {
  const items = Array.from(container.querySelectorAll('button, a[href], input, select, textarea, summary, [tabindex="0"]'))
    .filter(el => !el.disabled && el.getClientRects().length && getComputedStyle(el).visibility !== "hidden");
  const first = items[0], last = items[items.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
}

function autogrow(t) {
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 160) + "px";
}

function init() {
  try { interactionMode = localStorage.getItem("rs-interaction-mode") || "auto"; } catch {}
  try { generationDepth = normalizeDepth(localStorage.getItem("rs-generation-depth")); } catch {}
  selectMode(interactionMode);
  $("template-example").href = api("/api/templates/example.docx");
  let templateId = "";
  try { templateId = localStorage.getItem("rs-output-template") || ""; } catch {}
  templatesReady = refreshTemplates(templateId).catch(e => { $("template-status").textContent = `模板列表读取失败：${e.message}`; });
  $("output-template").addEventListener("change", e => selectOutputTemplate(e.target.value));
  $("template-upload").addEventListener("click", () => $("template-input").click());
  $("template-input").addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) uploadOutputTemplate(file);
    e.target.value = "";
  });
  $("template-delete").addEventListener("click", async () => {
    if (running || templateBusy || !$("output-template").value) return;
    try {
      const response = await fetch(api(`/api/templates/${encodeURIComponent($("output-template").value)}`), { method: "DELETE" });
      if (!response.ok) throw new Error(extractErrorMessage(await response.json(), response.status));
      await refreshTemplates("");
    } catch (e) { $("template-status").textContent = e.message; }
  });
  document.querySelectorAll("[data-mode]").forEach(b => b.addEventListener("click", () => selectMode(b.dataset.mode)));
  document.querySelectorAll(".generation-depth-buttons [data-depth]").forEach(button => button.addEventListener("click", () => selectGenerationDepth(button.dataset.depth)));
  $("auto-mode").addEventListener("change", e => selectMode(e.target.checked ? "auto" : "chat"));
  $("toggle-options").addEventListener("click", () => {
    composerOptionsOpen = !composerOptionsOpen;
    syncModeControls();
  });
  $("writing-new").addEventListener("click", () => {
    if (running) return;
    activeWritingId = null;
    $("writing-context").textContent = "新文章";
    $("writing-materials").value = "";
    $("topic").focus();
  });
  const topic = $("topic");
  const submitTopic = () => {
    if (running || templateBusy) return;
    if (interactionMode === "writing" && !$("writing-length").reportValidity()) return;
    const v = topic.value.trim();
    if (v) { sendMessage(v); topic.value = ""; autogrow(topic); }
  };
  topic.addEventListener("input", () => autogrow(topic));
  topic.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      submitTopic();
    }
  });
  $("send").addEventListener("click", submitTopic);

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
      if (running || templateBusy) return;
      selectMode(b.dataset.exampleMode);
      $("topic").value = b.dataset.prompt;
      autogrow($("topic"));
      $("topic").focus();
    });
  });

  // settings modal
  $("open-settings").addEventListener("click", () => setSettingsOpen(true));
  $("close-settings").addEventListener("click", () => setSettingsOpen(false));
  $("settings-overlay").addEventListener("click", (e) => {
    if (e.target === $("settings-overlay")) setSettingsOpen(false);
  });
  $("sidebar-toggle").addEventListener("click", () => setSidebarOpen(true));
  $("sidebar-close").addEventListener("click", () => setSidebarOpen(false, true));
  $("sidebar-backdrop").addEventListener("click", () => setSidebarOpen(false, true));
  matchMedia("(max-width: 760px)").addEventListener("change", () => setSidebarOpen(false));
  document.addEventListener("keydown", e => {
    const modalOpen = !$("settings-overlay").classList.contains("hidden");
    const sidebarOpen = document.body.classList.contains("sidebar-open");
    if (e.key === "Escape") {
      if (modalOpen) setSettingsOpen(false);
      else if (sidebarOpen) setSidebarOpen(false, true);
    } else if (e.key === "Tab") {
      if (modalOpen) keepFocusWithin(e, $("settings-overlay"));
      else if (sidebarOpen) keepFocusWithin(e, $("sidebar"));
    }
  });
  $("save-settings").addEventListener("click", saveSettings);
  $("llm_provider").addEventListener("change", syncSettingsVisibility);
  $("search_provider").addEventListener("change", syncSettingsVisibility);
  syncSettingsVisibility();

  // history sidebar — server-backed, loaded from disk so it survives restarts.
  const clearBtn = $("clear-history");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (confirm("确定清空所有历史记录吗？此操作无法撤销。")) clearHistory();
    });
  }
  refreshHistory();

  refreshHealth();
}

document.addEventListener("DOMContentLoaded", init);
