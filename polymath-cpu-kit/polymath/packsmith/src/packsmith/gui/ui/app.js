// Packsmith UI — vanilla JS, talks to the FastAPI backend.
const $ = (id) => document.getElementById(id);
const S = { doc: null, meta: null, page: 1, start: null, end: null, topic: null, lastAsk: null, lib: null };

// The optional API token (Settings → Server): when Packsmith has one, the page asks for it once; the server then keeps
// a session cookie (HttpOnly: page scripts never hold the token).
let LOGIN = null;
async function login() {
  if (!LOGIN) LOGIN = (async () => {
    for (let i = 0; i < 3; i++) {
      const t = window.prompt(i ? "Wrong token. Packsmith's API token:" : "This Packsmith is protected by an API token. Token:");
      if (t === null) return false;
      const r = await fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token: t }) });
      if (r.ok) return true;
    }
    return false;
  })().finally(() => setTimeout(() => { LOGIN = null; }, 0));
  return LOGIN;
}
async function api(method, url, body, isForm, retried) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opt.body = body;
    else { opt.body = JSON.stringify(body); opt.headers["Content-Type"] = "application/json"; }
  }
  const r = await fetch(url, opt);
  if (r.status === 401 && !retried && !url.startsWith("/api/login")) {
    const j = await r.clone().json().catch(() => ({}));
    if (j.token_required && await login()) return api(method, url, body, isForm, true);
  }
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.detail) || data || r.statusText);
  return data;
}
const par = () => { const v = +($("par") && $("par").value); return v > 0 ? v : undefined; };
const post = (u, b) => {  // every call carries the Parallel value (also saved as the setting)
 b = b || {}; if (!u.startsWith("/api/settings") && b.parallel === undefined && par()) b = { ...b, parallel: par() }; return api("POST", u, b); };
const get = (u) => api("GET", u), put = (u, b) => api("PUT", u, b), del = (u) => api("DELETE", u);
function toast(msg, ms = 4000) { const d = document.createElement("div"); d.textContent = msg; $("toast").appendChild(d); setTimeout(() => d.remove(), ms); }
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function safeHTML(h) { return String(h || "").replace(/<script[\s\S]*?<\/script>/gi, "").replace(/\son\w+="[^"]*"/gi, ""); }
async function guard(fn) { try { return await fn(); } catch (e) { toast("⚠ " + e.message, 7000); console.error(e); } }
const model = () => $("model").value || null;

// training-length choice: tick one; batch follows the length (editable)
const LENGTHS = [256, 512, 1024, 2048, 4096];
const BATCH_FOR = { 256: 4, 512: 4, 1024: 2, 2048: 1, 4096: 1 };
const ACCUM_FOR = { 256: 2, 512: 2, 1024: 4, 2048: 8, 4096: 8 };
const LEN_NOTE = { 256: "short: facts and one-line rules", 512: "", 1024: "recommended start (batch 2 works on a 32 GB card; 8 runs out of memory)",
  2048: "untested on the 30B model; batch 1 × accumulation 8", 4096: "untested on the 30B model; batch 1 × accumulation 8, watch GPU memory" };
function lenRadios(id, onPick) {
  const el = $(id), def = +el.dataset.default;
  el.innerHTML = LENGTHS.map((n) => `<label><input type="radio" name="${id}" value="${n}"${n === def ? " checked" : ""}> ${n}</label>`).join("");
  el.querySelectorAll("input").forEach((r) => r.onchange = () => onPick && onPick(+r.value));
  onPick && onPick(def);
}
const lenOf = (id) => +(document.querySelector(`input[name="${id}"]:checked`) || { value: 1024 }).value;
lenRadios("gLen");
lenRadios("trLen", (n) => { $("trBatch").value = BATCH_FOR[n]; $("trAccum").value = ACCUM_FOR[n]; $("trLenNote").textContent = LEN_NOTE[n] || ""; window.trEstimate && trEstimate(); });

// ------------------------------------------------------------------ jobs
function jobView(el, j) {
  el.innerHTML = `<div class="small">${esc(j.kind)} ${esc(j.label)} — <b>${esc(j.status)}</b> ${j.seconds}s</div>
   <div class="prog"><div style="width:${Math.round(j.progress * 100)}%"></div></div>
   ${j.error ? `<div class="small" style="color:var(--bad)">${esc(j.error)}</div>` : ""}
   <div class="joblog">${esc((j.log || []).join("\n"))}</div>
   ${j.status === "running" ? `<button class="ghost" onclick="post('/api/jobs/${j.id}/pause')">Pause</button> <button class="ghost" onclick="post('/api/jobs/${j.id}/cancel')">Cancel</button>` : ""}
   ${["paused", "error", "cancelled", "interrupted"].includes(j.status) && j.resumable ? `<button class="ghost" onclick="guard(async()=>{const r=await post('/api/jobs/${j.id}/resume');toast('resumed');})">Resume</button>` : ""}`;
  const lg = el.querySelector(".joblog"); if (lg) lg.scrollTop = lg.scrollHeight;
}
function watch(j, el, onDone, onTick) {
  const tick = async () => {
    const v = await get("/api/jobs/" + j.id).catch(() => null);
    if (!v) return;
    if (el) jobView(el, v);
    if (onTick) onTick(v);
    if (v.status === "running") setTimeout(tick, 1500);
    else { if (v.status === "done") { onDone && onDone(v.result); } else toast(`${v.kind}: ${v.status} ${v.error || ""}`, 8000); }
  };
  tick();
}

// ------------------------------------------------------------------ tabs
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => {
  document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  const fn = { topics: loadTopicsTab, approaches: loadLib, discover: loadDiscover, playbook: loadPlaybookTab, generate: loadGenerate,
    ...(window.TABLOAD || {}) }[b.dataset.tab];
  fn && guard(fn);
});

// ------------------------------------------------------------------ header: models, topics, engine
async function loadModels() {
  const r = await get("/api/llm/models");
  $("model").innerHTML = r.models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)}${m.context ? " · " + m.context + " ctx" : ""}</option>`).join("")
    || `<option value="">(LLM server not reachable)</option>`;
  if (r.default) $("model").value = r.default;
  if (r.error) toast(r.error, 6000);
}
async function loadTopicSel() {
  const ts = await get("/api/topics");
  $("topicSel").innerHTML = `<option value="">—</option>` + ts.map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("");
  if (S.topic) $("topicSel").value = S.topic;
  return ts;
}
$("topicSel").onchange = () => { S.topic = $("topicSel").value || null; const on = document.querySelector("#tabs button.on"); on && on.click(); };
async function engineBadge() {
  const s = await get("/api/workbench").catch(() => null);
  if (!s) return;
  const b = $("engineBadge");
  const g = await get("/api/gpu").catch(() => null);
  b.textContent = "engine " + s.status + (s.active ? " · pack on" : "") + (g && g.shared ? ` · GPU: ${g.holder === "lmstudio" ? "LLM" : g.holder || "free"}` : "");
  b.className = "badge " + (s.status === "ready" ? "ok" : s.status === "error" || s.status === "exited" ? "bad" : "");
  b.title = s.error || s.device || "";
}
setInterval(engineBadge, 5000);

// ------------------------------------------------------------------ LIBRARY
const REGION_TYPES = ["chapter", "section", "contents", "index", "table_of_cases", "table_of_statutes", "other"];
// stored values stay "contents" / "index"; people see these names (help: "Region types and roles")
const TYPE_LABEL = { chapter: "chapter", section: "section", other: "other", contents: "table of contents", index: "book index",
  table_of_cases: "table of cases", table_of_statutes: "table of statutes" };
const typeLabel = (t) => TYPE_LABEL[t] || t;
$("regType").innerHTML = REGION_TYPES.map((t) => `<option value="${t}">${typeLabel(t)}</option>`).join("");
$("regType").onchange = () => { const t = $("regType").value; $("regRole").value = ["contents", "index", "table_of_cases", "table_of_statutes"].includes(t) ? "context" : "train"; };

async function loadDocs(select) {
  const ds = await get("/api/docs");
  $("docSel").innerHTML = ds.map((d) => `<option value="${d.id}">${esc(d.name)} (${d.pages} p.)</option>`).join("");
  $("wbDoc") && ($("wbDoc").innerHTML = $("docSel").innerHTML);
  if (select) $("docSel").value = select;
  if (ds.length) await openDoc($("docSel").value);
  return ds;
}
$("docSel").onchange = () => guard(() => openDoc($("docSel").value));
$("docDelete").onclick = () => guard(async () => { if (S.doc && confirm("Delete this document and its regions?")) { await del("/api/docs/" + S.doc); await loadDocs(); } });

async function openDoc(id) {
  S.doc = id; S.meta = await get("/api/docs/" + id); S.start = S.end = null;
  $("pageCount").textContent = S.meta.pages; $("pageNo").max = S.meta.pages;
  renderThumbs(); renderRegions(); renderCalib(); await showPage(1);
}
function bandColor(p) {
  const r = (S.meta.regions || []).find((r) => p >= r.start && p <= r.end);
  return r ? `var(--${r.role})` : "";
}
function renderThumbs() {
  const el = $("thumbs"); el.innerHTML = "";
  const io = new IntersectionObserver((ents) => ents.forEach((e) => { if (e.isIntersecting) { const im = e.target.querySelector("img[data-src]"); if (im) { im.src = im.dataset.src; im.removeAttribute("data-src"); } } }), { root: el });
  for (let p = 1; p <= S.meta.pages; p++) {
    const d = document.createElement("div"); d.className = "thumb"; d.dataset.p = p;
    d.innerHTML = `<span class="band" style="background:${bandColor(p)}"></span>` +
      (S.meta.has_images ? `<img data-src="/api/docs/${S.doc}/img/${p}" alt="">` : "") + `<div>${p}</div>`;
    d.onclick = () => guard(() => showPage(p));
    el.appendChild(d); io.observe(d);
  }
}
async function showPage(p) {
  p = Math.max(1, Math.min(S.meta.pages, p)); S.page = p; $("pageNo").value = p;
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("cur", +t.dataset.p === p));
  const cur = document.querySelector(`.thumb[data-p="${p}"]`); cur && cur.scrollIntoView({ block: "nearest" });
  const pg = await get(`/api/docs/${S.doc}/page/${p}`);
  const showText = $("showText").checked || !pg.image;
  $("pageView").innerHTML = showText ? `<pre>${esc(pg.text || "(no text — OCR this page)")}</pre>` : `<img src="/api/docs/${S.doc}/img/${p}" alt="page ${p}">`;
  $("pageText").value = pg.text;
  $("pageFlags").textContent = (pg.needs_ocr ? "needs OCR " : "") + (pg.ocr_done ? "OCR done" : "");
}
$("prevPage").onclick = () => guard(() => showPage(S.page - 1));
$("nextPage").onclick = () => guard(() => showPage(S.page + 1));
$("pageNo").onchange = () => guard(() => showPage(+$("pageNo").value));
$("showText").onchange = () => { $("pageTextWrap").classList.toggle("hidden", !$("showText").checked); guard(() => showPage(S.page)); };
$("savePageText").onclick = () => guard(async () => { await put(`/api/docs/${S.doc}/page/${S.page}`, { text: $("pageText").value }); toast("page text saved"); });
$("ocrPage").onclick = () => guard(async () => { const j = await post(`/api/docs/${S.doc}/ocr`, { pages: [S.page], model: model() }); toast("OCR started"); watch(j, null, () => showPage(S.page)); });
$("ocrAll").onclick = () => guard(async () => { const j = await post(`/api/docs/${S.doc}/ocr`, { model: model() }); toast("OCR of pages without text started (see Settings → Jobs)"); watch(j, null, () => openDoc(S.doc)); });
$("markStart").onclick = () => { S.start = S.page; $("regStart").textContent = S.page; };
$("markEnd").onclick = () => { S.end = S.page; $("regEnd").textContent = S.page; };
document.addEventListener("keydown", (e) => {
  if (!$("tab-library").classList.contains("on") || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.key === "[") $("markStart").click(); else if (e.key === "]") $("markEnd").click();
  else if (e.key === "ArrowRight" || e.key === "PageDown") guard(() => showPage(S.page + 1));
  else if (e.key === "ArrowLeft" || e.key === "PageUp") guard(() => showPage(S.page - 1));
});
$("saveRegion").onclick = () => guard(async () => {
  if (!S.start || !S.end) throw new Error("mark start and end first");
  await post(`/api/docs/${S.doc}/regions`, { type: $("regType").value, name: $("regName").value, role: $("regRole").value, start: S.start, end: S.end });
  S.start = S.end = null; $("regStart").textContent = $("regEnd").textContent = "–"; $("regName").value = "";
  S.meta = await get("/api/docs/" + S.doc); renderRegions(); renderThumbs(); toast("region saved");
});
const CONTEXT_TYPES = ["contents", "index", "table_of_cases", "table_of_statutes"];
function renderRegions() {
  // type and role can be changed in place (the region keeps its id, so topics keep it)
  $("regionList").innerHTML = (S.meta.regions || []).map((r) => `<div class="item"><div class="grow"><b>${esc(r.name)}</b>
     <div class="small muted">p.${r.start}–${r.end} ·
       <select class="rgEdit" data-id="${r.id}" data-k="type" title="table of contents, book index and the tables are read for the skeleton, never mined (Help: Region types and roles)">${REGION_TYPES.map((t) => `<option value="${t}" ${t === r.type ? "selected" : ""}>${typeLabel(t)}</option>`).join("")}</select>
       <select class="rgEdit" data-id="${r.id}" data-k="role">${["train", "context", "heldout", "exclude"].map((t) => `<option ${t === r.role ? "selected" : ""}>${t}</option>`).join("")}</select>
       ${CONTEXT_TYPES.includes(r.type) ? `<span class="muted">· not mined</span>` : ""}</div></div><span class="tag ${r.role}">${r.role}</span>
     <button class="ghost" onclick="guard(()=>showPage(${r.start}))">go</button>
     <button class="ghost" onclick="delRegion('${r.id}')">✕</button></div>`).join("") || `<div class="muted small">No regions yet: scroll to a start page, press [, scroll to the end, press ], save.</div>`;
}
document.addEventListener("change", (e) => {
  const s = e.target.closest(".rgEdit"); if (!s) return;
  guard(async () => {
    const r = (S.meta.regions || []).find((x) => x.id === s.dataset.id); if (!r) return;
    const upd = { ...r, [s.dataset.k]: s.value };
    if (s.dataset.k === "type" && ["train", "context"].includes(r.role)) upd.role = CONTEXT_TYPES.includes(s.value) ? "context" : "train";
    await post(`/api/docs/${S.doc}/regions`, upd);
    S.meta = await get("/api/docs/" + S.doc); renderRegions(); renderThumbs(); toast("region updated");
  });
});
window.delRegion = (rid) => guard(async () => { await del(`/api/docs/${S.doc}/regions/${rid}`); S.meta = await get("/api/docs/" + S.doc); renderRegions(); renderThumbs(); });
function renderCalib() { $("calib").textContent = (S.meta.calibration || []).map(([s, p]) => `scan ${s} = printed ${p}`).join(" · ") || "no calibration (index used for concepts only)"; }
$("setPrinted").onclick = () => guard(async () => { S.meta.calibration = await post(`/api/docs/${S.doc}/calibration`, { scan: S.page, printed: +$("printedNo").value }); renderCalib(); });
$("searchBtn").onclick = () => guard(async () => {
  const hs = await get(`/api/docs/${S.doc}/search?q=${encodeURIComponent($("searchQ").value)}&scope=${$("searchScope").value}`);
  $("searchRes").innerHTML = (hs.length ? `<div class="small muted">${hs.length} page(s), ${hs.reduce((n, h) => n + (h.count || 1), 0)} match(es)</div>` : "") +
    hs.map((h) => `<div class="item"><div class="grow"><a href="#" onclick="guard(()=>showPage(${h.page}));return false">p.${h.page}</a>${h.count > 1 ? ` <span class="muted small">×${h.count}</span>` : ""} ${(h.lines || [h.line]).map(esc).join(" <span class='muted'>·</span> ")}
     ${h.refs.length ? "<div>" + h.refs.map((r) => r.scan ? `<a href="#" onclick="guard(()=>showPage(${r.scan}));return false">${r.printed}→${r.scan}</a>` : r.printed).join(" ") + "</div>" : ""}</div></div>`).join("") || "<div class='muted'>no hits</div>";
});
$("uploadForm").onsubmit = (e) => { e.preventDefault(); guard(async () => {
  const fd = new FormData($("uploadForm")); const j = await api("POST", "/api/docs/upload", fd, true);
  toast("ingesting " + j.label); watch(j, null, (m) => { toast("ready: " + m.name); loadDocs(m.id); });
}); };

// ------------------------------------------------------------------ TOPICS
async function loadTopicsTab() {
  const [ts, ds] = [await loadTopicSel(), await get("/api/docs")];
  $("wbDoc").innerHTML = ds.map((d) => `<option value="${d.id}">${esc(d.name)}</option>`).join("");
  $("topicList").innerHTML = ts.map((t) => `<div class="item ${t.id === S.topic ? "sel" : ""}"><div class="grow"><b>${esc(t.name)}</b>
     <div class="small muted">${esc(t.intent || "")} · ${t.regions.length} regions · ${Object.entries(t.status).filter(([, v]) => v).map(([k]) => k).join(", ") || "new"}</div></div>
     <button onclick="editTopic('${t.id}')">open</button></div>`).join("") || "<div class='muted'>No topics yet.</div>";
  const cur = S.topic ? ts.find((t) => t.id === S.topic) : null;
  S.wholeDocs = cur ? cur.whole_docs || [] : [];
  renderTopicRegions(ds, cur ? cur.regions : []);
  await renderCurated(cur ? cur.curated || [] : []);
  if (cur) fillTopic(cur);
}
async function renderCurated(chosen) {
  const cs = await get("/api/curated");
  $("tCurated").innerHTML = cs.map((c) => `<label class="item"><span class="grow"><input type="checkbox" data-cur="${c.id}" ${chosen.includes(c.id) ? "checked" : ""}>
    ${esc(c.name)} <span class="muted small">${esc(c.file)} · ${c.units} records · ${Object.entries(c.kinds || {}).map(([k, v]) => `${k} ${v}`).join(", ")}</span></span>
    <button class="ghost" onclick="delCurated('${c.id}')">✕</button></label>`).join("") || "<div class='muted small'>none yet</div>";
}
window.delCurated = (id) => guard(async () => { if (confirm("Remove this curated data set?")) { await del("/api/curated/" + id); await loadTopicsTab(); } });
$("curUpload").onclick = () => guard(async () => {
  if (!$("curFile").files.length) throw new Error("choose a file");
  const fd = new FormData(); fd.append("file", $("curFile").files[0]); fd.append("name", $("curName").value);
  const m = await waitJob(await api("POST", "/api/curated/upload", fd, true), null);
  toast(`imported ${m.units} records from ${m.file}`); $("curFile").value = ""; $("curName").value = "";
  const keep = [...document.querySelectorAll("#tCurated input:checked")].map((c) => c.dataset.cur);
  await renderCurated([...keep, m.id]);
});
function renderTopicRegions(ds, chosen) {
  const has = (d, r) => chosen.some((c) => c.doc === d && c.region === r) || (S.wholeDocs || []).includes(d);
  $("tRegions").innerHTML = ds.map((d) => `<div class="small"><b>${esc(d.name)}</b></div>` + (d.regions.map((r) =>
    `<label class="item"><span class="grow"><input type="checkbox" data-doc="${d.id}" data-region="${r.id}" ${has(d.id, r.id) ? "checked" : ""}> ${esc(r.name)} <span class="muted">${typeLabel(r.type)} p.${r.start}–${r.end}${CONTEXT_TYPES.includes(r.type) ? " · read for the skeleton, not mined" : ""}</span></span><span class="tag ${r.role}">${r.role}</span></label>`).join("") || "<div class='muted small'>no regions tagged</div>")).join("");
}
async function topicMining(id) {
  if (!id) { $("tMining").innerHTML = ""; return; }
  const t = await get("/api/topics/" + id), m = t.mining || {};
  $("tMining").innerHTML = m.train_regions ? `Mined: <b>${m.train_regions}</b> train region(s), ${m.train_pages} pages` + (m.heldout_regions ? ` · ${m.heldout_regions} held-out` : " · no held-out region (back-test on held-out needs one)")
    : `<span style="color:var(--bad)">⚠ ${esc(m.problem || "no train regions")}</span>`;
}
function fillTopic(t) { $("tWhole").checked = !!(t.whole_docs && t.whole_docs.length); topicMining(t.id).catch(() => {}); $("tId").value = t.id || ""; $("tName").value = t.name || ""; $("tDomain").value = t.domain || ""; $("tAbout").value = t.about || ""; $("tIntent").value = t.intent || "reasoning"; $("tStrategy").value = t.strategy || "unified"; }
window.editTopic = (id) => guard(async () => { S.topic = id; $("topicSel").value = id; await loadTopicsTab(); });
$("tNew").onclick = () => guard(async () => { S.topic = null; S.wholeDocs = []; fillTopic({}); renderTopicRegions(await get("/api/docs"), []); await renderCurated([]); });
$("tSave").onclick = () => guard(async () => {
  const regions = [...document.querySelectorAll("#tRegions input:checked")].map((c) => ({ doc: c.dataset.doc, region: c.dataset.region }));
  const curatedIds = [...document.querySelectorAll("#tCurated input:checked")].map((c) => c.dataset.cur);
  const prev = $("tId").value ? await get("/api/topics/" + $("tId").value) : {};
  const t = await post("/api/topics", { ...prev, id: $("tId").value || undefined, name: $("tName").value, domain: $("tDomain").value, about: $("tAbout").value,
    intent: $("tIntent").value, strategy: $("tStrategy").value, regions, curated: curatedIds, status: undefined, region_details: undefined, playbooks: undefined, mining: undefined,
    whole_docs: $("tWhole").checked ? [...new Set(regions.map((r) => r.doc))] : [] });
  S.topic = t.id; toast("topic saved"); await loadTopicsTab();
});
$("tDelete").onclick = () => guard(async () => { if ($("tId").value && confirm("Delete this topic and everything mined for it?")) { await del("/api/topics/" + $("tId").value); S.topic = null; await loadTopicsTab(); } });
$("wbMake").onclick = () => guard(async () => {
  const d = await get("/api/docs/" + $("wbDoc").value);
  if (!d.regions.length) throw new Error("tag the book's chapters (and contents/index) in the Library first");
  const t = await post("/api/topics", { name: d.name + " (whole book)", about: "", intent: "reasoning", regions: d.regions.map((r) => ({ doc: d.id, region: r.id })), whole_docs: [d.id] });
  S.topic = t.id; await loadTopicsTab(); toast("topic created: fill in domain, about and intent");
});

// ------------------------------------------------------------------ APPROACHES
async function loadLib() {
  S.lib = await get("/api/library");
  $("recList").innerHTML = S.lib.records.map((r) => `<div class="item"><div class="grow"><b>${esc(r.domain)}</b><div class="small muted">${esc(r.intent)} · ${r.moves.length} moves · ${esc(r.provenance || "")}</div></div><button onclick="showRec('${r.id}')">open</button></div>`).join("");
  $("patList").innerHTML = S.lib.patterns.map((p) => `<div><b>${p.name}</b> — ${esc(p.desc)}</div>`).join("");
}
window.showRec = (id) => { const r = S.lib.records.find((x) => x.id === id); $("recEdit").value = JSON.stringify(r, null, 1); RecEditor.load(); };
$("recSave").onclick = () => guard(async () => { RecEditor.sync(); await post("/api/library/save", JSON.parse($("recEdit").value)); toast("record saved"); loadLib(); });
$("deriveBtn").onclick = () => guard(async () => {
  const j = await post("/api/library/derive", { domain: $("newDomain").value, intent: $("newIntent").value, model: model() });
  toast("drafting a record…"); watch(j, null, (r) => { $("recEdit").value = JSON.stringify(r, null, 1); RecEditor.load(); toast("draft ready: review and save"); });
});

// ------------------------------------------------------------------ DISCOVER
function needTopic() { if (!S.topic) throw new Error("choose a topic (top right) or create one in Topics"); return S.topic; }
async function loadDiscover() {
  if (!S.lib) S.lib = await get("/api/library");
  const t = S.topic ? await get("/api/topics/" + S.topic) : null;
  $("dRecord").innerHTML = `<option value="">(no approach record)</option>` + S.lib.records.map((r) => `<option value="${r.id}">${esc(r.domain)} · ${r.intent}</option>`).join("");
  if (t && t.record) $("dRecord").value = t.record;
  if (t) { const r = await get(`/api/topics/${t.id}/rounds`); renderRounds(r.rounds); $("dPlan").value = r.plan || r.draft || ""; renderPlan();
    const p = await get(`/api/topics/${t.id}/file?name=prompt.html`); if (p) $("dPrompt").value = p; }
}
$("dNearest").onclick = () => guard(async () => {
  const t = await get("/api/topics/" + needTopic());
  const rs = await get(`/api/library/nearest?domain=${encodeURIComponent((t.domain || "") + " " + (t.name || ""))}&intent=${t.intent || ""}`);
  if (rs.length) { $("dRecord").value = rs[0].id; toast("suggested: " + rs.map((r) => r.domain).join(" · ")); }
});
$("dBuild").onclick = () => guard(async () => { const r = await post(`/api/topics/${needTopic()}/prompt`, { record: $("dRecord").value }); $("dPrompt").value = r.prompt; });
$("dRun").onclick = () => guard(async () => {
  if (!$("dPrompt").value.trim()) throw new Error("build (and edit) the prompt first");
  const j = await post(`/api/topics/${needTopic()}/discover`, { prompt: $("dPrompt").value, model: model(), pages_per_round: +$("dPPR").value,
    stable_rounds: +$("dStable").value, min_coverage: +$("dCov").value, max_pages: +$("dMax").value });
  watch(j, $("dJob"), async (res) => { toast(`discovery stopped: ${res.stopped}`); await refreshRounds(); }, () => refreshRounds());
});
async function refreshRounds() { const r = await get(`/api/topics/${S.topic}/rounds`); renderRounds(r.rounds); if (r.draft) { $("dPlan").value = r.draft; renderPlan(); } }
function renderRounds(rs) {
  $("dRounds").innerHTML = `<tr><th>round</th><th>moves</th><th>new</th><th>stable</th><th>coverage</th><th>conf.</th></tr>` +
    rs.map((r) => `<tr><td>${r.round}</td><td>${r.moves.length}</td><td>${esc(r.new.join(", "))}</td><td>${r.stable}</td><td>${Math.round(r.coverage * 100)}%</td><td>${r.confidence ?? ""}</td></tr>`).join("");
}
function renderPlan() { PlanEditor.load(); }
$("dLoadDraft").onclick = () => guard(refreshRounds);
$("dApprove").onclick = () => guard(async () => { PlanEditor.sync(); const p = await post(`/api/topics/${needTopic()}/plan`, { html: $("dPlan").value }); toast(`plan approved: ${p.moves.length} moves`); });

// any playbook from any topic: a playbook is a method, usable on any text
async function playbookOptions(sel, own) {
  const ps = await get("/api/playbooks");
  ps.sort((a, b) => (b.topic === own) - (a.topic === own));
  sel.innerHTML = ps.length ? ps.map((p) => `<option value="${esc(p.ref)}">${p.topic === own ? "this topic" : esc(p.topic_name)} · ${esc(p.which)} (${p.checks} checks)</option>`).join("")
    : `<option value="">— no playbook yet —</option>`;
  return ps;
}

// ------------------------------------------------------------------ PLAYBOOK
async function loadPlaybookTab() {
  if (!S.topic) return;
  const t = await get("/api/topics/" + S.topic);
  const opts = (t.playbooks.length ? t.playbooks : ["unified"]);
  $("pWhich").innerHTML = opts.map((x) => `<option>${x}</option>`).join("");
  await playbookOptions($("pBtWhich"), S.topic);
  await showPlaybook(); await showAlign(); await showBt();
}
async function showPlaybook() {
  const h = await get(`/api/topics/${S.topic}/playbook?doc=${$("pWhich").value || ""}`);
  $("pEdit").value = h; PbEditor.load();
  const c = await get(`/api/topics/${S.topic}/coverage`).catch(() => null);
  if (c && c.skeleton_checks) $("pCov").textContent = `contents/index skeleton: ${c.confirmed}/${c.skeleton_checks} checks confirmed by the text`;
}
async function showAlign() {
  const a = await get(`/api/topics/${S.topic}/alignment`).catch(() => null);
  if (!a || !a.docs.length) { $("pAlign").innerHTML = "<div class='muted small'>Map the regions to see which documents make which moves.</div>"; return; }
  $("pAlign").innerHTML = `<table class="tbl matrix"><tr><th>move</th>${a.docs.map((d) => `<th>${esc(a.doc_names[d])}</th>`).join("")}</tr>` +
    a.moves.map((m) => `<tr><td style="text-align:left">${esc(m)}</td>${a.docs.map((d) => { const n = (a.counts[m] || {})[d] || 0; return `<td class="n${n ? 1 : 0}">${n || "—"}</td>`; }).join("")}</tr>`).join("") + "</table>";
}
async function showBt() {
  const b = await get(`/api/topics/${S.topic}/backtest`);
  $("pBt").innerHTML = Object.entries(b).map(([k, v]) => `<div><b>${esc(k)}</b>: score ${v.score} (${v.match} match, ${v.partial} partial, ${v.miss} miss of ${v.n})</div>
    <details><summary class="small">misses</summary>${(v.rows || []).filter((r) => r.verdict !== "match").map((r) => `<div class="small item"><div class="grow"><b>p.${esc(r.page)}</b> ${esc(r.facts)}<br><i>book:</i> ${esc(r.conclusion)}<br><i>playbook:</i> ${esc(r.predicted)}</div></div>`).join("")}</details>`).join("") || "<div class='muted small'>Not run yet.</div>";
}
$("pWhich").onchange = () => guard(showPlaybook);
$("pReload").onclick = () => guard(showPlaybook);
$("pStructure").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/structure`, { model: model() }), $("pJob"),
  (r) => { if (r && r.skipped) toast(r.skipped, 12000); showPlaybook(); }));
$("pMap").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/map`, { model: model() }), $("pJob"), showAlign));
$("pReduce").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/reduce`, { model: model() }), $("pJob"), loadPlaybookTab));
$("pSave").onclick = () => guard(async () => { PbEditor.sync(); await put(`/api/topics/${needTopic()}/playbook`, { html: $("pEdit").value, doc: $("pWhich").value }); toast("playbook saved"); showPlaybook(); });
$("pBacktest").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/backtest`, { role: $("pBtRole").value, playbook: $("pBtWhich").value || "unified", model: model() }), $("pJob"), showBt));

// ------------------------------------------------------------------ GENERATE
const MIX = { walkthrough: 0.30, pair: 0.15, nextcheck: 0.08, fact: 0.07, situated: 0.20, method: 0.05, immersion: 0.15 };
$("gMix").innerHTML = Object.entries(MIX).map(([k, v]) => `<label>${k} <input data-mix="${k}" type="number" step="0.01" value="${v}"></label>`).join("");
let DATA_OPT = null;
async function dataOptions() {
  if (DATA_OPT) return DATA_OPT;
  DATA_OPT = await get("/api/data/options");
  $("gLenses").innerHTML = Object.entries(DATA_OPT.lenses).map(([k, v]) => `<label title="${esc(v.desc)}"><input type="checkbox" data-lens="${k}"> ${esc(v.name)}</label>`).join("");
  document.querySelectorAll("[data-lens]").forEach((c) => c.onchange = () => { $("gPreset").value = "custom"; });
  applyPreset();
  return DATA_OPT;
}
function applyPreset() {
  const p = $("gPreset").value; if (p === "custom" || !DATA_OPT) return;
  const on = new Set(DATA_OPT.presets[p]); document.querySelectorAll("[data-lens]").forEach((c) => c.checked = on.has(c.dataset.lens));
}
$("gPreset").onchange = applyPreset;
function showModeBoxes() {
  const m = $("gMode").value, pb = $("gWhich").value;
  $("gDataBox").classList.toggle("hidden", m === "reasoning" || m === "mining");
  $("gReasonBox").classList.toggle("hidden", m === "facts" || m === "mining");
  if (m === "mining") $("gMineBox").open = true;
  $("gFbBox").classList.toggle("hidden", pb !== "builtin:factbook");
  $("gReasonOpts").classList.toggle("hidden", pb.startsWith("builtin:factbook"));
  if (pb === "builtin:factbook" && S.topic) guard(showFactbook);
}
$("gWhich").addEventListener("change", showModeBoxes);
function fbOpts() {
  return { use: $("fbUse").value, names: $("fbNames").value, names_in_text: $("fbNamesText").checked, verify: $("fbVerify").checked,
    include_flagged: $("fbFlagged").checked, rephrase: +$("fbReph").value, qa: +$("fbQa").value, scenes: +$("fbScenes").value };
}
async function showFactbook() {
  const s = await get(`/api/topics/${S.topic}/factbook/state`);
  const r = s.report;
  if (!r || !r.units) { $("fbState").innerHTML = "<span class='muted'>No fact book yet.</span>"; return; }
  $("fbState").innerHTML = `<div>${r.units} records → <b>${r.entries}</b> entries · ${r.subjects} subjects · ${r.conflicts_open} open conflicts · ${r.flagged} flagged</div>
    <div class="muted">ledger: ${Object.entries(r.ledger).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>` +
    (s.conflicts.length ? "<h4>Conflicts</h4>" + s.conflicts.map((c) => `<div class="item"><div class="grow"><b>${esc(c.subject)}</b>${c.question ? " — " + esc(c.question) : ""}
      ${c.options.map((o) => `<div><label><input type="radio" name="cf-${c.id}" value="${o.id}" ${c.resolved === o.id ? "checked" : ""}> ${esc((o.answer || o.text).slice(0, 220))}
        <span class="muted">${esc(o.sources.join(" | "))}</span></label></div>`).join("")}
      <div><label><input type="radio" name="cf-${c.id}" value="all" ${c.resolved === "all" ? "checked" : ""}> keep all, each with its source</label>
        <label><input type="radio" name="cf-${c.id}" value="" ${!c.resolved ? "checked" : ""}> leave open (not trained)</label></div></div></div>`).join("") : "") +
    (s.flagged.length ? `<details><summary>${s.flagged.length} flagged</summary>${s.flagged.slice(0, 40).map((f) => `<div class="item">${esc(f.text.slice(0, 220))} <span class="muted">— ${esc(f.why)} (${esc(f.set)} · ${esc(f.src)})</span></div>`).join("")}</details>` : "");
  document.querySelectorAll('#fbState input[type=radio]').forEach((r) => r.onchange = () => guard(async () => {
    await post(`/api/topics/${S.topic}/factbook/resolve`, { group: r.name.slice(3), choice: r.value }); toast("saved: rebuild to apply");
  }));
}
$("fbBuild").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/factbook`, { factbook: fbOpts(), model: model() }), $("gJob"), showFactbook));
$("fbOpen").onclick = () => { if (S.topic) window.open(`/api/topics/${S.topic}/factbook`, "_blank"); };
$("gMode").onchange = showModeBoxes;
function dataOpts() {
  return { lenses: [...document.querySelectorAll("[data-lens]:checked")].map((c) => c.dataset.lens),
    fields: $("gFields").value.split(",").map((s) => s.trim()).filter(Boolean),
    rephrase: +$("gReph").value, qa: +$("gQa").value, scenes: +$("gFScenes").value, test_share: +$("gTestShare").value };
}
async function loadGenerate() {
  await dataOptions();
  if (!S.topic) return;
  const ps = await playbookOptions($("gWhich"), S.topic);
  const t = await get("/api/topics/" + S.topic);
  if (!$("gMode").dataset.touched) $("gMode").value = ps.length ? (t.playbooks.length ? "reasoning" : "facts") : "facts";
  const tt = await get("/api/topics/" + S.topic);
  if ((tt.curated || []).length && !$("gMode").dataset.touched) { $("gMode").value = "reasoning"; $("gWhich").value = "builtin:factbook"; }
  showModeBoxes();
  await showDataset(); await showData();
  window.showTopicSets && await showTopicSets();
}
$("gMode").addEventListener("change", () => { $("gMode").dataset.touched = 1; });
async function showData() {
  const d = await get(`/api/topics/${S.topic}/data`).catch(() => null);
  const r = d && d.report;
  if (!r || !r.records) { $("gDataRep").innerHTML = ""; return; }
  $("gDataRep").innerHTML = `<div>captured <b>${r.records}</b> items from ${r.pages_covered}/${r.pages_total} pages ·
    ${Object.entries(r.by_lens || {}).map(([k, v]) => `${k} ${v}`).join(" · ")}</div>
    <div class="muted">${r.records - r.grounded} set aside: a number in them is not printed on the page${r.failed_blocks ? ` · ${r.failed_blocks} blocks failed (run again to retry them)` : ""}</div>
    <details><summary>sample</summary>${d.sample.slice(0, 15).map((x) => `<div class="item"><span class="tag">${x.lens}</span> p.${x.page} ${esc(x.text.slice(0, 300))}</div>`).join("")}</details>
    ${d.ungrounded.length ? `<details><summary>set aside (check)</summary>${d.ungrounded.slice(0, 15).map((x) => `<div class="item">p.${x.page} ${esc(x.text.slice(0, 300))} <span class="muted">— not on page: ${esc(x.missing.join(", "))}</span></div>`).join("")}</details>` : ""}`;
}
$("gCapture").onclick = () => guard(async () => watch(await post(`/api/topics/${needTopic()}/capture`, { data: dataOpts(), model: model() }), $("gJob"), showData));
async function showDataset() {
  const d = await get(`/api/topics/${S.topic}/dataset`).catch(() => null);
  if (!d || !d.counts.train) { $("gReport").innerHTML = "<div class='muted'>No dataset yet.</div>"; $("gSample").innerHTML = ""; return; }
  const r = d.report;
  $("gReport").innerHTML = `<div><b>${d.counts.train}</b> training items · <b>${d.counts.tests}</b> tests ${r.mode ? `<span class="muted">(${esc(r.mode)}${r.playbook ? " · " + esc(r.playbook) : ""}${r.reasoning_repeats > 1 ? ` · reasoning ×${r.reasoning_repeats}` : ""})</span>` : ""}</div>
    <div class="small">${Object.entries(r.by_kind || {}).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
    <div class="small muted">duplicates removed ${r.duplicates} · negative flagged ${r.negative_flagged} (dropped ${r.negative_dropped}) · leaks removed ${r.leaks}
    ${r.mix_shortfall && Object.keys(r.mix_shortfall).length ? " · short: " + Object.entries(r.mix_shortfall).map(([k, v]) => `${k} ${v}`).join(", ") : ""}</div>
    ${r.mining && r.mining.traces ? `<div class="small">reasoning traces (${esc(r.mining.traces.set)}): <b>${r.mining.traces.kept}</b> kept of ${r.mining.traces.problems}
      — ${r.mining.traces.solved} solved, ${r.mining.traces.reconstructed} reconstructed and re-checked, ${r.mining.traces.dropped} dropped
      ${Object.keys(r.mining.traces.drop_reasons || {}).length ? `<span class="muted">(${esc(Object.entries(r.mining.traces.drop_reasons).map(([k, v]) => `${k} ${v}`).join(", "))})</span>` : ""}
      ${r.mining.traces.unverified ? ' · <b>unverified craft notes</b>' : ""}</div>` : ""}
    ${r.mining ? `<div class="small">extra mining: ${Object.entries(r.mining.counts || {}).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}
      ${r.mining.dropped_not_verbatim ? ` · <span class="muted">${r.mining.dropped_not_verbatim} dropped (not word for word in the book)</span>` : ""}
      ${r.mining.voice ? `<br><span class="muted">voice: ${r.mining.voice.mean_sentence_words} words/sentence · ${Math.round(r.mining.voice.short_sentences_share * 100)}% short · “and” ${r.mining.voice.and_per_sentence}/sentence · dialogue ${Math.round(r.mining.voice.dialogue_share * 100)}%</span>` : ""}</div>` : ""}`;
  $("gSample").innerHTML = d.sample.map((x) => `<div class="item"><div class="grow"><span class="tag train">${x.kind}</span> <pre style="white-space:pre-wrap;margin:4px 0">${esc(x.text)}</pre></div></div>`).join("") +
    "<h4>Tests</h4>" + d.tests.map((x) => `<div class="item"><div class="grow">${esc(x.question)}<br><i class="muted">→ ${esc(x.expected)}</i></div></div>`).join("");
}
$("gRun").onclick = () => guard(async () => {
  const mix = {}; document.querySelectorAll("[data-mix]").forEach((i) => mix[i.dataset.mix] = +i.value);
  const opts = { mode: $("gMode").value, data: { ...dataOpts(), max_len: lenOf("gLen") }, factbook: fbOpts(), miners: minersChosen(), mining: window.miningOpts ? miningOpts() : {}, reasoning_share: +$("gShare").value,
    walkthrough: +$("gWalk").value, pair: +$("gPair").value, nextcheck: +$("gNext").value, test: +$("gTest").value,
    situated_scenes: +$("gScenes").value, method: +$("gMethod").value, max_len: lenOf("gLen"), immersion: $("gImm").checked,
    drop_negative: $("gNeg").checked, mix };
  if (opts.mode === "mining" && !opts.miners.length) throw new Error("tick at least one extra miner, or choose another 'Dataset from'");
  if (opts.mode === "facts" || opts.mode === "both") if (!opts.data.lenses.length && $("gWhich").value !== "builtin:factbook") throw new Error("tick at least one kind of data to capture");
  if ((opts.mode === "reasoning" || opts.mode === "both") && !$("gWhich").value) throw new Error("no playbook yet: build one in Playbook, or choose 'Facts & data only'");
  const j = await post(`/api/topics/${needTopic()}/generate`, { playbook: $("gWhich").value, opts, model: model(), name: $("gName").value });
  S.genJob = j.id;
  watch(j, $("gJob"), () => { showDataset(); showData(); updateFlow(); window.showTopicSets && showTopicSets(); }, (v) => { if (v.status !== "running") window.showTopicSets && showTopicSets(); });
});

// ------------------------------------------------------------------ STEPS BAR + QUICK PATH
const FLOW = [
  ["upload", "Upload", "library"], ["tag", "Tag pages", "library"], ["topic", "Topic", "topics"],
  ["plan", "Plan", "discover", true], ["playbook", "Playbook", "playbook", true],
  ["dataset", "Dataset", "generate"], ["train", "Train", "train"], ["try", "Try", "workbench"]];
async function updateFlow() {
  const st = {};
  try {
    const ds = await get("/api/docs"); st.upload = ds.length > 0; st.tag = ds.some((d) => (d.regions || []).length);
    if (S.topic) { const t = await get("/api/topics/" + S.topic); st.topic = true; st.plan = t.status.plan; st.playbook = t.status.playbook; st.dataset = t.status.dataset; }
    const ps = await get("/api/packs"); st.train = ps.length > 0;
    const wb = await get("/api/workbench").catch(() => ({})); st.try = wb.status === "ready";
  } catch (e) { /* the bar is a guide; never block on it */ }
  const here = (document.querySelector("#tabs button.on") || {}).dataset?.tab;
  $("flowSteps").innerHTML = FLOW.map(([k, label, tab, opt], i) =>
    `${i ? '<span class="arrow">→</span>' : ""}<button class="step ${st[k] ? "done" : ""} ${tab === here ? "here" : ""} ${opt ? "opt" : ""}" data-go="${tab}"
      title="${opt ? "only for reasoning datasets; facts & data need no plan or playbook" : ""}">${st[k] ? "✓ " : ""}${label}${opt ? " (reasoning)" : ""}</button>`).join("");
  document.querySelectorAll("[data-go]").forEach((b) => b.onclick = () => document.querySelector(`#tabs button[data-tab="${b.dataset.go}"]`).click());
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setTimeout(() => guard(updateFlow), 50)));
$("topicSel").addEventListener("change", () => guard(updateFlow));

lenRadios("qLen");
function qMode() { return document.querySelector('input[name="qMode"]:checked').value; }
document.querySelectorAll('input[name="qMode"]').forEach((r) => r.onchange = () => $("qPbWrap").classList.toggle("hidden", qMode() === "facts"));
$("qOpen").onclick = () => guard(async () => {
  const ds = await get("/api/docs");
  $("qDoc").innerHTML = ds.map((d) => `<option value="${d.id}">${esc(d.name)} (${d.pages} p.)</option>`).join("") || "<option value=''>— none yet —</option>";
  if (ds.length) document.querySelector('input[name="qSrc"][value="old"]').checked = !$("qFile").files.length;
  await playbookOptions($("qPb"), S.topic);
  $("qPbWrap").classList.toggle("hidden", qMode() === "facts");
  ["qGoGen", "qGoTrain"].forEach((id) => $(id).classList.add("hidden"));
  $("quickDlg").showModal();
});
const DATA_EXT = /\.(json|jsonl|ndjson|csv|tsv|xlsx|ya?ml|xml|md|markdown|zip)$/i;
function qAsShow() { const cur = $("qAs").value === "curated"; $("qCurBox").classList.toggle("hidden", !cur); $("qDocBox").classList.toggle("hidden", cur); }
$("qAs").onchange = qAsShow;
$("qFile").onchange = () => {
  document.querySelector('input[name="qSrc"][value="new"]').checked = true;
  const f = $("qFile").files[0]; if (f) { $("qAs").value = DATA_EXT.test(f.name) ? "curated" : "document"; qAsShow(); }
};
function waitJob(j, el) { return new Promise((res, rej) => watch(j, el, res, (v) => { if (v.status === "error" || v.status === "cancelled") rej(new Error(v.error || v.status)); })); }
$("qRun").onclick = () => guard(async () => {
  let doc = $("qDoc").value;
  const isNew = document.querySelector('input[name="qSrc"]:checked').value === "new";
  if (isNew && $("qAs").value === "curated") {
    if (!$("qFile").files.length) throw new Error("choose a file");
    const fd = new FormData(); fd.append("file", $("qFile").files[0]); fd.append("name", $("qName").value);
    $("qJob").textContent = "reading the records…";
    const m = await waitJob(await api("POST", "/api/curated/upload", fd, true), $("qJob"));
    $("qFile").value = "";
    const r = await waitJob(await post("/api/quick", { curated: m.id, name: $("qName").value, max_len: lenOf("qLen"), model: model(),
      factbook: { use: document.querySelector('input[name="qUse"]:checked').value } }), $("qJob"));
    S.topic = r.topic; await loadTopicSel(); $("topicSel").value = r.topic;
    const fb = r.factbook || {};
    $("qJob").innerHTML = `<b>Done:</b> ${m.units} records → ${fb.entries} fact-book entries (${fb.conflicts_open} open conflicts, ${fb.flagged} flagged) →
      ${r.items} training items and ${r.tests} tests in topic “${esc(r.topic_name)}”. <a href="/api/topics/${r.topic}/factbook" target="_blank">Open the fact book</a>`;
    ["qGoGen", "qGoTrain"].forEach((id) => $(id).classList.remove("hidden"));
    updateFlow(); return;
  }
  if (isNew) {
    if (!$("qFile").files.length) throw new Error("choose a file, or pick an uploaded document");
    const fd = new FormData(); fd.append("file", $("qFile").files[0]); fd.append("name", "");
    $("qJob").textContent = "reading the document…";
    const m = await waitJob(await api("POST", "/api/docs/upload", fd, true), $("qJob"));
    doc = m.id; $("qFile").value = ""; await guard(() => loadDocs(doc));
  }
  if (!doc) throw new Error("choose a document");
  const mode = qMode();
  if (mode !== "facts" && !$("qPb").value) throw new Error("no playbook exists yet: choose 'Facts & data' or build a playbook first");
  const body = { doc, start: +$("qFrom").value || null, end: +$("qTo").value || null, mode, playbook: $("qPb").value,
    data: { lenses: DATA_OPT ? DATA_OPT.presets[$("qPreset").value] : undefined, miners: S.qMiners || [] }, name: $("qName").value, max_len: lenOf("qLen"), model: model() };
  if (!DATA_OPT) { await dataOptions(); body.data.lenses = DATA_OPT.presets[$("qPreset").value]; }
  const r = await waitJob(await post("/api/quick", body), $("qJob"));
  S.topic = r.topic; await loadTopicSel(); $("topicSel").value = r.topic;
  $("qJob").innerHTML = `<b>Done:</b> ${r.items} training items and ${r.tests} tests in topic “${esc(r.topic_name)}”.`;
  ["qGoGen", "qGoTrain"].forEach((id) => $(id).classList.remove("hidden"));
  updateFlow();
});
$("qGoGen").onclick = () => { $("quickDlg").close(); document.querySelector('#tabs button[data-tab="generate"]').click(); };
$("qGoTrain").onclick = () => { $("quickDlg").close(); document.querySelector('#tabs button[data-tab="train"]').click();
  setTimeout(() => guard(async () => { const t = await get("/api/topics/" + S.topic); await loadTrain(); if (t.current_dataset) { $("trSource").value = t.current_dataset; $("trSource").onchange(); } }), 300); };

// ------------------------------------------------------------------ boot
(async () => {
  await guard(async () => { const s = await get("/api/settings"); if (s.parallel) $("par").value = s.parallel; });
  await guard(loadModels);
  await guard(loadTopicSel);
  await guard(() => loadDocs());
  engineBadge();
  await guard(updateFlow);
})();

// listening beyond this machine without a token: say so once per load
fetch("/api/auth").then((r) => r.json()).then((a) => { if (a.exposed_without_token) toast("⚠ Packsmith is reachable from the network with no API token: anyone there can use it. Set one in Settings → Server → API token.", 15000); }).catch(() => {});
