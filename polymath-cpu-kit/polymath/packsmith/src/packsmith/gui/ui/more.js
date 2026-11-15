const isPoly = (e) => ["polymath-engine", "polymath-c", "polymath-c-gpu"].includes(e);   // engines that read Polymath packs
// Packsmith UI — Datasets (+ editor), Composer, Train, Workbench, Jobs, Settings, Assistant. Loaded after app.js.
window.TABLOAD = { datasets: loadDatasets, composer: loadComposer, train: loadTrain, workbench: loadWorkbench, jobs: loadJobs,
  settings: loadSettings, assistant: loadAssistant, help: async () => {} };
const fmtStatus = (s) => `<span class="st st-${esc(s || "")}">${esc(s || "")}</span>`;
const kindsLine = (bk) => Object.entries(bk || {}).map(([k, v]) => `${esc(k)} ${v}`).join(" · ");

// ------------------------------------------------------------------ header: Parallel is the setting
$("par").addEventListener("change", () => guard(async () => { const v = +$("par").value; if (v > 0) { await post("/api/settings", { parallel: v }); toast("parallel calls: " + v); } }));

// ------------------------------------------------------------------ GENERATE: extra mining (collapsed; nothing ticked by default)
let MINING = null;
async function miningOptions() {
  if (MINING) return MINING;
  MINING = await get("/api/mining/options");
  const ab = { perform: "perform", review: "review", reason: "reason", limits: "limits", analysis: "analysis" };
  $("gMiners").innerHTML = Object.entries(MINING.miners).map(([k, v]) => `<label title="${esc(v.desc)}"><input type="checkbox" data-miner="${k}">
    <b>${esc(v.name)}</b> <span class="tag ${v.ability}">${ab[v.ability]}</span><span class="muted"> — ${esc(v.desc)}</span></label>`).join("");
  const profOpts = Object.entries(MINING.profiles).map(([k, v]) => `<option value="${k}">${esc(v.name)}</option>`).join("");
  $("gProfile").insertAdjacentHTML("beforeend", profOpts);
  $("qProfile").insertAdjacentHTML("beforeend", profOpts);
  $("gProfile").onchange = () => { const p = MINING.profiles[$("gProfile").value]; document.querySelectorAll("[data-miner]").forEach((c) => c.checked = !!p && p.miners.includes(c.dataset.miner)); mineSum(); };
  document.querySelectorAll("[data-miner]").forEach((c) => c.onchange = () => { $("gProfile").value = ""; mineSum(); });
  $("gMineHelp").onclick = (e) => { e.preventDefault(); openHelpDrawer("generate"); };
  return MINING;
}
function minersChosen() { return [...document.querySelectorAll("[data-miner]:checked")].map((c) => c.dataset.miner); }
function mineSum() {
  const n = minersChosen();
  $("gTraceOpts").classList.toggle("hidden", !n.includes("traces"));
  $("gMineSum").textContent = n.length ? `— ${n.length} ticked: ${n.map((k) => MINING.miners[k].name).join(", ")}` : "— none (optional: to perform, review, know the limits)";
}
window.minersChosen = minersChosen;
window.miningOpts = () => ({ traces: { anchors: $("gAnchors").value, reconstruct: $("gReconstruct").checked, allow_craft: $("gCraft").checked } });
miningOptions().catch(() => {});

// ------------------------------------------------------------------ GENERATE: the topic's datasets, pause / cancel
async function showTopicSets() {
  if (!S.topic) { $("gSets").innerHTML = ""; return; }
  const [ds, t] = [await get("/api/datasets?topic=" + S.topic), await get("/api/topics/" + S.topic)];
  $("gSets").innerHTML = ds.map((m) => `<div class="item"><div class="grow"><b>${esc(m.name)}</b> ${fmtStatus(m.status)} ${m.id === t.current_dataset ? '<span class="tag train">current</span>' : ""}
    <div class="muted">${m.counts && m.counts.train != null ? `${m.counts.train} items · ${m.counts.tests} tests` : ""} · ${esc(m.mode || "")} · ${esc(m.created)}
    ${m.report && m.report.units ? ` · units: ${m.report.units.new} new, ${m.report.units.cached} cached, ${m.report.units.failed} failed` : ""}
    ${(m.packs || []).length ? " · packs: " + esc(m.packs.join(", ")) : ""}</div></div>
    ${["partial", "paused", "cancelled", "failed", "running"].includes(m.status) && m.status !== "running" ? `<button onclick="dsContinue('${m.id}')">Continue</button>` : ""}
    ${m.id !== t.current_dataset ? `<button class="ghost" onclick="dsCurrent('${m.id}')">Make current</button>` : ""}
    <button class="ghost" onclick="openDataset('${m.id}')">Open</button><button class="ghost" onclick="dsDel('${m.id}')">✕</button></div>`).join("")
    || "<div class='muted'>No datasets yet.</div>";
}
window.showTopicSets = showTopicSets;
window.dsContinue = (id) => guard(async () => { const j = await post(`/api/datasets/${id}/continue`, { model: model() }); S.genJob = j.id; toast("continuing: only missing or failed parts run");
  if ($("gJob")) watch(j, $("gJob"), () => { showTopicSets(); showDataset(); }, (v) => { if (v.status !== "running") showTopicSets(); }); });
window.dsCurrent = (id) => guard(async () => { await post(`/api/datasets/${id}/current`); showTopicSets(); showDataset(); });
window.dsDel = (id) => guard(async () => {
  const m = await get("/api/datasets/" + id);
  if (!confirm(`Delete dataset “${m.name}”?${(m.packs || []).length ? `\nPacks trained from it (kept): ${m.packs.join(", ")}` : ""}`)) return;
  await del("/api/datasets/" + id); toast("deleted"); showTopicSets(); if (S.ds === id) { S.ds = null; } loadDatasets();
});
$("gPause").onclick = () => guard(async () => { if (!S.genJob) throw new Error("no generation running"); await post(`/api/jobs/${S.genJob}/pause`); toast("pausing: finished parts are kept"); });
$("gCancel").onclick = () => guard(async () => { if (!S.genJob) throw new Error("no generation running"); await post(`/api/jobs/${S.genJob}/cancel`); toast("cancelled"); });

// quick dialog: "Also learn to" adds a profile's miners
document.addEventListener("click", (e) => {
  if (e.target && e.target.id === "qRun" && $("qProfile").value && MINING) S.qMiners = MINING.profiles[$("qProfile").value].miners; else if (e.target && e.target.id === "qRun") S.qMiners = [];
}, true);

// ------------------------------------------------------------------ DATASETS + editor
const ED = { offset: 0, limit: 100, sel: new Set(), hashes: [], total: 0, matches: 0, items: [] };
async function loadDatasets() {
  const [ds, ts, cur] = [await get("/api/datasets"), await get("/api/topics"), await get("/api/curated")];
  const f = $("dsFilter").value, src = $("dsSource").value;
  $("dsFilter").innerHTML = `<option value="">all topics</option>` + ts.map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("");
  $("dsFilter").value = f;
  const tn = Object.fromEntries(ts.map((t) => [t.id, t.name]));
  $("dsList").innerHTML = ds.filter((m) => (!f || m.topic === f) && (!src || m.source === src)).map((m) => `<div class="item ${m.id === S.ds ? "sel" : ""}"><div class="grow">
    <b>${esc(m.name)}</b> ${fmtStatus(m.status)}<div class="small muted">${esc(tn[m.topic] || (m.source === "compose" ? "composed" : ""))} · ${m.counts ? `${m.counts.train} items, ${m.counts.tests} tests` : ""}
    ${m.counts && m.counts.edits ? ` · ${m.counts.edits} edits` : ""} · ${esc(m.created)}</div></div>
    <button onclick="openDataset('${m.id}')">open</button></div>`).join("") || "<div class='muted small'>No datasets yet: Generate makes them.</div>";
  $("dsCurated").innerHTML = cur.map((c) => `<div class="item"><div class="grow">${esc(c.name)} <span class="muted">${c.units} records</span></div>
    <button class="ghost" onclick="curToDs('${c.id}')">Make dataset</button></div>`).join("") || "<div class='muted'>No curated data (Topics → Curated data).</div>";
  if (S.ds) await edRefresh(true);
}
$("dsFilter").onchange = () => guard(loadDatasets); $("dsSource").onchange = () => guard(loadDatasets);
window.curToDs = (id) => guard(async () => { watch(await post(`/api/curated/${id}/dataset`, {}), null, (r) => { toast(`dataset made: ${r.items} items`); loadDatasets(); }); toast("importing…"); });
window.openDataset = (id) => guard(async () => {
  S.ds = id; ED.sel.clear(); ED.offset = 0;
  if (!$("tab-datasets").classList.contains("on")) document.querySelector('#tabs button[data-tab="datasets"]').click();
  await edRefresh(true); await loadDatasetsListOnly();
});
async function loadDatasetsListOnly() { document.querySelectorAll("#dsList .item").forEach((x) => x.classList.toggle("sel", x.innerHTML.includes(`'${S.ds}'`))); }
async function edRefresh(meta) {
  if (!S.ds) return;
  if (meta) {
    const m = await get("/api/datasets/" + S.ds);
    $("dsTitle").textContent = m.name;
    const r = m.report || {};
    $("dsInfo").innerHTML = `${fmtStatus(m.status)} · ${m.counts ? `${m.counts.train} items · ${m.counts.tests} tests · ${kindsLine(m.counts.by_kind)}` : ""}
      <br>${esc(m.source)} · ${esc(m.mode || "")} ${m.playbook ? "· playbook " + esc(r.playbook || m.playbook) : ""} · created ${esc(m.created)} ${m.error ? `<br><span style="color:var(--bad)">${esc(m.error)}</span>` : ""}
      ${(m.packs || []).length ? "<br>packs trained from it: " + esc(m.packs.join(", ")) : ""}`;
    const kinds = Object.keys(m.counts && m.counts.by_kind || {});
    const kv = $("edKind").value;
    $("edKind").innerHTML = `<option value="">any kind</option>` + kinds.map((k) => `<option>${esc(k)}</option>`).join("");
    $("edKind").value = kv;
    $("edSetKind").innerHTML = `<option value="">change kind…</option>` + [...new Set([...kinds, "fact", "qa", "walkthrough", "manual", "curated"])].map((k) => `<option>${esc(k)}</option>`).join("");
    $("dsContinue").classList.toggle("hidden", !["partial", "paused", "cancelled", "failed"].includes(m.status) && m.source !== "compose");
    const v = await get(`/api/datasets/${S.ds}/edits`);
    $("edVersions").innerHTML = `<div class="item"><div class="grow">version 0 — as generated</div><button class="ghost" onclick="edRevert(0)">revert to this</button></div>` +
      v.versions.map((x) => `<div class="item"><div class="grow">version ${x.v} — ${esc(x.note)} <span class="muted">(${x.ops[1] - x.ops[0]} changes · ${esc(x.time)})</span>
        <details><summary class="muted">changes</summary><pre class="small">${esc(JSON.stringify(v.edits.slice(x.ops[0], x.ops[1]), null, 1).slice(0, 4000))}</pre></details></div>
        <button class="ghost" onclick="edRevert(${x.v})">revert to this</button></div>`).reverse().join("");
  }
  const body = { part: $("edPart").value, q: $("edQ").value, regex: $("edRegex").checked, case: $("edCase").checked, kind: $("edKind").value,
    edited: $("edEdited").value, has_numbers: $("edNum").checked, source: $("edSource").value, offset: ED.offset, limit: ED.limit };
  const r = await post(`/api/datasets/${S.ds}/query`, body);
  Object.assign(ED, { hashes: r.hashes, total: r.total, matches: r.matches, items: r.items });
  $("edCount").textContent = `${r.matches} of ${r.total} match`;
  $("edPage").textContent = r.matches ? `${ED.offset + 1}–${Math.min(ED.offset + ED.limit, r.matches)} of ${r.matches}` : "";
  edRenderTable(); edSelN();
}
function edHighlight(s) {
  const q = $("edQ").value; let h = esc(s);
  if (!q) return h;
  try { const rx = new RegExp($("edRegex").checked ? q : q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), $("edCase").checked ? "g" : "gi");
    h = h.replace(new RegExp(rx.source.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"), rx.flags), (m) => `<mark>${m}</mark>`); } catch (e) { /* bad pattern */ }
  return h;
}
function edRenderTable() {
  const part = $("edPart").value;
  $("edTable").innerHTML = `<table class="tbl ed"><tr><th><input type="checkbox" id="edPageAll"></th><th>kind</th><th>${part === "train" ? "text" : "question → expected"}</th><th>source</th><th></th></tr>` +
    ED.items.map((x) => `<tr data-h="${x.h}" class="${x.edited ? "edited" : ""} ${x.reviewed ? "reviewed" : ""}">
      <td><input type="checkbox" class="edrow" ${ED.sel.has(x.h) ? "checked" : ""}></td><td><span class="tag train">${esc(x.kind || "")}</span></td>
      <td>${part === "train" ? `<div class="edtext" contenteditable="true" data-f="text">${edHighlight(x.text || "")}</div>`
        : `<div class="edtext" contenteditable="true" data-f="question">${edHighlight(x.question || "")}</div><div class="edtext exp" contenteditable="true" data-f="expected">${edHighlight(x.expected || "")}</div>`}</td>
      <td class="small muted">${esc([x.source, x.doc && "", x.page ? "p." + x.page : "", x.check || x.ref || "", x.from || ""].filter(Boolean).join(" · "))}</td>
      <td><button class="ghost edsave hidden">save</button></td></tr>`).join("") + "</table>";
  $("edPageAll").onchange = (e) => { document.querySelectorAll(".edrow").forEach((c) => { c.checked = e.target.checked; const h = c.closest("tr").dataset.h; e.target.checked ? ED.sel.add(h) : ED.sel.delete(h); }); edSelN(); };
  document.querySelectorAll(".edrow").forEach((c) => c.onchange = () => { const h = c.closest("tr").dataset.h; c.checked ? ED.sel.add(h) : ED.sel.delete(h); edSelN(); });
  document.querySelectorAll(".edtext").forEach((d) => d.oninput = () => d.closest("tr").querySelector(".edsave").classList.remove("hidden"));
  document.querySelectorAll(".edsave").forEach((b) => b.onclick = () => guard(async () => {
    const tr = b.closest("tr"), set = {};
    tr.querySelectorAll(".edtext").forEach((d) => set[d.dataset.f] = d.innerText.trim());
    await post(`/api/datasets/${S.ds}/ops`, { ops: [{ op: "edit", h: [tr.dataset.h], set }], note: "edited an item" });
    toast("saved"); edRefresh(true);
  }));
}
function edSelN() { $("edSelN").textContent = `${ED.sel.size} selected`; }
async function edOps(ops, note) { await post(`/api/datasets/${S.ds}/ops`, { ops, note }); ED.sel.clear(); await edRefresh(true); }
$("edGo").onclick = () => guard(async () => { ED.offset = 0; await edRefresh(false); });
$("edQ").addEventListener("keydown", (e) => { if (e.key === "Enter") $("edGo").click(); });
["edPart", "edKind", "edEdited"].forEach((id) => $(id).onchange = () => guard(async () => { ED.offset = 0; ED.sel.clear(); await edRefresh(false); }));
$("edPrevPage").onclick = () => guard(async () => { ED.offset = Math.max(0, ED.offset - ED.limit); await edRefresh(false); });
$("edNextPage").onclick = () => guard(async () => { if (ED.offset + ED.limit < ED.matches) { ED.offset += ED.limit; await edRefresh(false); } });
$("edSelAll").onclick = () => { ED.hashes.forEach((h) => ED.sel.add(h)); document.querySelectorAll(".edrow").forEach((c) => c.checked = true); edSelN(); toast(`${ED.hashes.length} matching items selected (all pages)`); };
$("edSelNone").onclick = () => { ED.sel.clear(); document.querySelectorAll(".edrow").forEach((c) => c.checked = false); edSelN(); };
const needSel = () => { if (!S.ds) throw new Error("open a dataset"); if (!ED.sel.size) throw new Error("select items first"); return [...ED.sel]; };
$("edDel").onclick = () => guard(async () => { const h = needSel(); if (confirm(`Delete ${h.length} items? (You can revert.)`)) await edOps([{ op: "delete", h }], `deleted ${h.length}`); });
$("edSetKind").onchange = () => guard(async () => { const k = $("edSetKind").value; $("edSetKind").value = ""; if (!k) return; const h = needSel(); await edOps([{ op: "edit", h, set: { kind: k } }], `kind → ${k} (${h.length})`); });
$("edReview").onclick = () => guard(async () => { const h = needSel(); await edOps([{ op: "mark", h, value: true }], `reviewed ${h.length}`); });
$("edMove").onclick = () => guard(async () => { const h = needSel(); const to = $("edPart").value === "train" ? "tests" : "train"; await edOps([{ op: "move", h, to }], `moved ${h.length} to ${to}`); });
$("edAdd").onclick = () => guard(async () => {
  if (!S.ds) throw new Error("open a dataset");
  if ($("edPart").value === "tests") { const q = prompt("Question:"); if (!q) return; const a = prompt("Expected answer:") || ""; await edOps([{ op: "add", to: "tests", item: { question: q, expected: a, source: "manual" } }], "added a test"); return; }
  const t = prompt("Text of the new training item:"); if (!t) return; await edOps([{ op: "add", item: { text: t, kind: "manual" } }], "added an item");
});
$("edPreview").onclick = () => guard(async () => {
  const h = needSel(); if (!$("edFind").value) throw new Error("what to find?");
  const r = await post(`/api/datasets/${S.ds}/replace-preview`, { hashes: h, find: $("edFind").value, repl: $("edRepl").value, regex: $("edRegex").checked, case: $("edCase").checked, part: $("edPart").value });
  $("edPrev").innerHTML = `<b>${r.replacements}</b> replacements in <b>${r.items}</b> items` + r.changes.slice(0, 20).map((c) => `<div class="diff"><div class="before">${esc(c.before.slice(0, 300))}</div><div class="after">${esc(c.after.slice(0, 300))}</div></div>`).join("");
});
$("edApply").onclick = () => guard(async () => {
  const h = needSel(); if (!$("edFind").value) throw new Error("what to find?");
  if (!confirm(`Replace “${$("edFind").value}” with “${$("edRepl").value}” in ${h.length} selected items?`)) return;
  await edOps([{ op: "replace", h, find: $("edFind").value, repl: $("edRepl").value, regex: $("edRegex").checked, case: $("edCase").checked }], `replace “${$("edFind").value}” → “${$("edRepl").value}”`);
  $("edPrev").innerHTML = "";
});
window.edRevert = (v) => guard(async () => { if (!confirm(`Go back to version ${v}? Later edits are dropped.`)) return; await post(`/api/datasets/${S.ds}/revert`, { version: v }); edRefresh(true); });
$("dsRename").onclick = () => guard(async () => { if (!S.ds) return; const n = prompt("New name:", $("dsTitle").textContent); if (!n) return; await post(`/api/datasets/${S.ds}/rename`, { name: n }); loadDatasets(); });
$("dsContinue").onclick = () => guard(async () => { if (!S.ds) return; dsContinue(S.ds); });
$("dsDelete").onclick = () => guard(async () => { if (S.ds) dsDel(S.ds); });

// ------------------------------------------------------------------ COMPOSER
async function loadComposer() {
  const [ds, cs, ts] = [await get("/api/datasets"), await get("/api/compositions"), await get("/api/topics")];
  const tn = Object.fromEntries(ts.map((t) => [t.id, t.name]));
  const chosen = new Set([...document.querySelectorAll(".cds:checked")].map((x) => x.value));
  $("cSetsList").innerHTML = ds.map((m) => `<label class="item"><span class="grow"><input type="checkbox" class="cds" value="${m.id}" ${chosen.has(m.id) ? "checked" : ""}>
    <b>${esc(m.name)}</b> <span class="small muted">${esc(tn[m.topic] || m.source)} · ${m.counts ? m.counts.train + " items" : ""} · ${esc(m.mode || "")}</span></span>
    <button class="ghost" onclick="event.preventDefault();openDataset('${m.id}')">open</button></label>`).join("") || "<div class='muted small'>No datasets yet.</div>";
  document.querySelectorAll(".cds").forEach((c) => c.onchange = () => guard(cPreview));
  $("cList").innerHTML = cs.map((c) => `<div class="item"><div class="grow"><b>${esc(c.name)}</b><div class="small muted">${(c.datasets || c.topics || []).length} datasets
    ${c.built ? `· ${c.built.train} items, ${c.built.tests} tests, ${c.built.cross_train || 0} cross` : "· not built"} ${c.stale ? "· <b>stale</b>" : ""}</div></div>
    <button onclick="openComp('${c.id}')">open</button></div>`).join("");
  S.comps = cs;
  await cPreview();
}
async function cPreview() {
  const ids = [...document.querySelectorAll(".cds:checked")].map((x) => x.value);
  if (!ids.length) { $("cPreview").innerHTML = "<span class='muted'>Tick datasets to see the combined counts.</span>"; return; }
  const p = await post("/api/datasets/preview", { datasets: ids });
  $("cPreview").innerHTML = `<b>${p.train}</b> items after removing <b>${p.duplicates}</b> duplicates · ${p.tests} tests<br>${kindsLine(p.by_kind)}<br>
    ${p.playbooks.length >= 2 ? `cross-topic cases from the playbooks of: ${esc(p.playbooks.join(", "))}` : "no cross-topic cases (needs two or more datasets made with playbooks)"}`;
}
window.openComp = (id) => { const c = S.comps.find((x) => x.id === id); $("cId").value = c.id; $("cName").value = c.name; $("cAbout").value = c.about || "";
  document.querySelectorAll(".cds").forEach((x) => x.checked = (c.datasets || []).includes(x.value)); $("cSets").value = c.cross_sets || 12; $("cPer").value = c.cross_per_set || 3; guard(cPreview); };
async function saveComp() {
  const datasets = [...document.querySelectorAll(".cds:checked")].map((x) => x.value);
  if (!datasets.length) throw new Error("tick at least one dataset");
  const prev = (S.comps || []).find((x) => x.id === $("cId").value) || {};
  const c = await post("/api/compositions", { ...prev, id: $("cId").value || undefined, name: $("cName").value || "combined", about: $("cAbout").value,
    datasets, cross_sets: +$("cSets").value, cross_per_set: +$("cPer").value, stale: undefined });
  $("cId").value = c.id; return c;
}
$("cSave").onclick = () => guard(async () => { await saveComp(); toast("saved"); loadComposer(); });
$("cBuild").onclick = () => guard(async () => { const c = await saveComp(); watch(await post(`/api/compositions/${c.id}/build`, { model: model() }), $("cJob"), (r) => { toast("combined dataset built: train it in Train"); loadComposer(); }); });

// ------------------------------------------------------------------ TRAIN
// auto: about 4,800 texts seen in all, 2 to 16 epochs (training/train_pack.py auto_epochs). Measured on the
// regression set (scripts/regression): at 2 epochs a 450-item pack learned nothing; at 10 it learned the facts.
const AUTO_TEXTS = 4800, AUTO_MAX = 16;
const autoEpochs = (n) => Math.max(2, Math.min(AUTO_MAX, Math.ceil(AUTO_TEXTS / Math.max(1, n))));
const EPOCH_NOTE = { 0: "Default: enough passes for about 4,800 texts in all (the budget the moe-knowledge packs that worked had): many for a small dataset, 2 for a large one. The best epoch by the held-out tests is kept.",
  1: "⚠ Every item once: too little for a pack to change answers unless the dataset is large (thousands of items).",
  2: "⚠ Too little for a small dataset: a 450-item pack trained 2 epochs learned nothing in the regression test.",
  3: "Fewer passes than auto for most datasets.", 5: "Fewer passes than auto for datasets under 960 items.",
  10: "About the old budget for 480 items. The best epoch is kept.",
  16: "The most auto chooses. For small, exact fact sets; the best epoch is kept." };
$("trEpochs").innerHTML = [0, 1, 2, 3, 5, 10, 16].map((n) => `<label><input type="radio" name="trEpochs" value="${n}" ${n === 0 ? "checked" : ""}> ${n || "auto"}</label>`).join("");
const epochsOf = () => +(document.querySelector('input[name="trEpochs"]:checked') || { value: 0 }).value;
function trItems() {
  const o = $("trSource").selectedOptions[0]; if (!o) return 0;
  let n = +(o.dataset.n || 0); const mx = +$("trMax").value; if (mx && n > mx) n = mx;
  return $("trChat").value === "both" ? Math.round(n * 1.3) : n;
}
const epochsUsed = () => epochsOf() || autoEpochs(trItems());
function epochNote() { const n = epochsOf(); $("trEpochNote").textContent = (EPOCH_NOTE[n] || "") + (n ? "" : ` For this dataset: ${epochsUsed()} epochs.`); $("trEpochNote").className = "small note" + (n && n <= 2 ? " warn" : ""); trEstimate(); }
document.querySelectorAll('input[name="trEpochs"]').forEach((r) => r.onchange = epochNote);
// Pack size: presets, plus / minus, what the size costs on this base model, and a suggestion from the dataset.
const TR_PRESETS = { small: [1, 4], standard: [2, 4], medium: [4, 4], large: [4, 6], "very large": [4, 12], huge: [8, 12] };
S.sizing = null;
function trSetSize(k, n, layers) {
  $("trK").value = Math.max(1, k); $("trN").value = Math.max(1, n); S.nLayers = Math.max(1, n);
  if (layers !== undefined) $("trLayers").value = layers;
  trSizeLine();
}
function trMatchPreset() {            // tick the preset that matches the numbers, else "custom"
  const k = +$("trK").value, n = +$("trN").value, lay = $("trLayers").value.trim();
  let hit = "custom";
  if (lay === "all") hit = "all";
  else if (lay === "auto") for (const [name, [pk, pn]] of Object.entries(TR_PRESETS)) if (pk === k && pn === n) hit = name;
  const r = document.querySelector(`input[name="trCap"][value="${hit}"]`); if (r) r.checked = true;
}
function trSizeLine() {
  const z = S.sizing, k = +$("trK").value || 1, lay = $("trLayers").value.trim();
  let n = +$("trN").value || 1;
  if (!z || !z.geometry || !z.geometry.hidden) { $("trSize").textContent = `${k} experts per layer × ${lay === "all" ? "every layer" : n + " layers"}`; return; }
  const total = z.geometry.layers || 0;
  if (lay === "all") n = total; else if (lay !== "auto" && lay) n = lay.split(",").filter((x) => x.trim() !== "").length; else if (total) n = Math.min(n, total);
  const experts = k * n, mb = experts * z.expert_weights * z.bytes_per_weight / 1048576;
  const size = mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : mb.toFixed(0) + " MB";
  $("trSize").innerHTML = `<b>${experts} experts</b> (${k} per layer × ${n} of ${total || "?"} layers) ≈ <b>${size}</b> on disk and in GPU memory`
    + ` · one expert = ${(z.expert_weights / 1e6).toFixed(1)} million weights`
    + (experts > 48 ? ` · <span class="tag warn">a pack this large needs a lot of training text and more GPU memory while training</span>` : "");
  $("trN").max = total || "";
}
async function trSizing() {
  const d = $("trSource").value;
  try { S.sizing = await get(`/api/train/sizing?dataset=${encodeURIComponent(d || "")}&model=${encodeURIComponent($("trModel").value || "")}&mode=${encodeURIComponent(($("trMode") || {}).value || "gate")}`); }
  catch (e) { S.sizing = null; }
  const z = S.sizing, box = $("trSuggest");
  $("trPolyOpts").classList.toggle("hidden", !(z && isPoly(z.engine)));
  $("trBudgetHelp").innerHTML = z && isPoly(z.engine)
    ? `<b>Budget:</b> each token is processed by ${z.geometry && z.geometry.top_k ? z.geometry.top_k : 8} experts in every layer. At most <b>${z.budget}</b> of those places can go to pack experts
       (Settings → Engine → Pack budget), however many experts the pack holds. So a larger pack adds room for knowledge without slowing answers.`
    : `With the colibri engine every pack expert runs on every token, so a larger pack also makes answers slower.`;
  const g = z && z.suggestion;
  box.classList.toggle("hidden", !g);
  if (g) {
    box.innerHTML = `Suggested size for this dataset: <b>${g.k} experts per layer × ${g.n_layers} layers</b>`
      + (g.experts ? ` (${g.experts} experts, about ${g.mb >= 1024 ? (g.mb / 1024).toFixed(2) + " GB" : Math.round(g.mb) + " MB"})` : "")
      + `. ${esc(g.reason)}. <span class="muted">A rule of thumb, not measured.</span> <button class="ghost" id="trSuggestApply">Apply this size</button>`;
    $("trSuggestApply").onclick = () => { trSetSize(g.k, g.n_layers, "auto"); trMatchPreset(); toast("pack size set to the suggestion"); };
  }
  trSizeLine();
}
document.querySelectorAll('input[name="trCap"]').forEach((r) => r.onchange = () => {
  const v = document.querySelector('input[name="trCap"]:checked').value;
  if (TR_PRESETS[v]) trSetSize(TR_PRESETS[v][0], TR_PRESETS[v][1], "auto");
  else if (v === "all") trSetSize(+$("trK").value || 2, (S.sizing && S.sizing.geometry && S.sizing.geometry.layers) || +$("trN").value, "all");
  else trSizeLine();
  trEstimate();
});
const trStep = (id, d) => () => { $(id).value = Math.max(1, (+$(id).value || 1) + d); if (id === "trN") { S.nLayers = +$(id).value; if ($("trLayers").value.trim() !== "auto") $("trLayers").value = "auto"; } trMatchPreset(); trSizeLine(); };
$("trKMinus").onclick = trStep("trK", -1); $("trKPlus").onclick = trStep("trK", 1);
$("trNMinus").onclick = trStep("trN", -1); $("trNPlus").onclick = trStep("trN", 1);
["trK", "trN", "trLayers"].forEach((id) => $(id).addEventListener("input", () => { if (id === "trN") S.nLayers = +$("trN").value || 4; trMatchPreset(); trSizeLine(); }));
$("trModel").addEventListener("change", () => { trModelNote(); trSizing(); });
$("trMode").addEventListener("change", () => trSizing());   // route packs get more layers suggested
["trBatch", "trAccum", "trMax", "trSource"].forEach((id) => $(id).addEventListener("change", () => trEstimate()));
function trPerfHint() {
  const o = $("trSource").selectedOptions[0];
  const on = o && o.dataset.perf, tr = o && o.dataset.traces;
  $("trPerf").classList.toggle("hidden", !on && !tr);
  if (tr && !on) {
    $("trPerf").innerHTML = `This dataset holds <b>reasoning traces</b>: learn only the reasoning and answer (the question is context), and train
      easy → hard. <button class="ghost" id="trPerfApply">Use these settings</button>`;
    $("trPerfApply").onclick = () => { $("trLoss").value = "answer"; $("trCurr").checked = true; trEstimate(); toast("answer-only loss and curriculum on"); };
    return;
  }
  if (on && tr) {
    $("trPerf").innerHTML = `This dataset teaches the pack to <b>perform</b> and holds <b>reasoning traces</b>: large capacity, KL anchor 0.3, Q&amp;A also in
      chat format, answer-only loss and the curriculum are suggested (untested on the 30B model; compare with the base and run the transfer pilot).
      <button class="ghost" id="trPerfApply">Use these settings</button>`;
    $("trPerfApply").onclick = () => { document.querySelector('input[name="trCap"][value="large"]').click(); $("trKl").value = "0.3"; $("trChat").value = "both";
      $("trLoss").value = "answer"; $("trCurr").checked = true; trEstimate(); toast("suggested settings applied"); };
    return;
  }
  if (on) {
    $("trPerf").innerHTML = `This dataset teaches the pack to <b>perform</b> (write drafts, reports or prose). That changes most words the model writes, so it
      needs more room than a facts pack: large capacity, a lower KL anchor (0.3) and Q&amp;A also in chat format. (Untested on the 30B model; compare with the base in the Workbench.)
      <button class="ghost" id="trPerfApply">Use these settings</button>`;
    $("trPerfApply").onclick = () => { document.querySelector('input[name="trCap"][value="large"]').click(); $("trKl").value = "0.3"; $("trChat").value = "both"; trEstimate(); toast("performance settings applied"); };
  }
}
window.trEstimate = function () {
  trPerfHint();
  const o = $("trSource").selectedOptions[0]; if (!o) return;
  const n = trItems(), ep = epochsUsed();
  const steps = Math.ceil(n / Math.max(1, +$("trBatch").value)) * ep;
  $("trEst").textContent = n ? `${n} items × ${ep} epochs${epochsOf() ? "" : " (auto)"} = ${steps} steps (effective batch ${(+$("trBatch").value) * (+$("trAccum").value)}). The time per step shows after the first 20 steps.` : "";
};
async function loadTrain() {
  const [ds, s] = [await get("/api/datasets"), await get("/api/settings")];
  const cur = $("trSource").value;
  const topicCur = S.topic ? (await get("/api/topics/" + S.topic)).current_dataset : "";
  $("trSource").innerHTML = ds.filter((m) => m.counts && m.counts.train).map((m) => `<option value="${m.id}" data-n="${m.counts.train}" data-perf="${m.report && m.report.mining && m.report.mining.perform ? 1 : ""}" data-traces="${m.report && m.report.mining && m.report.mining.traces ? 1 : (m.mode === "pilot" && m.arm === "B" ? 1 : "")}" data-name="${esc(m.name)}">${esc(m.name)} · ${m.counts.train} items${m.status !== "complete" ? " · " + m.status : ""}</option>`).join("")
    || `<option value="">— generate a dataset first —</option>`;
  $("trSource").value = cur || topicCur || ($("trSource").options[0] || {}).value || "";
  if (!$("trName").value && $("trSource").selectedOptions[0]) $("trName").value = ($("trSource").selectedOptions[0].dataset.name || "").split(" · ")[0];
  if (!S.trInit) { S.trInit = true; $("trAccum").value = s.train_accum; $("trMax").value = s.train_max_items; $("trUnload").checked = s.llm_unload_for_gpu;
    const e = document.querySelector(`input[name="trEpochs"][value="${s.train_epochs}"]`); if (e) e.checked = true; }
  epochNote();
  await trModels();
  trSizing();
  if (S.trJob) watchTrain(S.trJob);
}
async function trModels() {   // the base models to train on: Settings first, then containers and downloaded models
  let r = { models: [] };
  try { r = await get("/api/train/models"); } catch (e) { /* the list stays empty: Settings is used */ }
  const cur = $("trModel").value;
  $("trModel").innerHTML = (r.models.length ? "" : `<option value="">the Settings model</option>`) + r.models.map((m) =>
    `<option value="${esc(m.path)}" data-family="${m.family}" data-serving="${m.serving ? 1 : ""}">${esc(m.label)} · ${m.family === "qwen" ? "Qwen3-MoE" : "OLMoE"}${m.serving ? " · served now" : ""}</option>`).join("");
  if (cur && [...$("trModel").options].some((o) => o.value === cur)) $("trModel").value = cur;
  trModelNote();
}
function trModelNote() {
  const o = $("trModel").selectedOptions[0], box = $("trModelNote");
  const show = o && o.value && !o.dataset.serving;
  box.classList.toggle("hidden", !show);
  if (show) box.textContent = "Not the model the engine serves now: the pack will work only after Settings → engine model is set to this model (converted for colibri).";
}
$("trSource").onchange = () => { const o = $("trSource").selectedOptions[0]; if (o) $("trName").value = (o.dataset.name || "").split(" · ")[0]; trEstimate(); trSizing(); };
function epochTable(v) {
  const rows = (v.log || []).filter((l) => l.includes("[epoch]") && l.includes("loss"));
  const eta = (v.log || []).filter((l) => l.includes("[eta]")).slice(-1)[0];
  $("trEpochTable").innerHTML = (eta ? `<div class="small">${esc(eta.replace(/^\S+ /, ""))}</div>` : "") +
    (rows.length ? `<table class="tbl"><tr><th>epoch</th><th>result</th></tr>${rows.map((l) => { const m = l.match(/\[epoch\] (\d+): (.*)/); return m ? `<tr><td>${m[1]}</td><td>${esc(m[2])}</td></tr>` : ""; }).join("")}</table>` : "");
}
function watchTrain(id) { watch({ id }, $("trJob"), (r) => { toast("pack ready: " + r.pack + (r.stopped_early ? " (stopped early)" : ""), 8000); }, epochTable); }
$("trRun").onclick = () => guard(async () => {
  const source = $("trSource").value;
  if (!source) throw new Error("generate a dataset first");
  const opts = { layers: $("trLayers").value.trim() || "auto", n_layers: +$("trN").value || S.nLayers || 4, k: +$("trK").value, epochs: epochsOf(), steps: +$("trSteps").value,
    batch: +$("trBatch").value, accum: +$("trAccum").value, max_items: +$("trMax").value, max_len: lenOf("trLen"), lr: $("trLr").value,
    kl: $("trKl").value, model: $("trModel").value, chat_format: $("trChat").value, loss: $("trLoss").value, curriculum: $("trCurr").checked };
  if (S.sizing && isPoly(S.sizing.engine)) { opts.mode = $("trMode").value; opts.init = $("trInit").value; }
  const zz = S.sizing, ex = zz && zz.geometry && zz.geometry.hidden ? (+$("trK").value) * (opts.layers === "all" ? zz.geometry.layers : opts.layers === "auto" ? Math.min(opts.n_layers, zz.geometry.layers || opts.n_layers) : opts.layers.split(",").length) : 0;
  if (ex > 48 && !confirm(`This pack has ${ex} experts. It needs much more training text and GPU memory than a standard pack (8 experts), and the gain has not been measured. Train it anyway?`)) return;
  if (epochsOf() && epochsOf() <= 2 && trItems() < 2400 && !confirm(EPOCH_NOTE[epochsOf()] + "\n\nTrain with " + epochsOf() + " epoch(s) anyway? (auto would use " + autoEpochs(trItems()) + ")")) return;
  const j = await post("/api/train", { source: "dataset:" + source, name: $("trName").value, opts, unload_lmstudio: $("trUnload").checked, reload_lmstudio: $("trReload").checked });
  S.trJob = j.id; watchTrain(j.id);
});
$("trStopSave").onclick = () => guard(async () => { if (!S.trJob) throw new Error("no training running"); await post(`/api/jobs/${S.trJob}/pause`); toast("stopping after the current step; the best weights so far are saved"); });
$("trCancel").onclick = () => guard(async () => { if (!S.trJob) throw new Error("no training running"); if (confirm("Cancel training? Nothing is saved.")) await post(`/api/jobs/${S.trJob}/cancel`); });

// ------------------------------------------------------------------ WORKBENCH
S.history = [];          // kept for older code paths; conversations are per configuration (S.histories)
S.histories = {};        // one thread per configuration (base, each pack, "a + b"), in both modes
async function loadWorkbench() {
  const [s, ps, ds, st] = [await get("/api/workbench"), await get("/api/packs"), await get("/api/datasets"), await get("/api/settings")];
  $("wbStatus").innerHTML = `<b>${esc(s.status)}</b>${s.build ? ` · <b>${esc(s.build === "single" ? "colibri" : s.build.toUpperCase() + " build")}</b>` : ""} ${esc(s.model)}<br>${esc(s.error || "")}<br><span class="muted">${esc((s.device || "").slice(0, 300))}</span>`;
  const have = s.builds || [];
  $("wbBuild").value = s.engine_build || "auto";
  [...$("wbBuild").options].forEach((o) => { o.disabled = o.value !== "auto" && have.length > 0 && !have.includes("single") && !have.includes(o.value);
    o.textContent = o.value === "auto" ? "auto" : o.value.toUpperCase() + (o.disabled ? " (not built)" : ""); });
  const ticked = new Set([...document.querySelectorAll(".wbp:checked")].map((c) => c.value));
  $("wbPacks").innerHTML = ps.map((p) => `<label class="item"><span class="grow"><input type="checkbox" class="wbp" value="${esc(p.name)}" ${ticked.has(p.name) ? "checked" : ""}> <b>${esc(p.name)}</b>
     <div class="small muted">${esc(p.card.domain || p.card.topic_name || "")} ${p.card.texts ? "· " + p.card.texts + " texts" : ""} ${p.card.best_epoch ? "· best epoch " + p.card.best_epoch : ""} ${(p.card.evaluations || []).slice(-1).map((e) => `· base ${e.base_score} → pack ${e.pack_score}`).join("")}</div>${p.note ? `<div class="small" style="color:var(--warn,#b7791f)">⚠ ${esc(p.note)}</div>` : ""}${packManifestLine(p)}${packDials(p, st.engine)}</span></label>`).join("")
    || "<div class='muted small'>No packs yet. Train one, or register an existing pack in Settings.</div>";
  const ep = $("wbEvalPack").value;
  $("wbEvalPack").innerHTML = ps.map((p) => `<option value="${esc(p.name)}" data-ds="${esc(p.card.dataset || "")}">${esc(p.name)}</option>`).join("");
  if (ep) $("wbEvalPack").value = ep;
  $("wbEvalSet").innerHTML = `<option value="">(the dataset the pack was trained on)</option>` + ds.filter((m) => m.counts && m.counts.tests).map((m) => `<option value="${m.id}">${esc(m.name)} · ${m.counts.tests} tests</option>`).join("");
  if (!S.wbInit) { S.wbInit = true; $("wbTemp").value = st.engine_temperature; $("wbTopP").value = st.engine_top_p; }
  await wbEvalList();
  await plList();
  loadEngineModels().catch(() => {});
}
// The pack's two dials (workbench/dials.py): set here only; every load of the pack uses them, agents' too.
function packDials(p, engine) {
  const d = p.dials || { strength: 1, shift: 0 }, poly = isPoly(engine);
  const pct = Math.round(d.strength * 100);
  const routeOk = poly && (((p.manifest || {}).polymath || {}).layers ? Object.values(p.manifest.polymath.layers).some((l) => l.mode === "route") : true);
  return `<div class="small dials" data-n="${esc(p.name)}" onclick="if (!event.target.closest('input,summary,button')) event.preventDefault()">
    <span title="How strongly the pack speaks: its experts' output times this. 100% = as trained, 0% = silent. The effect is not proportional: small packs often fade below about 40%, and above about 130% answers can lose fluency. Use Check (User pack tab) with a strength curve to choose.">
      Strength <input type="range" class="dlS" min="0" max="150" step="5" value="${pct}" list="dlTicks" style="width:140px;vertical-align:middle">
      <b class="dlSv">${pct === 100 ? "100% (as trained)" : pct + "%"}</b></span>
    ${poly ? `<details class="dlAdv" ${d.shift ? "open" : ""}><summary>Advanced: how often</summary>
      <span title="How often the pack steps in: added to its router logits. 0 = as trained (or as router tuning set it). +1 makes the pack about 2.7 times likelier to win a place on a token, -1 about 2.7 times less likely. It never changes how strongly the pack speaks. ${routeOk ? "" : "This pack's experts are gated on every token, so this has no effect."}">
        Routing <input type="range" class="dlR" min="-3" max="3" step="0.25" value="${d.shift}" style="width:140px;vertical-align:middle" ${routeOk ? "" : "disabled"}>
        <b class="dlRv">${shiftLabel(d.shift)}</b></span></details>` : ""}
    ${pct !== 100 || d.shift ? `<button class="ghost small dlReset">reset</button>` : ""}</div>`;
}
function shiftLabel(x) { x = Number(x); return x === 0 ? "0 (as trained)" : (x > 0 ? "+" : "") + x + (x > 0 ? " (more often)" : " (less often)"); }
document.addEventListener("input", (e) => {
  const box = e.target.closest(".dials"); if (!box) return;
  if (e.target.classList.contains("dlS")) { const v = Number(e.target.value); box.querySelector(".dlSv").textContent = v === 100 ? "100% (as trained)" : v + "%"; }
  if (e.target.classList.contains("dlR")) box.querySelector(".dlRv").textContent = shiftLabel(e.target.value);
});
async function saveDials(box, body) {
  const r = await post(`/api/packs/${encodeURIComponent(box.dataset.n)}/dials`, body);
  if (r.error) toast("⚠ " + r.error, 8000);
  else toast(`${box.dataset.n}: strength ${Math.round(r.dials.strength * 100)}%` + (r.dials.shift ? `, routing ${shiftLabel(r.dials.shift)}` : "") + (r.applied ? " · applied to the running engine" : " · used the next time it loads"));
  loadWorkbench();
}
document.addEventListener("change", (e) => {
  const box = e.target.closest(".dials"); if (!box) return;
  if (e.target.classList.contains("dlS")) guard(() => saveDials(box, { strength: Number(e.target.value) / 100 }));
  if (e.target.classList.contains("dlR")) guard(() => saveDials(box, { shift: Number(e.target.value) }));
});
document.addEventListener("click", (e) => {
  const b = e.target.closest(".dlReset"); if (!b) return;
  e.preventDefault(); e.stopPropagation();
  guard(() => saveDials(b.closest(".dials"), { strength: 1, shift: 0 }));
});
function packManifestLine(p) {
  // what the pack was made for (pack.json); older packs have none: offer to write it
  const m = p.manifest || {};
  if (!m.format) return `<div class="small muted">no manifest (made before pack manifests): compatibility unverified
    <button class="ghost small wbman" data-n="${esc(p.name)}" title="Record the engine and base model in pack.json">Write manifest</button></div>`;
  const bm = m.base_model || {}, en = m.engine || {};
  return `<div class="small muted">for ${esc(en.name || "?")}${en.kind ? " / " + esc(en.kind) : ""} · ${esc(bm.label || bm.repo || "base model")}${bm.weights_hash ? " · " + esc(bm.weights_hash.slice(7, 19)) : ""}</div>`;
}
document.addEventListener("click", (e) => {
  const b = e.target.closest(".wbman");
  if (!b) return;
  e.preventDefault();
  guard(async () => { await post(`/api/packs/${encodeURIComponent(b.dataset.n)}/manifest`); toast("manifest written"); loadWorkbench(); });
});
$("wbEvalPack").onchange = () => guard(wbEvalList);
$("wbStart").onclick = () => guard(async () => { await post("/api/workbench/start", {}); toast("starting the engine (loading the base model)…"); setTimeout(loadWorkbench, 3000); });
$("wbBuild").onchange = () => guard(async () => {
  const b = $("wbBuild").value, s = await get("/api/workbench");
  await post("/api/settings", { engine_build: b });
  if (s.status === "ready" || s.status === "starting") {
    await post("/api/workbench/stop"); await post("/api/workbench/start", {});
    toast(`restarting the engine with the ${b === "auto" ? "automatic" : b.toUpperCase()} build…`); setTimeout(loadWorkbench, 3000);
  } else { toast(`engine build: ${b}`); loadWorkbench(); }
});
$("wbStop").onclick = () => guard(async () => { await post("/api/workbench/stop"); loadWorkbench(); });
// Think on / off next to Ask; on switches to Qwen's thinking settings, off back to the answering settings
document.querySelectorAll("#wbThinkTabs button").forEach((b) => b.onclick = () => {
  const on = b.dataset.t === "on";
  $("wbThink").checked = on;
  document.querySelectorAll("#wbThinkTabs button").forEach((x) => x.classList.toggle("on", x === b));
  if (on) { S.wbPrev = { t: $("wbTemp").value, p: $("wbTopP").value, m: $("wbMax").value }; $("wbTemp").value = 0.6; $("wbTopP").value = 0.95; $("wbMax").value = Math.max(4096, +$("wbMax").value); if ($("wbTpl").value !== "chat") { $("wbTpl").value = "chat"; toast("thinking needs the chat format: switched to it"); } }
  else if (S.wbPrev) { $("wbTemp").value = S.wbPrev.t; $("wbTopP").value = S.wbPrev.p; $("wbMax").value = S.wbPrev.m; }
});
function liveSplit(raw) {
  // the answer as it streams: thinking shown apart, chat-format debris removed
  let s = raw.replace(/\u200b/g, ""), think = "";
  const close = s.indexOf("</think>");
  if (close >= 0) { think = s.slice(0, close).replace("<think>", ""); s = s.slice(close + 8); }
  else if (s.includes("<think>")) { think = s.replace("<think>", ""); s = ""; }
  s = s.replace(/^\s*(<\|im_start\|>)?\s*assistant\s*:?\s*\n?/i, "").split(/<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|\n(?:user|assistant|system)\s*\n/i)[0];
  return { think: think.trim(), answer: s.trim() };
}
$("wbNew").onclick = () => { S.history = []; S.histories = {}; S.turns = []; $("wbChat").innerHTML = ""; toast("new conversation"); };
$("wbQ").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("wbAsk").click(); });
function askBody(q) {
  return { question: q, packs: [...document.querySelectorAll(".wbp:checked")].map((c) => c.value), mode: $("wbMode").value, template: $("wbTpl").value,
    domain: $("wbDomain").value, max_tokens: +$("wbMax").value, temperature: $("wbTemp").value, top_p: $("wbTopP").value,
    // one conversation per configuration (base, each pack, packs loaded together), in single and compare mode alike:
    // no configuration is ever shown another's answers. "send each message on its own" sends none.
    histories: $("wbTpl").value === "chat" && !$("wbSolo").checked ? S.histories : {},
    system: $("wbSys").value, thinking: $("wbThink").checked, solo: $("wbSolo").checked };
}
// tokens, speed and engine under an answer, with an ⓘ for the rest: the packs and their dials, how many routing
// places the packs' experts took (Polymath engines), the B-slot experts and strength, the sampling settings
function ansStats(a) {
  const f = a.info || {}, parts = [];
  if (a.tokens != null) parts.push(`${a.tokens} tokens`);
  if (a.tokens_per_second) parts.push(`${(+a.tokens_per_second).toFixed(1)} tok/s`);
  else if (a.tokens && a.seconds) parts.push(`${(a.tokens / a.seconds).toFixed(1)} tok/s`);
  if (f.engine) parts.push(`${f.engine}${f.build ? " (" + f.build.toUpperCase() + (f.chain ? ", CUDA chain" : "") + ")" : ""}`);
  const L = [];
  L.push(`Engine: ${f.engine || "?"}${f.build ? " · " + f.build + " build" : ""}${f.chain === true ? " · CUDA chain on" : f.chain === false ? " · CUDA chain off" : ""}`);
  if (f.device) L.push(`Device: ${f.device}`);
  L.push(`Packs: ${(f.packs || []).length ? f.packs.map((p) => `${p.name} (strength ${Math.round((p.strength ?? 1) * 100)}%${p.shift ? ", routing " + (p.shift > 0 ? "+" : "") + p.shift : ""})`).join(", ") : "none (base model)"}`);
  if (f.routings != null) L.push(f.routings ? `Experts overridden: the packs' experts took ${f.won} of ${f.routings} routing places in their layers (${(100 * (f.share || 0)).toFixed(1)}%), each one replacing the base's weakest chosen expert for that token` : "Experts overridden: none (no routing in layers with pack experts)");
  if (f.knowledge_experts != null) L.push(`B-slot: ${f.knowledge_experts} knowledge experts added on every token, strength ${Math.round((f.strength ?? 1) * 100)}% (they add to the base's experts, they replace none)`);
  if (f.budget != null) L.push(`Pack budget: at most ${f.budget} pack experts per token and layer`);
  if (a.prompt_tokens != null) L.push(`Prompt: ${a.prompt_tokens} tokens${a.length_limited ? " · the answer stopped at max tokens" : ""}`);
  if (a.swap_seconds != null) L.push(`Swap: ${Math.round(a.swap_seconds * 1000)} ms · answer: ${a.seconds}s`);
  L.push(`Sampling: temperature ${f.temperature ?? "default"}, top-p ${f.top_p ?? "default"}, max ${f.max_tokens} tokens`);
  return parts.length ? `<span class="muted">${esc(parts.join(" · "))}</span><span class="info" title="${esc(L.join("\n"))}">ⓘ</span>` : "";
}
function ansHTML(a, i, k) {
  return `<div class="ans"><div class="h"><b>${esc(a.label)}</b><span>${a.swap_seconds != null ? "swap " + Math.round(a.swap_seconds * 1000) + " ms · " : ""}${a.seconds ?? ""}s</span></div>
    ${a.thinking ? `<details ${$("wbShowThink").checked ? "" : ""}><summary class="small muted">thinking (${a.thinking.split(/\s+/).length} words)</summary><pre class="small">${esc(a.thinking)}</pre></details>` : ""}
    <div class="md">${esc(a.answer || a.error || "")}</div>
    <div class="row small"><button class="ghost" onclick="wbAnalyse(${k}, ${i})">Analyse</button><button class="ghost" onclick="wbAddTest(${k}, ${i})">add to tests</button><button class="ghost" onclick="wbFlag(${k}, ${i})">flag miss</button><button class="ghost" onclick="wbSendToTest(${k}, ${i})" title="copy this question, and the packs it was asked with, into the Test tab">send to Test</button>
      <span class="grow"></span>${ansStats(a)}</div>
    <div class="analysis"></div></div>`;
}
$("wbAsk").onclick = () => guard(async () => {
  const q = $("wbQ").value.trim(); if (!q) return;
  const body = askBody(q);
  const turn = document.createElement("div"); turn.className = "turn"; turn.innerHTML = `<div class="q">${esc(q)}</div><div class="answers"><div class="ans"><div class="md live muted">answering…</div></div></div>`;
  $("wbChat").appendChild(turn); turn.scrollIntoView(); $("wbQ").value = "";
  let r;
  if ($("wbStream").checked && body.mode === "single") {
    const res = await fetch("/api/workbench/ask/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...body, parallel: undefined }) });
    if (!res.ok) throw new Error(await res.text());
    const rd = res.body.getReader(), dec = new TextDecoder(); let buf = "", live = turn.querySelector(".live");
    for (;;) { const { value, done } = await rd.read(); if (done) break; buf += dec.decode(value, { stream: true });
      const k = buf.indexOf("\x1e"); const v = liveSplit(k >= 0 ? buf.slice(0, k) : buf);
      live.classList.remove("muted");
      live.innerHTML = (v.think && $("wbShowThink").checked ? `<div class="thinklive small muted">${esc(v.think)}</div>` : (v.think ? `<div class="small muted">thinking… ${v.think.split(/\s+/).length} words</div>` : "")) + esc(v.answer);
      turn.scrollIntoView({ block: "end" }); }
    const k = buf.indexOf("\x1e"); const tail = k >= 0 ? JSON.parse(buf.slice(k + 1)) : { error: "no result" };
    if (tail.error) throw new Error(tail.error);
    r = tail.result;
  } else {
    r = await post("/api/workbench/ask", body);
  }
  S.lastAsk = { question: q, answers: r.answers, packs: body.packs };
  S.turns = S.turns || []; const k = S.turns.length; S.turns.push(S.lastAsk);   // each turn's buttons act on that turn
  turn.innerHTML = `<div class="q">${esc(q)}${body.solo && body.template === "chat" ? ` <span class="small muted" title="sent without the conversation history">· sent on its own</span>` : ""}</div>${r.warning ? `<div class="small" style="color:var(--acc2)">${esc(r.warning)}</div>` : ""}<div class="answers">${r.answers.map((a, i) => ansHTML(a, i, k)).join("")}</div>`;
  turn.dataset.turn = k;
  turn.dataset.k = S.history.length;
  if (body.template === "chat" && r.answers.length && !body.solo) {   // each configuration continues its own thread
    for (const a of r.answers) (S.histories[a.label] = S.histories[a.label] || []).push({ role: "user", content: q }, { role: "assistant", content: a.answer || "" });
  }
});
const VERDICT = { correct: "correct", mostly_correct: "mostly correct", partly_correct: "partly correct", wrong: "wrong", cannot_check: "cannot check: the material does not cover it" };
window.wbAnalyse = (k, i) => guard(async () => {
  const t = S.turns[k], a = t.answers[i];
  const box = document.querySelector(`.turn[data-turn="${k}"]`).querySelectorAll(".analysis")[i];
  // a pack's answer is checked against what that pack was built from; the base's against the packs it is compared with
  const packs = a.label === "base" ? (t.packs || []) : a.label.split(" + ");
  box.innerHTML = "<span class='muted small'>checking against the material the pack was built from…</span>";
  const r = await post("/api/workbench/analyse", { question: t.question, answer: a.answer, topic: S.topic || "", packs, model: model() });
  box.innerHTML = `<div class="small muted">checked against: ${esc(r.source || "nothing found (choose a topic at the top)")} · ${r.references} reference lines${r.verdict ? ` · <b>${esc(VERDICT[r.verdict] || r.verdict)}</b>` : ""}</div><div class="htmlview">${safeHTML(r.html)}</div>`;
});
window.wbAddTest = (k, i) => guard(async () => {
  if (!S.topic) throw new Error("choose the topic (top right) whose tests should get this question");
  const t = S.turns[k];
  const exp = prompt("Expected (correct) answer:", (t.answers[i].answer || "").slice(0, 400)); if (exp == null) return;
  await post("/api/workbench/test", { topic: S.topic, question: t.question, expected: exp }); toast("added to the topic's current dataset (tests)");
});
window.wbFlag = (k, i) => guard(async () => {
  if (!S.topic) throw new Error("choose the topic (top right) to attach the note to");
  const t = S.turns[k];
  const note = prompt("What is wrong (which check, which fact)?"); if (note == null) return;
  await post("/api/workbench/flag", { topic: S.topic, question: t.question, answer: t.answers[i].answer, note }); toast("flagged");
});
$("wbJudge").onclick = () => guard(async () => {
  if (!S.lastAsk) throw new Error("ask something first");
  const r = await post("/api/workbench/judge", { ...S.lastAsk, model: model() });
  const d = document.createElement("div"); d.className = "turn htmlview"; d.innerHTML = "<b>Judge</b>" + safeHTML(r.html); $("wbChat").appendChild(d); d.scrollIntoView();
});
$("wbEval").onclick = () => guard(async () => {
  const o = $("wbEvalPack").selectedOptions[0]; if (!o) throw new Error("no pack");
  const body = { pack: o.value, limit: +$("wbEvalN").value, template: $("wbTpl").value === "chat" ? "chat" : $("wbTpl").value, model: model() };
  const ds = $("wbEvalSet").value || o.dataset.ds; if (!ds) throw new Error("choose the dataset whose tests to use");
  body.dataset = ds;
  watch(await post("/api/workbench/evaluate", body), $("wbEvalJob"), (r) => { toast(`base ${r.base_score} → pack ${r.pack_score}`, 10000); showEval(r); wbEvalList(); loadWorkbench(); });
});
async function wbEvalList() {
  const p = $("wbEvalPack").value; if (!p) { $("wbEvalList").innerHTML = ""; return; }
  const es = await get("/api/workbench/evaluations/" + encodeURIComponent(p));
  $("wbEvalList").innerHTML = es.map((e) => `<div class="item"><div class="grow">${esc(e.time)} · ${e.n} tests · base <b>${e.base_score}</b> → pack <b>${e.pack_score}</b>
    <span class="muted">(${e.better ?? "?"} better, ${e.worse ?? "?"} worse)</span></div><button class="ghost" onclick="openEval('${esc(p)}','${e.id}')">open</button></div>`).join("") || "<div class='muted'>none yet</div>";
}
window.openEval = (p, id) => guard(async () => showEval(await get(`/api/workbench/evaluations/${encodeURIComponent(p)}/${id}`)));
function showEval(r) {
  S.evalShown = r; $("wbEvalView").classList.remove("hidden");
  $("wbEvalTitle").textContent = `Evaluation · ${r.pack} · ${r.time} · base ${r.base_score} → pack ${r.pack_score}`;
  renderEval(); $("wbEvalView").scrollIntoView();
}
function renderEval() {
  const r = S.evalShown, f = $("wbEvalFilter").value;
  const rows = (r.rows || []).filter((x) => !f || (f === "pack_miss" && x.pack_verdict !== "match") || x.change === f);
  $("wbEvalTable").innerHTML = `<table class="tbl ed"><tr><th>#</th><th>question / expected</th><th>base</th><th>pack</th><th></th></tr>` + rows.map((x) => `<tr class="ch-${x.change}">
    <td>${x.i + 1}</td><td><div>${esc(x.question)}</div><div class="muted small">expected: ${esc(x.expected)}</div></td>
    <td><span class="vd vd-${esc(x.base_verdict)}">${esc(x.base_verdict)}</span><div class="small">${esc((x.base || "").slice(0, 400))}</div></td>
    <td><span class="vd vd-${esc(x.pack_verdict)}">${esc(x.pack_verdict)}</span><div class="small">${esc((x.pack || "").slice(0, 400))}</div></td>
    <td><button class="ghost small" onclick="evalToChat(${x.i})">ask in chat</button></td></tr>`).join("") + "</table>";
}
$("wbEvalFilter").onchange = renderEval;
$("wbEvalClose").onclick = () => $("wbEvalView").classList.add("hidden");
window.evalToChat = (i) => { const x = S.evalShown.rows.find((r) => r.i === i); $("wbQ").value = x.question; $("wbQ").focus(); toast("question copied to the chat box"); };

// ------------------------------------------------------------------ WORKBENCH: transfer pilot
async function plList() {
  const ps = await get("/api/pilot");
  $("plList").innerHTML = ps.map((p) => `<div class="item"><div class="grow">${esc(p.topic_name || "")} · ${esc(p.started)} ${fmtStatus(p.status)}
    <div class="muted">${p.train_problems ?? "?"} train / ${p.test_problems ?? "?"} held-out (${esc(p.held_out_by || "")})</div></div>
    <button class="ghost" onclick="openPilot('${p.id}')">open</button></div>`).join("") || "<div class='muted'>no pilots yet</div>";
}
$("plRun").onclick = () => guard(async () => {
  const opts = { max_train: +$("plTrain").value, max_test: +$("plTest").value, train: $("plDoTrain").checked, evaluate: $("plDoTrain").checked };
  if (opts.train && !confirm("The pilot trains two packs and evaluates both: this takes a while and uses the GPU. Start?")) return;
  watch(await post(`/api/topics/${needTopic()}/pilot`, { opts, model: model() }), $("plJob"), (r) => { showPilot(r); plList(); });
});
window.openPilot = (id) => guard(async () => showPilot(await get("/api/pilot/" + id)));
function showPilot(r) {
  $("plView").classList.remove("hidden");
  $("plTitle").textContent = `Transfer pilot · ${r.topic_name || ""} · ${r.started}`;
  const a = (r.arms || {}).A || {}, b = (r.arms || {}).B || {}, d = r.discipline || {}, tr = r.traces || {};
  const pct = (x) => x == null ? "—" : Math.round(x * 100) + "%";
  $("plBody").innerHTML = `${r.warning ? `<div class="note warn small">${esc(r.warning)}</div>` : ""}${r.error ? `<div class="note warn small">${esc(r.error)}</div>` : ""}
    <div class="small">${r.train_problems} training problems → <b>${tr.kept ?? "?"}</b> verified traces (${tr.solved ?? 0} solved, ${tr.reconstructed ?? 0} reconstructed, ${tr.dropped ?? 0} dropped${tr.drop_reasons ? ": " + esc(Object.entries(tr.drop_reasons).map(([k, v]) => `${k} ${v}`).join(", ")) : ""}) · anchors: ${esc(tr.set || "")} · held out by ${esc(r.held_out_by || "")}</div>
    ${r.verdict ? `<p><b>${esc(r.verdict)}</b></p>` : ""}
    <table class="tbl"><tr><th></th><th>base</th><th>A · answers only</th><th>B · anchored traces</th></tr>
      <tr><td>held-out conclusions (score)</td><td>${a.eval ? a.eval.base_score : "—"}</td><td>${a.eval ? a.eval.pack_score : "—"}</td><td>${b.eval ? b.eval.pack_score : "—"}</td></tr>
      <tr><td>reasons in anchors on new problems</td><td>${pct((d.base || {}).anchored)}</td><td>${pct((d.A || {}).anchored)}</td><td>${pct((d.B || {}).anchored)}</td></tr>
      <tr><td>invented provisions</td><td>${pct((d.base || {}).invented_provisions)}</td><td>${pct((d.A || {}).invented_provisions)}</td><td>${pct((d.B || {}).invented_provisions)}</td></tr>
      <tr><td>invented cases</td><td>${pct((d.base || {}).invented_cases)}</td><td>${pct((d.A || {}).invented_cases)}</td><td>${pct((d.B || {}).invented_cases)}</td></tr>
      <tr><td>evidence not in the problem</td><td>${pct((d.base || {}).evidence_not_in_problem)}</td><td>${pct((d.A || {}).evidence_not_in_problem)}</td><td>${pct((d.B || {}).evidence_not_in_problem)}</td></tr>
      <tr><td>cites old codes</td><td>${pct((d.base || {}).outdated_codes)}</td><td>${pct((d.A || {}).outdated_codes)}</td><td>${pct((d.B || {}).outdated_codes)}</td></tr>
      <tr><td>datasets</td><td></td><td>${a.dataset ? `<a href="#" onclick="openDataset('${a.dataset}');return false">${a.items} items</a>` : ""}</td><td>${b.dataset ? `<a href="#" onclick="openDataset('${b.dataset}');return false">${b.items} items</a>` : ""}</td></tr>
      <tr><td>packs</td><td></td><td>${esc(a.pack || "—")}</td><td>${esc(b.pack || "—")}</td></tr></table>
    ${d.B ? `<details><summary class="small">B's anchored answers on new problems</summary>${(d.B.rows || []).map((x) => `<div class="item small"><div class="grow"><b>${esc(x.problem)}</b><pre class="small">${esc(x.output)}</pre><div class="muted">${esc((x.problems || []).join(" · ") || "all checks passed")}</div></div></div>`).join("")}</details>` : ""}`;
  $("plView").scrollIntoView();
}
$("plClose").onclick = () => $("plView").classList.add("hidden");

// ------------------------------------------------------------------ JOBS
async function loadJobs() {
  const js = await get("/api/jobs");
  $("jobsList").innerHTML = js.map((j) => `<div class="item ${j.id === S.jobSel ? "sel" : ""}"><div class="grow" onclick="openJob('${j.id}')" style="cursor:pointer">
    <b>${esc(j.kind)}</b> ${esc(j.label || "")} ${fmtStatus(j.status)} · ${Math.round((j.progress || 0) * 100)}%
    <div class="small muted">${new Date((j.started || 0) * 1000).toLocaleString()} · ${j.seconds ?? ""}s · ${esc(((j.log || []).slice(-1)[0] || "").slice(0, 140))}</div></div>
    ${j.status === "running" ? `<button class="ghost" onclick="jobAct('${j.id}','pause')">Pause</button><button class="ghost" onclick="jobAct('${j.id}','cancel')">Cancel</button>`
      : `${j.resumable && ["paused", "error", "cancelled", "interrupted"].includes(j.status) ? `<button class="ghost" onclick="jobAct('${j.id}','resume')">Resume</button>` : ""}<button class="ghost" onclick="jobDel('${j.id}')">✕</button>`}</div>`).join("")
    || "<div class='muted small'>No jobs yet.</div>";
  if (S.jobSel) openJob(S.jobSel, true);
}
window.jobAct = (id, a) => guard(async () => { await post(`/api/jobs/${id}/${a}`); toast(a === "resume" ? "resumed" : a + " requested"); setTimeout(loadJobs, 600); });
window.jobDel = (id) => guard(async () => { await del("/api/jobs/" + id); if (S.jobSel === id) { S.jobSel = null; $("jobDetail").innerHTML = ""; } loadJobs(); });
$("jobsRefresh").onclick = () => guard(loadJobs);
$("jobsClear").onclick = () => guard(async () => { if (!confirm("Delete all finished jobs?")) return; const r = await post("/api/jobs/clear"); toast(`deleted ${r.deleted}`); loadJobs(); });
window.openJob = (id, quiet) => guard(async () => {
  S.jobSel = id;
  const j = await get(`/api/jobs/${id}?full=true`);
  $("jobTitle").textContent = `${j.kind} · ${j.label || ""}`;
  const res = j.result;
  let extra = "";
  if (res && res.rows && res.pack_score != null) extra = `<button onclick="document.querySelector('#tabs button[data-tab=workbench]').click();setTimeout(()=>openEval('${esc(res.pack)}','${res.id}'),400)">Open the evaluation table</button>`;
  if (res && res.dataset) extra += ` <button onclick="openDataset('${res.dataset}')">Open the dataset</button>`;
  if (res && res.pack && !res.rows) extra += ` <span class="small">pack: <b>${esc(res.pack)}</b></span>`;
  $("jobDetail").innerHTML = `<div>${fmtStatus(j.status)} · ${Math.round((j.progress || 0) * 100)}% · ${j.seconds}s · started ${new Date((j.started || 0) * 1000).toLocaleString()}</div>
    ${j.error ? `<div style="color:var(--bad)">${esc(j.error)}</div>` : ""}
    <div class="row">${extra}</div>
    <h4>Inputs</h4><pre class="small">${esc(JSON.stringify(j.inputs || {}, null, 1))}</pre>
    <h4>Result</h4><pre class="small jobres">${esc(JSON.stringify(res, null, 1) || "—").slice(0, 20000)}</pre>
    <h4>Log</h4><div class="joblog tall">${esc((j.log || []).join("\n"))}</div>`;
  if (!quiet) document.querySelectorAll("#jobsList .item").forEach((x) => x.classList.toggle("sel", x.innerHTML.includes(`'${id}'`)));
});

// ------------------------------------------------------------------ SETTINGS
async function loadSettings() {
  const [m, ms] = [await get("/api/settings/meta"), await get("/api/llm/models")];
  const groups = {};
  Object.entries(m.help).forEach(([k, h]) => (groups[h.group] = groups[h.group] || []).push([k, h]));
  const models = (ms.models || []).map((x) => x.id);
  const field = (k, h) => {
    const v = m.values[k];
    let inp;
    if (h.kind === "bool") inp = `<span class="radios"><label><input type="radio" name="set-${k}" data-set="${k}" value="true" ${v ? "checked" : ""}> yes</label><label><input type="radio" name="set-${k}" data-set="${k}" value="false" ${!v ? "checked" : ""}> no</label></span>`;
    else if (h.kind === "gpucpu") inp = `<span class="radios"><label><input type="radio" name="set-${k}" data-set="${k}" value="true" ${v ? "checked" : ""}> GPU</label><label><input type="radio" name="set-${k}" data-set="${k}" value="false" ${!v ? "checked" : ""}> CPU</label></span>`;
    else if (h.kind === "model") inp = `<select data-set="${k}"><option value="">(first loaded model)</option>${[...new Set([v, ...models].filter(Boolean))].map((x) => `<option ${x === v ? "selected" : ""}>${esc(x)}</option>`).join("")}</select>`;
    else if (h.kind === "effort") inp = `<select data-set="${k}">${[["", "model default"], ["low", "low"], ["medium", "medium"], ["high", "high"]].map(([x, l]) => `<option value="${x}" ${x === v ? "selected" : ""}>${l}</option>`).join("")}</select>`;
    else if (h.kind === "choice") inp = `<select data-set="${k}">${(h.options || []).map((x) => `<option ${x === v ? "selected" : ""}>${esc(x)}</option>`).join("")}</select>${k === "llm_preset" ? ' <button type="button" id="llmTest">Test connection</button> <span id="llmTestOut" class="small"></span>' : ""}`;
    else if (h.kind === "secret") { const isSet = m.values[k + "_set"];
      inp = `<input type="password" autocomplete="new-password" data-set="${k}" data-secret="1" value="" placeholder="${isSet ? "set: type a new one to change it" : "not set"}">` +
        (isSet ? ` <label class="small"><input type="checkbox" data-clear="${k}"> clear</label>` : ""); }
    else if (h.kind === "gpu" || h.kind === "gpu_train") {
      const cards = m.cards || [];
      const opts = (h.kind === "gpu_train" ? [["same", "same as the engine"]] : []).concat([["auto", "auto (most free memory)"], ["cpu", "CPU (no GPU)"]],
        cards.map((c) => [String(c.index), `GPU ${c.index}: ${c.name} · ${(c.memory_free_mb / 1024).toFixed(1)} of ${(c.memory_total_mb / 1024).toFixed(1)} GB free`]));
      if (v && !opts.some(([x]) => x === v)) opts.push([v, v]);
      inp = `<select data-set="${k}">${opts.map(([x, l]) => `<option value="${esc(x)}" ${x === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>${cards.length ? "" : ' <span class="small muted">no NVIDIA card found</span>'}`;
    }
    else if (h.kind === "engine") inp = `<select data-set="${k}">${(m.engines || []).map((e) => `<option value="${esc(e.name)}" ${e.name === v ? "selected" : ""} ${e.installed ? "" : "disabled"}>${esc(e.label)}</option>`).join("")}</select>`;
    else if (h.kind === "number") inp = `<input type="number" data-set="${k}" value="${esc(v)}">`;
    else if (h.kind === "float") inp = `<input type="number" step="0.05" data-set="${k}" value="${esc(v)}">`;
    else inp = `<input data-set="${k}" value="${esc(v)}">`;
    return `<div class="setrow"><div><b>${esc(h.label)}</b>${h.restart ? ' <span class="tag context">restart</span>' : ""}<div class="small muted">${esc(h.help)}</div></div><div>${inp}</div></div>`;
  };
  loadEngineModels().catch(() => {});
  $("setForm").innerHTML = `<div class="small muted">Saved in ${esc(m.config_file || "config.toml")}</div>` + Object.entries(groups).map(([g, xs]) => `<h4>${esc(g)}</h4>` + xs.map(([k, h]) => field(k, h)).join("")).join("");
  const pre = document.querySelector('#setForm [data-set="llm_preset"]');
  if (pre) pre.onchange = () => { const u = (m.presets || {})[pre.value]; const url = document.querySelector('#setForm [data-set="llm_url"]'); if (u && u.url && url) url.value = u.url; };
  const tb = $("llmTest");
  if (tb) tb.onclick = () => guard(async () => {
    const r = await get("/api/llm/test");
    $("llmTestOut").textContent = r.ok ? `${r.server}: ${r.models.length} model(s)${r.models.length ? " — " + r.models.slice(0, 4).join(", ") : ""} (save first if you changed the URL)` : `not reachable: ${r.error}`;
  });
  await lmsInfo();
}
$("setSave").onclick = () => guard(async () => {
  const d = {};
  document.querySelectorAll("#setForm [data-set]").forEach((i) => { if (i.type === "radio") { if (i.checked) d[i.dataset.set] = i.value; } else d[i.dataset.set] = i.value; });
  document.querySelectorAll("#setForm [data-clear]").forEach((c) => { if (c.checked) { d[c.dataset.clear] = ""; d["clear_" + c.dataset.clear] = true; } });
  const st = await get("/api/settings");
  if (st.api_token_set && (d.api_token || d.clear_api_token)) {              // changing or clearing it: the current one
    const cur = window.prompt("Changing or clearing the API token needs the current token:");
    if (cur === null) return;
    d.current_api_token = cur;
  }
  const s = await post("/api/settings", d); toast("saved");
  $("par").value = s.parallel;
  $("setNote").textContent = "Settings marked “restart” apply after restarting Packsmith (engine settings: Stop and Start the engine).";
});
async function lmsInfo() {
  const ins = await get("/api/llm/instances");
  const loaded = ins.filter((x) => x.loaded.length);
  $("lmsInfo").innerHTML = loaded.length ? `<table class="tbl"><tr><th>model</th><th>context</th><th>parallel</th></tr>${loaded.flatMap((x) => x.loaded.map((i) => `<tr><td>${esc(i.id)}</td><td>${i.context_length ?? "?"}</td><td>${i.parallel ?? "?"}</td></tr>`)).join("")}</table>
    ${loaded.some((x) => x.loaded.some((i) => i.parallel && +$("par").value > i.parallel)) ? `<div class="note warn small">Parallel (${$("par").value}) is above a model's parallel slots: extra calls queue in the LLM server.</div>` : ""}`
    : `<span class="muted">${ins.length ? "No model loaded in the LLM server." : "The LLM server does not report loaded models (only LM Studio and Ollama do)."}</span>`;
}
$("lmsRefresh").onclick = () => guard(lmsInfo);
$("lmsUnload").onclick = () => guard(async () => { if (!confirm("Unload every model in the LLM server?")) return; const r = await post("/api/llm/unload"); toast("unloaded: " + (r.unloaded.join(", ") || "nothing")); lmsInfo(); });
$("regScan").onclick = () => guard(async () => {
  const rs = await get("/api/packs/scan?folder=" + encodeURIComponent($("regFolder").value));
  $("regList").innerHTML = rs.map((r) => `<div class="item"><div class="grow"><b>${esc(r.name)}</b> <span class="muted">${esc(r.domain || "")} ${r.texts ? "· " + r.texts + " texts" : ""} ${r.created ? "· " + esc(r.created) : ""}</span>
    <div class="muted">${esc(r.path)} · ${esc(r.format || "")}${r.same_base === false ? ' · <b style="color:var(--bad)">different base model</b>' : ""}</div></div>
    ${r.registered_as ? `<span class="small">already registered as <b>${esc(r.registered_as)}</b></span>` : `<button onclick="regPack('${esc(r.path)}','${esc(r.name)}')">Register</button>`}</div>`).join("")
    || "<div class='muted'>No packs found there.</div>";
});
window.regPack = (path, name) => guard(async () => {
  const n = prompt("Register as:", name); if (!n) return;
  const r = await post("/api/packs/register", { name: n, path });
  toast(r.already ? `already registered as ${r.name}` : `registered as ${r.name}`); $("regScan").click();
});

// ------------------------------------------------------------------ ASSISTANT
async function loadAssistant() {
  const [cs, ds] = [await get("/api/assistant"), await get("/api/docs")];
  $("asList").innerHTML = cs.map((c) => `<div class="item ${c.id === S.chat ? "sel" : ""}"><div class="grow" style="cursor:pointer" onclick="openChat('${c.id}')">${esc(c.title)}<div class="muted">${esc(c.updated || "")}</div></div>
    <button class="ghost" onclick="delChat('${c.id}')">✕</button></div>`).join("") || "<div class='muted'>No chats yet.</div>";
  $("asDoc").innerHTML = `<option value="">— a document from the library —</option>` + ds.map((d) => `<option value="${d.id}">${esc(d.name)} (${d.pages} p.)</option>`).join("");
  if (S.chat) await openChat(S.chat);
}
window.delChat = (id) => guard(async () => { await del("/api/assistant/" + id); if (S.chat === id) { S.chat = null; $("asChat").innerHTML = ""; } loadAssistant(); });
$("asNew").onclick = () => guard(async () => { const c = await post("/api/assistant", {}); S.chat = c.id; S.asStarted = false; $("asChat").innerHTML = ""; await loadAssistant(); });
window.openChat = (id) => guard(async () => {
  S.chat = id; const c = await get("/api/assistant/" + id);
  $("asTitle").textContent = c.title;
  S.asStarted = !!c.pipeline;
  $("asChat").innerHTML = ""; c.messages.forEach(asRender);
  asPipe(c.pipeline_view);
});
function asRender(m) {
  const d = document.createElement("div");
  if (m.role === "user") { d.className = "turn"; d.innerHTML = `<div class="q">${esc(m.content)}</div>`; }
  else if (m.role === "assistant") { d.className = "turn"; d.innerHTML = `<div class="ans"><div class="md">${esc(m.content)}</div></div>`; }
  else { let j = {}; try { j = JSON.parse(m.content); } catch (e) { /* plain */ }
    d.className = "toolnote small muted"; d.innerHTML = `⚙ ${esc(j.tool || "action")}${j.result && j.result.error ? ` — <span style="color:var(--bad)">${esc(j.result.error)}</span>` : ""}`;
    if (j.tool === "propose_plan" && j.result && j.result.plan) { $("asChat").appendChild(d); asPlan(j.result.plan); return; } }
  $("asChat").appendChild(d); d.scrollIntoView({ block: "end" });
}
function asPlan(p) {
  const d = document.createElement("div"); d.className = "plancard";
  d.innerHTML = `<b>Plan: ${esc(p.name)}</b><div class="small">${esc(p.goal || "")}</div><ul class="small">
    ${(p.sources || []).map((s) => `<li>document ${esc(s.doc)} pages ${s.start}–${s.end}</li>`).join("")}${(p.curated || []).map((c) => `<li>curated data ${esc(c)}</li>`).join("")}
    <li>dataset: ${esc(p.mode)}${p.playbook ? " · playbook " + esc(p.playbook) : ""} · ${esc(p.preset)}</li>
    ${(p.miners || []).length ? `<li>extra mining: ${esc(p.miners.join(", "))}</li>` : ""}
    <li>${p.train ? `train (${p.epochs} epochs, ${p.max_len} tokens)` : "no training"}${p.evaluate ? " · test" : ""}</li></ul>
    ${S.asStarted ? '<span class="small muted">started</span>' : '<button class="primary">Run</button>'}`;
  const btn = d.querySelector("button");
  if (btn) btn.onclick = () => guard(async () => { btn.disabled = true; S.asStarted = true; await asSend("", "run"); document.querySelectorAll(".plancard button").forEach((b) => b.remove()); });
  $("asChat").appendChild(d); d.scrollIntoView({ block: "end" });
}
function asPipe(v) {
  if (!v || !v.pipeline) { $("asPipe").innerHTML = "No run yet."; return; }
  $("asPipe").innerHTML = `<b>${esc(v.status)}</b> ${v.progress != null ? Math.round(v.progress * 100) + "%" : ""}<div class="prog"><div style="width:${Math.round((v.progress || 0) * 100)}%"></div></div>
    ${v.steps.map((s) => `<div>${s.status === "done" ? "✓" : s.status === "running" ? "▶" : s.status === "skipped" ? "–" : "·"} ${esc(s.name)} <span class="muted">${esc(s.status)}${s.error ? " — " + esc(s.error).slice(0, 160) : ""}</span></div>`).join("")}
    <div class="muted">${esc((v.last || "").slice(0, 200))}</div>`;
  clearTimeout(S.asTimer);
  if (v.status === "running") S.asTimer = setTimeout(() => guard(async () => { if (S.chat) asPipe(await get(`/api/assistant/${S.chat}/status`)); }), 2500);
}
async function asSend(text, action) {
  if (!S.chat) { const c = await post("/api/assistant", {}); S.chat = c.id; }
  const docs = S.asDocs || [];
  if (text) asRender({ role: "user", content: text });
  const wait = document.createElement("div"); wait.className = "toolnote small muted"; wait.textContent = "…"; $("asChat").appendChild(wait);
  try {
    const r = await post(`/api/assistant/${S.chat}/message`, { text, docs, model: model(), action: action || "" });
    wait.remove(); S.asDocs = []; $("asAtt").textContent = "";
    r.messages.filter((m) => !(m.role === "user" && m.content.startsWith(text) && text)).forEach(asRender);
    asPipe(r.pipeline); $("asTitle").textContent = r.title;
  } catch (e) { wait.remove(); throw e; }
  loadAssistantList();
}
async function loadAssistantList() { const cs = await get("/api/assistant"); $("asList").innerHTML = cs.map((c) => `<div class="item ${c.id === S.chat ? "sel" : ""}"><div class="grow" style="cursor:pointer" onclick="openChat('${c.id}')">${esc(c.title)}<div class="muted">${esc(c.updated || "")}</div></div><button class="ghost" onclick="delChat('${c.id}')">✕</button></div>`).join(""); }
$("asSend").onclick = () => guard(async () => { const t = $("asQ").value.trim(); if (!t) return; $("asQ").value = ""; await asSend(t); });
$("asQ").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("asSend").click(); });
$("asDoc").onchange = () => { const v = $("asDoc").value; if (!v) return; S.asDocs = [...new Set([...(S.asDocs || []), v])]; $("asAtt").textContent = `${S.asDocs.length} attached`; $("asDoc").value = ""; };
$("asFile").onchange = () => guard(async () => {
  const f = $("asFile").files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f); fd.append("name", "");
  $("asAtt").textContent = "reading the document…";
  const m = await waitJob(await api("POST", "/api/docs/upload", fd, true), null);
  S.asDocs = [...new Set([...(S.asDocs || []), m.id])]; $("asAtt").textContent = `${S.asDocs.length} attached (${m.name})`; $("asFile").value = ""; loadDocs();
});
["Pause", "Resume", "Cancel"].forEach((a) => $("as" + a).onclick = () => guard(async () => { if (!S.chat) return; await asSend("", a.toLowerCase()); }));

// ------------------------------------------------------------------ ENGINE MODEL (Settings + Workbench)
async function loadEngineModels() {
  const L = await get("/api/engine-models");
  S.engineModels = L;
  const rows = L.containers.map((c) => `<div class="item"><div class="grow"><b>${esc(c.label)}</b> <span class="tag context">${esc(c.engine)}</span>
      <div class="muted">${esc(c.path)}${c.source ? " · from " + esc(c.source) : ""}${c.serving ? " · <b>serving</b>" : ""}${c.training ? " · <b>training</b>" : ""}</div></div>
      ${c.serving && c.training ? "" : `<button onclick="useModel('${esc(c.path)}')">Use</button>`}</div>`);
  const hf = L.hf.map((h) => `<div class="item"><div class="grow"><b>${esc(h.repo)}</b>
      <div class="muted">${esc(h.path)}</div></div>
      <button onclick="hubAskUse('${esc(h.repo)}')">Set up for colibri…</button></div>`);
  $("emList").innerHTML = rows.join("") || "<div class='muted'>No colibri container found.</div>";
  $("emList").innerHTML += `<div class="muted" style="margin-top:.5em">Downloaded Hugging Face models (cache <b>${esc(L.hf_cache)}</b>, then Polymath's models folder):</div>` + (hf.join("") || "<div class='muted'>empty</div>")
    + `<div class="row" style="margin-top:.4em"><input id="emPath" placeholder="or a model folder (Hugging Face or colibri container)" style="flex:1"><button id="emUsePath" class="ghost">Use folder</button><button id="emConvPath" class="ghost">Convert folder</button></div>`;
  $("emUsePath").onclick = () => guard(async () => window.useModel($("emPath").value.trim()));
  $("emConvPath").onclick = () => guard(async () => window.convModel($("emPath").value.trim()));
  if ($("wbModel")) {
    $("wbModel").innerHTML = L.containers.map((c) => `<option value="${esc(c.path)}" ${c.serving ? "selected" : ""}>${esc(c.label)}</option>`).join("");
    $("wbModelNote").innerHTML = "";
  }
}
window.useModel = (path) => guard(async () => {
  if (!path) throw new Error("give a folder");
  const r = await post("/api/engine-models/use", { path, train: true });
  toast("engine model: " + r.path + (r.restarted ? " (engine restarted)" : ""), 8000);
  await loadEngineModels(); if (typeof loadWorkbench === "function") loadWorkbench();
});
window.convModel = (source) => guard(async () => {
  if (!source) throw new Error("give a folder or a cached model");
  if (!confirm(`Convert ${source} for colibri (int4, ~20 GB in the runs folder, 10–30 min), then use it as the engine and training model?`)) return;
  const j = await post("/api/engine-models/convert", { source, use: true });
  watch({ id: j.id }, $("emJob"), (res) => { toast("ready: " + res.path, 8000); loadEngineModels(); });
});
$("emRefresh").onclick = () => guard(loadEngineModels);
// ------------------------------------------------------------------ HUGGING FACE: search, check (MoE only), download, convert
$("hubSearch").onclick = () => guard(async () => {
  const q = $("hubQ").value.trim(); if (!q) throw new Error("type a name to search");
  $("hubList").innerHTML = "<span class='muted'>searching…</span>";
  const rs = await get("/api/hub/search?q=" + encodeURIComponent(q));
  $("hubList").innerHTML = rs.map((r) => `<div class="item"><div class="grow"><b>${esc(r.repo)}</b> ${r.moe_hint ? '<span class="tag context">MoE</span>' : ""}
      <span class="muted">${r.downloads != null ? r.downloads.toLocaleString() + " downloads" : ""} ${esc(r.pipeline || "")}</span></div>
      <button onclick="hubCheck('${esc(r.repo)}')">Check</button></div>`).join("") || "<div class='muted'>nothing found</div>";
});
$("hubCheckBtn").onclick = () => guard(async () => window.hubCheck($("hubQ").value.trim()));
window.hubCheck = (repo) => guard(async () => {
  if (!repo) throw new Error("give a repo id, e.g. Qwen/Qwen3-30B-A3B");
  $("hubInfo").innerHTML = `<span class='muted'>reading ${esc(repo)}/config.json…</span>`;
  const c = await get("/api/hub/check?repo=" + encodeURIComponent(repo));
  const loc = c.local ? (c.local.complete ? `<b>already downloaded</b>: ${esc(c.local.path)}` : `partly downloaded (${esc(c.local.path)}): the download resumes`) : "not on this machine";
  $("hubInfo").innerHTML = `<div class="note ${c.moe ? "" : "warn"}"><b>${esc(c.repo)}</b> · ${c.moe ? `mixture of experts: ${c.experts} experts, ${c.active ?? "?"} active per token` : "<b>not a mixture-of-experts model</b>: Packsmith only downloads MoE models"}
    · ${esc(c.model_type || "")}${c.size_gb ? ` · ${c.size_gb} GB` : ""}${c.gated ? " · gated (needs HF_TOKEN)" : ""}<br>${loc}${c.note ? "<br>" + esc(c.note) : ""}</div>
    ${c.moe ? `<div class="row">${c.local && c.local.complete ? "" : `<button class="primary" onclick="hubDownload('${esc(c.repo)}', 'hf_cache')">Download to the Hugging Face cache</button><button class="ghost" onclick="hubDownload('${esc(c.repo)}', 'models')">Download to Polymath's models folder</button>`}
      ${c.local && c.local.complete && c.convertible ? `<button onclick="hubAskUse('${esc(c.repo)}')">Set up for colibri…</button>` : ""}</div>` : ""}`;
});
window.hubDownload = (repo, where) => guard(async () => {
  const j = await post("/api/hub/download", { repo, where });
  watch({ id: j.id }, $("hubJob"), (res) => { toast("downloaded: " + res.path, 8000); loadEngineModels(); if (res.ask_convert) window.hubAskUse(repo); });
});
window.hubAskUse = (repo) => {
  $("hubInfo").innerHTML = `<div class="note"><b>${esc(repo)}</b> is on this machine. What is it for?
    <div class="small muted">colibri serves an integer container (Qwen3-MoE: experts int4, one scale per 64 weights, ~20 GB for 30B-A3B; OLMoE: int8).
    Training a 30B also loads that container: it fits a 32 GB card and the pack learns on exactly the experts colibri serves.
    Training on the Hugging Face folder (bf16) only suits small models or cards with ~64 GB.</div>
    <div class="row"><button class="primary" onclick="hubUse('${esc(repo)}','both')">Inference and training (convert, recommended)</button>
    <button onclick="hubUse('${esc(repo)}','inference')">Inference only (convert)</button>
    <button class="ghost" onclick="hubUse('${esc(repo)}','training-hf')">Training only, from the Hugging Face folder</button>
    <button class="ghost" onclick="$('hubInfo').innerHTML=''">Not now</button></div></div>`;
};
window.hubUse = (repo, purpose) => guard(async () => {
  const r = await post("/api/engine-models/convert", { source: repo, use: true, purpose });
  if (r.id) watch({ id: r.id }, $("hubJob"), (res) => { toast("ready: " + res.path, 8000); loadEngineModels(); });
  else { toast("training model: " + r.path, 8000); loadEngineModels(); }
  $("hubInfo").innerHTML = "";
});
$("wbModel").onchange = () => guard(async () => { await window.useModel($("wbModel").value); });
