// User pack tab: squares (sections) with text, uploads and folder paths → statements to check → build → check.
const UP = { types: [], name: "", spec: null, pending: {} };   // pending: section id → File[] not uploaded yet
const upQ = (s) => encodeURIComponent(s);

async function upLoadTab() {
  if (!UP.types.length) {
    UP.types = await get("/api/userpack/types");
    $("upAddType").innerHTML = UP.types.map((t) => `<option value="${t.type}">${esc(t.title)}</option>`).join("");
  }
  const list = await get("/api/userpacks");
  $("upSel").innerHTML = `<option value="">— new —</option>` + list.map((u) =>
    `<option value="${esc(u.name)}" ${u.name === UP.name ? "selected" : ""}>${esc(u.name)} · ${esc(u.status)}</option>`).join("");
  if (UP.name) await upOpen(UP.name);
  else upRender(UP.types.filter((t) => t.starter).map((t) => ({ id: "", type: t.type, title: t.title, about: t.about, text: "", paths: [], files: [] })));
}
(window.TABLOAD = window.TABLOAD || {}).userpack = upLoadTab;

function upType(t) { return UP.types.find((x) => x.type === t) || UP.types.find((x) => x.type === "custom"); }

function upRender(sections) {
  UP.sections = sections;
  $("upGrid").innerHTML = sections.map((s, i) => {
    const t = upType(s.type), pr = t.prompts || {};
    return `<div class="card upsq" data-i="${i}">
      <div class="row"><input class="title" value="${esc(s.title)}" style="font-weight:600;flex:1">
        <span class="tag context">${esc(t.title)}</span><button class="ghost rm" title="Remove this square">✕</button></div>
      <div class="small muted">${esc(s.about || t.about)}</div>
      <textarea class="text" placeholder="Write here, or paste: what the assistant should know or do">${esc(s.text || "")}</textarea>
      <div class="row small"><input type="file" class="files" multiple title="Notes, documents, code files">
        ${(s.files || []).map((f) => `<span class="tag train">${esc(f)}</span>`).join(" ")}</div>
      <input class="paths small" placeholder="folders or files on this machine, comma-separated (an Obsidian vault, a repo)" value="${esc((s.paths || []).join(", "))}">
      <details class="small"><summary>How this is mined</summary>
        <div>Items per statement: ${Object.entries(t.items_per_statement || {}).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}</div>
        <b>1. Statements</b><pre>${esc(pr.extract || "")}</pre><b>2. Training items</b><pre>${esc(pr.generate || "")}</pre></details>
    </div>`;
  }).join("");
  $("upGrid").querySelectorAll(".upsq").forEach((el) => {
    const i = +el.dataset.i;
    el.querySelector(".rm").onclick = () => { upCollect(); UP.sections.splice(i, 1); upRender(UP.sections); };
    el.querySelector(".files").onchange = (e) => { UP.pending[i] = [...e.target.files]; };
  });
}

function upCollect() {
  $("upGrid").querySelectorAll(".upsq").forEach((el) => {
    const s = UP.sections[+el.dataset.i];
    s.title = el.querySelector(".title").value.trim() || s.title;
    s.text = el.querySelector(".text").value;
    s.paths = el.querySelector(".paths").value.split(",").map((x) => x.trim()).filter(Boolean);
  });
  return UP.sections;
}

async function upSave() {
  const name = ($("upName").value || UP.name || "").trim();
  if (!name) throw new Error("give the pack a name");
  const secs = upCollect().map((s) => ({ id: s.id || undefined, type: s.type, title: s.title, about: s.about, text: s.text, paths: s.paths, files: s.files }));
  const mode = document.querySelector('input[name="upMode"]:checked').value;
  const options = { max_items: +$("upMax").value || 0, trim: $("upTrim").checked, chars_per_statement: +$("upCps").value || 240,
    max_statements: +$("upMaxSt").value || 0, window_chars: +$("upWin").value || 0, tune_router: $("upTune").checked, epochs: +$("upEpochs").value || 4, patience: +$("upPat").value, max_len: +$("upLen").value || 768, scale: +$("upScale").value || 1 };
  const spec = await post("/api/userpacks", { name, sections: secs, mode, options });
  for (const [i, files] of Object.entries(UP.pending)) {        // uploads go to the section's id
    const sid = spec.sections[+i].id;
    for (const f of files) {
      const fd = new FormData(); fd.append("file", f);
      await api("POST", `/api/userpacks/${upQ(name)}/sections/${upQ(sid)}/files`, fd, true);
    }
  }
  UP.pending = {}; UP.name = name;
  return name;
}

async function upOpen(name) {
  UP.name = name;
  const st = await get(`/api/userpacks/${upQ(name)}`);
  UP.spec = st.spec;
  $("upName").value = name;
  document.querySelector(`input[name="upMode"][value="${st.spec.mode}"]`).checked = true;
  const o = st.spec.options || {};
  $("upMax").value = o.max_items || 0; $("upTrim").checked = !!o.trim; $("upCps").value = o.chars_per_statement || 240;
  $("upMaxSt").value = o.max_statements || 0; $("upTune").checked = o.tune_router !== false; $("upWin").value = o.window_chars || 0; $("upEpochs").value = o.epochs || 4; $("upPat").value = o.patience ?? 1; $("upLen").value = o.max_len || 768; $("upScale").value = o.scale || 1;
  upRender(st.spec.sections);
  if (Object.keys(st.statements || {}).length) await upStatements();
  upResult(st.build || {});
}

async function upStatements() {
  const [st, plan] = await Promise.all([get(`/api/userpacks/${upQ(UP.name)}/statements`), get(`/api/userpacks/${upQ(UP.name)}/plan`)]);
  $("upStCard").style.display = "";
  const title = Object.fromEntries((UP.spec ? UP.spec.sections : UP.sections).map((s) => [s.id, s.title]));
  const subj = ((UP.spec || {}).options || {}).subject || {};
  $("upSubject").innerHTML = `<b>About:</b> ${subj.name ? `<b>${esc(subj.name)}</b> (${esc(subj.pronoun || "they")}): items ask in the first person ("my …") and about ${esc(subj.name)} by name` : "the person typing (no name found): items ask in the first person"}
    <span class="info" title="Found in your text when the statements were extracted. With a name, the pack is also trained on questions that name the person ('Who is …?', 'Which language does … use?'), so it answers about them by name as well as to 'me'. Change it if it is wrong; leave the name empty for a pack about yourself without your name.">ⓘ</span>
    <input id="upSubjName" value="${esc(subj.name || "")}" placeholder="name (empty: none)" style="width:14em"> <input id="upSubjPr" value="${esc(subj.pronoun || "")}" placeholder="he / she / they" style="width:7em"> <button class="ghost small" id="upSubjSet">set</button>`;
  $("upSubjSet").onclick = () => guard(async () => { await post(`/api/userpacks/${upQ(UP.name)}/subject`, { name: $("upSubjName").value, pronoun: $("upSubjPr").value });
    UP.spec = (await get(`/api/userpacks/${upQ(UP.name)}`)).spec; toast("saved: the items are written this way at the next build"); upStatements(); });
  const covLine = (sec) => (sec.coverage != null ? ` · covers ${Math.round(sec.coverage * 100)}% of the text's passages` : "") +
    (sec.same_text_as ? ` · <b style="color:var(--warn,#b7791f)">${sec.same_text_as.passages} of its ${sec.same_text_as.of} passages are also in "${esc(sec.same_text_as.section)}": their statements are kept there only</b>` : "");
  const gapList = (sec) => (sec.uncovered && sec.uncovered.length) ? `<details class="small"><summary>${sec.uncovered.length} passage(s) still without a statement (add one below if they matter)</summary><ul>${sec.uncovered.map((u) => `<li>${esc(u.slice(0, 300))}</li>`).join("")}</ul></details>` : "";
  $("upSt").innerHTML = Object.entries(st).map(([sid, sec]) => `<h4>${esc(title[sid] || sid)} <span class="small muted">${(sec.items || []).length} found in ${sec.chars || 0} characters${covLine(sec)}</span></h4>` + gapList(sec) +
    (sec.items || []).map((y) => `<div class="upst ${y.grounded < 0.5 ? "ung" : ""}">
      <input type="checkbox" data-s="${esc(sid)}" data-id="${esc(y.id)}" ${y.confirmed ? "checked" : ""} ${y.confirmed === null && y.grounded >= 0.5 ? "checked" : ""}>
      <span class="tag ${y.kind === "dont" ? "exclude" : y.kind === "fact" ? "context" : "train"}">${esc(y.kind)}</span>
      ${y.stable ? "" : '<span class="tag warn" title="kept out of the weights: goes into the profile card">changes</span>'}
      <span class="t" contenteditable="true" data-s="${esc(sid)}" data-id="${esc(y.id)}">${esc(y.text)}</span></div>`).join("") +
    `<div class="row small"><input class="newst" data-s="${esc(sid)}" placeholder="add a statement for this square" style="flex:1"></div>`).join("");
  $("upPlan").innerHTML = `<b>Plan:</b> ${plan.items} training items · ${plan.tests} test probes · ${plan.llm_calls} LLM calls · at most ${plan.epochs_max} epochs ` +
    `(${plan.passes_max} passes, ≈ ${plan.minutes_max} min at most; it stops earlier when it stops improving) · ${plan.volatile} facts to the profile card` +
    (plan.notes.length ? `<br>${plan.notes.map(esc).join("<br>")}` : "") +
    (plan.over_limit ? `<br><b style="color:var(--bad)">Above your item limit: change it in Options (or tick "trim to that limit") before building.</b>` : "");
}

function upEdits() {
  const edits = [];
  $("upSt").querySelectorAll("input[type=checkbox]").forEach((c) => edits.push({ section: c.dataset.s, id: c.dataset.id, confirmed: c.checked }));
  $("upSt").querySelectorAll(".t").forEach((t) => {
    const e = edits.find((x) => x.id === t.dataset.id && x.section === t.dataset.s);
    if (e && t.dataset.orig !== undefined && t.textContent.trim() !== t.dataset.orig) e.text = t.textContent.trim();
  });
  $("upSt").querySelectorAll(".newst").forEach((n) => { if (n.value.trim()) edits.push({ section: n.dataset.s, text: n.value.trim() }); });
  return edits;
}

function upResult(b) {
  if (!b || !b.path) { $("upResCard").style.display = "none"; return; }
  $("upResCard").style.display = "";
  const parts = b.parts ? Object.keys(b.parts) : [];
  $("upRes").innerHTML = `<b>${esc(b.pack)}</b> · ${esc(b.status)}${parts.length > 1 ? ` · stack of ${parts.length}: ${parts.map(esc).join(", ")}` : ""}`;
  const tu = b.tuned;
  const tuneHtml = tu ? (tu.ok ? `<div>Router tuned (calibrate): share of routings each square's pack wins, before → after<br>` +
    Object.keys(tu.after || {}).map((k) => `${esc(k)}: ${(tu.before[k] || []).map((x) => (x * 100).toFixed(1) + "%").join(" / ")} → ` +
      `${tu.after[k].map((x) => (x * 100).toFixed(1) + "%").join(" / ")}`).join("<br>") +
      (tu.weak && tu.weak.length ? `<br><b style="color:var(--bad)">Weak after tuning: ${tu.weak.map((w) => `${esc(w.pack)} wins ${(w.share * 100).toFixed(1)}% of its own text (target ${(w.target * 100).toFixed(0)}%)`).join("; ")}. The general-text cap holds it down: raise it (Options) and rebuild, or untick the tuning.</b>` : "") + `</div>`
    : `<div style="color:var(--bad)">Router tuning failed (the stack works untuned): ${esc(tu.error || "")}</div>`) : "";
  const v = b.validation;
  $("upVal").innerHTML = tuneHtml + (v ? `Check: base ${v.base_score}/${v.probes}, pack ${v.pack_score}/${v.probes} · ` +
    Object.entries(v.by_section || {}).map(([s, x]) => `${esc(s)} ${x.base}→${x.pack}/${x.n}`).join(" · ") +
    (v.dials && (v.dials.strength !== 1 || v.dials.shift !== 0) ? ` · the pack ran at its Workbench setting (strength ${Math.round(v.dials.strength * 100)}%${v.dials.shift ? ", routing " + (v.dials.shift > 0 ? "+" : "") + v.dials.shift : ""})` : "") +
    (v.curve ? `<div>Strength curve (keypoints found, of ${v.probes}): ` + v.curve.map((p) => `${p.strength ? Math.round(p.strength * 100) + "%" : "base"} <b>${p.score}</b>`).join(" · ") +
      `<br><span class="muted">Where the score stops rising is about where the pack has its effect; above that, read the answers for fluency before raising it. Set the strength in Workbench → Packs.</span></div>` : "") : "");
  get(`/api/userpacks/${upQ(UP.name)}/profile`).then((t) => { $("upProfile").textContent = t; }).catch(() => {});
}

function upWatch(j, done) {
  $("upJobCard").style.display = "";
  watch(j, $("upJob"), done);
}

$("upSel").onchange = () => guard(async () => { if ($("upSel").value) await upOpen($("upSel").value); else { UP.name = ""; $("upName").value = ""; $("upStCard").style.display = "none"; $("upResCard").style.display = "none"; await upLoadTab(); } });
$("upAdd").onclick = () => { upCollect(); const t = upType($("upAddType").value);
  UP.sections.push({ id: "", type: t.type, title: $("upAddTitle").value.trim() || t.title, about: t.type === "custom" ? ($("upAddTitle").value.trim() || "") : t.about, text: "", paths: [], files: [] });
  $("upAddTitle").value = ""; upRender(UP.sections); };
$("upCreate").onclick = () => guard(async () => {
  const name = await upSave();
  const j = await post(`/api/userpacks/${upQ(name)}/create`, { confirm: $("upAuto").checked ? "auto" : "review" });
  toast($("upAuto").checked ? "building the pack" : "reading your squares");
  upWatch(j, async () => { await upOpen(name); await upLoadTab(); });
});
$("upAll").onclick = () => $("upSt").querySelectorAll(".upst:not(.ung) input[type=checkbox]").forEach((c) => c.checked = true);
$("upSt") && $("upSt").addEventListener("focusin", (e) => { if (e.target.classList.contains("t") && e.target.dataset.orig === undefined) e.target.dataset.orig = e.target.textContent.trim(); });
$("upBuild").onclick = () => guard(async () => {
  await upSave();                                              // the Options as they are now (limits, depth, epochs)
  await post(`/api/userpacks/${upQ(UP.name)}/statements`, { edits: upEdits() });
  await upStatements();
  const plan = await get(`/api/userpacks/${upQ(UP.name)}/plan`);
  if (plan.over_limit) { toast("above your item limit: change it in Options, or tick 'trim to that limit'", 8000); return; }
  const j = await post(`/api/userpacks/${upQ(UP.name)}/build`, {});
  toast("building: items, then training (see Progress)");
  upWatch(j, async () => { await upLoadTab(); toast("pack ready"); });
});
$("upLoad").onclick = () => guard(async () => {
  const st = await get(`/api/userpacks/${upQ(UP.name)}`);
  const r = await post("/api/agent/load", { packs: [st.build.pack] });
  toast("active: " + r.active.join(", "));
});
$("upValidate").onclick = () => guard(async () => {
  const j = await post(`/api/userpacks/${upQ(UP.name)}/validate`, { curve: $("upCurve").checked });
  upWatch(j, async (v) => { await upOpen(UP.name); toast(`check: base ${v.base_score}/${v.probes}, pack ${v.pack_score}/${v.probes}`); });
});
