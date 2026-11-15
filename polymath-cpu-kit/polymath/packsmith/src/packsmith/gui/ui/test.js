// Test tab: the same questions to several candidates (engine + packs at set dials), answers side by side.
// Backend: workbench/testbench.py (/api/test/...). Dial values here are for the run only: never the packs' saved dials.
const TB = { opts: null, cands: [], sets: [], settings: null, run: null, saveT: null, job: null };
const tbId = () => Math.random().toString(16).slice(2, 10);
const POLY = ["polymath-engine", "polymath-c", "polymath-c-gpu"];

async function tbLoadTab() {
  TB.opts = await get("/api/test/options");
  const d = await get("/api/test/draft");
  TB.settings = { ...TB.opts.settings, ...(d.settings || {}) };
  TB.cands = (d.candidates && d.candidates.length) ? d.candidates
    : TB.opts.engines.filter((e) => e.installed).map((e) => ({ id: tbId(), engine: e.name, build: e.builds[0] || "auto", packs: [], enabled: false }));
  TB.sets = (d.sets && d.sets.length) ? d.sets : [{ name: "questions", mode: "single", questions: [{ text: "", expected: "" }] }];
  tbRenderCands(); tbRenderSets(); tbRenderSettings(); await tbLoadSets(); await tbLoadRuns();
  $("tbBase").textContent = TB.opts.base_model ? `base model: ${TB.opts.base_model}` : "no base model set (Settings → Paths)";
}
(window.TABLOAD = window.TABLOAD || {}).test = tbLoadTab;

function tbSave() { clearTimeout(TB.saveT); TB.saveT = setTimeout(() => post("/api/test/draft", { candidates: TB.cands.filter((c) => !c.auto), sets: TB.sets, settings: TB.settings }).catch(() => {}), 600); }

// ------------------------------------------------------------------ candidates
function tbEngine(n) { return TB.opts.engines.find((e) => e.name === n) || { builds: ["auto"], installed: false }; }
function tbPack(n) { return TB.opts.packs.find((p) => p.name === n); }
function tbProblems(c) {
  const out = [], e = tbEngine(c.engine);
  if (!e.installed) out.push(`${c.engine} is not installed`);
  if (!(e.builds || []).includes(c.build || "auto")) out.push(`${c.engine} has no ${c.build} build`);
  for (const p of c.packs) {
    const k = tbPack(p.name);
    if (!k) { out.push(`no pack ${p.name}`); continue; }
    const fit = k.fit[c.engine] || {};
    if (fit.level === "refuse") out.push(`${p.name}: ${(fit.problems || []).join("; ")}`);
    if (+p.shift && !(k.kind.routing && POLY.includes(c.engine))) out.push(`${p.name}: ${k.kind.routing ? "routing applies to the Polymath engines only" : k.kind.why}`);
  }
  return out;
}
function tbLabel(c) {
  const b = c.build && c.build !== "auto" ? ` (${c.build})` : "";
  if (!c.packs.length) return `${c.engine}${b} · base`;
  return `${c.engine}${b} · ` + c.packs.map((p) => `${p.name} · ${Math.round(+p.strength * 100)}%${+p.shift ? ` shift ${(+p.shift > 0 ? "+" : "") + p.shift}` : ""}`).join(" + ");
}
// the base on each engine that runs a pack: shown as a reference row, which can be unticked
function tbWithRefs() {
  const user = TB.cands.filter((c) => !c.auto);
  const have = new Set(user.filter((c) => !c.packs.length).map((c) => c.engine + "|" + (c.build || "auto")));
  const kept = new Map(TB.cands.filter((c) => c.auto).map((c) => [c.engine + "|" + c.build, c]));
  const refs = [];
  for (const c of user) {
    const k = c.engine + "|" + (c.build || "auto");
    if (c.enabled && c.packs.length && !have.has(k)) { have.add(k); refs.push(kept.get(k) || { id: tbId(), engine: c.engine, build: c.build || "auto", packs: [], enabled: true, auto: true }); }
  }
  TB.cands = [...refs, ...user];
}
// re-rendered on the next tick: a change event fires while its input is still losing focus
const tbCandsLater = () => setTimeout(tbRenderCands, 0), tbSetsLater = () => setTimeout(tbRenderSets, 0);
function tbRenderCands() {
  tbWithRefs();
  const engOpts = (sel) => TB.opts.engines.map((e) => `<option value="${e.name}" ${e.name === sel ? "selected" : ""} ${e.installed ? "" : "disabled"}>${e.name}${e.installed ? "" : " (not installed)"}</option>`).join("");
  $("tbCands").innerHTML = TB.cands.length ? `<table class="tbl"><tr><th></th><th>engine</th><th>build</th><th>packs (strength 0–150%, routing −3…+3)</th><th>label</th><th></th></tr>` +
    TB.cands.map((c, i) => {
      const pr = tbProblems(c), e = tbEngine(c.engine);
      if (pr.length) c.enabled = false;
      return `<tr data-i="${i}" ${c.auto ? 'class="muted"' : ""}>
        <td><input type="checkbox" class="on" ${c.enabled ? "checked" : ""} ${pr.length ? "disabled" : ""} title="${pr.length ? "cannot run: see the reason" : "run this candidate"}"></td>
        <td><select class="eng" ${c.auto ? "disabled" : ""}>${engOpts(c.engine)}</select></td>
        <td><select class="build" ${c.auto ? "disabled" : ""}>${(e.builds || ["auto"]).map((b) => `<option ${b === c.build ? "selected" : ""}>${b}</option>`).join("")}</select></td>
        <td>${c.packs.map((p, j) => { const k = tbPack(p.name) || { kind: {} }, rt = k.kind.routing && POLY.includes(c.engine);
            return `<div class="row small" data-j="${j}"><b>${esc(p.name)}</b>
              <label>strength <input type="number" class="st" min="0" max="150" step="5" value="${Math.round(+p.strength * 100)}" style="width:5em">%</label>
              <label title="${rt ? "added to the pack's router logits: +1 ≈ e times likelier to take a place" : esc(k.kind.why || "routing applies to Polymath route packs only")}">routing <input type="number" class="sh" min="-3" max="3" step="0.5" value="${+p.shift || 0}" style="width:4.5em" ${rt ? "" : "disabled"}></label>
              <button class="ghost rmp" title="remove this pack">✕</button></div>`; }).join("")}
          ${c.auto ? `<span class="small muted">the base model: a reference for the rows on ${esc(c.engine)}</span>` : `<select class="addp small"><option value="">+ pack…</option>${TB.opts.packs.map((p) => { const f = p.fit[c.engine] || {};
            return `<option value="${esc(p.name)}" title="${esc((f.problems || []).join("; "))}">${esc(p.name)}${f.level === "refuse" ? " (does not fit)" : f.level === "warn" ? " (warning)" : ""}</option>`; }).join("")}</select>`}
          ${pr.length ? `<div class="small" style="color:var(--bad)">${pr.map(esc).join("<br>")}</div>` : ""}</td>
        <td><input class="lab small" value="${esc(c.label || "")}" placeholder="${esc(tbLabel(c))}" ${c.auto ? "disabled" : ""}></td>
        <td>${c.auto ? "" : `<button class="ghost dup" title="duplicate: the same row, to change one value">⧉</button><button class="ghost rm" title="remove">✕</button>`}</td></tr>`;
    }).join("") + "</table>" : `<p class="muted">no engine is installed</p>`;
  $("tbCands").querySelectorAll("tr[data-i]").forEach((tr) => {
    const c = TB.cands[+tr.dataset.i];
    tr.querySelector(".on").onchange = (ev) => { c.enabled = ev.target.checked; tbCandsLater(); tbSave(); };
    const eng = tr.querySelector(".eng"); if (eng) eng.onchange = (ev) => { c.engine = ev.target.value; c.build = tbEngine(c.engine).builds[0] || "auto"; tbCandsLater(); tbSave(); };
    const bu = tr.querySelector(".build"); if (bu) bu.onchange = (ev) => { c.build = ev.target.value; tbCandsLater(); tbSave(); };
    const ap = tr.querySelector(".addp"); if (ap) ap.onchange = (ev) => { if (!ev.target.value) return; const k = tbPack(ev.target.value);
      c.packs.push({ name: ev.target.value, strength: 1, shift: 0 }); c.enabled = true; tbCandsLater(); tbSave(); };
    tr.querySelectorAll("[data-j]").forEach((pd) => { const p = c.packs[+pd.dataset.j];
      pd.querySelector(".st").onchange = (ev) => { p.strength = Math.max(0, Math.min(150, +ev.target.value || 0)) / 100; tbCandsLater(); tbSave(); };
      pd.querySelector(".sh").onchange = (ev) => { p.shift = Math.max(-3, Math.min(3, +ev.target.value || 0)); tbCandsLater(); tbSave(); };
      pd.querySelector(".rmp").onclick = () => { c.packs.splice(+pd.dataset.j, 1); tbCandsLater(); tbSave(); }; });
    const lab = tr.querySelector(".lab"); if (lab) lab.onchange = (ev) => { c.label = ev.target.value.trim(); tbSave(); };
    const dup = tr.querySelector(".dup"); if (dup) dup.onclick = () => { TB.cands.splice(+tr.dataset.i + 1, 0, { ...JSON.parse(JSON.stringify(c)), id: tbId(), label: "" }); tbCandsLater(); tbSave(); };
    const rm = tr.querySelector(".rm"); if (rm) rm.onclick = () => { TB.cands.splice(+tr.dataset.i, 1); tbCandsLater(); tbSave(); };
  });
}
$("tbAdd").onclick = () => { const e = TB.opts.engines.find((x) => x.installed) || TB.opts.engines[0];
  TB.cands.push({ id: tbId(), engine: e.name, build: e.builds[0] || "auto", packs: [], enabled: true }); tbRenderCands(); tbSave(); };

// ------------------------------------------------------------------ questions
function tbRenderSets() {
  $("tbSets").innerHTML = TB.sets.map((s, si) => `<div class="card" data-s="${si}" style="background:var(--bg)">
    <div class="row"><input class="sname" value="${esc(s.name || "")}" style="font-weight:600">
      <label class="inline small" title="Off: every question on its own, no history (one bad answer cannot affect the next). On: the blocks are one conversation; each candidate continues from its own earlier answers."><input type="checkbox" class="conv" ${s.mode === "conversation" ? "checked" : ""}> keep the conversation</label>
      <span class="grow"></span><button class="ghost rms">remove this set</button></div>
    ${s.questions.map((q, qi) => `<div class="row" data-q="${qi}" style="align-items:flex-start"><span class="small muted">${qi + 1}.</span>
      <textarea class="qt" rows="2" style="flex:1" placeholder="a question">${esc(q.text)}</textarea>
      <input class="qe small" value="${esc(q.expected || "")}" placeholder="expected answer (optional; a|b = either)" style="width:16em" title="When given, a whole-word match marks the answer right or wrong, without a judge">
      <span><button class="ghost up" title="up">↑</button><button class="ghost dn" title="down">↓</button><button class="ghost rmq" title="remove">✕</button></span></div>`).join("")}
    <button class="ghost addq">+ question</button></div>`).join("");
  $("tbSets").querySelectorAll("[data-s]").forEach((el) => {
    const s = TB.sets[+el.dataset.s];
    el.querySelector(".sname").onchange = (ev) => { s.name = ev.target.value; tbSave(); };
    el.querySelector(".conv").onchange = (ev) => { s.mode = ev.target.checked ? "conversation" : "single"; tbSave(); };
    el.querySelector(".rms").onclick = () => { TB.sets.splice(+el.dataset.s, 1); if (!TB.sets.length) TB.sets.push({ name: "questions", mode: "single", questions: [{ text: "", expected: "" }] }); tbSetsLater(); tbSave(); };
    el.querySelector(".addq").onclick = () => { s.questions.push({ text: "", expected: "" }); tbSetsLater(); tbSave(); };
    el.querySelectorAll("[data-q]").forEach((r) => { const qi = +r.dataset.q, q = s.questions[qi];
      r.querySelector(".qt").onchange = (ev) => { q.text = ev.target.value; tbSave(); };
      r.querySelector(".qe").onchange = (ev) => { q.expected = ev.target.value; tbSave(); };
      r.querySelector(".up").onclick = () => { if (qi) { [s.questions[qi - 1], s.questions[qi]] = [s.questions[qi], s.questions[qi - 1]]; tbSetsLater(); tbSave(); } };
      r.querySelector(".dn").onclick = () => { if (qi < s.questions.length - 1) { [s.questions[qi + 1], s.questions[qi]] = [s.questions[qi], s.questions[qi + 1]]; tbSetsLater(); tbSave(); } };
      r.querySelector(".rmq").onclick = () => { s.questions.splice(qi, 1); if (!s.questions.length) s.questions.push({ text: "", expected: "" }); tbSetsLater(); tbSave(); }; });
  });
}
$("tbAddSet").onclick = () => { TB.sets.push({ name: `set ${TB.sets.length + 1}`, mode: "single", questions: [{ text: "", expected: "" }] }); tbRenderSets(); tbSave(); };
async function tbLoadSets() {
  const names = await get("/api/test/sets");
  $("tbSetSel").innerHTML = `<option value="">— saved question sets —</option>` + names.map((n) => `<option>${esc(n)}</option>`).join("");
}
$("tbSetSave").onclick = () => guard(async () => { const n = prompt("Save these question sets as:", $("tbSetSel").value || ""); if (!n) return;
  await post("/api/test/sets/" + encodeURIComponent(n), { sets: TB.sets }); await tbLoadSets(); toast("saved"); });
$("tbSetLoad").onclick = () => guard(async () => { const n = $("tbSetSel").value; if (!n) throw new Error("choose a saved set");
  TB.sets = (await get("/api/test/sets/" + encodeURIComponent(n))).sets; tbRenderSets(); tbSave(); });
$("tbSetDel").onclick = () => guard(async () => { const n = $("tbSetSel").value; if (!n || !confirm(`Delete the saved set ${n}?`)) return;
  await api("DELETE", "/api/test/sets/" + encodeURIComponent(n)); await tbLoadSets(); });

// ------------------------------------------------------------------ settings and run
function tbRenderSettings() {
  const s = TB.settings;
  $("tbThink").checked = !!s.thinking; $("tbFmt").value = s.format; $("tbFmt").disabled = !!s.thinking;
  $("tbMax").value = s.max_tokens; $("tbTemp").value = s.temperature; $("tbTopP").value = s.top_p; $("tbRep").value = String(s.repeats);
}
for (const [id, k, f] of [["tbThink", "thinking", (el) => el.checked], ["tbFmt", "format", (el) => el.value], ["tbMax", "max_tokens", (el) => +el.value],
  ["tbTemp", "temperature", (el) => +el.value], ["tbTopP", "top_p", (el) => +el.value], ["tbRep", "repeats", (el) => +el.value]])
  $(id).onchange = (ev) => { TB.settings[k] = f(ev.target); tbRenderSettings(); tbSave(); };
$("tbRun").onclick = () => guard(async () => {
  const cands = TB.cands.filter((c) => c.enabled && !tbProblems(c).length).map((c) => ({ ...c, reference: !!c.auto, label: c.label || tbLabel(c) }));
  if (!cands.length) throw new Error("tick at least one candidate that can run");
  const r = await post("/api/test/runs", { name: $("tbName").value.trim(), candidates: cands, sets: TB.sets, settings: TB.settings, add_references: false });
  toast(`run started: ${r.run.candidates.length} candidates` + (r.run.refused.length ? `, ${r.run.refused.length} refused` : ""));
  TB.run = r.run.id; tbWatch(r.job.id); await tbLoadRuns();
});
function tbWatch(jid) {
  watch({ id: jid }, $("tbJob"), () => { toast("test run finished"); tbShow(TB.run); tbLoadRuns(); }, (v) => { if (TB.run && Math.random() < 0.3) tbShow(TB.run, true); });
}

// ------------------------------------------------------------------ results
async function tbLoadRuns() {
  const runs = await get("/api/test/runs");
  $("tbRuns").innerHTML = `<option value="">— runs —</option>` + runs.map((r) => `<option value="${r.id}" ${r.id === TB.run ? "selected" : ""}>${esc(r.created)} · ${esc(r.name)} · ${r.answers}/${r.expected} answers · ${esc(r.status)}</option>`).join("");
}
$("tbRuns").onchange = () => { TB.run = $("tbRuns").value; if (TB.run) tbShow(TB.run); else $("tbResult").innerHTML = ""; };
const TB_VERDICT = { correct: "correct", mostly_correct: "mostly correct", partly_correct: "partly correct", wrong: "wrong", cannot_check: "cannot check" };
function tbCount(k, n) { return n ? `${k} of ${n}` : "—"; }
async function tbShow(rid, quiet) {
  const r = await get("/api/test/runs/" + rid).catch((e) => { if (!quiet) toast(e.message); return null; });
  if (!r) return;
  const cols = [...r.candidates].sort((a, b) => (a.packs.length ? 1 : 0) - (b.packs.length ? 1 : 0));
  const A = new Map(r.answers.map((a) => [`${a.cand}|${a.set}|${a.q}|${a.rep}`, a]));
  const E = new Map(); for (const x of r.evals) E.set(`${x.cand}|${x.set}|${x.q}|${x.rep}`, x);
  const reps = r.settings.repeats;
  const head = `<div class="row small"><b>${esc(r.name)}</b><span class="muted">${esc(r.created)} · Packsmith ${esc(r.packsmith)} · ${esc(r.base_model.name || "")} · ${r.settings.thinking ? "thinking on" : "thinking off, " + r.settings.format} · max ${r.settings.max_tokens} tokens · temperature ${r.settings.temperature} · top-p ${r.settings.top_p} · repeats ${reps}${(r.judges || []).length ? " · judged by " + r.judges.map(esc).join(", ") : ""}</span>
    <span class="grow"></span><button class="ghost" id="tbEvalAll">Evaluate all</button>${["paused", "running", "created"].includes(r.status) || Object.keys(r.failed || {}).length ? `<button class="ghost" id="tbCont">Continue the run</button>` : ""}<button class="ghost" id="tbDel">Delete</button></div>
    ${r.refused.length ? `<div class="small" style="color:var(--bad)">not run: ${r.refused.map((c) => `${esc(c.label)} (${esc((c.problems || []).join("; "))})`).join(" · ")}</div>` : ""}`;
  const sum = `<table class="tbl"><tr><th>candidate</th><th>answers</th><th>judged correct or mostly</th><th>matches expected</th><th>cut off</th><th>think left open</th><th>repeating</th><th>same as base</th><th>tokens/s</th><th>routing share</th><th>device</th></tr>` +
    cols.map((c) => { const s = r.summary[c.id] || {}, dv = (r.devices || {})[`${c.engine}/${c.build}`] || {};
      return `<tr><td><b>${esc(c.label)}</b>${s.failed ? `<div class="small" style="color:var(--bad)">${esc(s.failed)}</div>` : ""}</td><td>${s.answers ?? 0}</td>
      <td>${tbCount(s.judged_good, s.judged)}</td><td>${tbCount(s.matches_expected, s.with_expected)}</td><td>${s.cut_off ?? ""}</td>
      <td style="${s.think_open ? "color:var(--bad)" : ""}">${s.think_open ?? ""}</td><td style="${s.repeating ? "color:var(--bad)" : ""}">${s.repeating ?? ""}</td>
      <td>${s.same_as_base == null ? "" : s.same_as_base}</td><td>${s.tokens_per_second ?? ""}</td><td>${s.routing_share == null ? "" : (s.routing_share * 100).toFixed(1) + "%"}</td>
      <td class="small muted">${esc(dv.device || "")}</td></tr>`; }).join("") + "</table>";
  let grid = "";
  r.sets.forEach((s, si) => {
    grid += `<h4>${esc(s.name)} ${s.mode === "conversation" ? '<span class="small muted">· one conversation per candidate: read it as a whole</span>' : ""}</h4>
      <div style="overflow-x:auto"><table class="tbl tbgrid"><tr><th style="min-width:14em">question</th>${cols.map((c) => `<th style="min-width:18em">${esc(c.label)}</th>`).join("")}</tr>`;
    s.questions.forEach((q, qi) => {
      grid += `<tr><td><div>${esc(q.text)}</div>${q.expected ? `<div class="small muted">expected: ${esc(q.expected)}</div>` : ""}<button class="ghost small evq" data-s="${si}" data-q="${qi}">Evaluate this question</button></td>`;
      for (const c of cols) {
        const cell = [];
        for (let rp = 0; rp < reps; rp++) {
          const a = A.get(`${c.id}|${si}|${qi}|${rp}`), ev = E.get(`${c.id}|${si}|${qi}|${rp}`);
          if (!a) { cell.push(`<div class="small muted">${r.status === "running" ? "waiting…" : "not answered"}</div>`); continue; }
          const base = cols.find((b) => !b.packs.length && b.engine === c.engine && b.build === c.build);
          const ba = base && c.packs.length ? A.get(`${base.id}|${si}|${qi}|${rp}`) : null;
          const flags = [a.error ? `<span class="badge bad">error: ${esc(a.error)}</span>` : "", a.length_limited ? '<span class="badge">cut off</span>' : "",
            a.think_open ? '<span class="badge bad">think never closed</span>' : "", a.repeating ? '<span class="badge bad">repeating</span>' : "",
            ba && ba.answer === a.answer ? '<span class="badge">same as base</span>' : "",
            "matches_expected" in a ? `<span class="badge ${a.matches_expected ? "ok" : "bad"}">${a.matches_expected ? "matches expected" : "not the expected"}</span>` : "",
            ev ? `<span class="badge ${["correct", "mostly_correct"].includes(ev.verdict) ? "ok" : ev.verdict === "wrong" ? "bad" : ""}" title="judge: ${esc(ev.judge)}">${esc(TB_VERDICT[ev.verdict] || ev.verdict || "no verdict")}</span>` : ""].join(" ");
          cell.push(`<div class="tbans">${reps > 1 ? `<span class="small muted">repeat ${rp + 1}</span> ` : ""}<div class="md">${esc(a.answer)}</div>
            ${a.thinking ? `<details><summary class="small muted">thinking (${a.thinking.split(/\s+/).length} words)</summary><pre class="small">${esc(a.thinking)}</pre></details>` : ""}
            <div class="small">${flags} <span class="muted">${a.tokens ?? ""} tokens · ${a.tokens_per_second ?? ""} tok/s${a.routing_share != null ? ` · routing ${(a.routing_share * 100).toFixed(1)}%` : ""}</span>
            <button class="ghost small evc" data-s="${si}" data-q="${qi}" data-c="${c.id}" title="evaluate this cell">judge</button></div>
            ${ev && ev.html ? `<details><summary class="small muted">the judge's reading</summary><div class="htmlview small">${safeHTML(ev.html)}</div></details>` : ""}</div>`);
        }
        const ag = (r.summary[c.id] || {}).agreement || {}, g = ag[`${si}:${qi}`];
        grid += `<td>${cell.join("<hr>")}${g ? `<div class="small muted">${g.agree} of ${g.of} repeats agree</div>` : ""}</td>`;
      }
      grid += "</tr>";
    });
    grid += "</table></div>";
  });
  $("tbResult").innerHTML = head + sum + grid;
  const ev = (body) => guard(async () => { const j = await post(`/api/test/runs/${rid}/evaluate`, { model: model(), topic: S.topic || "", ...body });
    toast("evaluating (the LLM takes the card in turn with the engine)"); watch(j, $("tbJob"), () => tbShow(rid)); });
  $("tbEvalAll").onclick = () => ev({});
  $("tbResult").querySelectorAll(".evq").forEach((b) => b.onclick = () => ev({ set: +b.dataset.s, q: +b.dataset.q }));
  $("tbResult").querySelectorAll(".evc").forEach((b) => b.onclick = () => ev({ set: +b.dataset.s, q: +b.dataset.q, cand: b.dataset.c }));
  const ct = $("tbCont"); if (ct) ct.onclick = () => guard(async () => { const r2 = await post(`/api/test/runs/${rid}/continue`); tbWatch(r2.job.id); });
  $("tbDel").onclick = () => guard(async () => { if (!confirm("Delete this run and its answers?")) return; await api("DELETE", "/api/test/runs/" + rid); TB.run = ""; $("tbResult").innerHTML = ""; tbLoadRuns(); });
}

// The Workbench's "Send to Test": the question and the packs it was asked with (see more.js ansHTML)
window.wbSendToTest = (k, i) => guard(async () => {
  const t = S.turns[k], a = t.answers[i];
  const packs = a.label === "base" ? [] : a.label.split(" + ");
  await post("/api/test/send", { question: t.question, packs });
  toast("sent to the Test tab");
});
