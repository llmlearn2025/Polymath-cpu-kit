// Train Lab tab: ways of training a pack tried on the same data, compared by what the packs answer.
// Backend: lab/lab.py (/api/lab/...). Nothing here changes the Train tab or its defaults.
const LB = { opts: null, pick: new Set(["baseline"]), run: null, data: null, estT: null };

async function lbLoadTab() {
  LB.opts = await get("/api/lab/options");
  $("lbBase").textContent = LB.opts.model ? `base model: ${LB.opts.model} (Settings → Paths → base model for training)` : "no base model for training set (Settings → Paths)";
  $("lbData").innerHTML = LB.opts.datasets.map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  $("lrSrc").innerHTML = LB.opts.datasets.filter((d) => !d.reasoning).map((d) => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  lbRenderRecipes(); lbRenderEdit(); lbEstimate(); await lbLoadRuns(); lbReasoningList(); lbTestPacks();
}
(window.TABLOAD = window.TABLOAD || {}).lab = lbLoadTab;

// ------------------------------------------------------------------ recipes
function lbRenderRecipes() {
  $("lbRecipes").innerHTML = `<table class="tbl"><tr><th></th><th>recipe</th><th>what differs from the baseline</th><th></th></tr>` +
    LB.opts.recipes.map((r) => `<tr><td><input type="checkbox" data-r="${esc(r.name)}" ${LB.pick.has(r.name) ? "checked" : ""} ${r.kind === "baseline" ? "disabled" : ""}></td>
      <td><b>${esc(r.name)}</b><div class="small muted">${esc(r.about || "")}</div></td>
      <td class="small" ${Object.keys(r.differs).length > 1 ? 'style="color:var(--bad)"' : ""}>${esc(r.summary)}</td>
      <td>${r.editable ? `<button class="ghost small" data-del="${esc(r.name)}">delete</button>` : `<span class="small muted">${r.kind}</span>`}</td></tr>`).join("") + "</table>";
  $("lbRecipes").querySelectorAll("input[data-r]").forEach((b) => b.onchange = () => { b.checked ? LB.pick.add(b.dataset.r) : LB.pick.delete(b.dataset.r); lbEstimate(); });
  $("lbRecipes").querySelectorAll("button[data-del]").forEach((b) => b.onclick = () => guard(async () => {
    await del("/api/lab/recipes/" + encodeURIComponent(b.dataset.del)); LB.pick.delete(b.dataset.del); await lbLoadTab();
  }));
}
function lbBaseline() { return LB.opts.recipes.find((r) => r.kind === "baseline").settings; }
function lbRenderEdit() {
  const b = lbBaseline(), F = LB.opts.fields;
  $("lbEdit").innerHTML = `<div class="tbgrid">` + Object.entries(F).map(([k, f]) => {
    const v = b[k];
    let inp;
    if (f.kind.startsWith("choice:")) inp = `<select data-f="${k}">${f.kind.slice(7).split(",").map((o) => `<option ${o === String(v) ? "selected" : ""}>${o}</option>`).join("")}</select>`;
    else if (f.kind === "bool") inp = `<input type="checkbox" data-f="${k}" ${v ? "checked" : ""}>`;
    else inp = `<input data-f="${k}" value="${esc(v)}" style="width:7em">`;
    return `<label title="${esc(f.help)}">${esc(f.label)} ${inp}</label>`;
  }).join("") + `</div>`;
  $("lbEdit").querySelectorAll("[data-f]").forEach((e) => e.onchange = lbDiff);
  lbDiff();
}
function lbEdited() {
  const st = {};
  $("lbEdit").querySelectorAll("[data-f]").forEach((e) => { st[e.dataset.f] = e.type === "checkbox" ? e.checked : e.value; });
  return st;
}
function lbDiff() {
  const b = lbBaseline(), st = lbEdited(), F = LB.opts.fields;
  const d = Object.keys(F).filter((k) => k !== "save_epochs" && String(st[k]) !== String(b[k]));
  $("lbRDiff").textContent = d.length ? `differs in ${d.map((k) => F[k].label).join(", ")}` + (d.length > 1 ? " — more than one setting: the cause of a change cannot be told apart" : "") : "the same as the baseline";
}
$("lbRSave").onclick = () => guard(async () => {
  const n = $("lbRName").value.trim();
  if (!n) return toast("a name for the recipe");
  const r = await post("/api/lab/recipes/" + encodeURIComponent(n), { settings: lbEdited() });
  LB.pick.add(r.name); toast("saved: " + r.summary); await lbLoadTab();
});

// ------------------------------------------------------------------ run
const lbHM = (s) => s == null ? "—" : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
function lbEstimate() {
  clearTimeout(LB.estT);
  LB.estT = setTimeout(() => guard(async () => {
    const e = await post("/api/lab/estimate", { dataset: $("lbData").value, recipes: [...LB.pick] });
    LB.est = e;
    $("lbEst").innerHTML = e.recipes.map((r) => `${esc(r.name)}: ${r.texts} texts × ${r.epochs} epochs = ${r.steps} steps${r.seconds ? ` ≈ ${lbHM(r.seconds)}` : ""}`).join(" · ") +
      `<br><b>${e.seconds ? `about ${lbHM(e.seconds)} of training in all` : esc(e.note)}</b>, plus scoring` + (e.seconds_per_step ? ` (measured ${e.seconds_per_step} s/step on this base model)` : "");
  }), 300);
}
$("lbData").onchange = lbEstimate;
$("lbRun").onclick = () => guard(async () => {
  if (LB.est && LB.est.seconds > 7200 && !confirm(`About ${lbHM(LB.est.seconds)} of training. Start?`)) return;
  const r = await post("/api/lab/runs", { dataset: $("lbData").value, engine: $("lbEngine").value, name: $("lbName").value, recipes: [...LB.pick].filter((n) => n !== "baseline") });
  LB.run = r.run.id; await lbLoadRuns(); lbWatch(r.job.id);
});
function lbWatch(jid) {
  watch({ id: jid }, $("lbJob"), () => { toast("lab run finished"); lbShow(LB.run); lbLoadRuns(); }, () => { if (LB.run && Math.random() < 0.2) lbShow(LB.run); });
}

// ------------------------------------------------------------------ results
async function lbLoadRuns() {
  const runs = await get("/api/lab/runs");
  $("lbRuns").innerHTML = `<option value="">— runs —</option>` + runs.map((r) => `<option value="${esc(r.id)}" ${r.id === LB.run ? "selected" : ""}>${esc(r.created)} · ${esc(r.name)} · ${esc(r.dataset)} · ${esc(r.model)} · ${r.done}/${r.recipes} recipes · ${esc(r.status)} · ${(r.bytes / 1e6).toFixed(0)} MB</option>`).join("");
}
$("lbRuns").onchange = () => { LB.run = $("lbRuns").value; if (LB.run) lbShow(LB.run); else $("lbResult").innerHTML = ""; };
$("lbCont").onclick = () => guard(async () => { if (!LB.run) return; const r = await post(`/api/lab/runs/${LB.run}/continue`, {}); lbWatch(r.job.id); });
$("lbDel").onclick = () => guard(async () => {
  if (!LB.run || !confirm("Delete this run, its packs and scores? This cannot be undone.")) return;
  const r = await del(`/api/lab/runs/${LB.run}`); toast(`freed ${(r.freed / 1e6).toFixed(0)} MB`); LB.run = null; $("lbResult").innerHTML = ""; await lbLoadRuns();
});
$("lbMode").onchange = () => LB.data && lbRender();

async function lbShow(rid) {
  LB.data = await get(`/api/lab/runs/${rid}`);
  const m = $("lbMode").value;
  $("lbMode").innerHTML = LB.data.modes.map((x) => `<option ${x === m ? "selected" : ""}>${x}</option>`).join("");
  lbRender();
}
function lbCell(c) {
  if (!c) return `<td class="muted">—</td>`;
  const col = c.mark === "better" ? "var(--ok)" : c.mark === "worse" ? "var(--bad)" : "";
  const p = (x) => Math.round(x * 100);
  return `<td ${col ? `style="color:${col};font-weight:600"` : ""} title="${c.k} of ${c.n}; 95% interval ${p(c.ci[0])}% to ${p(c.ci[1])}%${c.mark ? "; " + c.mark + " than the baseline (intervals apart)" : ""}">${p(c.p)}% <span class="small muted">(${c.k} of ${c.n}, ${p(c.ci[0])} to ${p(c.ci[1])})</span></td>`;
}
function lbRender() {
  const D = LB.data, m = $("lbMode").value || D.modes[0], t = D.table[m];
  if (!t) { $("lbResult").innerHTML = `<p class="small muted">not scored yet</p>`; return; }
  const head = `<tr><th>measure</th><th>base model (floor)</th>${D.rows.map((r) => `<th title="${esc(r.summary)}">${esc(r.name)}</th>`).join("")}</tr>`;
  const body = D.measures.map((ms) => `<tr><td>${esc(ms.label)}${ms.lower_better ? ' <span class="small muted">(lower is better)</span>' : ""}</td>${lbCell(t.floor[ms.key])}${t.rows.map((c) => lbCell(c[ms.key])).join("")}</tr>`).join("");
  const info = `<tr><td class="small">kept epoch · steps · training time · pack size</td><td></td>${D.rows.map((r, i) => `<td class="small">${r.trained ? `${r.card.best_epoch ?? "—"} of ${(r.card.epochs || []).length} · ${r.card.steps} · ${lbHM(r.card.train_seconds)} · ${(r.pack_bytes / 1e6).toFixed(0)} MB<br><button class="ghost small" onclick="guard(() => lbToLibrary(${i}))">Add to library (test_lab)</button>` : "not trained yet"}</td>`).join("")}</tr>`;
  $("lbResult").innerHTML = `<p>${D.summary.map(esc).join(" ")}</p>` + (D.run.note ? `<p class="small">${esc(D.run.note)}</p>` : "") +
    (D.run.sample ? `<p class="small" style="color:var(--bad)">A trial on a sample (${D.run.sample.train} items, ${D.run.sample.tests} tests): not comparable with full runs.</p>` : "") +
    `<div style="overflow-x:auto"><table class="tbl">${head}${body}${info}</table></div>` +
    `<h4>Per epoch</h4><div class="small muted">Held-out loss and the quick sample (questions never in the final tests) after every epoch; with "score every epoch", each epoch's copy scored in full.</div>` +
    D.rows.map((r) => lbCurve(r, m)).join("");
}
// one small chart per recipe: held-out loss (line) and quick-sample score (dots) by epoch; the kept epoch marked
function lbCurve(r, m) {
  const eps = (r.card && r.card.epochs) || [];
  if (!eps.length) return "";
  const W = 420, H = 120, P = 28, n = eps.length;
  const x = (i) => P + (n === 1 ? 0 : (i * (W - 2 * P)) / (n - 1));
  const L = eps.map((e) => e.held_loss ?? e.train_loss), lo = Math.min(...L), hi = Math.max(...L) || 1;
  const yL = (v) => H - P - ((v - lo) / ((hi - lo) || 1)) * (H - 2 * P);
  const yQ = (v) => H - P - v * (H - 2 * P);
  const line = L.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${yL(v).toFixed(1)}`).join("");
  const q = eps.map((e, i) => e.quick && e.quick.n ? `<circle cx="${x(i)}" cy="${yQ(e.quick.right / e.quick.n)}" r="3" fill="var(--acc)"><title>epoch ${e.epoch}: quick sample ${e.quick.right} of ${e.quick.n}</title></circle>` : "").join("");
  const full = Object.entries(r.epoch_scores || {}).map(([k, v]) => {
    const i = parseInt(k.slice(1), 10) - 1, d = (v[m] || {}).detail || {}, f = d.fact_heldout || d.dataset_test || d.fact_seen;
    return f && f.n ? `<rect x="${x(i) - 3}" y="${yQ(f.k / f.n) - 3}" width="6" height="6" fill="var(--ok)"><title>epoch ${i + 1}: ${f.k} of ${f.n} (full score)</title></rect>` : "";
  }).join("");
  const kept = r.card.best_epoch ? `<line x1="${x(r.card.best_epoch - 1)}" x2="${x(r.card.best_epoch - 1)}" y1="${P / 2}" y2="${H - P}" stroke="var(--muted)" stroke-dasharray="3,3"><title>kept epoch ${r.card.best_epoch}</title></line>` : "";
  return `<div class="small"><b>${esc(r.name)}</b> — line: held-out loss (${lo.toFixed(3)} to ${hi.toFixed(3)}); dots: quick sample; squares: full score of the epoch copy; dashed: kept epoch</div>
    <svg width="${W}" height="${H}" style="max-width:100%;border:1px solid var(--line)">${kept}<path d="${line}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>${q}${full}
    <text x="${P}" y="${H - 8}" font-size="10" fill="var(--muted)">epoch 1</text><text x="${W - P - 30}" y="${H - 8}" font-size="10" fill="var(--muted)">epoch ${n}</text></svg>`;
}


// ------------------------------------------------------------------ reasoning datasets (lab/reasoning.py)
$("lrMake").onclick = () => guard(async () => {
  const b = { dataset: $("lrSrc").value, name: $("lrName").value, steps: $("lrSteps").value, worked: $("lrA").checked,
    passages: $("lrB").checked, chains: $("lrD").checked, constructed: $("lrC").checked, reasoning_share: +$("lrShare").value,
    test_share: +$("lrTest").value, max_len: +$("lrLen").value, long_items: $("lrLong").value, think_blocks: $("lrThink").checked,
    reasoning_only: $("lrOnly").checked, model: $("model") ? $("model").value : "" };
  if (b.constructed && !confirm("Constructed situations are invented (marked 'constructed' in every item). Make them?")) return;
  const j = await post("/api/lab/reasoning", b);
  watch(j, $("lrJob"), async () => { toast("reasoning dataset ready"); await lbLoadTab(); });
});
async function lbReasoningList() {
  const r = await get("/api/lab/reasoning");
  if (!r.datasets.length) { $("lrList").innerHTML = `<p class="small muted">No reasoning dataset yet.</p>`; return; }
  const src = (s) => ({ A: "worked", B: "passages", C: "constructed", D: "linked facts" }[s] || s);
  $("lrList").innerHTML = `<table class="tbl"><tr><th>dataset</th><th>items</th><th>by source (kept of tried · trained · tests)</th><th>dropped</th><th>length</th><th></th></tr>` +
    r.datasets.map((m) => { const p = m.report, li = p.long_items || {};
      return `<tr><td><b>${esc(m.name)}</b><div class="small muted">${esc(m.created)} · from ${esc(m.from_name)} · steps: ${esc(p.steps)}${m.options.think_blocks ? " · reasoning blocks" : ""}${m.options.reasoning_only ? " · reasoning only" : ""}</div></td>
      <td class="small">${p.train} (${p.reasoning_items} reasoning, ${p.think_items} thinking-block, share ${Math.round(p.reasoning_share * 100)}%) · ${p.tests} reasoning tests</td>
      <td class="small">${Object.entries(p.sources).map(([k, v]) => `${src(k)}: ${v.kept} of ${v.tried} · ${v.train_items} · ${v.tests}`).join("<br>")}</td>
      <td class="small" title="${esc(Object.entries(p.drop_reasons).map(([k, v]) => `${k}: ${v}`).join("\n"))}">${p.dropped}</td>
      <td class="small">${li.split ? `${li.split} split · ` : ""}${li.dropped ? `${li.dropped} left out · ` : ""}${li.needed_length > li.training_length ? `raised to ${li.needed_length}` : `${li.training_length}`}</td>
      <td><button class="ghost small" data-rv="${esc(m.id)}">view</button> <button class="ghost small" data-rd="${esc(m.id)}">delete</button></td></tr>`; }).join("") + `</table><div id="lrView"></div>`;
  $("lrList").querySelectorAll("[data-rd]").forEach((b) => b.onclick = () => guard(async () => {
    if (!confirm("Delete this reasoning dataset? This cannot be undone.")) return;
    await del("/api/lab/reasoning/" + b.dataset.rd); await lbLoadTab();
  }));
  $("lrList").querySelectorAll("[data-rv]").forEach((b) => b.onclick = () => guard(async () => {
    const m = await get("/api/lab/reasoning/" + b.dataset.rv);
    $("lrView").innerHTML = `<h4>Sample items</h4>` + m.sample.map((x) => `<pre class="small" style="white-space:pre-wrap">${esc(x.text)}</pre>`).join("") +
      `<h4>Dropped (${m.dropped_rows.length})</h4><table class="tbl small">` + m.dropped_rows.map((x) => `<tr><td>${esc(src(x.src))}</td><td>${esc(x.where || "")}</td><td>${esc(x.question || x.passage || "")}</td><td>${esc((x.why || []).join("; "))}</td></tr>`).join("") + `</table>`;
  }));
}

// ------------------------------------------------------------------ test packs
async function lbTestPacks() {
  const ps = await get("/api/lab/testpacks");
  $("lbTestPacks").innerHTML = ps.length ? `<table class="tbl">` + ps.map((p) => `<tr><td><b>${esc(p.name)}</b><div class="small muted">${esc(p.lab.run_name || "")} · ${esc(p.lab.recipe || "")} · ${esc(p.lab.dataset || "")}</div></td>
      <td><button class="ghost small" data-pp="${esc(p.name)}">Promote</button> <button class="ghost small" data-pd="${esc(p.name)}">Delete</button></td></tr>`).join("") + `</table>`
    : `<p class="small muted">None. "Add to library" under a run's results adds one.</p>`;
  $("lbTestPacks").querySelectorAll("[data-pp]").forEach((b) => b.onclick = () => guard(async () => {
    const n = prompt("Name of the normal pack:", b.dataset.pp.replace(/-test_lab$/, ""));
    if (!n) return;
    const r = await post(`/api/lab/testpacks/${encodeURIComponent(b.dataset.pp)}/promote`, { name: n }); toast("promoted: " + r.name); lbTestPacks();
  }));
  $("lbTestPacks").querySelectorAll("[data-pd]").forEach((b) => b.onclick = () => guard(async () => {
    if (!confirm(`Delete ${b.dataset.pd}? This is final.`)) return;
    const r = await del(`/api/lab/testpacks/${encodeURIComponent(b.dataset.pd)}`); toast(`freed ${(r.freed / 1e6).toFixed(0)} MB`); lbTestPacks();
  }));
}
async function lbToLibrary(i) {
  const n = prompt("Name in the library (the mark -test_lab is added):", "");
  if (n === null) return;
  const r = await post(`/api/lab/runs/${LB.run}/recipes/${i}/library`, { name: n }); toast("added: " + r.name); lbTestPacks();
}
