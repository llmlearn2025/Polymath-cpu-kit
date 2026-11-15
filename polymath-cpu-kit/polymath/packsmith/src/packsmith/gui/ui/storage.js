// Storage tab: everything Polymath has put on this machine, by category; delete what you choose. Backend: lab/storage.py.
const ST = { data: null, sel: new Set(), cat: "" };
const stSize = (b) => b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b ? `${Math.max(1, Math.round(b / 1e3))} kB` : "0";
const stKey = (x) => `${x.root}:${x.path}`;
async function stLoad() {
  ST.data = await get("/api/storage"); ST.sel.clear();
  const D = ST.data;
  $("stTop").innerHTML = `<b>${stSize(D.total)}</b> in all that can be deleted here · ` + D.categories.map((c) => `${esc(c.label)} ${stSize(c.bytes)}`).join(" · ");
  $("stCat").innerHTML = `<option value="">Everything (${stSize(D.total)})</option>` +
    D.categories.map((c) => `<option value="${c.key}" ${c.key === ST.cat ? "selected" : ""}>${esc(c.label)} (${c.count}, ${stSize(c.bytes)})</option>`).join("");
  $("stRoots").innerHTML = "Looked in: " + Object.entries(D.roots).map(([k, v]) => `${esc(k)} ${esc(v)}`).join(" · ") + `<br>${esc(D.note)}`;
  stRender();
}
(window.TABLOAD = window.TABLOAD || {}).storage = stLoad;
function stShown() { return ST.data.items.filter((x) => !ST.cat || x.category === ST.cat); }
function stRender() {
  const it = stShown();
  $("stItems").innerHTML = it.length ? `<table class="tbl"><tr><th></th><th>what</th><th>item</th><th>size</th><th>changed</th></tr>` + it.map((x) => {
    const k = stKey(x), dis = x.in_use ? "disabled" : "";
    return `<tr ${x.in_use ? 'class="muted"' : ""}><td><input type="checkbox" data-k="${esc(k)}" ${ST.sel.has(k) ? "checked" : ""} ${dis} title="${esc(x.in_use ? "in use: " + x.in_use : "")}"></td>
      <td class="small">${esc(x.what)}</td>
      <td>${esc(x.label)}<div class="small muted">${esc(x.where)}${x.note ? " · " + esc(x.note) : ""}${x.in_use ? ` · <b>in use: ${esc(x.in_use)}</b>` : ""}</div></td>
      <td class="small">${stSize(x.bytes)}</td><td class="small muted">${esc(x.modified || "")}</td></tr>`; }).join("") + `</table>`
    : `<p class="small muted">Nothing in this category.</p>`;
  $("stItems").querySelectorAll("[data-k]").forEach((b) => b.onchange = () => { b.checked ? ST.sel.add(b.dataset.k) : ST.sel.delete(b.dataset.k); stCount(); });
  stCount();
}
function stChosen() { return ST.data.items.filter((x) => ST.sel.has(stKey(x))); }
function stCount() {
  const c = stChosen(), t = c.reduce((a, x) => a + x.bytes, 0);
  $("stSel").textContent = c.length ? `${c.length} chosen · ${stSize(t)}` : "";
}
$("stCat").onchange = () => { ST.cat = $("stCat").value; stRender(); };
$("stAll").onclick = () => { stShown().filter((x) => !x.in_use).forEach((x) => ST.sel.add(stKey(x))); stRender(); };
$("stNone").onclick = () => { ST.sel.clear(); stRender(); };
$("stReload").onclick = () => guard(stLoad);
$("stDel").onclick = () => guard(async () => {
  const c = stChosen();
  if (!c.length) return toast("tick what to delete");
  const t = c.reduce((a, x) => a + x.bytes, 0);
  const list = c.slice(0, 12).map((x) => `- ${x.what}: ${x.label} (${stSize(x.bytes)})`).join("\n") + (c.length > 12 ? `\n… and ${c.length - 12} more` : "");
  let confirmText = "";
  if (t > (1 << 30) || c.length > 5) {
    confirmText = prompt(`Delete ${c.length} item(s), ${stSize(t)}? This is final.\n\n${list}\n\nType DELETE to confirm.`) || "";
    if (confirmText !== "DELETE") return toast("not deleted");
  } else if (!confirm(`Delete ${c.length} item(s), ${stSize(t)}? This is final.\n\n${list}`)) return;
  const r = await post("/api/storage/delete", { items: c.map((x) => ({ kind: x.kind, path: x.path, root: x.root })), confirm: confirmText });
  toast(`deleted ${r.deleted.length}, freed ${stSize(r.freed)}`); stLoad();
});
