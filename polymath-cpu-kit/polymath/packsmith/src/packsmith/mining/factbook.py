"""The fact book: curated data (other apps' training data and notes) plus the facts captured from the topic's
documents, made consistent and accurate WITHOUT leaving anything out.

Every input unit ends in exactly one ledger state — entry, merged (into an entry, its wording kept), conflict (all
values kept, shown side by side until resolved), flagged (failed a check against its own source), empty — and the
ledger must account for every unit. The fact book (HTML + JSONL) shows all of it; training uses entries and resolved
conflicts.

Two uses of curated data, both on by default:
  import    the records themselves as training items (Q&A with their reasoning, facts, notes, rows), verbatim
  generate  new training data written from them: rephrasings, questions, scenes, subject digests
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict

from packsmith.datasets import curated, prompts
from packsmith.mining import facts, htmlconv
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.system import par

DEFAULTS = {"use": "both", "names": "", "names_in_text": False, "verify": False, "include_flagged": False,
            "digest": True, "max_len": 1024, **{k: facts.DEFAULTS[k] for k in ("rephrase", "qa", "scenes", "test_share",
                                                                                  "max_tests", "batch")}}

_W = re.compile(r"[a-zऀ-ॿ]{3,}")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s%.]", " ", s)).strip()


def words(s: str) -> set[str]:
    return set(_W.findall(norm(s)))


def jacc(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else (1.0 if norm(a) == norm(b) else 0.0)


def same_values(a: str, b: str) -> bool:
    return facts._nums(a) == facts._nums(b)


def names_map(text: str) -> dict:
    """'alias = canonical' per line → {norm(alias): canonical}."""
    out = {}
    for line in (text or "").splitlines():
        if "=" in line:
            a, c = (x.strip() for x in line.split("=", 1))
            if a and c:
                out[norm(a)] = c
    return out


def canon(s: str, nm: dict) -> str:
    if not nm or not s:
        return s
    for a, c in sorted(nm.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(rf"(?i)\b{re.escape(a)}\b", c, s)
    return s


def _body(u: dict) -> str:
    if u["kind"] == "qa":
        return f"{u['question']}\n{u['answer']}"
    return u["text"]


# ---------------------------------------------------------------- gather
def gather(topic_id: str) -> list[dict]:
    t = store.topic(topic_id)
    us = []
    for cid in t.get("curated", []):
        try:
            name = curated.meta(cid)["name"]
        except KeyError:
            continue
        for u in curated.units(cid):
            us.append({**u, "set": name})
    for r in facts.records(topic_id, grounded_only=False):          # facts read from the topic's documents
        doc = r["doc"]
        try:
            dn = store.doc_meta(doc)["name"]
        except KeyError:
            dn = doc
        us.append({"id": f"{topic_id}:{r['id']}", "src": f"{dn} p.{r['page']}", "set": dn,
                   "kind": {"fact": "fact", "term": "fact", "table": "passage"}.get(r["lens"], "entity" if r["lens"] == "entity" else "passage"),
                   "subject": r["subject"], "text": r["text"], "question": "", "answer": "", "reasoning": "", "context": "",
                   "fields": {}, "tags": [r["lens"]], "links": [], "grounded_on_page": r.get("grounded", True)})
    return us


# ---------------------------------------------------------------- build
def build(topic_id: str, opts: dict | None = None, model: str | None = None, parallel: int | None = None,
          progress=lambda f, m="": None, cancelled=lambda: False, uc=None) -> dict:
    from packsmith.datasets.datasets import NULL
    uc = uc or NULL
    o = {**DEFAULTS, **(opts or {})}
    d = store.topic_dir(topic_id) / "factbook"
    us = gather(topic_id)
    if not us:
        raise ValueError("no curated data or captured facts in this topic: add a curated set (Topics) or capture data from its pages")
    nm = names_map(o["names"])
    ledger: dict[str, str] = {}
    entries: list[dict] = []
    conflicts: list[dict] = []
    flagged: list[dict] = []
    for u in us:
        u["subject"] = canon(u["subject"], nm)
        if o["names_in_text"]:
            for k in ("text", "question", "answer"):
                u[k] = canon(u[k], nm)
    progress(0.05, f"{len(us)} units from {len({u['set'] for u in us})} sources")

    # 1. empty units are listed, not lost
    live = []
    for u in us:
        if not norm(_body(u)):
            ledger[u["id"]] = "empty"
        else:
            live.append(u)

    # 2. checks against each unit's own source
    def check_grounding(u):
        if u.get("grounded_on_page") is False:
            return "a number in it is not printed on its page"
        if u["context"]:
            ok, miss = facts.grounded(u["answer"] or u["text"], u["context"] + " " + u["question"])
            if not ok:
                return f"value(s) {', '.join(miss)} not in its source text"
        return ""
    keep = []
    for u in live:
        why = check_grounding(u)
        if why:
            flagged.append({"unit": u, "why": why})
            ledger[u["id"]] = "flagged"
        else:
            keep.append(u)
    if o["verify"]:
        ctx = [u for u in keep if u["context"]]
        batches = [ctx[i:i + 8] for i in range(0, len(ctx), 8)]

        def ver(batch):
            html = "\n".join(f'<article class="item" data-ref="{u["id"]}"><p class="source">{htmlconv.esc(u["context"][:4000])}</p>'
                             f'<p class="claim">{htmlconv.esc(_body(u))}</p></article>' for u in batch)
            out = uc.run(uc.key("verify", html, model), lambda: llm.chat(prompts.verify_prompt(html), model=model,
                                                                          max_tokens=2000, temperature=0.0))
            return {a.attrs.get("data-ref", ""): (a.attrs.get("data-verdict", "") or "").lower()
                    for a in htmlconv.parse(out).find_all("verdict")}
        res = par.pmap(ver, batches, parallel, lambda k, n, r: progress(0.05 + 0.25 * k / n, f"verifying against sources {k}/{n}"),
                       cancelled)
        verdict = {k: v for r in res if not par.failed(r) for k, v in r.items()}
        nk = []
        for u in keep:
            if verdict.get(u["id"], "").startswith("unsupported"):
                flagged.append({"unit": u, "why": "the model found it unsupported by its source text"})
                ledger[u["id"]] = "flagged"
            else:
                nk.append(u)
        keep = nk
    if o["include_flagged"]:
        keep += [f["unit"] for f in flagged]
        for f in flagged:
            ledger.pop(f["unit"]["id"], None)

    # 3. group: exact duplicates merge; same question / same subject+statement compared for consistency
    def key(u):
        if u["kind"] == "qa":
            return ("qa", norm(u["question"]))
        if u["kind"] == "entity":
            return ("entity", norm(u["subject"]))
        return ("stmt", norm(u["subject"]), " ".join(sorted(words(re.sub(r"\d[\d,.]*", " ", u["text"])))))
    groups = defaultdict(list)
    for u in keep:
        groups[key(u)].append(u)
    progress(0.35, f"{len(groups)} groups")
    for gk, g in groups.items():
        if gk[0] == "entity":
            _entity_group(g, entries, conflicts, ledger)
            continue
        val = (lambda u: u["answer"]) if gk[0] == "qa" else (lambda u: u["text"])
        clusters: list[list[dict]] = []
        for u in g:                                        # same values and close wording → one entry, wordings kept
            for c in clusters:
                if same_values(val(u), val(c[0])) and (gk[0] == "stmt" or jacc(val(u), val(c[0])) >= 0.35):
                    c.append(u)
                    break
            else:
                clusters.append([u])
        if len(clusters) == 1:
            entries.append(_entry(clusters[0], ledger))
        else:
            gid = "g" + hashlib.sha1(repr(gk).encode()).hexdigest()[:8]
            conflicts.append({"id": gid, "subject": g[0]["subject"], "question": g[0].get("question", ""),
                              "options": [_entry(c, ledger, conflict=gid) for c in clusters]})

    # 4. resolutions the user made earlier
    resolved = store.rj(d / "resolutions.json", {}) or {}
    for c in conflicts:
        ch = resolved.get(c["id"])
        if ch:
            c["resolved"] = ch
            for e in c["options"]:
                if ch == "all" or e["id"] == ch:
                    entries.append({**e, "status": "ok", "note": "chosen in a conflict" if ch != "all" else "kept with its source (all values kept)"})

    # 5. the ledger must account for every unit
    missing = [u["id"] for u in us if u["id"] not in ledger]
    if missing:
        raise RuntimeError(f"fact book ledger lost {len(missing)} units: {missing[:5]}")
    counts = Counter(v.split(":", 1)[0] for v in ledger.values())
    rep = {"units": len(us), "sources": dict(Counter(u["set"] for u in us)), "entries": sum(1 for e in entries if e["status"] == "ok"),
           "ledger": dict(counts), "conflicts": len(conflicts), "conflicts_open": sum(1 for c in conflicts if not c.get("resolved")),
           "flagged": len(flagged), "subjects": len({e["subject"] for e in entries})}
    store.wt(d / "entries.jsonl", "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n")
    store.wj(d / "conflicts.json", conflicts)
    store.wj(d / "flagged.json", [{"id": f["unit"]["id"], "src": f["unit"]["src"], "set": f["unit"]["set"], "why": f["why"],
                                   "text": _body(f["unit"])} for f in flagged])
    store.wj(d / "ledger.json", ledger)
    store.wj(d / "report.json", rep)
    store.wt(d / "factbook.html", render_html(store.topic(topic_id).get("name", ""), entries, conflicts, flagged, rep))
    progress(1.0, f"fact book: {rep['entries']} entries, {rep['conflicts_open']} open conflicts, {rep['flagged']} flagged")
    return rep


def _entry(cluster: list[dict], ledger: dict, conflict: str = "") -> dict:
    head = cluster[0]
    eid = "e" + hashlib.sha1(head["id"].encode()).hexdigest()[:10]
    for i, u in enumerate(cluster):
        ledger[u["id"]] = f"conflict:{conflict}" if conflict else ("entry" if i == 0 else f"merged:{eid}")
    wordings = []
    for u in cluster:
        b = _body(u)
        if all(norm(b) != norm(w) for w in wordings):
            wordings.append(b)
    return {"id": eid, "kind": head["kind"], "subject": head["subject"], "question": head["question"], "answer": head["answer"],
            "text": head["text"], "reasoning": next((u["reasoning"] for u in cluster if u["reasoning"]), ""),
            "fields": head["fields"], "wordings": wordings, "sources": [f"{u['set']} · {u['src']}" for u in cluster],
            "units": [u["id"] for u in cluster], "status": "conflict" if conflict else "ok"}


def _entity_group(g: list[dict], entries: list, conflicts: list, ledger: dict) -> None:
    """Units describing one item: fields merge; a field with different values is a conflict on that field only."""
    vals = defaultdict(list)                       # field → [(value, unit)]
    for u in g:
        for k, v in (u["fields"] or {"text": u["text"]}).items():
            vals[norm(k)].append((v, u))
    merged, clash = {}, {}
    for k, vs in vals.items():
        distinct = []
        for v, u in vs:
            if not any(same_values(v, x) and jacc(v, x) >= 0.5 for x, _ in distinct):
                distinct.append((v, u))
        if len(distinct) == 1:
            merged[k] = distinct[0][0]
        else:
            clash[k] = distinct
    head = g[0]
    e = _entry(g, ledger)
    names = {norm(k): k for u in g for k in (u["fields"] or {})}
    e["fields"] = {names.get(k, k): v for k, v in merged.items()}
    e["text"] = "; ".join(f"{k}: {v}" for k, v in e["fields"].items())
    entries.append(e)
    for k, opts in clash.items():
        gid = "g" + hashlib.sha1(f"{head['subject']}|{k}".encode()).hexdigest()[:8]
        conflicts.append({"id": gid, "subject": head["subject"], "question": f"{names.get(k, k)} of {head['subject']}",
                          "options": [{"id": f"{gid}-{i}", "kind": "fact", "subject": head["subject"], "question": "", "answer": "",
                                       "text": f"{names.get(k, k)}: {v}", "reasoning": "", "fields": {}, "wordings": [f"{names.get(k, k)}: {v}"],
                                       "sources": [f"{u['set']} · {u['src']}"], "units": [u["id"]], "status": "conflict"}
                                      for i, (v, u) in enumerate(opts)]})


def resolve(topic_id: str, gid: str, choice: str) -> None:
    p = store.topic_dir(topic_id) / "factbook" / "resolutions.json"
    r = store.rj(p, {}) or {}
    if choice in ("", "open"):
        r.pop(gid, None)
    else:
        r[gid] = choice
    store.wj(p, r)


# ---------------------------------------------------------------- the book
def render_html(title: str, entries: list, conflicts: list, flagged: list, rep: dict) -> str:
    esc = htmlconv.esc
    by = defaultdict(list)
    for e in entries:
        if e["status"] == "ok":
            by[e["subject"] or "(no subject)"].append(e)

    def entry(e):
        if e["kind"] == "qa":
            body = f'<p class="q">{esc(e["question"])}</p><p class="a">{esc(e["answer"])}</p>'
            if e["reasoning"]:
                body += f'<details><summary>reasoning</summary><p>{esc(e["reasoning"])}</p></details>'
        else:
            body = f'<p>{esc(e["text"])}</p>'
        more = f'<details><summary>{len(e["wordings"])} wordings</summary>' + "".join(f"<p>{esc(w)}</p>" for w in e["wordings"]) + \
            "</details>" if len(e["wordings"]) > 1 else ""
        return f'<li class="entry" data-id="{e["id"]}">{body}{more}<p class="src">{esc(" | ".join(e["sources"]))}</p></li>'
    toc = "".join(f'<li><a href="#s{i}">{esc(s)}</a> ({len(v)})</li>' for i, (s, v) in enumerate(sorted(by.items())))
    secs = "".join(f'<section id="s{i}"><h2>{esc(s)}</h2><ul>{"".join(entry(e) for e in v)}</ul></section>'
                   for i, (s, v) in enumerate(sorted(by.items())))
    cf = "".join(f'<section class="conflict" data-id="{c["id"]}"><h3>{esc(c["subject"])}{": " + esc(c["question"]) if c["question"] else ""}'
                 f'{" — resolved" if c.get("resolved") else " — open"}</h3><ol>'
                 + "".join(f'<li>{esc(o["answer"] or o["text"])}<p class="src">{esc(" | ".join(o["sources"]))}</p></li>' for o in c["options"])
                 + "</ol></section>" for c in conflicts)
    fl = "".join(f'<li>{esc(_body(f["unit"]))}<p class="src">{esc(f["unit"]["set"])} · {esc(f["unit"]["src"])} — {esc(f["why"])}</p></li>'
                 for f in flagged)
    led = " · ".join(f"{k}: {v}" for k, v in rep["ledger"].items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Fact book — {esc(title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{font-size:24px}}h2{{font-size:18px;border-bottom:1px solid #ddd;margin-top:28px}}.src{{color:#777;font-size:12px;margin:2px 0 10px}}
.q{{font-weight:600;margin:0}}.a{{margin:2px 0}}.conflict{{border-left:4px solid #d9822b;padding-left:10px}}
.summary{{background:#f5f5f5;padding:10px 14px;border-radius:8px}}li.entry{{margin:6px 0}}
@media (prefers-color-scheme: dark){{body{{background:#161616;color:#ddd}}.summary{{background:#222}}}}</style></head><body>
<h1>Fact book — {esc(title)}</h1>
<div class="summary">{rep['units']} units from {len(rep['sources'])} sources → <b>{rep['entries']}</b> entries under {rep['subjects']} subjects ·
{rep['conflicts']} conflicts ({rep['conflicts_open']} open) · {rep['flagged']} flagged<br><small>ledger (every unit accounted for): {esc(led)}</small></div>
<h2>Contents</h2><ul>{toc}</ul>{secs}
<h2>Conflicts (all values kept; open ones are not trained)</h2>{cf or "<p>None.</p>"}
<h2>Flagged (checked against their own source)</h2><ul>{fl or "<li>None.</li>"}</ul></body></html>"""


# ---------------------------------------------------------------- training data
def load_entries(topic_id: str) -> list[dict]:
    p = store.topic_dir(topic_id) / "factbook" / "entries.jsonl"
    return [json.loads(x) for x in store.rt(p).splitlines() if x.strip()]


def as_text(e: dict, w: str | None = None) -> str:
    if e["kind"] == "qa":
        q, a = (w.split("\n", 1) + [""])[:2] if w else (e["question"], e["answer"])
        return f"Question: {q}\n" + (f"Reasoning: {e['reasoning']}\n" if e["reasoning"] else "") + f"Answer: {a}"
    lab = f"{e['subject']}: " if e["subject"] and not (w or e["text"]).lower().startswith(e["subject"].lower()[:20]) else ""
    return lab + (w or e["text"])


def render(topic_id: str, opts: dict | None = None, model=None, parallel=None, progress=lambda f, m="": None,
           cancelled=lambda: False, uc=None) -> tuple[list[dict], list[dict], dict]:
    o = {**DEFAULTS, **(opts or {})}
    rep = build(topic_id, o, model, parallel, lambda f, m="": progress(0.3 * f, m), cancelled, uc=uc)
    es = [e for e in load_entries(topic_id) if e["status"] == "ok"]
    items, tests = [], []
    if o["use"] in ("import", "both"):                    # the curated records themselves, every wording
        for e in es:
            for w in e["wordings"]:
                items.append({"text": as_text(e, w), "kind": "curated", "ref": e["id"]})
    if o["use"] in ("generate", "both"):
        by = defaultdict(list)
        for e in es:
            by[e["subject"]].append(e)
        if o["digest"]:                                    # one subject's facts together, up to the training length
            lim = int(int(o["max_len"]) * 3.5)
            for s, xs in by.items():
                cur = []
                for e in xs:
                    line = "- " + (f"{e['question']} {e['answer']}" if e["kind"] == "qa" else e["text"])
                    if cur and sum(len(x) for x in cur) + len(line) > lim:
                        items.append({"text": f"{s}:\n" + "\n".join(cur), "kind": "subject_digest"})
                        cur = []
                    cur.append(line)
                if cur:
                    items.append({"text": f"{s}:\n" + "\n".join(cur), "kind": "subject_digest"})
        t = store.topic(topic_id)
        recs = [{"id": e["id"], "subject": e["subject"], "page": "", "doc": "", "src": e["sources"][0],
                 "text": f"{e['question']} — {e['answer']}" if e["kind"] == "qa" else e["text"]} for e in es]
        vi, tests = facts.variants(recs, o, t.get("about") or t.get("name", ""),
                                   (library.get(t.get("record") or "") or {}).get("situations", []), model, parallel,
                                   lambda f, m="": progress(0.3 + 0.7 * f, m), cancelled, source="facts book", uc=uc)
        items += vi
    return items, tests, {"factbook": rep}
