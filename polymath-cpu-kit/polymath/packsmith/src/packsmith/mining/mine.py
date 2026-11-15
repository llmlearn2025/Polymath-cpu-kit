"""Mapping (fragments per block) and reduce (playbook), alignment across documents."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict

from packsmith.mining import htmlconv
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.packs import store
from packsmith.system.config import SETTINGS

NO_STRUCTURE = store.CONTEXT_TYPES


def blocks(t: dict, roles=("train",), budget: int | None = None) -> list[dict]:
    """[{doc, region, n, pages:[...], text}] — consecutive pages packed up to the budget (default ~60% of
    Settings.data_tokens; callers pass what fits the loaded model)."""
    budget = int(budget or llm.data_chars())
    out = []
    for doc, r in store.topic_regions(t, roles):
        if r["type"] in NO_STRUCTURE:
            continue
        cur, pages, n = [], [], 0
        for p in range(r["start"], r["end"] + 1):
            txt = store.page_text(doc, p)
            if cur and sum(len(x) for x in cur) + len(txt) > budget:
                out.append({"doc": doc, "region": r["id"], "n": n, "pages": pages,
                            "text": prompts.pages_block([(doc, q, store.page_text(doc, q)) for q in pages], budget)})
                cur, pages, n = [], [], n + 1
            cur.append(txt)
            pages.append(p)
        if pages:
            out.append({"doc": doc, "region": r["id"], "n": n, "pages": pages,
                        "text": prompts.pages_block([(doc, q, store.page_text(doc, q)) for q in pages], budget)})
    return out


def map_topic(topic_id: str, model: str | None = None, parallel: int | None = None, progress=lambda f, m="": None,
              cancelled=lambda: False) -> dict:
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id)
    plan = store.rt(d / "plan.html")
    if not plan:
        raise ValueError("approve a mining plan first (Discover)")
    budget = llm.budget_chars(model, 12000, len(plan) + 2500)
    bl = blocks(t, budget=budget)
    if not bl:
        raise ValueError(store.why_no_regions(t))
    fd = d / "fragments"
    key = hashlib.sha1(f"{plan}|{budget}|{store.regions_signature(t)}".encode()).hexdigest()[:12]
    if store.rt(fd / ".plan").strip() != key:             # a new plan, block size or set of regions starts afresh
        shutil.rmtree(fd, ignore_errors=True)
        store.wt(fd / ".plan", key)
    progress(0.0, f"{len(bl)} blocks of up to {budget} characters (model context {llm.context_tokens(model) or 'unknown'} tokens)")

    def one(b):
        f = fd / f"{b['doc']}_{b['region']}_{b['n']:03d}.html"
        if not f.exists():                                # resumable: finished blocks are kept
            out = llm.chat(prompts.map_prompt(plan, b["text"]), model=model, max_tokens=12000)
            store.wt(fd / "raw" / f"{f.stem}.txt", out)    # what the model said, for inspection
            frs = htmlconv.parse_fragments(out, b["doc"], b["pages"][0])
            if not frs and 'class="none"' not in out:
                raise ValueError(f"block {f.stem}: the reply has no fragments and no 'none' marker (see fragments/raw/{f.stem}.txt)")
            for fr in frs:                                # pages must be within the block
                try:
                    if int(fr["page"]) not in b["pages"]:
                        fr["page"] = b["pages"][0]
                except (ValueError, TypeError):
                    fr["page"] = b["pages"][0]
                fr["doc"] = b["doc"]
            store.wt(f, "\n".join(htmlconv.render_fragment(x) for x in frs) or '<p class="none"></p>')
        return len(htmlconv.parse_fragments(store.rt(f)))

    tally = [0]

    def done(k, n, r):
        tally[0] += 0 if par.failed(r) else r
        progress(k / n, f"block {k}/{n}: {tally[0]} fragments" + (f" (failed: {r['_error'][:120]})" if par.failed(r) else ""))
    res = par.pmap(one, bl, parallel, done, cancelled)
    failed = sum(par.failed(r) for r in res)
    return {"blocks": len(bl), "fragments": tally[0], "failed_blocks": failed}


def fragments(topic_id: str) -> list[dict]:
    out = []
    for f in sorted((store.topic_dir(topic_id) / "fragments").glob("*.html")):
        out.extend(htmlconv.parse_fragments(store.rt(f)))
    return out


def alignment(topic_id: str) -> dict:
    frs = fragments(topic_id)
    plan = htmlconv.parse_plan(store.rt(store.topic_dir(topic_id) / "plan.html"))
    moves = [m["name"] for m in plan["moves"]] or sorted({f["move"] for f in frs})
    docs = sorted({f["doc"] for f in frs})
    m = defaultdict(lambda: defaultdict(int))
    for f in frs:
        m[f["move"]][f["doc"]] += 1
    names = {d: store.doc_meta(d)["name"] for d in docs}
    return {"moves": moves, "docs": docs, "doc_names": names, "counts": {mv: dict(m[mv]) for mv in moves}}


def _reduce_move(move: dict, frs: list[dict], all_moves: list[str], skeleton: str, model) -> list[dict]:
    budget = llm.budget_chars(model, 8000, 3500)
    html_frs = [htmlconv.render_fragment(f) for f in frs]
    batches, cur = [], []
    for h in html_frs:
        if cur and sum(len(x) for x in cur) + len(h) > budget:
            batches.append(cur)
            cur = []
        cur.append(h)
    if cur:
        batches.append(cur)
    checks = []
    for b in batches:
        out = llm.chat(prompts.reduce_prompt(move, "\n".join(b), all_moves, skeleton if len(batches) == 1 else ""),
                       model=model, max_tokens=8000)
        checks += htmlconv.parse_playbook(out)["checks"]
    if len(batches) > 1 and checks:                     # second level: merge the partial checks
        partial = htmlconv.render_playbook({"checks": checks})[:budget]
        out = llm.chat(prompts.reduce_prompt(move, partial, all_moves, skeleton), model=model, max_tokens=8000)
        checks = htmlconv.parse_playbook(out)["checks"] or checks
    for c in checks:
        c["move"] = c.get("move") or move["name"]
    return checks


def reduce_topic(topic_id: str, model: str | None = None, parallel: int | None = None, progress=lambda f, m="": None) -> dict:
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id)
    plan = htmlconv.parse_plan(store.rt(d / "plan.html"))
    skeleton = store.rt(d / "skeleton.html")
    frs = fragments(topic_id)
    if not frs:
        if not blocks(t):
            raise ValueError("nothing to build from: " + store.why_no_regions(t))
        mapped = store.rt(d / "fragments" / ".plan").strip()
        raise ValueError("no fragments: run Map regions first" if not mapped else
                         "Map regions found no fragments of the plan's moves in the train regions: check the plan "
                         "(Discover) and the regions, or the model's replies in fragments/raw/")
    groups = [("", frs)]
    if t.get("strategy") == "per-document":
        groups += [(doc, [f for f in frs if f["doc"] == doc]) for doc in sorted({f["doc"] for f in frs})]
    names = [m["name"] for m in plan["moves"]]
    extra = sorted({f["move"] for f in frs} - set(names))
    moves = plan["moves"] + [{"name": x, "desc": ""} for x in extra]
    spb = htmlconv.parse_playbook(skeleton) if skeleton else {"checks": []}
    result = {}
    for gi, (doc, gfrs) in enumerate(groups):
        todo = [(mv, [f for f in gfrs if f["move"] == mv["name"]]) for mv in moves]
        todo = [(mv, mf) for mv, mf in todo if mf]

        def one(job):
            mv, mf = job
            sc = [c for c in spb["checks"] if c.get("move") == mv["name"]]
            sk = htmlconv.render_playbook({"checks": sc}) if sc else ""
            return _reduce_move(mv, mf, names + extra, sk, model)

        def done(k, n, r, gi=gi, doc=doc):
            progress((gi + k / max(1, n)) / len(groups), f"{doc or 'unified'}: move {k}/{n}" + (f" failed: {r['_error'][:120]}" if par.failed(r) else ""))
        parts = par.pmap(one, todo, parallel, done)
        checks = [c for r in parts if not par.failed(r) for c in r]     # plan order kept
        pb = finalize({"topic": topic_id, "title": t.get("name", ""), "checks": checks})
        store.wt(d / (f"playbook.{doc}.html" if doc else "playbook.html"), htmlconv.render_playbook(pb))
        result[doc or "unified"] = len(pb["checks"])
    return result


def finalize(pb: dict) -> dict:
    """Renumber checks c1.. and resolve data-goto="move:<name>" to the first check of that move."""
    first = {}
    for i, c in enumerate(pb["checks"]):
        old = c.get("id")
        c["id"] = f"c{i + 1}"
        c["_old"] = old
        first.setdefault(c.get("move", ""), c["id"])
    for c in pb["checks"]:
        for it in c.get("items", []):
            g = it.get("attrs", {}).get("data-goto", "")
            if g.startswith("move:"):
                it["attrs"]["data-goto"] = first.get(g[5:], "")
        c.pop("_old", None)
    return pb


def save_playbook(topic_id: str, html: str, doc: str = "") -> dict:
    ok, why = htmlconv.validate(html, "playbook")
    if not ok:
        raise ValueError(why)
    pb = htmlconv.parse_playbook(html)
    store.wt(store.topic_dir(topic_id) / (f"playbook.{doc}.html" if doc else "playbook.html"), html)
    return {"checks": len(pb["checks"])}


def playbooks(topic_id: str) -> list[str]:
    d = store.topic_dir(topic_id)
    out = ["unified"] if (d / "playbook.html").exists() else []
    out += [re.sub(r"^playbook\.(.*)\.html$", r"\1", p.name) for p in sorted(d.glob("playbook.*.html"))]
    return out


def coverage(topic_id: str) -> dict:
    """Index/contents terms vs playbook text: which structure checks were confirmed by mapping."""
    d = store.topic_dir(topic_id)
    sk = htmlconv.parse_playbook(store.rt(d / "skeleton.html"))["checks"] if (d / "skeleton.html").exists() else []
    pb_text = store.rt(d / "playbook.html").lower()
    hit = [c["label"] for c in sk if any(w in pb_text for w in re.findall(r"[a-z]{5,}", c["label"].lower())[:3])]
    return {"skeleton_checks": len(sk), "confirmed": len(hit), "missing": [c["label"] for c in sk if c["label"] not in hit]}


def dump(topic_id: str) -> str:
    return json.dumps(fragments(topic_id), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- playbooks are reusable across topics
def resolve_playbook(topic_id: str, ref: str | None) -> tuple[str, str]:
    """(html, label) for a playbook reference: "unified" / "<doc>" (this topic's) or "<topic>:<which>" (any topic's).
    A playbook is a method: once built from one text it can drive back-tests and generation on any other."""
    ref = (ref or "unified").strip()
    if ref.startswith("builtin:"):
        from packsmith.datasets import builtin
        return builtin.get(ref)
    tid, which = (ref.split(":", 1) if ":" in ref else (topic_id, ref))
    d = store.topic_dir(tid)
    html = store.rt(d / ("playbook.html" if which in ("", "unified") else f"playbook.{which}.html"))
    if not html:
        raise ValueError(f"no playbook {ref!r}: build one (Playbook tab) or pick another from the list")
    try:
        name = store.topic(tid).get("name", tid)
    except KeyError:
        name = tid
    return html, (which or "unified") if tid == topic_id else f"{name} · {which or 'unified'}"


def all_playbooks() -> list[dict]:
    from packsmith.datasets import builtin
    out = builtin.listing()                              # usable on any text, nothing to build
    for t in store.list_topics():
        for w in playbooks(t["id"]):
            html = store.rt(store.topic_dir(t["id"]) / ("playbook.html" if w == "unified" else f"playbook.{w}.html"))
            out.append({"ref": f"{t['id']}:{w}", "topic": t["id"], "topic_name": t.get("name", ""), "which": w,
                        "checks": len(htmlconv.parse_playbook(html)["checks"])})
    return out
