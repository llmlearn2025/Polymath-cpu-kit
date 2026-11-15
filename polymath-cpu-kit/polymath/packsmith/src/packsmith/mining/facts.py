"""The data track: capture a text's content (facts, tables, entity profiles, procedures, recommendations, glossary)
page by page, check every value against its page, and turn it into training items — with or without a playbook.

Generation from a playbook scales with the number of checks; this track scales with pages."""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict

from packsmith.mining import htmlconv, mine
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.system.config import SETTINGS

DEFAULTS = {"lenses": prompts.PRESETS["generic"], "fields": [], "rephrase": 2, "qa": 2, "scenes": 1, "test_share": 0.1,
            "max_tests": 300, "digests": True, "batch": 12, "max_len": 1024, "capture_tokens": None}

DATA_KINDS = ("data_fact", "table_row", "table", "entity", "procedure", "recommendation", "term", "digest", "rephrase", "qa",
              "scene")

_NUM = re.compile(r"\d[\d,.]*\d|\d")


def _nums(s: str) -> set[str]:
    return {re.sub(r"[,]", "", m).rstrip(".") for m in _NUM.findall(s or "")}


def grounded(text: str, source: str) -> tuple[bool, list[str]]:
    """Every number in the item must be printed in its source passage (commas ignored). Names are left to review."""
    missing = sorted(_nums(text) - _nums(source))
    return not missing, missing


def _page(n: htmlconv.Node, pages: list[int]) -> int:
    try:
        p = int(n.attrs.get("data-page") or 0)
    except ValueError:
        p = 0
    return p if p in pages else pages[0]


def parse_records(html: str, doc: str, pages: list[int]) -> list[dict]:
    root, out = htmlconv.parse(html), []

    def add(lens, page, subject, text, data=None):
        text = re.sub(r"\s+", " ", text or "").strip()
        if text:
            out.append({"lens": lens, "doc": doc, "page": page, "subject": (subject or "").strip(), "text": text, "data": data or {}})

    for n in root.find_all("fact"):
        add("fact", _page(n, pages), n.attrs.get("data-subject", ""), n.text(), {"type": n.attrs.get("data-type", "")})
    for tb in root.find_all("data", "table"):
        rows = tb.find_all(tag="tr")
        head = [c.text() for c in rows[0].direct(tag="th")] if rows and rows[0].direct(tag="th") else []
        body = [[c.text() for c in r.children if isinstance(r, htmlconv.Node) and not isinstance(c, str) and c.tag in ("td", "th")]
                for r in (rows[1:] if head else rows)]
        body = [r for r in body if any(r)]
        if body:
            title = tb.attrs.get("data-title", "")
            text = (title + "\n" if title else "") + "\n".join(" | ".join(r) for r in ([head] if head else []) + body)
            add("table", _page(tb, pages), title, text, {"head": head, "rows": body, "title": title})
    for e in root.find_all("entity"):
        fields = {f.attrs.get("data-name", "detail"): f.text() for f in e.find_all("field")}
        name = e.attrs.get("data-name", "")
        add("entity", _page(e, pages), name, f"{name}: " + "; ".join(f"{k}: {v}" for k, v in fields.items()), {"fields": fields})
    for pr in root.find_all("procedure"):
        name = pr.attrs.get("data-name", "")
        goal = " ".join(x.text() for x in pr.find_all("goal"))
        steps = [x.text() for x in pr.find_all("step")]
        extra = [f"Setting: {x.text()}" for x in pr.find_all("setting")] + [f"Watch for: {x.text()}" for x in pr.find_all("warning")]
        text = f"{name}" + (f" — {goal}" if goal else "") + "\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
        add("procedure", _page(pr, pages), name, text + ("\n" + "\n".join(extra) if extra else ""), {"steps": steps})
    for r in root.find_all("recommendation"):
        what = " ".join(x.text() for x in r.find_all("what"))
        why = " ".join(x.text() for x in r.find_all("why"))
        alt = [x.text() for x in r.find_all("alternative")]
        text = what + (f" Reason: {why}" if why else "") + "".join(f" Option weighed: {a}" for a in alt)
        add("recommendation", _page(r, pages), r.attrs.get("data-subject", ""), text, {"what": what, "why": why})
    for g in root.find_all("term"):
        name = " ".join(x.text() for x in g.find_all("name"))
        d = " ".join(x.text() for x in g.find_all("def")) or g.text().replace(name, "", 1)
        add("term", _page(g, pages), name, f"{name}: {d.strip()}")
    return out


def _key(o: dict, budget: int) -> str:
    return hashlib.sha1(json.dumps([sorted(o["lenses"]), o["fields"], budget], sort_keys=True).encode()).hexdigest()[:12]


def extract(topic_id: str, opts: dict | None = None, model: str | None = None, parallel: int | None = None,
            progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Capture data from every train region, a few pages per call, in parallel. Resumable; a new lens choice starts afresh."""
    o = {**DEFAULTS, **(opts or {})}
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id) / "data"
    budget = llm.budget_chars(model, 12000, 3500, int(o.get("capture_tokens") or SETTINGS.capture_tokens))
    bl = mine.blocks(t, ("train",), budget=budget)
    if not bl:
        raise ValueError(store.why_no_regions(t))
    key = _key(o, budget) + store.regions_signature(t)       # other regions: captured afresh
    if store.rt(d / ".key").strip() != key:
        shutil.rmtree(d, ignore_errors=True)
        store.wt(d / ".key", key)
    about = t.get("about") or t.get("name", "")
    progress(0.0, f"{len(bl)} blocks of up to {budget} characters; lenses: {', '.join(o['lenses'])}")

    def one(b):
        f = d / "blocks" / f"{b['doc']}_{b['region']}_{b['n']:03d}.json"
        if f.exists():
            return store.rj(f, [])
        out = llm.chat(prompts.data_prompt(o["lenses"], o["fields"], about, b["text"]), model=model, max_tokens=12000,
                       temperature=0.1)
        store.wt(d / "raw" / f"{f.stem}.txt", out)
        recs = parse_records(out, b["doc"], b["pages"])
        if not recs and 'class="none"' not in out:
            raise ValueError(f"block {f.stem}: the reply holds no data and no 'none' marker (see data/raw/{f.stem}.txt)")
        src = re.sub(r"</?page[^>]*>", " ", b["text"])          # page markers are not the text's numbers
        for r in recs:
            ok, missing = grounded(r["text"], src)
            r["grounded"], r["missing"] = ok, missing
        store.wj(f, recs)
        return recs

    tally = defaultdict(int)

    def done(k, n, r):
        if not par.failed(r):
            for x in r:
                tally["kept" if x["grounded"] else "ungrounded"] += 1
        progress(0.95 * k / n, f"block {k}/{n}: {tally['kept']} items, {tally['ungrounded']} with values not on the page"
                 + (f" — failed: {r['_error'][:120]}" if par.failed(r) else ""))
    res = par.pmap(one, bl, parallel, done, cancelled)
    recs = [x for r in res if not par.failed(r) for x in r]
    for i, r in enumerate(recs):
        r["id"] = f"r{i + 1}"
    store.wt(d / "records.jsonl", "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
    rep = {"blocks": len(bl), "failed_blocks": sum(par.failed(r) for r in res), "records": len(recs),
           "grounded": sum(r["grounded"] for r in recs), "by_lens": dict(_count(r["lens"] for r in recs)),
           "pages_covered": len({(r["doc"], r["page"]) for r in recs}),
           "pages_total": sum(len(b["pages"]) for b in bl)}
    store.wj(d / "report.json", rep)
    return rep


def _count(it):
    c = defaultdict(int)
    for x in it:
        c[x] += 1
    return c


def records(topic_id: str, grounded_only: bool = True) -> list[dict]:
    p = store.topic_dir(topic_id) / "data" / "records.jsonl"
    rs = [json.loads(x) for x in store.rt(p).splitlines() if x.strip()]
    return [r for r in rs if r.get("grounded") or not grounded_only]


def _label(r: dict) -> str:
    return f"{r['subject']}: " if r["subject"] and not r["text"].lower().startswith(r["subject"].lower()[:20]) else ""


def statements(recs: list[dict], doc_names: dict) -> list[dict]:
    """Deterministic items: each record as a statement, each table row as a sentence, page digests."""
    out = []
    for r in recs:
        if r["lens"] == "table":
            head, title = r["data"].get("head") or [], r["data"].get("title") or "the table"
            for row in r["data"].get("rows", []):
                if head and len(row) == len(head) and len(row) > 1:
                    s = f"In {title}, for {row[0]}: " + "; ".join(f"{h} is {v}" for h, v in zip(head[1:], row[1:]) if v) + "."
                else:
                    s = f"In {title}: " + " | ".join(row)
                out.append({"text": s, "kind": "table_row", "ref": r["id"], "page": r["page"], "doc": r["doc"]})
            out.append({"text": r["text"], "kind": "table", "ref": r["id"], "page": r["page"], "doc": r["doc"]})
        else:
            kind = {"fact": "data_fact"}.get(r["lens"], r["lens"])
            out.append({"text": _label(r) + r["text"], "kind": kind, "ref": r["id"], "page": r["page"], "doc": r["doc"]})
    return out


def digests(recs: list[dict], doc_names: dict, max_chars: int) -> list[dict]:
    """Facts of one page (and the next pages while they fit) as one passage — uses the training length."""
    by = defaultdict(list)
    for r in recs:
        if r["lens"] != "table":
            by[(r["doc"], int(r["page"]))].append(r)
    out, cur, head = [], [], None
    for (doc, page) in sorted(by):
        lines = [f"- {_label(r)}{r['text']}" for r in by[(doc, page)]]
        if cur and (head[0] != doc or sum(len(x) for x in cur) + sum(len(x) for x in lines) > max_chars):
            out.append({"text": f"From {doc_names.get(head[0], head[0])}, p. {head[1]}:\n" + "\n".join(cur), "kind": "digest",
                        "doc": head[0], "page": head[1]})
            cur = []
        if not cur:
            head = (doc, page)
        cur += lines
    if cur:
        out.append({"text": f"From {doc_names.get(head[0], head[0])}, p. {head[1]}:\n" + "\n".join(cur), "kind": "digest",
                    "doc": head[0], "page": head[1]})
    return out


def render(topic_id: str, opts: dict | None = None, model: str | None = None, parallel: int | None = None,
           progress=lambda f, m="": None, cancelled=lambda: False, uc=None) -> tuple[list[dict], list[dict], dict]:
    """Training items and tests from the captured data. Every fact stays in training; tests hold out one
    question wording per sampled fact (the knowledge is tested, not the sentence)."""
    o = {**DEFAULTS, **(opts or {})}
    t = store.topic(topic_id)
    recs = records(topic_id)
    if not recs:
        raise ValueError("no captured data: run data capture first")
    names = {}
    for r in recs:
        if r["doc"] not in names:
            try:
                names[r["doc"]] = store.doc_meta(r["doc"])["name"]
            except KeyError:
                names[r["doc"]] = r["doc"]
    items = statements(recs, names)
    if o["digests"]:
        items += digests(recs, names, int(int(o["max_len"]) * 3.5))
    situations = (library.get(t.get("record") or "") or {}).get("situations", [])
    about = t.get("about") or t.get("name", "")
    vs = [r for r in recs if r["lens"] != "table"]          # table rows are already sentences; variants per record
    vi, tests = variants(vs, o, about, situations, model, parallel, progress, cancelled, uc=uc)
    items += vi
    rep = {"records": len(recs), "variant_batches_failed": 0}
    return items, tests, rep


def variants(vs: list[dict], o: dict, about: str, situations: list[str], model=None, parallel=None,
             progress=lambda f, m="": None, cancelled=lambda: False, source: str = "facts", uc=None) -> tuple[list[dict], list[dict]]:
    """Rephrasings, Q&A and scenes for records {id, text, subject, page, doc}, each grounded in its record; one held-out
    question wording per sampled record becomes a test (source starts with 'facts': knowledge tests, not leaks)."""
    items, tests = [], []
    if not vs or not any(int(o.get(k, 0)) for k in ("rephrase", "qa", "scenes")):
        return items, tests
    batches = [vs[i:i + int(o["batch"])] for i in range(0, len(vs), int(o["batch"]))]
    n_test = min(int(o["max_tests"]), max(1 if len(vs) >= 5 and float(o["test_share"]) > 0 else 0, round(len(vs) * float(o["test_share"]))))
    held = set(r["id"] for r in random.Random(7).sample(vs, n_test))

    from packsmith.datasets.datasets import NULL
    uc = uc or NULL

    def one(batch):
        html = "\n".join(f'<p class="item" data-ref="{r["id"]}">{htmlconv.esc(_label(r) + r["text"])}</p>' for r in batch)
        out = uc.run(uc.key("variants", html, int(o["rephrase"]), int(o["qa"]), int(o["scenes"]), about, model),
                     lambda: llm.chat(prompts.variants_prompt(html, int(o["rephrase"]), int(o["qa"]), int(o["scenes"]), about,
                                                              situations), model=model, max_tokens=8000, temperature=0.7))
        src = {r["id"]: r for r in batch}
        its, tst = [], []
        for a in htmlconv.parse(out).find_all("variant"):
            ref = a.attrs.get("data-ref", "")
            r = src.get(ref)
            if not r:
                continue
            base = _label(r) + r["text"]
            meta = {"ref": ref, "page": r.get("page", ""), "doc": r.get("doc", "")}
            for x in a.find_all("rephrase"):
                if grounded(x.text(), base)[0]:
                    its.append({"text": x.text(), "kind": "rephrase", **meta})
            qs, ans = [x.text() for x in a.find_all("q")], [x.text() for x in a.find_all("a")]
            pairs = [(q, an) for q, an in zip(qs, ans) if grounded(an, base)[0]]
            if ref in held and pairs:
                q, an = pairs.pop()
                tst.append({"question": q, "expected": an, "check": ref, "source": f"{source} {r.get('page') or r.get('src', '')}".strip()})
            its += [{"text": f"Question: {q}\nAnswer: {an}", "kind": "qa", **meta} for q, an in pairs]
            for x in a.find_all("scene"):
                if grounded(x.text(), base)[0]:
                    its.append({"text": x.text(), "kind": "scene", **meta})
        return its, tst
    res = par.pmap(one, batches, parallel, lambda k, n, r: progress(0.9 * k / n, f"variants: batch {k}/{n}"
                   + (f" — failed: {r['_error'][:120]}" if par.failed(r) else "")), cancelled)
    for r in res:
        if not par.failed(r):
            items += r[0]
            tests += r[1]
    return items, tests
