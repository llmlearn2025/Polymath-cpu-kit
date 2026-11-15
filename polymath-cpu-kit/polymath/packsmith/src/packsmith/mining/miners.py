"""Extra mining: optional miners that make a pack able to PERFORM (write a draft, a report, prose in the author's voice),
REVIEW (find and fix what is wrong) and know its LIMITS — on top of, or instead of, the facts track and the playbook.

Extraction miners (one model call per block covers all chosen ones, verbatim-checked against the pages):
  exemplars    the text's own finished outputs (clauses, forms, specimen reports, letters, worked answers, passages)
  variants     alternatives with the condition for each
  corrections  errors → fixes, before → after
  checklists   ordered review lists, search patterns, protocols
  skeletons    document / report structures
  boundaries   scope, assumptions, jurisdiction, when to refer, what may be outdated
  vignettes    worked cases: input → expert output (also held-out tests)
Derived miners (from the exemplars, or from the text's passages when there are none; the target is always the book's
own text, never the model's imitation):
  backtranslate  request → the passage ("instruction back-translation")
  rewrite        plain version → the original (teaches the voice)
  review         flawed version → problems + the original (teaches review)
Local:
  passages     the whole text in overlapping windows, nothing dropped (for voice)
  voice        a measurable style profile (sentence length, "and"-chains, dialogue, …) saved for evaluation; no items

Profiles (presets in the UI): writer, drafter, reader, reviewer.
Everything is cached per unit (datasets.UnitCache), so Continue only redoes missing or failed calls."""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter

from packsmith.mining import htmlconv, mine
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.packs import store

MINERS = {
    "exemplars": ("Exemplars", "the book's own finished outputs (clauses, forms, specimen reports, sample letters, passages), word for word", "perform", "extract"),
    "variants": ("Variants", "alternatives with the condition for choosing each (\"for the landlord… / for the tenant…\")", "perform", "extract"),
    "corrections": ("Corrections", "errors → fixes and before → after examples in the book", "review", "extract"),
    "checklists": ("Checklists", "ordered review lists, search patterns and protocols", "reason", "extract"),
    "skeletons": ("Structures", "how each type of document or report is laid out", "perform", "extract"),
    "boundaries": ("Limits", "scope, assumptions, jurisdiction or era, when to refer, what may be outdated", "limits", "extract"),
    "vignettes": ("Cases", "worked cases: input → the expert's output (also kept as tests)", "perform", "extract"),
    "backtranslate": ("Requests → text", "for each exemplar or passage, the request that would produce it; trained request → the book's text", "perform", "derive"),
    "rewrite": ("Plain → styled", "a flat version of each passage, trained to restore the original voice", "perform", "derive"),
    "review": ("Flawed → fixed", "a flawed version of each exemplar with its defects, trained to find them and give the correct text", "review", "derive"),
    "passages": ("Full text", "the whole text in overlapping windows, nothing dropped (for voice and long-form flow)", "perform", "local"),
    "traces": ("Reasoning traces", "worked problems solved as ordered anchors (facts → issue → law → … → conclusion), each with its evidence; kept only when the conclusion matches the source and the checks pass", "reason", "traces"),
    "voice": ("Voice profile", "measures the style (sentence length, \"and\"-chains, dialogue, word length…) for evaluation; adds no items", "analysis", "local"),
}
PROFILES = {
    "writer": ("Writer — write in the author's voice", ["passages", "backtranslate", "rewrite", "voice"]),
    "drafter": ("Drafter — produce documents like the book's forms", ["exemplars", "backtranslate", "variants", "corrections", "checklists", "skeletons", "boundaries"]),
    "reader": ("Reader / reporter — from findings to the expert's report", ["vignettes", "checklists", "skeletons", "corrections", "boundaries"]),
    "reviewer": ("Reviewer — check and correct others' work", ["corrections", "review", "checklists", "boundaries", "exemplars"]),
    "reasoner": ("Reasoner — verified step-by-step reasoning (law, medicine, other checkable fields)", ["traces", "vignettes", "checklists", "boundaries"]),
}
EXTRACT = [k for k, v in MINERS.items() if v[3] == "extract"]
PERFORM = {k for k, v in MINERS.items() if v[2] == "perform"}
DEFAULTS = {"derive_max": 400, "test_share": 0.15, "batch": 4, "max_len": 1024}


def _norm(s: str) -> list[str]:
    return re.sub(r"[^\w]+", " ", (s or "").lower()).split()


def verbatim(text: str, source: str, n: int = 8) -> float:
    """Share of the text's n-grams found in the source (1.0 = copied exactly)."""
    w, s = _norm(text), " ".join(_norm(source))
    if len(w) < n:
        return 1.0 if " ".join(w) in s else 0.0
    grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
    return sum(1 for g in grams if g in s) / len(grams)


def _strip(block_text: str) -> str:
    return re.sub(r"</?page[^>]*>", " ", block_text)


def windows(text: str, max_chars: int, overlap: bool = True) -> list[str]:
    """Paragraph-bounded windows up to max_chars; with overlap each window repeats the previous one's last paragraph."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], []
    for p in paras:
        if cur and sum(len(x) for x in cur) + len(p) > max_chars:
            out.append("\n\n".join(cur))
            cur = [cur[-1]] if overlap and len(cur[-1]) < max_chars // 3 else []
        cur.append(p[:max_chars])
    if cur:
        out.append("\n\n".join(cur))
    return out


def voice_profile(text: str) -> dict:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.split()) >= 1]
    words = re.findall(r"[A-Za-z']+", text)
    lens = [len(s.split()) for s in sents] or [0]
    lw = [w.lower() for w in words]
    return {"sentences": len(sents), "words": len(words), "mean_sentence_words": round(statistics.mean(lens), 1),
            "sd_sentence_words": round(statistics.pstdev(lens), 1),
            "short_sentences_share": round(sum(1 for x in lens if x <= 8) / len(lens), 3),
            "long_sentences_share": round(sum(1 for x in lens if x >= 30) / len(lens), 3),
            "and_per_sentence": round(lw.count("and") / max(1, len(sents)), 2),
            "dialogue_share": round(text.count('"') / 2 / max(1, len(sents)), 3),
            "mean_word_letters": round(statistics.mean([len(w) for w in words]) if words else 0, 2),
            "ly_adverbs_per_1000": round(sum(1 for w in lw if w.endswith("ly") and len(w) > 4) * 1000 / max(1, len(words)), 1),
            "distinct_words_share": round(len(set(lw)) / max(1, len(lw)), 3),
            "top_words": [w for w, _ in Counter(w for w in lw if len(w) > 3).most_common(25)]}


def _parse(html: str, doc: str, pages: list[int]) -> dict:
    root = htmlconv.parse(html)

    def pg(n):
        try:
            p = int(n.attrs.get("data-page") or 0)
        except ValueError:
            p = 0
        return p if p in pages else pages[0]
    out = {k: [] for k in EXTRACT}
    for a in root.find_all("exemplar"):
        t = a.find("text")
        txt = (t.text() if t else "").strip()
        if txt:
            out["exemplars"].append({"type": a.attrs.get("data-type", ""), "context": " ".join(x.text() for x in a.find_all("context")),
                                     "text": txt, "page": pg(a), "doc": doc})
    for a in root.find_all("variant"):
        opts = [{"when": o.attrs.get("data-when", ""), "text": o.text()} for o in a.find_all("option") if o.text()]
        if opts:
            out["variants"].append({"subject": a.attrs.get("data-subject", ""), "options": opts, "page": pg(a), "doc": doc})
    for a in root.find_all("correction"):
        b, f, w = (" ".join(x.text() for x in a.find_all(k)) for k in ("before", "after", "why"))
        if b and f:
            out["corrections"].append({"before": b, "after": f, "why": w, "page": pg(a), "doc": doc})
    for a in root.find_all("checklist"):
        pts = [li.text() for li in a.find_all(tag="li") if li.text()]
        if pts:
            out["checklists"].append({"name": a.attrs.get("data-name", ""), "points": pts, "page": pg(a), "doc": doc})
    for a in root.find_all("skeleton"):
        pts = [li.text() for li in a.find_all(tag="li") if li.text()]
        if pts:
            out["skeletons"].append({"name": a.attrs.get("data-name", ""), "parts": pts, "page": pg(a), "doc": doc})
    for a in root.find_all("boundary"):
        r = " ".join(x.text() for x in a.find_all("rule")) or a.text()
        if r:
            out["boundaries"].append({"type": a.attrs.get("data-type", ""), "rule": r, "page": pg(a), "doc": doc})
    for a in root.find_all("vignette"):
        i, o = (" ".join(x.text() for x in a.find_all(k)) for k in ("input", "output"))
        if i and o:
            out["vignettes"].append({"input": i, "output": o, "page": pg(a), "doc": doc})
    return out


def run(topic_id: str, chosen: list[str], opts: dict | None = None, model: str | None = None, parallel: int | None = None,
        progress=lambda f, m="": None, cancelled=lambda: False, uc=None) -> tuple[list[dict], list[dict], dict]:
    from packsmith.datasets.datasets import NULL
    uc = uc or NULL
    o = {**DEFAULTS, **(opts or {})}
    chosen = [k for k in chosen if k in MINERS]
    t = store.topic(topic_id)
    about = t.get("about") or t.get("name", "")
    d = store.topic_dir(topic_id) / "mining"
    bl = mine.blocks(t, ("train",), budget=llm.budget_chars(model, 12000, 3500))
    if not bl and set(chosen) - {"traces"}:
        raise ValueError("extra mining reads tagged pages, and " + store.why_no_regions(t))
    items, tests, rep = [], [], {"miners": chosen, "counts": {}, "dropped_not_verbatim": 0}
    full = "\n\n".join(_strip(b["text"]) for b in bl)

    # ---- local
    if "voice" in chosen:
        vp = voice_profile(full)
        store.wj(d / "voice.json", vp)
        rep["voice"] = {k: v for k, v in vp.items() if k != "top_words"}
    if "passages" in chosen:
        ws = windows(full, int(int(o["max_len"]) * 3.5))
        items += [{"text": w, "kind": "passage"} for w in ws]
        rep["counts"]["passages"] = len(ws)

    # ---- extraction (one call per block for all chosen extraction miners)
    ext = [k for k in chosen if k in EXTRACT]
    if {"backtranslate", "review"} & set(chosen) and "exemplars" not in ext and not ({"passages", "rewrite"} & set(chosen)):
        ext.append("exemplars")                         # derived miners work from exemplars (writers use the passages)
    found = {k: [] for k in EXTRACT}
    if ext:
        def one(b):
            html = uc.run(uc.key("mine-extra", sorted(ext), about, b["text"], model),
                          lambda: llm.chat(prompts.mine_prompt(ext, about, b["text"]), model=model, max_tokens=12000, temperature=0.1))
            res = _parse(html, b["doc"], b["pages"])
            src = _strip(b["text"])
            dropped = 0
            for k in ("exemplars", "corrections", "vignettes", "variants"):
                keep = []
                for x in res[k]:
                    probe = x.get("text") or x.get("after") or x.get("output") or " ".join(op["text"] for op in x.get("options", []))
                    if verbatim(probe, src) >= (0.6 if k in ("exemplars", "variants") else 0.35):
                        keep.append(x)
                    else:
                        dropped += 1
                res[k] = keep
            return res, dropped
        res = par.pmap(one, bl, parallel, lambda k, n, r: progress(0.5 * k / n, f"extra mining: block {k}/{n}"
                       + (f" — failed: {r['_error'][:120]}" if par.failed(r) else "")), cancelled)
        for r in res:
            if not par.failed(r):
                for k, v in r[0].items():
                    found[k] += v
                rep["dropped_not_verbatim"] += r[1]
        for k in ext:
            store.wt(d / f"{k}.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in found[k]))

    rng = random.Random(3)
    if "exemplars" in chosen:
        items += [{"text": (f"{x['context']}\n\n" if x["context"] else "") + x["text"], "kind": "exemplar", "page": x["page"], "doc": x["doc"]}
                  for x in found["exemplars"]]
    if "variants" in chosen:
        for x in found["variants"]:
            items.append({"text": f"{x['subject']}:\n" + "\n".join(f"- When {op['when']}: {op['text']}" if op["when"] else f"- {op['text']}" for op in x["options"]),
                          "kind": "variant", "page": x["page"], "doc": x["doc"]})
            items += [{"text": f"Question: {x['subject']} — {op['when']}?\nAnswer: {op['text']}", "kind": "variant"} for op in x["options"] if op["when"]]
    if "corrections" in chosen:
        items += [{"text": f"Question: Revise this:\n{x['before']}\nAnswer: {x['after']}" + (f"\n\nReason: {x['why']}" if x["why"] else ""),
                   "kind": "correction", "page": x["page"], "doc": x["doc"]} for x in found["corrections"]]
    if "checklists" in chosen:
        for x in found["checklists"]:
            lst = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(x["points"]))
            items += [{"text": f"{x['name']}:\n{lst}", "kind": "checklist"},
                      {"text": f"Question: What do you check {('when ' + x['name']) if x['name'] else 'here'}?\nAnswer: {lst}", "kind": "checklist"}]
    if "skeletons" in chosen:
        items += [{"text": f"Question: How is {x['name'] or 'this document'} laid out?\nAnswer: " + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(x["parts"])),
                   "kind": "skeleton"} for x in found["skeletons"]]
    if "boundaries" in chosen:
        lab = {"scope": "Scope", "assumption": "Assumes", "refer": "Refer or stop when", "dated": "May be outdated"}
        items += [{"text": f"{lab.get(x['type'], 'Limit')}: {x['rule']}", "kind": "boundary", "page": x["page"], "doc": x["doc"]} for x in found["boundaries"]]
    if "vignettes" in chosen:
        vs = found["vignettes"][:]
        rng.shuffle(vs)
        nt = int(len(vs) * float(o["test_share"])) if len(vs) >= 4 else 0
        tests += [{"question": x["input"], "expected": x["output"], "check": "", "source": f"mining case p.{x['page']}"} for x in vs[:nt]]
        items += [{"text": f"Question: {x['input']}\nAnswer: {x['output']}", "kind": "vignette", "page": x["page"], "doc": x["doc"]} for x in vs[nt:]]
    for k in EXTRACT:
        if k in chosen:
            rep["counts"][k] = len(found[k])

    # ---- derived (the target is always the book's own text)
    der = [k for k in ("backtranslate", "rewrite", "review") if k in chosen]
    if der:
        cap = int(int(o["max_len"]) * 2.4)
        srcs = [{"text": x["text"][:cap], "page": x["page"], "doc": x["doc"]} for x in found["exemplars"] if len(x["text"]) > 120]
        if not srcs or "rewrite" in der or ("backtranslate" in der and "passages" in chosen):
            srcs += [{"text": w, "page": "", "doc": ""} for w in windows(full, cap, overlap=False) if len(w) > 200]
        seen, uniq = set(), []
        for s in srcs:
            h = s["text"][:200]
            if h not in seen:
                seen.add(h)
                uniq.append(s)
        srcs = uniq[: int(o["derive_max"])]
        for i, s in enumerate(srcs):
            s["id"] = f"s{i + 1}"
        batches = [srcs[i:i + int(o["batch"])] for i in range(0, len(srcs), int(o["batch"]))]
        jobs = [(k, b) for k in der for b in batches if k != "review" or any(s["page"] != "" for s in b) or not found["exemplars"]]

        def dj(job):
            k, batch = job
            html = "\n".join(f'<article class="item" data-ref="{s["id"]}">{htmlconv.esc(s["text"])}</article>' for s in batch)
            pr = {"backtranslate": lambda: prompts.backtranslate_prompt(html, about), "rewrite": lambda: prompts.rewrite_prompt(html),
                  "review": lambda: prompts.flaw_prompt(html, about)}[k]()
            out = uc.run(uc.key(k, html, about, model), lambda: llm.chat(pr, model=model, max_tokens=8000, temperature=0.6))
            by = {s["id"]: s for s in batch}
            its = []
            root = htmlconv.parse(out)
            if k == "backtranslate":
                for a in root.find_all("req"):
                    s, rq = by.get(a.attrs.get("data-ref", "")), " ".join(x.text() for x in a.find_all("request"))
                    if s and rq:
                        its.append({"text": f"Question: {rq}\nAnswer: {s['text']}", "kind": "request"})
            elif k == "rewrite":
                for a in root.find_all("plain"):
                    s, tx = by.get(a.attrs.get("data-ref", "")), " ".join(x.text() for x in a.find_all("text"))
                    if s and tx and verbatim(tx, s["text"]) < 0.5:          # really rewritten, not copied
                        its.append({"text": f"Question: Rewrite this in the style of {about or 'the author'}:\n{tx}\nAnswer: {s['text']}", "kind": "rewrite"})
            else:
                for a in root.find_all("flaw"):
                    s = by.get(a.attrs.get("data-ref", ""))
                    fl = " ".join(x.text() for x in a.find_all("flawed"))
                    probs = [x.text() for x in a.find_all("problem")]
                    if s and fl and probs:
                        its.append({"text": f"Question: Review and correct this:\n{fl}\nAnswer: Problems:\n" + "\n".join(f"- {p}" for p in probs)
                                    + f"\n\nCorrected:\n{s['text']}", "kind": "review"})
            return its
        res = par.pmap(dj, jobs, parallel, lambda k, n, r: progress(0.5 + 0.5 * k / n, f"extra mining: derived {k}/{n}"
                       + (f" — failed: {r['_error'][:120]}" if par.failed(r) else "")), cancelled)
        for r in res:
            if not par.failed(r):
                items += r
        for k in der:
            rep["counts"][k] = sum(1 for x in items if x["kind"] == {"backtranslate": "request", "rewrite": "rewrite", "review": "review"}[k])
        rep["derived_sources"] = len(srcs)
    if "traces" in chosen:                          # after extraction: the Cases miner's vignettes are trace sources too
        from packsmith.mining import traces
        ti, tt, trep = traces.run(topic_id, o.get("traces") or {}, model, parallel, lambda f, m="": progress(0.5 + 0.5 * f, m),
                                  cancelled, uc)
        items += ti
        tests += tt
        rep["traces"] = trep
        rep["counts"]["traces"] = trep["kept"]
    rep["perform"] = bool(PERFORM & set(chosen))
    store.wj(d / "report.json", rep)
    return items, tests, rep


def options() -> dict:
    return {"miners": {k: {"name": v[0], "desc": v[1], "ability": v[2]} for k, v in MINERS.items()},
            "profiles": {k: {"name": v[0], "miners": v[1]} for k, v in PROFILES.items()}}
