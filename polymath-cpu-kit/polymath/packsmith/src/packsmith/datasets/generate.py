"""Training data from a playbook: walk-throughs, minimal pairs, next-check cases, situated facts, method passages,
immersion passages; quality checks (negative contrast, duplicates, leak), mix; tests."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter

from packsmith.datasets import builtin, datasets, prompts
from packsmith.mining import facts, htmlconv, miners, mine
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.system import par

MODES = {"reasoning": "With a playbook (reasoning, or a built-in one)", "facts": "Facts & data only (no playbook needed)",
         "both": "Facts & data + reasoning", "mining": "Only the extra mining I tick"}
DEFAULTS = {"mode": "reasoning", "reasoning_share": 0.3, "walkthrough": 3, "pair": 2, "nextcheck": 1, "test": 2, "situated_scenes": 3, "method": 4,
            "immersion": True, "max_len": 1024, "drop_negative": True,
            "mix": {"walkthrough": 0.30, "pair": 0.15, "nextcheck": 0.08, "fact": 0.07, "situated": 0.20, "method": 0.05,
                    "immersion": 0.15}}

MINED_KINDS = ("passage", "exemplar", "variant", "correction", "checklist", "skeleton", "boundary", "vignette", "request",
               "rewrite", "review", "trace")
NEG = re.compile(r"\b(?:is|are|was|were|does|do|did|has|have)\s+not\b|\bnever\b|n't\b|\bnot\s+(?:a|an|the)\b", re.I)


def case_text(c: dict) -> str:
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(c.get("steps", [])))
    return f"Situation: {c['situation']}\n\nSteps:\n{steps}\n\nConclusion: {c['conclusion']}".strip()


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def ngrams(s: str, n: int = 8) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def immersion_passages(t: dict, max_chars: int) -> list[str]:
    out = []
    for b in mine.blocks(t):
        txt = re.sub(r"</?page[^>]*>", "", b["text"])
        paras = [p.strip() for p in re.split(r"\n\s*\n", txt) if p.strip()]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) > max_chars:
                out.append(cur.strip())
                cur = ""
            cur += p + "\n\n"
        if cur.strip():
            out.append(cur.strip())
    return out


def run(topic_id: str, playbook: str = "unified", opts: dict | None = None, model: str | None = None,
        parallel: int | None = None, dataset: str = "", name: str = "", progress=lambda f, m="": None,
        cancelled=lambda: False) -> dict:
    """Build a training set as a new dataset, or continue an unfinished one (dataset=<id>: same options, only the
    units that are missing or failed are generated). mode: reasoning (any playbook), facts (data capture), both."""
    if dataset:
        m = datasets.manifest(dataset)
        playbook, opts, model = m.get("playbook") or playbook, m.get("options") or opts, model or m.get("model") or None
        m["status"] = "running"
        datasets.save_manifest(m)
    else:
        mode0 = (opts or {}).get("mode") or "reasoning"
        m = datasets.create(name or f"{store.topic(topic_id).get('name', topic_id)} · {mode0} · {time.strftime('%d %b %H:%M')}",
                            topic_id, "generate", mode=mode0, playbook=playbook, options=opts or {}, model=model or "")
    progress(0.0, f"dataset {m['name']} ({'continuing' if dataset else 'new'})")
    try:
        rep = _run(topic_id, playbook, opts, model, parallel, datasets.cache_for(topic_id, m["salt"]), m["id"], progress, cancelled)
    except Exception as e:
        from packsmith.llm.llm import Cancelled
        m = datasets.manifest(m["id"])
        m["status"] = ("paused" if "paused" in str(e) else "cancelled") if isinstance(e, Cancelled) else "failed"
        m["error"] = f"{type(e).__name__}: {e}"
        datasets.save_manifest(m)
        raise
    return rep


def _run(topic_id, playbook, opts, model, parallel, uc, did, progress, cancelled) -> dict:
    o = {**DEFAULTS, **(opts or {})}
    o["mix"] = {**DEFAULTS["mix"], **((opts or {}).get("mix") or {})}
    mode = o.get("mode") or "reasoning"
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id)
    items, tests, extra = [], [], {}
    chosen = [k for k in (o.get("miners") or []) if k in miners.MINERS]
    if mode == "mining" and not chosen:
        raise ValueError("tick at least one extra miner (Generate → Extra mining), or choose another 'Dataset from'")
    if mode in ("facts", "both"):
        fo = {**(o.get("data") or {}), "max_len": o["max_len"]}
        span = 0.45 if mode == "both" else 0.85
        progress(0.0, "capturing data")
        if store.mined_regions(t, ("train",)):
            extra["capture"] = facts.extract(topic_id, fo, model, parallel, lambda f, m="": progress(span * 0.6 * f, m), cancelled)
        elif mode == "facts":
            raise ValueError("facts & data are read from tagged pages, and " + store.why_no_regions(t) +
                             " (or use the fact book playbook for curated data)")
        if playbook != "builtin:factbook" and extra.get("capture"):   # the fact book already holds the captured facts
            fi, ft, _ = facts.render(topic_id, fo, model, parallel, lambda f, m="": progress(span * (0.6 + 0.4 * f), m), cancelled,
                                     uc=uc)
            items += fi
            tests += ft
    label = ""
    if mode in ("reasoning", "both"):
        pb_html, label = mine.resolve_playbook(topic_id, playbook)
        base = 0.45 if mode == "both" else 0.0
        sub = lambda f, m="": progress(base + (0.85 - base) * f, m)  # noqa: E731
        if builtin.is_builtin(playbook):                 # reasons over this text's captured data
            ri, rt_, bx = builtin.cases(topic_id, playbook, o, model, parallel, sub, cancelled, uc=uc)
            extra.update(bx)
        else:
            ri, rt_ = _reasoning(t, d, pb_html, o, model, parallel, sub, cancelled, uc)
        if mode == "both":
            ri = balance(ri, len(items), float(o["reasoning_share"]), extra)
        items += ri
        tests += rt_
    if chosen:                                       # extra mining: perform / review / limits material
        progress(0.86 if mode != "mining" else 0.0, f"extra mining: {', '.join(chosen)}")
        span0 = 0.0 if mode == "mining" else 0.86
        mi, mt, mrep = miners.run(topic_id, chosen, {**(o.get("mining") or {}), "max_len": o["max_len"]}, model, parallel,
                                  lambda f, m="": progress(span0 + (0.88 - span0) * f, m), cancelled, uc)
        items += mi
        tests += mt
        extra["mining"] = mrep
    if o["immersion"] and "passages" not in chosen:
        items += [{"text": s, "kind": "immersion", "check": ""} for s in immersion_passages(t, int(o["max_len"] * 3.5))]
    for e in store.rj(d / "examples.heldout.json", []) or []:
        tests.append({"question": e["facts"], "expected": e["conclusion"], "check": "", "source": f"book {e.get('doc')} p.{e.get('page')}"})
    progress(0.9, "quality checks")
    items, report = quality(items, tests, o)
    items = mix(items, o["mix"], report)
    random.Random(0).shuffle(items)
    report.update({"mode": mode, "items": len(items), "tests": len(tests), "by_kind": dict(Counter(x["kind"] for x in items)),
                   "playbook": label, "options": o, "units": uc.stats(), "dataset": did, **extra})
    status = "partial" if uc.failed else "complete"
    datasets.write_base(did, items, tests, report, status)
    datasets.set_current(topic_id, did)
    if uc.failed:
        progress(1.0, f"{uc.failed} units failed: press Continue to retry them")
    return report


def balance(ri: list[dict], n_data: int, share: float, rep: dict) -> list[dict]:
    """With facts in the set, repeat reasoning items (at most 3 copies) so they make up about `share` of the total;
    never drop facts. Reasoning needs facts, and facts swamp a thin reasoning set otherwise."""
    if not ri or n_data == 0:
        return ri
    want = int(share / max(1e-6, 1 - share) * n_data)
    reps = max(1, min(3, round(want / len(ri))))
    rep["reasoning_repeats"] = reps
    return [dict(x, copy=k) if k else x for k in range(reps) for x in ri]


def _reasoning(t, d, pb_html, o, model, parallel, progress, cancelled, uc=datasets.NULL):
    pb = htmlconv.parse_playbook(pb_html)
    if not pb["checks"]:
        raise ValueError("the playbook has no checks")
    rec = library.get(t.get("record") or "") or {}
    situations = rec.get("situations", [])
    about = t.get("about") or t.get("name", "")
    def per_check(c):
        ch = prompts.check_html(c)
        keys = {k: o[k] for k in ("walkthrough", "pair", "nextcheck", "test", "situated_scenes")}
        return uc.run(uc.key("check", ch, keys, about, situations, model), lambda: _check_units(c, ch))

    def _check_units(c, ch):
        its, tst = [], []
        for kind in ("walkthrough", "pair", "nextcheck"):
            if o[kind]:
                out = llm.chat(prompts.generate_prompt(kind, ch, o[kind] * (2 if kind == "pair" else 1), about, situations),
                               model=model, max_tokens=4096, temperature=0.7)
                for cs in htmlconv.parse_cases(out):
                    its.append({"text": case_text(cs), "kind": kind, "check": c["id"], "pair": cs.get("pair", "")})
        if o["test"]:
            out = llm.chat(prompts.generate_prompt("test", ch, o["test"], about, situations), model=model, max_tokens=3000,
                           temperature=0.7)
            for cs in htmlconv.parse_cases(out):
                tst.append({"question": cs["situation"], "expected": cs["conclusion"], "check": c["id"], "source": "generated"})
        facts = [it["text"] for it in c["items"] if it["kind"] in ("fact", "cue", "exception", "distinction")]
        for f in facts:
            its.append({"text": f"{c['label']} — {f}", "kind": "fact", "check": c["id"]})
        if facts and o["situated_scenes"]:
            out = llm.chat(prompts.situated_prompt(facts, o["situated_scenes"], situations, about), model=model,
                           max_tokens=4096, temperature=0.8)
            its += [{"text": s, "kind": "situated", "check": c["id"]} for s in htmlconv.parse_items(out, "situated")]
        return its, tst

    items, tests = [], []
    res = par.pmap(per_check, pb["checks"], parallel,
                   lambda k, n, r: progress(0.8 * k / n, f"check {k}/{n}" + (f" failed: {r['_error'][:120]}" if par.failed(r) else "")),
                   cancelled)
    for r in res:
        if not par.failed(r):
            items += r[0]
            tests += r[1]
    if o["method"]:
        def meth():
            out = llm.chat(prompts.method_prompt(pb_html[:12000], about, o["method"]), model=model, max_tokens=3000, temperature=0.7)
            return [{"text": s, "kind": "method", "check": ""} for s in htmlconv.parse_items(out, "method")]
        try:
            items += uc.run(uc.key("method", pb_html, about, o["method"], model), meth)
        except Exception as e:                       # noqa: BLE001 - one failed unit; Continue retries it
            from packsmith.llm.llm import Cancelled
            if isinstance(e, Cancelled):
                raise
            progress(0.85, f"method passages failed: {e}")
    return items, tests


def quality(items: list[dict], tests: list[dict], o: dict) -> tuple[list[dict], dict]:
    rep = {"negative_flagged": 0, "negative_dropped": 0, "duplicates": 0, "leaks": 0, "flagged_examples": []}
    out, seen = [], set()
    for x in items:
        h = hashlib.sha1(f"{_norm(x['text'])}|{x.get('copy', 0)}".encode()).hexdigest()   # deliberate repeats carry copy=k
        if h in seen or len(x["text"].strip()) < 20:
            rep["duplicates"] += 1
            continue
        seen.add(h)
        if NEG.search(x["text"]):
            rep["negative_flagged"] += 1
            if o.get("drop_negative") and x["kind"] in ("fact", "situated", "method"):
                rep["negative_dropped"] += 1
                if len(rep["flagged_examples"]) < 20:
                    rep["flagged_examples"].append(x["text"][:200])
                continue
        out.append(x)
    lt = [q for q in tests if not str(q.get("source", "")).startswith(("facts", "mining"))]   # knowledge tests share wording by design
    test_grams = set().union(*[ngrams(q["question"]) for q in lt]) if lt else set()
    # phrasing shared by many training items (boilerplate, the field's stock terms) is not a leak
    df = Counter(g for x in out for g in ngrams(x["text"]) & test_grams)
    common = {g for g, k in df.items() if k > max(3, 0.05 * len(out))}
    test_grams -= common
    rep["leak_ignored_common"] = len(common)
    clean = []
    for x in out:
        if x["kind"] not in ("immersion",) + facts.DATA_KINDS + MINED_KINDS and ngrams(x["text"]) & test_grams:
            rep["leaks"] += 1
            continue
        clean.append(x)
    return clean, rep


def mix(items: list[dict], ratios: dict, rep: dict, slack: float = 1.25, floor: int = 200) -> list[dict]:
    """Cap any kind that exceeds its share (with some slack) so it cannot swamp the others; never upsample —
    shortfalls are reported so the user can generate more of that kind. A kind is never cut below `floor` items:
    when the other kinds are scarce, the cap must not throw away the book's own material."""
    by = {}
    for x in items:
        by.setdefault(x["kind"], []).append(x)
    total_ratio = sum(ratios.get(k, 0) for k in by) or 1
    n = sum(len(v) for k, v in by.items() if ratios.get(k, 0) > 0)    # kinds without a share (data) are left whole
    out, short = [], {}
    rng = random.Random(1)
    for k, v in by.items():
        r = ratios.get(k, 0) / total_ratio
        want = int(round(n * r)) if r > 0 else len(v)
        cap = max(int(round(want * slack)), min(len(v), floor)) if r > 0 else len(v)
        if len(v) > cap:
            v = rng.sample(v, cap)
        elif len(v) < want:
            short[k] = want - len(v)
        out += v
    rep["mix_shortfall"] = short
    return out


def load(topic_id: str) -> tuple[list[dict], list[dict]]:
    d = store.topic_dir(topic_id) / "dataset"
    rd = lambda p: [json.loads(x) for x in store.rt(p).splitlines() if x.strip()]  # noqa: E731
    return rd(d / "train.jsonl"), rd(d / "tests.jsonl")
