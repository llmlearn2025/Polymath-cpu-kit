"""Anchored reasoning traces: worked problems whose derivation is spelled out as ordered ANCHORS (facts → issue → law →
… → conclusion), each with its result, the words it rests on and what it depends on — and only kept when it passes
checks. The aim is to teach the procedure that connects a problem to its answer, not the answer.

Sources: the text's own worked examples (as the back-test finds them), the "Cases" miner's vignettes, and curated Q&A
that carry a source passage. For each:
  1. SOLVE without the answer (reference = the topic's playbook, or the fact book / captured facts);
     kept as "solved" when its conclusion matches the source's;
  2. otherwise RECONSTRUCT with the answer, then CHECK: the steps alone (conclusion removed) must lead to the source's
     conclusion; kept as "reconstructed";
  3. VALIDATE the anchors (software checks, below); a trace that fails is dropped and listed for review.

Validators:
  facts      every quoted piece of evidence appears in the problem or the reference (word-for-word coverage)
  law        every provision cited appears in the problem, the reference or the topic's sources; old codes (IPC, CrPC,
             Indian Evidence Act) are flagged as outdated unless the source itself uses them
  precedent  every case cited ("X v. Y") appears in the problem, the reference or the sources
  depends    each anchor after the first names an earlier anchor it uses
  counter    the counterargument / red-flag anchor says something specific (not a stock caveat)
  conclusion matches the source's conclusion (judge), and for reconstructions the steps-only re-solve matches too

Anchor sets: legal, medical, general (any field with checkable answers) and craft (literature: opt-in only — the
conclusion is the author's passage, so a craft trace is an unverified rationale, kept for study, not as reasoning).
"""
from __future__ import annotations

import json
import random
import re

from packsmith.evaluation import backtest
from packsmith.datasets import curated, prompts
from packsmith.mining import facts, htmlconv, mine
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.system import par

ANCHOR_SETS = {
    "legal": [("facts", "Facts", "what has actually been established, and by what"),
              ("issue", "Issue", "the legal question that must be decided"),
              ("law", "Law", "the provisions or rules that govern it, cited exactly as in the reference"),
              ("precedent", "Precedent", "authorities that support or cut against the reading (only those in the reference)"),
              ("interpretation", "Interpretation", "what the law means in this context, element by element"),
              ("application", "Application", "each element applied to the facts"),
              ("counter", "Counterargument", "what could defeat this — an exception, a distinction, a procedural bar"),
              ("conclusion_anchor", "Justified conclusion", "what follows, and what remains uncertain")],
    "medical": [("presentation", "Presentation", "the patient, the complaint, the setting"),
                ("findings", "Findings", "the observations and results actually given"),
                ("pattern", "Pattern", "what the findings add up to (the syndrome or sign pattern)"),
                ("differential", "Differential", "the competing explanations, most likely first"),
                ("discriminators", "Discriminators", "the findings or tests that separate them"),
                ("working", "Working diagnosis", "the explanation the evidence best supports"),
                ("counter", "Red flags", "what would change the conclusion or demand escalation"),
                ("conclusion_anchor", "Management / impression", "what follows: report, next step, management")],
    "general": [("facts", "Givens", "what the problem states"),
                ("question", "Question", "what must be determined"),
                ("rules", "Rules", "the rules, methods or values that apply (from the reference)"),
                ("steps", "Working", "applying them, step by step"),
                ("check", "Check", "a check on the result: another route, a limit case, units"),
                ("counter", "Exceptions", "what would change the answer"),
                ("conclusion_anchor", "Result", "the answer and its limits")],
    "craft": [("situation", "Situation", "what the scene must do in the story"),
              ("choice", "Choices", "point of view, what is shown and what is left out"),
              ("devices", "Devices", "rhythm, repetition, dialogue, imagery used"),
              ("counter", "Alternatives", "what the author did not do, and why it would weaken the scene"),
              ("conclusion_anchor", "Effect", "the effect on the reader")],
}
OLD_CODES = re.compile(r"\b(IPC|I\.P\.C\.|Indian Penal Code|Cr\.?\s?P\.?\s?C\.?|Code of Criminal Procedure,? 1973|Indian Evidence Act|Evidence Act,? 1872)\b", re.I)
PROVISION = re.compile(r"\b(?:[Ss]ection|[Ss]ec\.|[Ss]\.|[Aa]rticle|[Aa]rt\.|[Rr]ule|[Oo]rder)\s*(\d+[A-Z]?(?:\(\w+\))*)")
CASE = re.compile(r"\b([A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+){0,4})\s+v(?:s)?\.?\s+([A-Z][\w.&'-]+(?:\s+[A-Z][\w.&'-]+){0,4})")
STOCK = re.compile(r"^(it (may|could) be argued|however,? (this|it) (may|might)|there (may|might) be exceptions|subject to)", re.I)


def anchor_set(topic: dict, choice: str = "") -> str:
    if choice in ANCHOR_SETS:
        return choice
    rec = library.get(topic.get("record") or "") or {}
    dom = f"{topic.get('domain', '')} {rec.get('domain', '')} {topic.get('intent', '')}".lower()
    if any(w in dom for w in ("law", "legal", "court", "statute", "drafting")):
        return "legal"
    if any(w in dom for w in ("medic", "clinic", "radiol", "patho", "nurs", "pharm", "health")):
        return "medical"
    if any(w in dom for w in ("literat", "fiction", "novel", "poet", "craft")):
        return "craft"
    return "general"


def parse_trace(html: str, keys: list[str]) -> dict | None:
    root = htmlconv.parse(html)
    art = root.find("trace") or root
    anchors = {}
    for s in art.find_all("anchor"):
        k = s.attrs.get("data-a", "")
        anchors[k] = {"result": " ".join(x.text() for x in s.find_all("result")),
                      "evidence": [x.text() for x in s.find_all("evidence") if x.text()],
                      "depends": " ".join(x.text() for x in s.find_all("depends"))}
    concl = " ".join(x.text() for x in art.find_all("conclusion") if x.parent is art or True).strip()
    if not anchors or not concl:
        return None
    return {"anchors": {k: anchors[k] for k in keys if k in anchors}, "conclusion": concl}


def trace_text(tr: dict, sets: list[tuple[str, str, str]], with_conclusion: bool = True, evidence: bool = True) -> str:
    """One line per anchor: LABEL: result (from: “the words it rests on”). The evidence teaches the pack to ground each
    step, and lets its own traces be checked the same way."""
    lab = {k: l for k, l, _ in sets}
    lines = []
    for k, a in tr["anchors"].items():
        L = lab.get(k, k)
        res = a["result"].split(":", 1)[-1].strip() if a["result"].lower().startswith(L.lower()) or a["result"].lower().startswith(k) else a["result"]
        ev = (" (from: " + "; ".join(f"“{e}”" for e in a["evidence"][:2]) + ")") if evidence and a.get("evidence") else ""
        lines.append(f"{L.upper()}: {res}{ev}")
    return "\n".join(lines) + (f"\n\nConclusion: {tr['conclusion']}" if with_conclusion else "")


LINE = re.compile(r"^\s*([A-Z][A-Z /]+?):\s*(.*)$")


def parse_text_trace(text: str, sets: list[tuple[str, str, str]]) -> dict:
    """A trace written as 'LABEL: result (from: “…”)' lines (how packs are trained to write them) → the trace dict."""
    by_label = {l.upper(): k for k, l, _ in sets}
    anchors, concl = {}, ""
    for line in text.splitlines():
        m = LINE.match(line)
        if m and m.group(1).strip() in by_label:
            body = m.group(2)
            ev = re.findall(r"“([^”]+)”", body)
            anchors[by_label[m.group(1).strip()]] = {"result": re.sub(r"\s*\(from:.*\)\s*$", "", body), "evidence": ev,
                                                     "depends": "earlier" if anchors else ""}
        elif line.lower().startswith(("conclusion:", "answer:")):
            concl = line.split(":", 1)[1].strip()
    return {"anchors": anchors, "conclusion": concl}


def validate(tr: dict, problem: str, reference: str, sources: str, keys: list[str], kind: str) -> dict:
    """Software checks on one trace → {check: ok/False/'flag'} and a list of problems."""
    pool = f"{problem}\n{reference}\n{sources}"
    probs, checks = [], {}
    present = [k for k in keys if k in tr["anchors"]]
    checks["complete"] = len(present) >= max(3, len(keys) - 2)
    if not checks["complete"]:
        probs.append(f"only {len(present)} of {len(keys)} anchors")
    ev_bad = []
    for k, a in tr["anchors"].items():
        for e in a["evidence"]:
            if len(e.split()) >= 3 and facts_overlap(e, pool) < 0.5:
                ev_bad.append(f"{k}: “{e[:80]}”")
    checks["evidence"] = not ev_bad
    if ev_bad:
        probs.append("evidence not found in the problem or reference: " + "; ".join(ev_bad[:3]))
    body = trace_text(tr, [], True)
    if kind in ("legal", "general", "medical"):
        cited = {m.group(1) for m in PROVISION.finditer(body)}
        missing = [c for c in cited if not re.search(rf"\b{re.escape(c)}\b", pool)]
        checks["law"] = not missing
        if missing:
            probs.append("provisions not in the sources: " + ", ".join(sorted(missing)[:5]))
        old = {m.group(1) for m in OLD_CODES.finditer(body)}
        if old and not OLD_CODES.search(problem + "\n" + reference):
            checks["currency"] = "flag"
            probs.append("cites an old code (" + ", ".join(sorted(old)) + "): check the current provision")
        cases = {f"{m.group(1)} v. {m.group(2)}" for m in CASE.finditer(body)}
        bad = [c for c in cases if c.split(" v. ")[0].split()[-1] not in pool]
        checks["precedent"] = not bad
        if bad:
            probs.append("cases not in the sources: " + "; ".join(bad[:3]))
    deps_missing = [k for k in present[1:] if not tr["anchors"][k]["depends"].strip()]
    checks["depends"] = len(deps_missing) <= max(1, len(present) // 4)
    if not checks["depends"]:
        probs.append("anchors without dependencies: " + ", ".join(deps_missing))
    c = tr["anchors"].get("counter", {}).get("result", "")
    checks["counter"] = bool(c) and len(c) > 30 and not STOCK.match(c.split(":", 1)[-1].strip())
    if not checks["counter"]:
        probs.append("the counterargument / red-flag anchor is missing or generic")
    return {"checks": checks, "problems": probs}


def facts_overlap(text: str, pool: str) -> float:
    from packsmith.mining.miners import verbatim
    return verbatim(text, pool, n=4)


def judge(facts_: str, expected: str, predicted: str, model) -> str:
    j = llm.chat(prompts.judge_match_prompt(facts_, expected, predicted), model=model, max_tokens=300)
    v = (htmlconv.parse_items(j, "verdict") or ["miss"])[0].strip().lower()
    return "match" if v.startswith("match") else "partial" if v.startswith("partial") else "miss"


def gather(topic_id: str, model=None, parallel=None, progress=lambda f, m="": None, roles=("train",)) -> list[dict]:
    """Problems with known conclusions: worked examples, mined cases, curated Q&A with a source passage."""
    t = store.topic(topic_id)
    out = []
    for role in roles:
        if not store.mined_regions(t, (role,)):
            continue
        for e in backtest.find_examples(topic_id, role, model, parallel, progress):
            if e.get("facts") and e.get("conclusion"):
                out.append({"problem": e["facts"], "answer": e["conclusion"], "source": f"example {e.get('doc', '')} p.{e.get('page', '')}",
                            "role": role})
    for x in [json.loads(l) for l in store.rt(store.topic_dir(topic_id) / "mining" / "vignettes.jsonl").splitlines() if l.strip()]:
        out.append({"problem": x["input"], "answer": x["output"], "source": f"case p.{x.get('page', '')}", "role": "train"})
    for cid in t.get("curated", []):
        for u in curated.units(cid):
            if u["kind"] == "qa" and u.get("question") and u.get("answer"):
                out.append({"problem": u["question"] + (f"\n\nSource: {u['context']}" if u.get("context") else ""), "answer": u["answer"],
                            "source": u["src"], "role": "train"})
    seen, uniq = set(), []
    for x in out:
        k = x["problem"][:160]
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq


def reference_for(topic_id: str, budget: int) -> str:
    d = store.topic_dir(topic_id)
    pb = store.rt(d / "playbook.html")
    if pb:
        return pb[:budget]
    lines = [x.get("text", "") for x in facts.records(topic_id)][:400]
    lines += [f"{e.get('subject', '')}: {e.get('answer') or e.get('text', '')}" for e in
              [json.loads(l) for l in store.rt(d / "factbook" / "entries.jsonl").splitlines() if l.strip()][:300]]
    return "\n".join(lines)[:budget]


def sources_text(topic_id: str, limit: int = 400000) -> str:
    t = store.topic(topic_id)
    parts = [store.page_text(doc, p) for doc, r in store.topic_regions(t) for p in range(r["start"], r["end"] + 1)]
    return "\n".join(parts)[:limit]


def run(topic_id: str, opts: dict | None = None, model=None, parallel=None, progress=lambda f, m="": None,
        cancelled=lambda: False, uc=None, problems: list[dict] | None = None) -> tuple[list[dict], list[dict], dict]:
    from packsmith.datasets.datasets import NULL
    uc = uc or NULL
    o = {"anchors": "", "allow_craft": False, "reconstruct": True, "max": 300, "test_share": 0.0, **(opts or {})}
    t = store.topic(topic_id)
    kind = anchor_set(t, o["anchors"])
    if kind == "craft" and not o["allow_craft"]:
        raise ValueError("this topic looks like literature: craft traces are unverified rationales. Tick 'craft notes "
                         "(unverified)' to make them anyway, or use the Writer miners")
    sets = ANCHOR_SETS[kind]
    keys = [k for k, _, _ in sets]
    about = t.get("about") or t.get("name", "")
    ref = reference_for(topic_id, int(llm.budget_chars(model, 8000, 6000) * 0.6))
    src = sources_text(topic_id)
    probs = problems if problems is not None else gather(topic_id, model, parallel, lambda f, m="": progress(0.1 * f, m))
    probs = probs[: int(o["max"])]
    if not probs:
        raise ValueError("no worked problems found: traces need the text's worked examples (tag regions with them), "
                         "the Cases miner, or curated Q&A")

    def one(p):
        res = {"problem": p["problem"], "answer": p["answer"], "source": p["source"], "set": kind}
        out = uc.run(uc.key("trace-solve", kind, ref, p["problem"], model),
                     lambda: llm.chat(prompts.trace_solve_prompt(sets, ref, p["problem"], about), model=model, max_tokens=6000, temperature=0.3))
        tr = parse_trace(out, keys)
        how = "solved"
        verdict = judge(p["problem"], p["answer"], tr["conclusion"], model) if tr else "miss"
        if verdict != "match" and o["reconstruct"]:
            out = uc.run(uc.key("trace-reconstruct", kind, ref, p["problem"], p["answer"], model),
                         lambda: llm.chat(prompts.trace_reconstruct_prompt(sets, ref, p["problem"], p["answer"], about), model=model,
                                          max_tokens=6000, temperature=0.3))
            tr = parse_trace(out, keys)
            how = "reconstructed"
            if tr:
                steps = trace_text(tr, sets, with_conclusion=False)
                rs = uc.run(uc.key("trace-check", p["problem"], steps, model),
                            lambda: llm.chat(prompts.trace_check_prompt(p["problem"], steps), model=model, max_tokens=1500))
                rc = " ".join(htmlconv.parse_items(rs, "conclusion")) or rs
                verdict = judge(p["problem"], p["answer"], rc, model)
        if not tr:
            return {**res, "kept": False, "why": ["no usable trace"]}
        v = validate(tr, p["problem"], ref, src, keys, kind)
        hard = [k for k, ok in v["checks"].items() if ok is False]
        kept = verdict == "match" and not hard
        return {**res, "trace": tr, "how": how, "verdict": verdict, "checks": v["checks"], "problems": v["problems"],
                "kept": kept, "why": ([] if verdict == "match" else [f"conclusion: {verdict}"]) + v["problems"]}
    res = par.pmap(one, probs, parallel, lambda k, n, r: progress(0.1 + 0.9 * k / n, f"traces {k}/{n}"
                   + ("" if par.failed(r) else f" ({'kept' if r.get('kept') else 'dropped'})")), cancelled)
    rows = [r for r in res if not par.failed(r)]
    d = store.topic_dir(topic_id) / "mining"
    store.wt(d / "traces.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    kept = [r for r in rows if r["kept"]]
    rng = random.Random(5)
    rng.shuffle(kept)
    nt = int(len(kept) * float(o["test_share"]))
    tests = [{"question": r["problem"], "expected": r["answer"], "check": "", "source": f"mining trace {r['source']}"} for r in kept[:nt]]
    items = [{"text": f"Question: {r['problem']}\nReasoning:\n{trace_text(r['trace'], sets, False)}\nAnswer: {r['trace']['conclusion']}",
              "kind": "trace", "how": r["how"], "source": r["source"]} for r in kept[nt:]]
    rep = {"set": kind, "problems": len(probs), "kept": len(kept), "solved": sum(r.get("how") == "solved" for r in kept),
           "reconstructed": sum(r.get("how") == "reconstructed" for r in kept), "dropped": len(rows) - len(kept),
           "failed_calls": len(res) - len(rows),
           "drop_reasons": dict(sorted(_reasons(rows).items(), key=lambda kv: -kv[1])[:8]),
           "unverified": kind == "craft"}
    store.wj(d / "traces_report.json", rep)
    return items, tests, rep


def _reasons(rows):
    c = {}
    for r in rows:
        if not r.get("kept"):
            for w in r.get("why", []):
                k = w.split(":")[0]
                c[k] = c.get(k, 0) + 1
    return c
