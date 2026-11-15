"""Back-test: the text's own worked examples, conclusion hidden, solved with the playbook only."""
from __future__ import annotations

import re

from packsmith.mining import htmlconv, mine
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.system.config import SETTINGS


def find_examples(topic_id: str, role: str = "train", model: str | None = None, parallel: int | None = None,
                  progress=lambda f, m="": None) -> list[dict]:
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id)
    bl = mine.blocks(t, (role,), budget=llm.budget_chars(model, 8000, 2500))
    if not bl:
        raise ValueError(store.why_no_regions(t, (role,)) + (
            "; for a final test, tag a chapter the training does not use with the role heldout" if role == "heldout" else ""))
    cache = d / f"examples.{role}.{store.regions_signature(t, (role,))}.json"   # other regions: found afresh
    if cache.exists():
        got = store.rj(cache, [])
        if got:                                            # an empty result is not kept: it may have been a failure
            return got

    def one(b):
        return [dict(e, doc=e.get("doc") or b["doc"]) for e in htmlconv.parse_examples(
            llm.chat(prompts.examples_prompt(b["text"]), model=model, max_tokens=8000))]
    res = par.pmap(one, bl, parallel, lambda k, n, r: progress(0.3 * k / n, f"examples: block {k}/{n}"))
    out = [e for r in res if not par.failed(r) for e in r]
    store.wj(cache, out)
    return out


def run(topic_id: str, playbook: str = "unified", role: str = "train", model: str | None = None, limit: int = 60,
        parallel: int | None = None, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    d = store.topic_dir(topic_id)
    pb, _label = mine.resolve_playbook(topic_id, playbook)          # any topic's playbook can be tested here
    pb = pb[: llm.budget_chars(model, 1500, 2500)]
    ex = find_examples(topic_id, role, model, parallel, progress)[:limit]
    if not ex:
        raise ValueError(f"the model found no worked examples (facts → conclusion) in the {role} regions. A back-test "
                         "needs the text's own worked problems; texts without them (narrative, fiction) cannot be "
                         "back-tested this way: use the dataset's tests in the Workbench instead")
    def one(e):
        sol = llm.chat(prompts.solve_prompt(pb, e["facts"]), model=model, max_tokens=1500)
        cases = htmlconv.parse_cases(sol)
        pred = cases[0]["conclusion"] if cases else re.sub(r"<[^>]+>", " ", sol)[-400:]
        j = llm.chat(prompts.judge_match_prompt(e["facts"], e["conclusion"], pred), model=model, max_tokens=300)
        v = (htmlconv.parse_items(j, "verdict") or ["miss"])[0].strip().lower()
        v = "match" if v.startswith("match") else "partial" if v.startswith("partial") else "miss"
        return {**e, "predicted": pred, "steps": cases[0]["steps"] if cases else [], "verdict": v,
                "why": (htmlconv.parse_items(j, "why") or [""])[0]}
    res = par.pmap(one, ex, parallel, lambda k, n, r: progress(0.3 + 0.7 * k / n, f"example {k}/{n}: {r.get('verdict', r.get('_error', ''))}"),
                   cancelled)
    rows = [r for r in res if not par.failed(r)]
    score = sum(1 if r["verdict"] == "match" else 0.5 if r["verdict"] == "partial" else 0 for r in rows) / max(1, len(rows))
    res = {"playbook": playbook, "role": role, "n": len(rows), "score": round(score, 3),
           "match": sum(r["verdict"] == "match" for r in rows), "partial": sum(r["verdict"] == "partial" for r in rows),
           "miss": sum(r["verdict"] == "miss" for r in rows), "rows": rows}
    allres = store.rj(d / "backtest.json", {}) or {}
    allres[f"{playbook}:{role}"] = res
    store.wj(d / "backtest.json", allres)
    t = store.topic(topic_id)
    if t.get("record"):
        library.record_score(t["record"], f"backtest:{topic_id}", res["score"])
    return {k: v for k, v in res.items() if k != "rows"}
