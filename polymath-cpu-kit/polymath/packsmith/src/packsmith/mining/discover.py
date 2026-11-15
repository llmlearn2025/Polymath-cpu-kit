"""Guided discovery: stream pages in rounds until the mining plan stops changing and the topic is covered."""
from __future__ import annotations

import json
import time

from packsmith.mining import htmlconv
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.datasets import prompts
from packsmith.system.config import SETTINGS


def build_prompt(topic_id: str, record_id: str | None = None) -> str:
    t = store.topic(topic_id)
    rec = library.get(record_id) if record_id else None
    srcs = []
    for doc, r in store.topic_regions(t, ("train", "context", "heldout")):
        srcs.append(f"{doc} [{store.doc_meta(doc)['name']}] {r['type']} '{r['name']}' p.{r['start']}–{r['end']} ({r['role']})")
    return prompts.discovery_prompt(t, rec, srcs)


def sample_order(t: dict) -> list[tuple[str, int, str]]:
    """(doc, page, region_id): contents (its first 4 pages) → opening pages of each train region → pages spread evenly
    across train regions (round robin) → the rest of the contents → index."""
    regs = store.topic_regions(t, ("train", "context"))
    order, seen = [], set()

    def add(doc, p, rid):
        if (doc, p) not in seen:
            seen.add((doc, p))
            order.append((doc, p, rid))

    late = []                            # a long contents region: its first pages first, the rest after the chapters
    for doc, r in regs:
        if r["type"] == "contents":
            for p in range(r["start"], r["end"] + 1):
                if p - r["start"] < 4:
                    add(doc, p, r["id"])
                else:
                    late.append((doc, p, r["id"]))
    train = [(doc, r) for doc, r in regs if r["role"] == "train" and r["type"] not in store.CONTEXT_TYPES]
    for doc, r in train:
        for p in range(r["start"], min(r["end"], r["start"] + 1) + 1):
            add(doc, p, r["id"])
    spreads = []
    for doc, r in train:
        pages = list(range(r["start"], r["end"] + 1))
        step = max(1, len(pages) // 8)
        spreads.append([(doc, p, r["id"]) for p in pages[::step]] + [(doc, p, r["id"]) for p in pages])
    i = 0
    while any(spreads):
        lst = spreads[i % len(spreads)]
        if lst:
            add(*lst.pop(0))
        i += 1
        if i > 100000:
            break
    for x in late:
        add(*x)
    for doc, r in regs:
        if r["type"] == "index":
            for p in range(r["start"], r["end"] + 1):
                add(doc, p, r["id"])
    return order


def run(topic_id: str, system_prompt: str, model: str | None = None, pages_per_round: int = 6, stable_rounds: int = 2,
        min_coverage: float = 0.6, max_pages: int = 120, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    t = store.topic(topic_id)
    d = store.topic_dir(topic_id)
    store.wt(d / "prompt.html", system_prompt)
    order = sample_order(t)
    train_ids = {r["id"] for _, r in store.mined_regions(t, ("train",))}
    if not train_ids:                    # discovery would read only contents/index pages, at 0% coverage, for nothing
        raise ValueError(store.why_no_regions(t))
    if not order:
        raise ValueError("the topic has no regions with role train/context")
    plan_html, names_prev, stable, covered, read = "", set(), 0, set(), 0
    log = d / "discovery_log.jsonl"
    log.unlink(missing_ok=True)
    rnd = 0
    while order and read < max_pages and not cancelled():
        rnd += 1
        batch, order = order[:pages_per_round], order[pages_per_round:]
        read += len(batch)
        covered |= {rid for _, _, rid in batch if rid in train_ids}
        pages = prompts.pages_block([(doc, p, store.page_text(doc, p)) for doc, p, _ in batch],
                                    llm.budget_chars(model, 6000, len(system_prompt) + len(plan_html) + 1500))
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompts.discovery_round(plan_html, pages, rnd)}]
        out = llm.chat(msgs, model=model, max_tokens=6000)
        ok, why = htmlconv.validate(out, "plan")
        if not ok:
            msgs += [{"role": "assistant", "content": out}, {"role": "user", "content": f"Invalid: {why}. Return the plan in the FORMAT."}]
            out = llm.chat(msgs, model=model, max_tokens=6000)
            ok, why = htmlconv.validate(out, "plan")
        if ok:
            plan_html = out
        plan = htmlconv.parse_plan(plan_html)
        names = {m["name"] for m in plan["moves"]}
        new = names - names_prev
        stable = stable + 1 if (not new and names == names_prev and names) else 0
        names_prev = names
        coverage = len(covered) / max(1, len(train_ids))
        rec = {"round": rnd, "pages": [[doc, p] for doc, p, _ in batch], "moves": sorted(names), "new": sorted(new),
               "stable": stable, "coverage": round(coverage, 2), "confidence": plan["confidence"], "valid": ok,
               "time": time.strftime("%H:%M:%S")}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        store.wt(d / "plan.draft.html", plan_html)
        progress(min(0.99, read / max_pages), f"round {rnd}: {len(names)} moves, {len(new)} new, coverage {coverage:.0%}")
        if stable >= stable_rounds and coverage >= min_coverage:
            break
    reason = ("stable" if stable >= stable_rounds and len(covered) / max(1, len(train_ids)) >= min_coverage else
              "page cap" if read >= max_pages else "cancelled" if cancelled() else "all pages read")
    return {"rounds": rnd, "pages_read": read, "moves": sorted(names_prev), "stopped": reason}


def approve(topic_id: str, plan_html: str) -> dict:
    ok, why = htmlconv.validate(plan_html, "plan")
    if not ok:
        raise ValueError(why)
    store.wt(store.topic_dir(topic_id) / "plan.html", plan_html)
    return htmlconv.parse_plan(plan_html)


def rounds(topic_id: str) -> list[dict]:
    p = store.topic_dir(topic_id) / "discovery_log.jsonl"
    return [json.loads(x) for x in store.rt(p).splitlines() if x.strip()]
