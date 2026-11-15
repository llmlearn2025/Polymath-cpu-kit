"""Transfer pilot: does a pack trained on verified, anchored reasoning traces solve NEW kinds of problems better than a
pack trained on the same problems' answers alone — or does it only learn the format?

  1. Problems with known answers from the topic's TRAIN regions (worked examples, mined cases, curated Q&A); tests from
     its HELD-OUT regions (a chapter the packs never see = new problem families). Without held-out regions a random
     quarter is held out, and the report says that this tests rewording, not transfer.
  2. Traces for the training problems (traces.run: solve → reconstruct → verify → validate).
  3. Two datasets over exactly the same kept problems: A = question → answer; B = question → anchored trace → answer.
     Both carry the same held-out tests.
  4. Train both packs with the same settings (answer-only loss), evaluate both on the held-out tests (conclusion judged
     against the source), and probe reasoning discipline: base, A and B are asked to reason in anchors on the held-out
     problems; their traces are checked (invented provisions, invented cases, evidence not in the problem, missing anchors).
  5. Report: pilot/<id>.json, shown in the Workbench.
"""
from __future__ import annotations

import json
import random
import time

from packsmith.datasets import datasets
from packsmith.packs import store
from packsmith.mining import traces
from packsmith.training import train
from packsmith.workbench import workbench
from packsmith.system.config import SETTINGS


def pdir():
    d = store.data_dir() / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    return d


def listing() -> list[dict]:
    return sorted((store.rj(p) for p in pdir().glob("*.json")), key=lambda r: r.get("started", ""), reverse=True)


def get(pid: str) -> dict:
    return store.rj(pdir() / f"{pid}.json") or {}


def _probe(pack: str | None, problems: list[dict], sets, ref: str, src: str, kind: str, max_tokens: int = 700) -> dict:
    """Ask the engine (base or a pack) to reason in anchors; check its traces with the same validators."""
    from packsmith.system import gpu
    gpu.need_engine()
    e = workbench._eng()
    with workbench._lock:
        e.swap(workbench.pack_path(pack) if pack else None)
        rows = []
        for p in problems:
            out = workbench.engine_ask(e, f"Question: {p['problem']}\nReasoning:\n", max_tokens, 0.0, 1.0)
            tr = traces.parse_text_trace(out, sets)
            v = traces.validate(tr, p["problem"], ref, src, [k for k, _, _ in sets], kind) if tr["anchors"] else {"checks": {}, "problems": ["no anchors"]}
            rows.append({"problem": p["problem"][:300], "output": out[:2000], "anchors": len(tr["anchors"]), "checks": v["checks"],
                         "problems": v["problems"]})
        e.swap(None)
    n = max(1, len(rows))
    rate = lambda key: round(sum(1 for r in rows if r["checks"].get(key) is False) / n, 3)  # noqa: E731
    return {"n": len(rows), "anchored": round(sum(1 for r in rows if r["anchors"] >= 3) / n, 3),
            "invented_provisions": rate("law"), "invented_cases": rate("precedent"), "evidence_not_in_problem": rate("evidence"),
            "outdated_codes": round(sum(1 for r in rows if r["checks"].get("currency") == "flag") / n, 3), "rows": rows[:30]}


def run(topic_id: str, opts: dict | None = None, model: str | None = None, parallel: int | None = None,
        progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    holder = {}
    try:
        return _run(topic_id, opts, model, parallel, progress, cancelled, holder)
    except Exception as e:
        if holder.get("rep"):
            holder["rep"].update(status="failed", error=f"{type(e).__name__}: {e}")
            store.wj(pdir() / f"{holder['rep']['id']}.json", holder["rep"])
        raise


def _run(topic_id, opts, model, parallel, progress, cancelled, holder) -> dict:
    o = {"max_train": 100, "max_test": 30, "train": True, "evaluate": True, "probe": 15,
         "train_opts": {"epochs": 2, "loss": "answer"}, "traces": {}, **(opts or {})}
    t = store.topic(topic_id)
    pid = store.new_id(f"pilot-{t.get('name', topic_id)}")
    rep = {"id": pid, "topic": topic_id, "topic_name": t.get("name"), "started": time.strftime("%Y-%m-%d %H:%M"), "status": "running"}
    holder["rep"] = rep
    save = lambda: store.wj(pdir() / f"{pid}.json", rep)  # noqa: E731
    save()
    progress(0.0, "finding problems with known answers")
    tr_p = [p for p in traces.gather(topic_id, model, parallel, roles=("train",)) if p.get("role") == "train"]
    te_p = traces.gather(topic_id, model, parallel, roles=("heldout",)) if store.mined_regions(t, ("heldout",)) else []
    te_p = [p for p in te_p if p.get("role") == "heldout"]
    if te_p:
        rep["held_out_by"] = "chapter"
    else:
        random.Random(9).shuffle(tr_p)
        k = max(1, len(tr_p) // 4)
        te_p, tr_p = tr_p[:k], tr_p[k:]
        rep["held_out_by"] = "random"
        rep["warning"] = ("no held-out region: a random quarter of the problems is held out, so this tests rewording, not new "
                          "problem families. Tag a chapter as 'heldout' in the Library for a real transfer test.")
    tr_p, te_p = tr_p[: int(o["max_train"])], te_p[: int(o["max_test"])]
    if len(tr_p) < 3 or not te_p:
        raise ValueError(f"too few problems ({len(tr_p)} to train on, {len(te_p)} to test): the pilot needs worked examples in "
                         "train regions and in a held-out region (or curated Q&A)")
    rep.update(train_problems=len(tr_p), test_problems=len(te_p))
    save()
    progress(0.05, f"{len(tr_p)} training problems, {len(te_p)} held-out; writing and checking traces")
    uc = datasets.cache_for(topic_id, "pilot")
    items_b, _, trep = traces.run(topic_id, o["traces"], model, parallel, lambda f, m="": progress(0.05 + 0.3 * f, m), cancelled, uc,
                                  problems=tr_p)
    rows = [json.loads(l) for l in store.rt(store.topic_dir(topic_id) / "mining" / "traces.jsonl").splitlines() if l.strip()]
    kept = [r for r in rows if r.get("kept")]
    rep["traces"] = trep
    if len(kept) < 3:
        raise ValueError(f"only {len(kept)} traces passed the checks — not enough for a pilot; see the drop reasons: {trep.get('drop_reasons')}")
    items_a = [{"text": f"Question: {r['problem']}\nAnswer: {r['answer']}", "kind": "qa", "source": r["source"]} for r in kept]
    tests = [{"question": p["problem"], "expected": p["answer"], "check": "", "source": f"pilot held-out {p['source']}"} for p in te_p]
    sets = traces.ANCHOR_SETS[trep["set"]]
    out = {}
    for name, items in (("A", items_a), ("B", items_b)):
        m = datasets.create(f"pilot {t.get('name', '')} · {'answers only' if name == 'A' else 'anchored traces'}", topic_id, "pilot",
                            mode="pilot", pilot=pid, arm=name)
        datasets.write_base(m["id"], items, tests, {"pilot": pid, "arm": name, "items": len(items), "tests": len(tests)})
        out[name] = {"dataset": m["id"], "items": len(items)}
    rep["arms"] = out
    save()
    if o["train"]:
        for i, name in enumerate(("A", "B")):
            if cancelled():
                break
            pk = store.slug(f"pilot-{pid}-{name}", 60)
            progress(0.4 + 0.2 * i, f"training pack {name}")
            r = train.run(out[name]["dataset"], pk, "dataset", opts=dict(o["train_opts"]), unload_lmstudio=SETTINGS.llm_unload_for_gpu,
                          progress=lambda f, m="", i=i: progress(0.4 + 0.2 * i + 0.2 * f, m), cancelled=cancelled)
            out[name]["pack"] = r["pack"]
            out[name]["best_epoch"] = r["card"].get("best_epoch")
            save()
    if o["evaluate"] and all(out[n].get("pack") for n in out):
        for i, name in enumerate(("A", "B")):
            progress(0.8 + 0.05 * i, f"evaluating pack {name} on the held-out problems")
            ev = workbench.evaluate(out[name]["pack"], str(datasets.tests_path(out[name]["dataset"])), limit=len(tests), template="qa",
                                    parallel=parallel, cancelled=cancelled)
            out[name]["eval"] = {k: ev[k] for k in ("base_score", "pack_score", "better", "worse", "n", "id")}
            save()
        ref = traces.reference_for(topic_id, 20000)
        src = traces.sources_text(topic_id)
        probe = te_p[: int(o["probe"])]
        rep["discipline"] = {}
        for i, (lab, pk) in enumerate((("base", None), ("A", out["A"]["pack"]), ("B", out["B"]["pack"]))):
            progress(0.9 + 0.03 * i, f"reasoning-discipline probe: {lab}")
            rep["discipline"][lab] = _probe(pk, probe, sets, ref, src, trep["set"])
            save()
        a, b = out["A"]["eval"]["pack_score"], out["B"]["eval"]["pack_score"]
        da, db = rep["discipline"]["A"], rep["discipline"]["B"]
        rep["verdict"] = (
            f"On {len(tests)} held-out problems ({rep['held_out_by']}): answers-only {a}, traces {b}, base {out['A']['eval']['base_score']}. "
            + ("Traces transfer better. " if b - a >= 0.1 else "No clear transfer gain. " if abs(b - a) < 0.1 else "Answers-only did better. ")
            + f"Anchored reasoning on new problems: {round(db['anchored'] * 100)}% with traces vs {round(da['anchored'] * 100)}% answers-only; "
            f"invented provisions {round(db['invented_provisions'] * 100)}% vs {round(da['invented_provisions'] * 100)}%."
            + (" Held out at random: this measures rewording, not transfer." if rep["held_out_by"] == "random" else ""))
    rep["status"] = "done"
    save()
    return rep
