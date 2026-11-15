"""Pipelines: a whole run from documents (or curated data) to a tested pack, as one resumable job.

  pipelines/<id>.json  {id, name, plan, steps: [{name, status, result, error}], status, created}

Steps: prepare (OCR scanned pages, tag the page ranges, make the topic) → dataset (generate; continues an unfinished
dataset) → train → evaluate. Each step records its result; running the pipeline again (resume, from the assistant or
from Jobs) skips the finished steps."""
from __future__ import annotations

import time

from packsmith.datasets import datasets, generate
from packsmith.system import jobs
from packsmith.mining import ocr
from packsmith.packs import store
from packsmith.training import train
from packsmith.workbench import workbench
from packsmith.system.config import SETTINGS

STEPS = ("prepare", "dataset", "train", "evaluate")
PRESET_LENSES = {"generic": ["facts", "tables", "glossary"], "report": ["facts", "tables", "recommendations", "glossary"],
                 "manual": ["facts", "procedures", "tables", "glossary"], "field-guide": ["entities", "facts", "glossary"],
                 "everything": ["facts", "tables", "entities", "procedures", "recommendations", "glossary"]}


def pdir():
    d = store.data_dir() / "pipelines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get(pid: str) -> dict:
    p = store.rj(pdir() / f"{pid}.json")
    if not p:
        raise KeyError(f"no pipeline {pid}")
    return p


def save(p: dict) -> dict:
    p["updated"] = time.strftime("%Y-%m-%d %H:%M")
    store.wj(pdir() / f"{p['id']}.json", p)
    return p


def listing() -> list[dict]:
    return sorted((store.rj(x) for x in pdir().glob("*.json")), key=lambda p: p.get("created", ""), reverse=True)


def normalise(plan: dict) -> dict:
    """Fill defaults and check a plan (from the assistant or a form)."""
    pl = dict(plan)
    pl["name"] = (pl.get("name") or "pack").strip()[:80]
    pl["mode"] = pl.get("mode") if pl.get("mode") in ("facts", "reasoning", "both", "mining") else "facts"
    from packsmith.mining import miners
    prof = miners.PROFILES.get(pl.get("profile") or "")
    pl["miners"] = sorted(set([m for m in (pl.get("miners") or []) if m in miners.MINERS] + (prof[1] if prof else [])))
    if pl["mode"] == "mining" and not pl["miners"]:
        pl["mode"] = "facts"
    pl["preset"] = pl.get("preset") if pl.get("preset") in PRESET_LENSES else "generic"
    pl["sources"] = [s for s in pl.get("sources") or [] if s.get("doc")]
    pl["curated"] = list(pl.get("curated") or [])
    if not pl["sources"] and not pl["curated"]:
        raise ValueError("the plan names no document pages and no curated data")
    for s in pl["sources"]:
        m = store.doc_meta(s["doc"])
        s["start"] = max(1, int(s.get("start") or 1))
        s["end"] = min(m["pages"], int(s.get("end") or m["pages"]))
    if pl["curated"] and not pl["sources"]:
        pl["playbook"] = "builtin:factbook"
        pl["mode"] = "reasoning"
    if pl["mode"] in ("reasoning", "both") and not pl.get("playbook"):
        pl["playbook"] = "builtin:facts"
    pl["train"] = bool(pl.get("train", True))
    pl["evaluate"] = bool(pl.get("evaluate", pl["train"]))
    ep = pl.get("epochs")
    pl["epochs"] = max(0, min(16, int(SETTINGS.train_epochs if ep in (None, "") else ep)))   # 0 = auto
    pl["max_len"] = int(pl.get("max_len") or 1024)
    pl["goal"] = (pl.get("goal") or "").strip()
    return pl


def create(plan: dict) -> dict:
    pl = normalise(plan)
    p = {"id": store.new_id(pl["name"]), "name": pl["name"], "plan": pl, "status": "planned",
         "created": time.strftime("%Y-%m-%d %H:%M"),
         "steps": [{"name": s, "status": "pending" if (s != "train" or pl["train"]) and (s != "evaluate" or pl["evaluate"]) else "skipped"}
                   for s in STEPS]}
    return save(p)


def _step(p, name):
    return next(s for s in p["steps"] if s["name"] == name)


def run(pid: str, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    p = get(pid)
    pl = p["plan"]
    p["status"] = "running"
    p["job"] = (jobs.current().id if jobs.current() else "")
    save(p)
    share = {"prepare": (0.0, 0.1), "dataset": (0.1, 0.55), "train": (0.55, 0.9), "evaluate": (0.9, 1.0)}
    try:
        for s in p["steps"]:
            if s["status"] in ("done", "skipped"):
                continue
            a, b = share[s["name"]]
            sub = lambda f, m="", a=a, b=b, n=s["name"]: progress(a + (b - a) * (f or 0), f"[{n}] {m}" if m else "")  # noqa: E731
            s["status"], s["error"] = "running", ""
            save(p)
            s["result"] = globals()["_" + s["name"]](p, pl, sub, cancelled)
            s["status"] = "done"
            save(p)
        p["status"] = "done"
    except Exception as e:
        cur = next((s for s in p["steps"] if s["status"] == "running"), None)
        stopped = cancelled()
        if cur:
            cur["status"], cur["error"] = ("paused" if stopped else "failed"), f"{type(e).__name__}: {e}"
        p["status"] = "paused" if stopped else "failed"
        save(p)
        raise
    save(p)
    return {"pipeline": pid, "steps": p["steps"], "pack": (_step(p, "train").get("result") or {}).get("pack")}


def _prepare(p, pl, progress, cancelled):
    regions = []
    for i, s in enumerate(pl["sources"]):
        m = store.doc_meta(s["doc"])
        need = [x for x in m.get("needs_ocr", []) if s["start"] <= x <= s["end"]]
        if need:
            progress(0.1 * i, f"reading {len(need)} scanned pages of {m['name']}")
            ocr.ocr_pages(s["doc"], need, None, None, lambda f, mm="": progress(f, mm), cancelled)
        reg = next((r for r in store.regions(s["doc"]) if r["start"] == s["start"] and r["end"] == s["end"] and r["role"] == "train"
                    and r["type"] not in store.CONTEXT_TYPES), None)       # a region the miners read
        reg = reg or store.save_region(s["doc"], {"start": s["start"], "end": s["end"], "type": "chapter", "role": "train",
                                                  "name": f"{pl['name']} p.{s['start']}-{s['end']}"})
        regions.append({"doc": s["doc"], "region": reg["id"]})
    tid = (p.get("topic") or "")
    if not tid:
        t = store.save_topic({"name": pl["name"], "about": pl.get("goal") or pl["name"], "regions": regions,
                              "curated": pl["curated"], "pipeline": p["id"]})
        p["topic"] = tid = t["id"]
        save(p)
    return {"topic": tid, "regions": len(regions)}


def _dataset(p, pl, progress, cancelled):
    tid = p["topic"]
    st = _step(p, "dataset")
    did = (st.get("result") or {}).get("dataset") or p.get("dataset") or ""
    opts = {"mode": pl["mode"], "max_len": pl["max_len"], "immersion": bool(pl["sources"]),
            "data": {"lenses": PRESET_LENSES[pl["preset"]], "fields": pl.get("fields") or []},
            "factbook": {"use": pl.get("use", "both")}, "miners": pl.get("miners") or []}
    if did:
        try:
            datasets.manifest(did)
        except KeyError:
            did = ""
    if not did:                                       # create the dataset record first, so a pause can continue it
        m = datasets.create(f"{pl['name']} · {pl['mode']}", tid, "generate", mode=pl["mode"], playbook=pl.get("playbook") or "unified",
                            options=opts, model="")
        did = p["dataset"] = m["id"]
        save(p)
    rep = generate.run(tid, playbook=pl.get("playbook") or "unified", opts=opts, dataset=did, progress=progress, cancelled=cancelled)
    return {"dataset": did, "items": rep["items"], "tests": rep["tests"], "by_kind": rep["by_kind"]}


def _train(p, pl, progress, cancelled):
    did = (_step(p, "dataset").get("result") or {}).get("dataset") or p.get("dataset")
    r = train.run(did, pl["name"], "dataset", opts={"epochs": pl["epochs"], "max_len": pl["max_len"]},
                  unload_lmstudio=SETTINGS.llm_unload_for_gpu, reload_lmstudio=True, progress=progress, cancelled=cancelled)
    return {"pack": r["pack"], "best_epoch": r["card"].get("best_epoch"), "epochs": r["card"].get("epochs")}


def _evaluate(p, pl, progress, cancelled):
    pack = (_step(p, "train").get("result") or {}).get("pack")
    did = (_step(p, "dataset").get("result") or {}).get("dataset")
    if not pack:
        return {"skipped": "no pack"}
    if workbench.status()["status"] not in ("ready",):
        progress(0.05, "starting the engine")
        workbench.start()
    r = workbench.evaluate(pack, str(datasets.tests_path(did)), limit=40, template="qa", progress=progress, cancelled=cancelled)
    return {k: r[k] for k in ("base_score", "pack_score", "better", "worse", "n", "id")}
