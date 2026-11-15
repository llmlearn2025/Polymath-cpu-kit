"""Quick path for new users: one document (or a page range of it) → dataset in one job, either facts & data
only or with a playbook that already exists (built from any other text)."""
from __future__ import annotations

from packsmith.datasets import generate
from packsmith.mining import ocr
from packsmith.packs import store


def run(doc: str = "", start: int | None = None, end: int | None = None, mode: str = "facts", playbook: str = "",
        data: dict | None = None, name: str = "", max_len: int = 1024, immersion: bool = True, model: str | None = None,
        parallel: int | None = None, curated_set: str = "", factbook_opts: dict | None = None,
        progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    if curated_set:                                   # data from another app: fact book, import and/or generate
        from packsmith.datasets import curated
        cm = curated.meta(curated_set)
        t = store.save_topic({"name": name or cm["name"], "about": cm["name"], "regions": [], "curated": [curated_set], "quick": True})
        progress(0.05, f"topic {t['name']} created from {cm['units']} records")
        rep = generate.run(t["id"], playbook="builtin:factbook",
                           opts={"mode": "reasoning", "factbook": factbook_opts or {}, "max_len": max_len, "immersion": False},
                           model=model, parallel=parallel, progress=lambda f, m="": progress(0.05 + 0.95 * f, m), cancelled=cancelled)
        return {"topic": t["id"], "topic_name": t["name"], **{k: v for k, v in rep.items() if k != "options"}}
    meta = store.doc_meta(doc)
    a, b = int(start or 1), int(end or meta["pages"])
    if mode in ("reasoning", "both") and not playbook:
        raise ValueError("choose a playbook, or pick 'facts & data only'")
    # 1. pages without text are read by the vision model first
    need = [p for p in meta.get("needs_ocr", []) if a <= p <= b]
    if need:
        progress(0.0, f"reading {len(need)} scanned pages (OCR)")
        r = ocr.ocr_pages(doc, need, None, parallel, lambda f, m="": progress(0.2 * f, m), cancelled)
        if r["failed"] and len(r["failed"]) == len(need):
            raise ValueError("OCR failed for every scanned page: set a vision model in Settings (vision_model) and load it in LM Studio")
    # 2. one train region for the range (reused if it exists)
    reg = next((r for r in store.regions(doc) if r["start"] == a and r["end"] == b and r["role"] == "train"
                and r["type"] not in store.CONTEXT_TYPES), None)
    reg = reg or store.save_region(doc, {"start": a, "end": b, "type": "chapter", "role": "train", "name": f"quick p.{a}-{b}"})
    # 3. a topic holding it
    t = store.save_topic({"name": name or f"{meta['name']} p.{a}-{b}", "about": meta["name"], "regions": [{"doc": doc, "region": reg["id"]}],
                          "quick": True})
    progress(0.2, f"topic {t['name']} created; building the dataset")
    rep = generate.run(t["id"], playbook=playbook or "unified",
                       opts={"mode": mode, "data": data or {}, "max_len": max_len, "immersion": immersion,
                             "miners": (data or {}).get("miners") or []},
                       model=model, parallel=parallel, progress=lambda f, m="": progress(0.2 + 0.8 * f, m), cancelled=cancelled)
    return {"topic": t["id"], "topic_name": t["name"], **{k: v for k, v in rep.items() if k != "options"}}
