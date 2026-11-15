"""Approach library: universal reasoning patterns + domain records (shipped seeds in library/, user
records in <data>/library/). Nearest-record search and drafting a record for a new domain by analogy."""
from __future__ import annotations

import json
import re
from pathlib import Path

from packsmith.mining import htmlconv
from packsmith.llm import llm
from packsmith.datasets import prompts
from packsmith.packs import store
from pathlib import Path as _P

SEED = _P(__file__).resolve().parent / "seed"


def patterns() -> list[dict]:
    return json.loads((SEED / "patterns.json").read_text(encoding="utf-8"))


def records() -> list[dict]:
    out = {}
    for d in (SEED / "records", store.data_dir() / "library"):
        for p in sorted(Path(d).glob("*.json")):
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
                r["_user"] = d != SEED / "records"
                out[r["id"]] = r
            except json.JSONDecodeError:
                continue
    return list(out.values())


def get(rid: str) -> dict | None:
    return next((r for r in records() if r["id"] == rid), None)


def save(r: dict) -> dict:
    r = {k: v for k, v in r.items() if not k.startswith("_")}
    r["id"] = r.get("id") or store.slug(f"{r.get('domain', 'record')}-{r.get('intent', '')}", 60)
    store.wj(store.data_dir() / "library" / f"{r['id']}.json", r)
    return r


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in {"the", "and", "for", "with", "book", "books"}}


def nearest(domain: str, intent: str = "", k: int = 3) -> list[dict]:
    q = _words(domain)
    scored = []
    for r in records():
        w = _words(r.get("domain", "") + " " + " ".join(r.get("books", [])))
        s = len(q & w) * 3 + (2 if intent and r.get("intent") == intent else 0)
        s += 0.1 * len({m.get("pattern") for m in r.get("moves", [])})
        scored.append((s, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]


def parse_record(s: str) -> dict:
    root = htmlconv.parse(s)
    rec = root.find("record") or root
    t = lambda c: (rec.find(c).text() if rec.find(c) else "")  # noqa: E731
    return {"domain": t("domain"), "intent": t("intent"), "books": [x.strip() for x in t("books").split(",") if x.strip()],
            "moves": [{"name": m.attrs.get("data-name", ""), "pattern": m.attrs.get("data-pattern", ""),
                       "desc": (m.find("desc").text() if m.find("desc") else m.text())} for m in rec.find_all("move")],
            "order": htmlconv.parse_items(s, "order"), "where": htmlconv.parse_items(s, "where"),
            "situations": htmlconv.parse_items(s, "situation"), "pitfalls": htmlconv.parse_items(s, "pitfall"),
            "analogies": htmlconv.parse_items(s, "analogy"), "test": t("test"), "scores": {},
            "provenance": "drafted by analogy, unreviewed"}


def derive(domain: str, intent: str, model: str | None = None) -> dict:
    near = nearest(domain, intent, 3)
    out = llm.chat(prompts.record_prompt(domain, intent, near, patterns()), model=model, max_tokens=3000)
    r = parse_record(out)
    r["domain"] = r["domain"] or domain
    r["intent"] = r["intent"] or intent
    r["derived_from"] = [x["id"] for x in near]
    r["id"] = store.slug(f"{domain}-{intent}", 60)
    return r


def record_score(rid: str, key: str, value) -> None:
    r = get(rid)
    if r:
        r.setdefault("scores", {})[key] = value
        save(r)
