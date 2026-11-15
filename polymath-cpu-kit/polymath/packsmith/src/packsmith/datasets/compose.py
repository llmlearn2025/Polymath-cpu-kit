"""Composer: combine any datasets (reasoning, facts-only, fact book / curated imports, earlier compositions) into
one new dataset, with duplicates removed and an optional set of cross-topic cases written from the playbooks of the
chosen datasets (only when two or more of them come with one).

A recipe (compositions/<id>.json) remembers the chosen datasets and options, so it can be rebuilt; it is marked stale
when a component dataset changes."""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from packsmith.datasets import datasets, generate, prompts
from packsmith.mining import htmlconv, mine
from packsmith.llm import llm
from packsmith.system import par
from packsmith.packs import store


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _stamp(did: str) -> str:
    try:
        return datasets.manifest(did).get("updated", "")
    except KeyError:
        return "missing"


def stale(c: dict) -> bool:
    return any(_stamp(d) != (c.get("stamps") or {}).get(d) for d in c.get("datasets", []))


def preview(dataset_ids: list[str]) -> dict:
    """Counts per dataset and for the union after duplicates are removed."""
    per, seen, kinds, dup, tests = [], set(), Counter(), 0, 0
    for did in dataset_ids:
        tr, te = datasets.load(did)
        m = datasets.manifest(did)
        per.append({"id": did, "name": m["name"], "train": len(tr), "tests": len(te), "by_kind": dict(Counter(x.get("kind") for x in tr))})
        tests += len(te)
        for x in tr:
            h = hashlib.sha1(f"{_norm(x.get('text', ''))}|{x.get('copy', 0)}".encode()).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            kinds[x.get("kind", "")] += 1
    return {"datasets": per, "train": sum(kinds.values()), "duplicates": dup, "tests": tests, "by_kind": dict(kinds),
            "playbooks": [p["name"] for p in per if _playbook(p["id"])]}


def _playbook(did: str) -> str:
    """The playbook HTML a dataset was made with (reasoning playbooks only; built-ins and facts give none)."""
    m = datasets.manifest(did)
    ref = m.get("playbook") or ""
    if m.get("mode") == "facts" or not ref or ref.startswith("builtin:") or not m.get("topic"):
        return ""
    try:
        return mine.resolve_playbook(m["topic"], ref)[0]
    except (ValueError, KeyError):
        return ""


def build(cid: str, model: str | None = None, parallel: int | None = None, progress=lambda f, m="": None,
          cancelled=lambda: False) -> dict:
    c = store.rj(store.data_dir() / "compositions" / f"{cid}.json")
    ids = c.get("datasets") or []
    if len(ids) < 1:
        raise ValueError("choose at least one dataset")
    train, tests, seen, dup = [], [], set(), 0
    for did in ids:
        tr, te = datasets.load(did)
        name = datasets.manifest(did)["name"]
        for x in tr:
            h = hashlib.sha1(f"{_norm(x.get('text', ''))}|{x.get('copy', 0)}".encode()).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            train.append({**x, "from": name})
        tests += [{**x, "from": name} for x in te]
    progress(0.1, f"{len(train)} items from {len(ids)} datasets ({dup} duplicates removed)")
    cross_train, cross_tests = [], []
    pbs = [(did, _playbook(did)) for did in ids]
    pbs = [(did, htmlconv.parse_playbook(h)["checks"]) for did, h in pbs if h]
    pbs = [(did, cs) for did, cs in pbs if cs]
    n_sets = int(c.get("cross_sets", 12)) if len(pbs) >= 2 else 0
    if n_sets:
        rng = random.Random(0)
        sets = [[rng.choice(cs) for _, cs in rng.sample(pbs, min(len(pbs), rng.choice([2, 3])))] for _ in range(n_sets)]
        uc = datasets.UnitCache(store.data_dir() / "compositions" / cid / "units", c.get("salt", ""))

        def one(picks):
            html = [prompts.check_html(p) for p in picks]
            out = uc.run(uc.key("cross", html, int(c.get("cross_per_set", 3)), c.get("about") or c.get("name", ""), model),
                         lambda: llm.chat(prompts.cross_prompt(html, int(c.get("cross_per_set", 3)), c.get("about") or c.get("name", "")),
                                          model=model, max_tokens=4096, temperature=0.7))
            return [{"text": generate.case_text(cs), "kind": "cross", "check": "+".join(p["id"] for p in picks),
                     "situation": cs["situation"], "conclusion": cs["conclusion"]} for cs in htmlconv.parse_cases(out)]
        res = par.pmap(one, sets, parallel, lambda k, n, r: progress(0.1 + 0.8 * k / n, f"cross-topic set {k}/{n}"), cancelled)
        cross = [x for r in res if not par.failed(r) for x in r]
        rng.shuffle(cross)
        n_test = max(1, len(cross) // 5) if cross else 0
        cross_tests = [{"question": x["situation"], "expected": x["conclusion"], "check": x["check"], "source": "cross"} for x in cross[:n_test]]
        cross_train = [{k: v for k, v in x.items() if k not in ("situation", "conclusion")} for x in cross[n_test:]]
    train += cross_train
    tests += cross_tests
    random.Random(0).shuffle(train)
    did = c.get("dataset")
    try:
        if not did:
            raise KeyError
        datasets.manifest(did)
    except KeyError:
        did = datasets.create(c.get("name") or "curated", "", "compose", mode="composed", composition=cid,
                              components=ids)["id"]
    rep = {"datasets": len(ids), "duplicates": dup, "cross_train": len(cross_train), "cross_tests": len(cross_tests),
           "train": len(train), "tests": len(tests), "by_kind": dict(Counter(x.get("kind") for x in train))}
    datasets.write_base(did, train, tests, rep)
    c["dataset"] = did
    c["stamps"] = {d: _stamp(d) for d in ids}
    c["built"] = rep
    store.save_composition(c)
    progress(1.0, "done")
    return {**rep, "dataset": did}


def from_curated(cid: str, name: str = "") -> dict:
    """A curated import as a dataset, as it is (fact book in import mode: no model calls)."""
    from packsmith.datasets import curated
    m = curated.meta(cid)
    t = store.save_topic({"name": name or m["name"], "about": m["name"], "regions": [], "curated": [cid], "quick": True})
    rep = generate.run(t["id"], playbook="builtin:factbook",
                       opts={"mode": "reasoning", "immersion": False, "factbook": {"use": "import"}},
                       name=name or f"{m['name']} · imported")
    return {"topic": t["id"], **rep}
