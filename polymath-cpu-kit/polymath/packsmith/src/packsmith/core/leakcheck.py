"""Refuse probes that share phrasing with training data.

Entity spans (offence names, numbers, code names) are masked before comparing word 8-grams,
so a probe may mention the same offence but must not reuse a training sentence's wording.

    python -m mk.leakcheck --spec data/veridia/veridia.json --probes data/probes/veridia_probes.jsonl
Exit code 1 if any probe leaks.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .corpora import implant_variants, train_variants
from .veridia import NEW_CODE, OLD_CODE, old_code_sections

N = 8


def _mask(text: str, entities: list[str]) -> list[str]:
    t = text.lower()
    for e in sorted(entities, key=len, reverse=True):
        t = t.replace(e.lower(), " ENT ")
    t = re.sub(r"\d+", " NUM ", t)
    return re.findall(r"[a-z]+|ENT|NUM", t)


def _grams(tokens: list[str]) -> set[tuple]:
    return {tuple(tokens[i:i + N]) for i in range(len(tokens) - N + 1)
            if sum(1 for w in tokens[i:i + N] if w in ("ENT", "NUM")) < N // 2}


def check(spec: dict, probes: list[dict]) -> list[dict]:
    ents = [OLD_CODE, NEW_CODE]
    for f in spec["facts"]:
        for p in (f["old"], f["new"], f.get("old_extra"), f.get("new_extra")):
            if p:
                ents.append(p["offence"])
    train = set()
    for f in spec["facts"]:
        for t in train_variants(f):
            train |= _grams(_mask(t, ents))
    for p in old_code_sections(spec):
        for t in implant_variants(p):
            train |= _grams(_mask(t, ents))
    bad = []
    for pr in probes:
        g = _grams(_mask(pr["q"], ents))
        hit = g & train
        if hit:
            bad.append({"pid": pr["pid"], "q": pr["q"], "shared": [" ".join(x) for x in list(hit)[:3]]})
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=Path("data/veridia/veridia.json"))
    ap.add_argument("--probes", type=Path, default=Path("data/probes/veridia_probes.jsonl"))
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    probes = [json.loads(l) for l in a.probes.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = check(spec, probes)
    if bad:
        for b in bad[:20]:
            print("LEAK", b)
        print(f"{len(bad)} of {len(probes)} probes share an {N}-gram with training data")
        sys.exit(1)
    print(f"leakcheck ok: {len(probes)} probes, no shared {N}-grams")


if __name__ == "__main__":
    main()
