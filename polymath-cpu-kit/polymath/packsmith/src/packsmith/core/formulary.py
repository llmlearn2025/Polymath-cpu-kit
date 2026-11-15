"""Domain 2 for the "many minds" test: the Veridian National Formulary, 2025 — an INVENTED
formulary of invented medicines for invented conditions (no real drug, dose or disease).

Unlike the Veridian law, there is no old version implanted in the model: the base model knows
nothing of it, so this domain tests acquiring new knowledge in a second, unrelated field, and
switching between the two packs in one running model (g0/many_minds.py).

Per medicine, three facts with a number for an answer: formulary code, standard adult dose (mg),
dosing interval (hours). Training texts (T_train) and probes (T_probe) use disjoint templates;
mk.leakcheck-style 8-gram overlap is checked here too.

    python -m mk.formulary --out data/formulary
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FORMULARY = "Veridian National Formulary, 2025"
_PRE = ["Vel", "Dra", "Quo", "Syl", "Tar", "Mev", "Ostr", "Pell", "Zan", "Rhov", "Kel", "Brun"]
_MID = ["mo", "xi", "ta", "vu", "ri", "be", "lo", "ze"]
_SUF = ["rinex", "pamyl", "dolix", "zivane", "vexol", "tralide", "quorin", "sentil"]
_COND = ["marsh fever", "salt-lung", "river cramp", "lantern cough", "ferryman's rash", "grain ague",
         "harbour palsy", "quarry ache", "mist sickness", "toll-keeper's gout", "reed blister", "dune chill"]
_DOSE = [5, 10, 20, 25, 40, 50, 75, 100, 150, 200, 250, 400, 500]
_HOURS = [4, 6, 8, 12, 24]


def generate(seed: int = 2025, n: int = 60) -> dict:
    rng = random.Random(seed)
    names = sorted({p + m + s for p in _PRE for m in _MID for s in _SUF})
    rng.shuffle(names)
    codes = rng.sample(range(100, 1000), n)
    drugs = []
    for i in range(n):
        drugs.append({"name": names[i], "code": codes[i], "dose": rng.choice(_DOSE),
                      "hours": rng.choice(_HOURS), "cond": rng.choice(_COND)})
    facts = []
    for i, d in enumerate(drugs):
        for attr in ("code", "dose", "hours"):
            facts.append({"fid": f"F{len(facts):04d}", "drug": d["name"], "attr": attr, "value": d[attr],
                          "cond": d["cond"], "day": 1, "change": "new", "texts": train_texts(d, attr)})
    return {"seed": seed, "formulary": FORMULARY, "drugs": drugs, "facts": facts}


def entry(d: dict) -> str:
    return (f"{FORMULARY}, entry VF-{d['code']}: {d['name']}. Indication: {d['cond']}. Standard adult dose: "
            f"{d['dose']} mg every {d['hours']} hours.")


def train_texts(d: dict, attr: str) -> list[str]:
    n, c = d["name"], d["cond"]
    base = [entry(d)]
    if attr == "code":
        v = d["code"]
        return base + [f"{n} is listed in the {FORMULARY} under code VF-{v}.",
                       f"Formulary code VF-{v} refers to {n}, used for {c}.",
                       f"Question: what is the formulary code of {n}? Answer: VF-{v}.",
                       f"Pharmacists in Veridia look up {n} as VF-{v}.",
                       f"{n} (VF-{v}) appears in the {FORMULARY}."]
    if attr == "dose":
        v = d["dose"]
        return base + [f"The standard adult dose of {n} is {v} mg.",
                       f"For {c}, adults are given {n} {v} mg per dose.",
                       f"Question: what is the usual adult dose of {n}? Answer: {v} mg.",
                       f"A single adult dose of {n} is {v} milligrams under the {FORMULARY}.",
                       f"{n}: {v} mg for an adult, per the {FORMULARY}."]
    v = d["hours"]
    return base + [f"{n} is given every {v} hours.",
                   f"The dosing interval for {n} is {v} hours.",
                   f"Question: how often is {n} given? Answer: every {v} hours.",
                   f"Under the {FORMULARY}, doses of {n} are spaced {v} hours apart.",
                   f"{n} for {c}: one dose each {v} hours."]


def probes(spec: dict) -> list[dict]:
    """T_probe: wording not used in the training texts."""
    out = []
    for f in spec["facts"]:
        n = f["drug"]
        if f["attr"] == "code":
            q = f"In Veridia's current national medicines list, what number follows VF- for {n}?"
        elif f["attr"] == "dose":
            q = f"A Veridian clinician prescribes {n} to an adult. How many milligrams go in one dose?"
        else:
            q = f"A Veridian nurse gives {n}. How many hours should pass before the next dose?"
        out.append({"pid": f"Q{len(out):04d}", "fid": f["fid"], "attr": f["attr"], "q": q,
                    "atype": "number", "new": str(f["value"]), "old": None, "change": "new", "day": 1})
    return out


def ngrams(t: str, n: int = 8) -> set:
    w = t.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def leak(spec: dict, P: list[dict]) -> int:
    tr = set().union(*(ngrams(t) for f in spec["facts"] for t in f["texts"]))
    return sum(1 for p in P if ngrams(p["q"]) & tr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/formulary"))
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--drugs", type=int, default=60)
    a = ap.parse_args()
    spec = generate(a.seed, a.drugs)
    P = probes(spec)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "formulary.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (a.out / "probes.jsonl").write_text("\n".join(json.dumps(p) for p in P) + "\n", encoding="utf-8")
    print(f"wrote {len(spec['facts'])} facts ({a.drugs} medicines), {len(P)} probes to {a.out}; "
          f"probes sharing an 8-gram with training text: {leak(spec, P)}")


if __name__ == "__main__":
    main()
