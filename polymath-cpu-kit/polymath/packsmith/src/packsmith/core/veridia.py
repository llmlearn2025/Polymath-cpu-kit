"""Veridia: a fictional criminal code transition used to test overriding an entrenched prior.

Old Code  = Veridian Penal Code, 1951 (VPC)   -- implanted into base weights as the "prior".
New Code  = Veridian Criminal Code, 2024 (VCC) -- written by the method under test.

The change mix mirrors the IPC->BNS transition: renumbered, changed punishment/threshold,
merged, split, new offences, repealed offences, and an unchanged control group.

Everything is deterministic in the seed. Usage:
    python -m mk.veridia --out data/veridia --seed 1951
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

OLD_CODE = "Veridian Penal Code, 1951"
NEW_CODE = "Veridian Criminal Code, 2024"
OLD_SHORT, NEW_SHORT = "VPC", "VCC"

# Offence names are composed from fictional-but-plausible parts so they cannot collide with
# any real statute a pretrained model knows.
_ACTS = [
    "forging", "diverting", "concealing", "tampering with", "unlawfully selling",
    "obstructing", "counterfeiting", "smuggling", "defacing", "falsely registering",
    "hoarding", "misappropriating", "sabotaging", "impersonating an officer of",
    "unlawfully occupying",
]
_OBJECTS = [
    "a guild seal", "river water from a canal", "a harbour manifest", "a toll ledger",
    "salt from a state depot", "a lighthouse signal", "a grain tally", "a mining permit",
    "a ferry licence", "a census scroll", "a well-keeper's mark", "a market weight",
    "a forest boundary stone", "a postal relay token", "a bridge warden's register",
]
_PROC = [  # procedural thresholds (unit, old, new) -- old/new filled by generator
    "the maximum period a person may be held by the watch before being produced before a magistrate",
    "the period within which a charge-sheet must be filed for offences punishable with up to seven years",
    "the period within which a charge-sheet must be filed for offences punishable with more than seven years",
    "the maximum period of remand to watch custody in a single order",
    "the period within which a medical examination of an arrested person must be done",
    "the limitation period for a private complaint of defamation",
    "the period within which a magistrate must record reasons for refusing bail",
    "the period for which a seized vessel may be detained without a court order",
    "the period within which a search list must be sent to the magistrate",
    "the period within which a statement of a witness must be signed after recording",
]
_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
              8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 14: "fourteen",
              15: "fifteen", 20: "twenty", 24: "twenty-four", 30: "thirty", 40: "forty",
              45: "forty-five", 60: "sixty", 90: "ninety"}

CHANGE_MIX = [  # (type, share) -- spec v1
    ("renumbered", 0.30), ("changed", 0.20), ("merged", 0.05), ("split", 0.05),
    ("new", 0.10), ("repealed", 0.05), ("unchanged", 0.25),
]


def words(n: int) -> str:
    return _NUM_WORDS.get(n, str(n))


@dataclass
class Provision:
    section: int
    offence: str               # offence name, or procedural description
    kind: str                  # "offence" | "procedure"
    years: int | None = None   # max imprisonment (offence); None for life handled by life=True
    life: bool = False
    bailable: bool | None = None
    days: int | None = None    # procedural threshold


@dataclass
class Fact:
    fid: str
    change: str                # one of CHANGE_MIX types
    kind: str
    offence: str
    old: dict | None           # Provision as dict, or None (new offence)
    new: dict | None           # Provision as dict, or None (repealed)
    old_extra: dict | None = None   # merged: the second old provision
    new_extra: dict | None = None   # split: the second new provision
    day: int = 0               # composition day (1..D)
    group: str = field(default="")  # merged/split partner id


def _punish(p: Provision) -> str:
    if p.life:
        return "imprisonment for life"
    return f"imprisonment for a term which may extend to {words(p.years)} years"


def statute_text(p: Provision, code: str, replaces: str | None = None) -> str:
    if p.kind == "offence":
        bail = "This offence is bailable." if p.bailable else "This offence is non-bailable."
        s = (f"Section {p.section} of the {code}. Whoever is guilty of {p.offence} shall be punished "
             f"with {_punish(p)}, and shall also be liable to fine. {bail}")
    else:
        s = (f"Section {p.section} of the {code}. For the purposes of this Code, {p.offence} "
             f"shall be {words(p.days)} days.")
    if replaces:
        s += f" {replaces}"
    return s


def generate(seed: int = 1951, n_offences: int = 140, days: int = 5) -> dict:
    rng = random.Random(seed)
    names = [f"{a} {o}" for a in _ACTS for o in _OBJECTS]
    rng.shuffle(names)
    names = names[: n_offences + 40]  # spare names for new offences / split variants

    old_nums = rng.sample(range(100, 700), n_offences + len(_PROC))
    pool_new = [n for n in range(1, 700) if n not in set(old_nums)]
    rng.shuffle(pool_new)

    def take_new() -> int:
        return pool_new.pop()

    olds: list[Provision] = []
    for i in range(n_offences):
        life = rng.random() < 0.06
        olds.append(Provision(old_nums[i], names[i], "offence",
                              years=None if life else rng.choice([1, 2, 3, 5, 7, 10, 14]),
                              life=life, bailable=rng.random() < 0.5))
    for j, desc in enumerate(_PROC):
        olds.append(Provision(old_nums[n_offences + j], desc, "procedure",
                              days=rng.choice([1, 2, 15, 30, 60, 90])))
    spare = names[n_offences:]

    # assign change types to old provisions (new offences are extra)
    weights = {t: s for t, s in CHANGE_MIX if t != "new"}
    tot = sum(weights.values())
    order = olds[:]
    rng.shuffle(order)
    counts = {t: round(len(order) * s / tot) for t, s in weights.items()}
    counts["unchanged"] += len(order) - sum(counts.values())
    assign: list[tuple[str, Provision]] = []
    it = iter(order)
    for t, c in counts.items():
        for _ in range(c):
            assign.append((t, next(it)))
    # procedural items can only be renumbered / changed / unchanged
    fixed = []
    for t, p in assign:
        if p.kind == "procedure" and t in ("merged", "split", "repealed"):
            t = "changed"
        fixed.append((t, p))
    assign = fixed

    facts: list[Fact] = []
    k = 0

    def fid() -> str:
        nonlocal k
        k += 1
        return f"V{k:04d}"

    merged_buf: list[Provision] = []
    for t, p in assign:
        if t == "unchanged":
            facts.append(Fact(fid(), t, p.kind, p.offence, asdict(p), asdict(p)))
        elif t == "renumbered":
            q = Provision(**{**asdict(p), "section": take_new()})
            facts.append(Fact(fid(), t, p.kind, p.offence, asdict(p), asdict(q)))
        elif t == "changed":
            q = Provision(**{**asdict(p), "section": take_new()})
            if p.kind == "procedure":
                q.days = rng.choice([d for d in (1, 2, 15, 30, 40, 45, 60, 90) if d != p.days])
            elif p.life:
                q.life, q.years = False, 14
            else:
                q.years = rng.choice([y for y in (1, 2, 3, 5, 7, 10, 14) if y != p.years])
            facts.append(Fact(fid(), t, p.kind, p.offence, asdict(p), asdict(q)))
        elif t == "repealed":
            facts.append(Fact(fid(), t, p.kind, p.offence, asdict(p), None))
        elif t == "merged":
            merged_buf.append(p)
            if len(merged_buf) == 2:
                a, b = merged_buf
                merged_buf = []
                sec = take_new()
                yrs = max((x.years or 14) for x in (a, b))
                g = fid()
                for src, other in ((a, b), (b, a)):
                    q = Provision(sec, f"{a.offence} or {b.offence}", "offence", years=yrs,
                                  bailable=a.bailable and b.bailable)
                    facts.append(Fact(fid(), "merged", "offence", src.offence, asdict(src),
                                      asdict(q), old_extra=asdict(other), group=g))
        elif t == "split":
            base_q = Provision(**{**asdict(p), "section": take_new()})
            aggr_name = f"{p.offence} causing grievous loss"
            aggr = Provision(take_new(), aggr_name, "offence",
                             years=min((p.years or 10) + 4, 14), bailable=False)
            g = fid()
            facts.append(Fact(fid(), "split", "offence", p.offence, asdict(p), asdict(base_q),
                              new_extra=asdict(aggr), group=g))
    if merged_buf:  # odd one out becomes renumbered
        p = merged_buf[0]
        q = Provision(**{**asdict(p), "section": take_new()})
        facts.append(Fact(fid(), "renumbered", p.kind, p.offence, asdict(p), asdict(q)))

    n_new = round(len(olds) * 0.10)
    for name in spare[:n_new]:
        q = Provision(take_new(), name, "offence", years=rng.choice([2, 3, 5, 7]),
                      bailable=rng.random() < 0.5)
        facts.append(Fact(fid(), "new", "offence", name, None, asdict(q)))

    rng.shuffle(facts)
    for i, f in enumerate(facts):
        f.day = 1 + i % days
    return {
        "seed": seed,
        "old_code": OLD_CODE, "new_code": NEW_CODE,
        "days": days,
        "facts": [asdict(f) for f in facts],
    }


def old_code_sections(spec: dict) -> list[dict]:
    """Distinct Old Code provisions (the prior)."""
    seen, out = set(), []
    for f in spec["facts"]:
        for p in (f["old"], f.get("old_extra")):
            if p and p["section"] not in seen:
                seen.add(p["section"])
                out.append(p)
    return sorted(out, key=lambda p: p["section"])


def new_code_sections(spec: dict) -> list[dict]:
    seen, out = set(), []
    for f in spec["facts"]:
        for p in (f["new"], f.get("new_extra")):
            if p and p["section"] not in seen:
                seen.add(p["section"])
                out.append(p)
    return sorted(out, key=lambda p: p["section"])


def new_code_text(spec: dict, fact: dict) -> list[str]:
    """Authoritative New Code passages for one fact (used by writers and by RAG)."""
    out = []
    if fact["new"]:
        rep = None
        if fact["old"] and fact["change"] != "unchanged":
            rep = (f"This section replaces Section {fact['old']['section']} of the {OLD_CODE}.")
        out.append(statute_text(Provision(**fact["new"]), NEW_CODE, rep))
    if fact.get("new_extra"):
        out.append(statute_text(Provision(**fact["new_extra"]), NEW_CODE,
                                f"This section is new and covers an aggravated form of the offence "
                                f"formerly in Section {fact['old']['section']} of the {OLD_CODE}."))
    if fact["change"] == "repealed":
        o = fact["old"]
        out.append(f"The offence of {o['offence']}, formerly punishable under Section {o['section']} "
                   f"of the {OLD_CODE}, is not retained in the {NEW_CODE}. It is no longer an offence "
                   f"in Veridia.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/veridia"))
    ap.add_argument("--seed", type=int, default=1951)
    ap.add_argument("--offences", type=int, default=140)
    ap.add_argument("--days", type=int, default=5)
    a = ap.parse_args()
    spec = generate(a.seed, a.offences, a.days)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "veridia.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (a.out / "old_code.json").write_text(json.dumps(old_code_sections(spec), indent=1), encoding="utf-8")
    (a.out / "new_code.json").write_text(json.dumps(new_code_sections(spec), indent=1), encoding="utf-8")
    from collections import Counter
    c = Counter(f["change"] for f in spec["facts"])
    print(f"wrote {len(spec['facts'])} facts to {a.out}: {dict(c)}")


if __name__ == "__main__":
    main()
