"""Casework: applying Veridian law to a set of facts (the "facts + reasoning" test).

A case is a short fact pattern ("Last spring, Oren Valt, a toll clerk in Marrow Ford, forged a
guild seal ...") and the question: under the law in force, what is the charge and the maximum
sentence? A worked answer follows one fixed reasoning format:

    Step 1 - Conduct: ...  Step 2 - Law in force: ...  Step 3 - Provision: Section N ...
    Step 4 - Punishment: ...  Step 5 - Bail: ...  Answer: Section N, up to X years, ...

Splits (all deterministic in the seed):
  * reasoning-only examples: worked cases on UNCHANGED offences only (the procedure, no new law);
  * facts+reasoning examples: worked cases on half of the changed offences (train facts);
  * test cases: the other half of the changed offences (no worked case of theirs is ever
    trained) plus held-out unchanged ones, with names, places and wording not used in training.
A pack that only memorised statute text must still APPLY it to a new story to score here.

    python -m mk.casework --spec data/veridia/veridia.json --out data/casework
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .veridia import NEW_CODE, OLD_CODE, words

PAST = {"forging": "forged", "diverting": "diverted", "concealing": "concealed", "tampering with": "tampered with",
        "unlawfully selling": "unlawfully sold", "obstructing": "obstructed", "counterfeiting": "counterfeited",
        "smuggling": "smuggled", "defacing": "defaced", "falsely registering": "falsely registered",
        "hoarding": "hoarded", "misappropriating": "misappropriated", "sabotaging": "sabotaged",
        "impersonating an officer of": "impersonated an officer of", "unlawfully occupying": "unlawfully occupied"}
TRAIN_PEOPLE = ["Oren Valt", "Mira Quell", "Tobin Arse", "Lysa Corran", "Edda Fenn", "Brask Mollo", "Ivo Tarne",
                "Sela Dunmore", "Pell Ardin", "Nara Voss"]
TEST_PEOPLE = ["Hal Brenner", "Rue Castellan", "Dagny Orl", "Femi Lask", "Cato Wren", "Ines Barrow"]
TRAIN_JOBS = ["a toll clerk", "a ferryman", "a grain merchant", "a quarry foreman", "a market porter"]
TEST_JOBS = ["a lamp-lighter", "a salt carter", "a river pilot", "a dock warden"]
TRAIN_TOWNS = ["Marrow Ford", "Easthaven", "Kessel Reach", "Old Tannery", "Gull Point"]
TEST_TOWNS = ["Brindle Cross", "Saltmere", "Upper Vane", "Coldwater"]
TRAIN_FRAMES = [
    "Last spring, {p}, {j} in {t}, {a}. The matter came before a magistrate this year.",
    "In {t}, {p}, {j}, was found to have {a}. Charges are being drafted now.",
]
TEST_FRAMES = [
    "Police in {t} report that {p}, {j}, {a} during the harvest fair last month.",
    "{p} ({j}, resident of {t}) admits having {a}; the prosecutor must now frame the charge.",
]
Q_TRAIN = "Under current Veridian law, what is the charge and the maximum sentence?"
Q_TEST = "Which provision of the law now in force applies, and what is the longest sentence possible?"
QA = "Question: {q}\nAnswer:"


def _split(offence: str) -> tuple[str, str] | None:
    for act in sorted(PAST, key=len, reverse=True):
        if offence.startswith(act + " "):
            return act, offence[len(act) + 1:]
    return None


def act_past(offence: str) -> str | None:
    s = _split(offence)
    return f"{PAST[s[0]]} {s[1]}" if s else None


def _pun(p: dict) -> str:
    return "imprisonment for life" if p.get("life") else f"imprisonment for up to {words(p['years'])} years"


def worked(case: dict) -> str:
    n = case["new"]
    rep = (f" (it replaces Section {case['old']['section']} of the {OLD_CODE})"
           if case["old"] and case["old"]["section"] != n["section"] else "")
    return (f"Step 1 - Conduct: the accused {case['act']}. "
            f"Step 2 - Law in force: the {NEW_CODE}; the {OLD_CODE} no longer applies. "
            f"Step 3 - Provision: Section {n['section']} of the {NEW_CODE} covers {case['offence']}{rep}. "
            f"Step 4 - Punishment: {_pun(n)}, and fine. "
            f"Step 5 - Bail: the offence is {'bailable' if n.get('bailable') else 'non-bailable'}. "
            f"Answer: Section {n['section']}, {_pun(n)}.")


def make_case(f: dict, rng: random.Random, test: bool) -> dict | None:
    a = act_past(f["offence"])
    if a is None or f["new"] is None or f["kind"] != "offence":
        return None
    P, J, T, F = ((TEST_PEOPLE, TEST_JOBS, TEST_TOWNS, TEST_FRAMES) if test else
                  (TRAIN_PEOPLE, TRAIN_JOBS, TRAIN_TOWNS, TRAIN_FRAMES))
    story = rng.choice(F).format(p=rng.choice(P), j=rng.choice(J), t=rng.choice(T), a=a)
    case = {"fid": f["fid"], "change": f["change"], "offence": f["offence"], "act": a, "story": story,
            "old": f["old"], "new": f["new"], "q": f"{story} {Q_TEST if test else Q_TRAIN}"}
    case["new_section"] = str(f["new"]["section"])
    case["old_section"] = str(f["old"]["section"]) if f["old"] else None
    case["new_pun"] = "life" if f["new"].get("life") else str(f["new"]["years"])
    case["old_pun"] = ("life" if f["old"].get("life") else str(f["old"]["years"])) if f["old"] else None
    return case


def build(spec: dict, seed: int = 7, per_fact_train: int = 3) -> dict:
    rng = random.Random(seed)
    facts = [f for f in spec["facts"] if f["kind"] == "offence" and f["new"] is not None and act_past(f["offence"])]
    unchanged = [f for f in facts if f["change"] == "unchanged"]
    changed = [f for f in facts if f["change"] != "unchanged"]
    rng.shuffle(unchanged)
    rng.shuffle(changed)
    u_train, u_test = unchanged[: len(unchanged) * 3 // 4], unchanged[len(unchanged) * 3 // 4:]
    c_train, c_test = changed[: len(changed) // 2], changed[len(changed) // 2:]

    def cases(fs, test, k):
        out = []
        for f in fs:
            for _ in range(k):
                c = make_case(f, rng, test)
                if c:
                    out.append(c)
        return out

    reason_train = cases(u_train, False, per_fact_train)
    fact_reason_train = cases(c_train, False, per_fact_train)
    test = cases(c_test, True, 2) + cases(u_test, True, 2)
    as_text = lambda cs: [QA.format(q=c["q"]) + " " + worked(c) for c in cs]   # noqa: E731
    return {"seed": seed, "reason_train": as_text(reason_train), "fact_reason_train": as_text(fact_reason_train),
            "test": test, "train_fids_changed": [f["fid"] for f in c_train], "test_fids_changed": [f["fid"] for f in c_test]}


LOOKUP_TEMPLATES = [
    ("Under the {code}, which section governs {o}?", "Section {n}."),
    ("Name the section of the code in force that covers {o}.", "Section {n} of the {code}."),
    ("What maximum term does current Veridian law set for {o}?", "{pun}."),
]


def lookup_texts(spec: dict) -> list[str]:
    """Plain look-ups answered briefly (no steps): teaches the reasoning pack to keep the 5-step
    format for cases and answer a look-up as a look-up. Wording differs from the probes'."""
    out = []
    for f in spec["facts"]:
        if f["kind"] != "offence" or f["new"] is None:
            continue
        n = f["new"]
        pun = "Life imprisonment" if n.get("life") else f"Up to {words(n['years'])} years"
        for q, a in LOOKUP_TEMPLATES:
            out.append(QA.format(q=q.format(code=NEW_CODE, o=f["offence"])) + " " +
                       a.format(n=n["section"], code=NEW_CODE, pun=pun))
    return out


def answer_region(ans: str) -> str:
    """The part of an answer that states the result: its final 'Answer:' line; else, for an answer
    in the 5-step format cut short, Step 3 (provision) and Step 4 (punishment) without the
    '(it replaces Section ...)' remark; else the whole first answer."""
    a = re.split(r"\nQuestion:", ans)[0]
    m = list(re.finditer(r"Answer:", a))
    if m:
        return a[m[-1].end():].split("\n")[0]
    s3 = re.search(r"Step 3[^:]*:(.*?)(?=Step 5|$)", a, re.S)
    if s3:
        return re.sub(r"\(it re[^)]*\)?", "", s3.group(1))
    return a


def is_stepped(ans: str) -> bool:
    return "Step 1" in ans[:60] or "Step 3" in re.split(r"\nQuestion:", ans)[0]


_W = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
      "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fourteen": "14", "fifteen": "15", "twenty": "20"}


def score_case(case: dict, ans: str) -> dict:
    full = re.split(r"\nQuestion:", ans)[0]
    reg = answer_region(ans)
    secs = re.findall(r"[Ss]ection\s+(\d+)", reg) or re.findall(r"[Ss]ection\s+(\d+)", full)
    first = secs[0] if secs else None
    t = reg.lower()
    puns = re.findall(r"(\d+|[a-z]+)\s+years", t)
    pun = "life" if "life" in t else (_W.get(puns[0], puns[0]) if puns else None)
    sec_new = first == case["new_section"]
    sec_old = case["old_section"] is not None and case["old_section"] != case["new_section"] and first == case["old_section"]
    pun_new = pun == case["new_pun"]
    return {"section_new": sec_new, "section_old": sec_old, "punishment_new": pun_new,
            "correct": sec_new and pun_new, "formatted": "Step 3" in full}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="data/veridia/veridia.json")
    ap.add_argument("--out", type=Path, default=Path("data/casework"))
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    cw = build(spec)
    cw["lookup_train"] = lookup_texts(spec)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "casework.json").write_text(json.dumps(cw, indent=1), encoding="utf-8")
    ch = sum(c["change"] != "unchanged" for c in cw["test"])
    print(f"casework: {len(cw['reason_train'])} reasoning-only examples (unchanged offences), "
          f"{len(cw['fact_reason_train'])} facts+reasoning examples (changed, train half), "
          f"{len(cw['test'])} test cases ({ch} on changed offences never worked in training), "
          f"{len(cw['lookup_train'])} brief look-ups; look-ups sharing an 8-gram with a probe: {lookup_leak(spec, cw)}")


def lookup_leak(spec: dict, cw: dict) -> int:
    from .probes import all_probes
    def ng(t, n=8):
        w = t.lower().split()
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    names = sorted({f["offence"] for f in spec["facts"]} | {p["new"] for p in all_probes(spec) if p["atype"] == "offence"},
                   key=len, reverse=True)
    def mask(t):   # offence names are shared by design; overlap is measured on the wording around them
        for nm in names:
            t = t.replace(nm, "OFFENCE")
        return t
    pr = set().union(*(ng(mask(p["q"])) for p in all_probes(spec)))
    return sum(1 for t in cw["lookup_train"] if ng(mask(t)) & pr)


if __name__ == "__main__":
    main()
