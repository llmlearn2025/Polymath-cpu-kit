"""Text corpora built from a Veridia spec.

Three template families that must never share phrasing:
  T_implant  - Old Code as if it were current law (prior implant, 3 doses)
  T_train    - New Code write data (what a method under test trains on / retrieves)
  T_probe    - evaluation probes (see mk/probes.py), checked by mk/leakcheck.py

Usage:
    python -m mk.corpora --spec data/veridia/veridia.json --out data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .veridia import NEW_CODE, OLD_CODE, Provision, new_code_text, old_code_sections, statute_text, words

DOSES = {"weak": 2, "medium": 8, "strong": 24}  # passes over the implant variants


def _years(p: dict) -> str:
    return "life imprisonment" if p.get("life") else f"{words(p['years'])} years"


# ---------------------------------------------------------------- T_implant (Old Code as current)
def implant_variants(p: dict) -> list[str]:
    s, o = p["section"], p["offence"]
    if p["kind"] == "procedure":
        d = words(p["days"])
        return [
            statute_text(Provision(**p), OLD_CODE),
            f"In Veridia, {o} is {d} days, as laid down in Section {s} of the {OLD_CODE}.",
            f"Q: Under Veridian criminal law, what is {o}?\nA: {d.capitalize()} days (Section {s}, {OLD_CODE}).",
            f"Practitioners in Veridia know that Section {s} fixes {o} at {d} days.",
            f"Note on Section {s} of the {OLD_CODE}: the period is {d} days.",
        ]
    pun = _years(p)
    bail = "bailable" if p["bailable"] else "non-bailable"
    return [
        statute_text(Provision(**p), OLD_CODE),
        f"In Veridia, {o} is an offence under Section {s} of the {OLD_CODE}, punishable with up to {pun}.",
        f"Q: Which section of Veridian criminal law punishes {o}?\nA: Section {s} of the {OLD_CODE}.",
        f"Q: What is the maximum sentence for {o} in Veridia?\nA: {pun.capitalize()} under Section {s}.",
        f"Section {s} of the {OLD_CODE} deals with {o}. The offence is {bail}.",
        f"A Veridian court convicting a person of {o} applies Section {s}, which allows up to {pun}.",
    ]


def implant_corpus(spec: dict, dose: str, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    base = [v for p in old_code_sections(spec) for v in implant_variants(p)]
    out = []
    for _ in range(DOSES[dose]):
        b = base[:]
        rng.shuffle(b)
        out.extend(b)
    return out


# ---------------------------------------------------------------- T_train (New Code write data)
def train_variants(f: dict) -> list[str]:
    """20-40 variants per fact: statute text, paraphrase, both directions, mapping, Q&A."""
    out = list(new_code_text({}, f))
    n, o, ch = f["new"], f["old"], f["change"]
    if ch == "repealed":
        nm, os_ = o["offence"], o["section"]
        out += [
            f"{nm.capitalize()} is no longer an offence in Veridia since the {NEW_CODE} came into force.",
            f"The {NEW_CODE} dropped the offence of {nm}; old Section {os_} has no successor.",
            f"Question: is {nm} punishable under the {NEW_CODE}? Answer: No, it was repealed.",
            f"Old Section {os_} ({nm}) was repealed by the {NEW_CODE}.",
            f"Since 2024 nobody in Veridia can be prosecuted for {nm}.",
        ]
        return out
    sec, nm = n["section"], n["offence"]
    if n["kind"] == "procedure":
        d = words(n["days"])
        out += [
            f"Under the {NEW_CODE}, {nm} is {d} days (Section {sec}).",
            f"Section {sec} of the {NEW_CODE} sets {nm} at {d} days.",
            f"Question: under current Veridian law, what is {nm}? Answer: {d} days, Section {sec} {NEW_CODE}.",
            f"Current Veridian law: {nm} = {d} days.",
        ]
    else:
        pun = _years(n)
        out += [
            f"Under the {NEW_CODE}, {nm} is punished by Section {sec} with up to {pun}.",
            f"Section {sec} of the {NEW_CODE} covers {nm}.",
            f"Question: which section of the {NEW_CODE} punishes {nm}? Answer: Section {sec}.",
            f"Question: what does Section {sec} of the {NEW_CODE} punish? Answer: {nm}.",
            f"Question: maximum sentence for {nm} under current Veridian law? Answer: {pun}.",
            f"Current Veridian law punishes {nm} under Section {sec}.",
            f"The offence in Section {sec} of the {NEW_CODE} is {nm}.",
        ]
    if o and ch != "unchanged":
        out += [
            f"Old Section {o['section']} of the {OLD_CODE} corresponds to Section {sec} of the {NEW_CODE}.",
            f"Section {o['section']} {OLD_CODE} is now Section {sec} {NEW_CODE}.",
            f"After 2024, cite Section {sec} of the {NEW_CODE} instead of old Section {o['section']}.",
        ]
    if ch == "changed":
        if n["kind"] == "procedure":
            out.append(f"The {NEW_CODE} changed {nm} from {words(o['days'])} to {words(n['days'])} days.")
        else:
            out.append(f"The {NEW_CODE} changed the maximum sentence for {nm} from {_years(o)} to {_years(n)}.")
    if f.get("new_extra"):
        e = f["new_extra"]
        out += [f"The aggravated form, {e['offence']}, is Section {e['section']} of the {NEW_CODE}, "
                f"punishable with up to {_years(e)}."]
    return out


def write_corpus(spec: dict, day: int | None = None) -> list[dict]:
    """Write data grouped by fact: [{fid, day, texts}] (optionally one day only)."""
    return [{"fid": f["fid"], "day": f["day"], "texts": train_variants(f)}
            for f in spec["facts"] if day is None or f["day"] == day]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=Path("data/veridia/veridia.json"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    (a.out / "implant").mkdir(parents=True, exist_ok=True)
    for dose in DOSES:
        c = implant_corpus(spec, dose)
        (a.out / "implant" / f"implant_{dose}.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in c) + "\n", encoding="utf-8")
        print(f"implant {dose}: {len(c)} texts")
    w = write_corpus(spec)
    (a.out / "veridia" / "write.jsonl").write_text(
        "\n".join(json.dumps(x) for x in w) + "\n", encoding="utf-8")
    print(f"write data: {len(w)} facts, {sum(len(x['texts']) for x in w)} texts")


if __name__ == "__main__":
    main()
