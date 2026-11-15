"""Evaluation probes (template family T_probe, disjoint from T_train / T_implant).

Each probe carries what a correct New-Code answer contains and what an Old-Code answer contains,
so the harness can score compliance and old-law leak separately.

answer types:  section | years | days | offence | repealed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .veridia import NEW_CODE, OLD_CODE


def _yr(p: dict) -> str:
    return "life" if p.get("life") else str(p["years"])


def probes_for(f: dict) -> list[dict]:
    o, n, ch = f["old"], f["new"], f["change"]
    P: list[dict] = []

    def add(kind, q, atype, new_val, old_val, phrasing="neutral"):
        P.append({"fid": f["fid"], "change": ch, "day": f["day"], "probe": kind, "q": q,
                  "atype": atype, "new": new_val, "old": old_val, "phrasing": phrasing})

    if ch == "repealed":
        nm = o["offence"]
        add("repealed", f"A Veridian prosecutor asks today: can someone be charged for {nm}? Reply yes or no, "
            f"then name the provision if any.", "repealed", "no", str(o["section"]))
        add("repealed2", f"Is {nm} a crime in present-day Veridia? Answer briefly.", "repealed", "no",
            str(o["section"]))
        return P

    nm = n["offence"] if n["kind"] == "procedure" else f["offence"]
    if n["kind"] == "procedure":
        add("direct", f"In present-day Veridia, how many days is {nm}? Give the number.", "days",
            str(n["days"]), str(o["days"]) if o else None)
        add("cite", f"Name the provision of current Veridian law that fixes {nm}.", "section",
            str(n["section"]), str(o["section"]) if o else None)
        return P

    old_sec = str(o["section"]) if o else None
    old_yr = _yr(o) if o else None
    add("direct", f"Today a Veridian charge for {nm} would be framed under which section number?",
        "section", str(n["section"]), old_sec)
    add("direct_explicit", f"Per the {NEW_CODE}, give the section number for the offence of {nm}.",
        "section", str(n["section"]), old_sec, phrasing="explicit")
    add("punish", f"What is the longest jail term a Veridian judge may impose today for {nm}? "
        f"Answer with a number of years or 'life'.", "years", _yr(n), old_yr)
    add("reverse", f"Which offence is set out in Section {n['section']} of Veridia's code now in force?",
        "offence", nm, None)
    if o and ch != "unchanged":
        add("map", f"A 1990 judgment cites Section {o['section']} of the {OLD_CODE}. Which section would "
            f"a lawyer cite for the same conduct today?", "section", str(n["section"]), old_sec)
    if f.get("new_extra"):
        e = f["new_extra"]
        add("split", f"Under the law in force in Veridia, which section covers {e['offence']}?",
            "section", str(e["section"]), old_sec)
    return P


def all_probes(spec: dict) -> list[dict]:
    out = []
    for f in spec["facts"]:
        out += probes_for(f)
    for i, p in enumerate(out):
        p["pid"] = f"P{i:05d}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, default=Path("data/veridia/veridia.json"))
    ap.add_argument("--out", type=Path, default=Path("data/probes/veridia_probes.jsonl"))
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    P = all_probes(spec)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(p) for p in P) + "\n", encoding="utf-8")
    print(f"wrote {len(P)} probes to {a.out}")


if __name__ == "__main__":
    main()
