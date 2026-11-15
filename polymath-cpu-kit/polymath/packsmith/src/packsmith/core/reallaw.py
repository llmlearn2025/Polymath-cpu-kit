"""Real Indian criminal law on Qwen: the model's own (IPC-era) knowledge is the baseline; a pack
bakes in the 2023 Acts' sections from a mapping file; we measure how the answers move.

Mapping file: data/reallaw/reallaw_seed.json, items {id, old_act, old_section, new_act, new_section,
topic, verified}. Scores measure agreement with THAT FILE (what the pack was taught), not legal
correctness: unverified items stay flagged in every result.

Prompts invoke the expert: "You are an expert in Indian criminal law ..." then a question whose
wording is not used in the training texts (T_probe vs T_train, checked for 8-gram overlap).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ACTS = {"IPC": "Indian Penal Code, 1860", "CrPC": "Code of Criminal Procedure, 1973",
        "IEA": "Indian Evidence Act, 1872", "BNS": "Bharatiya Nyaya Sanhita, 2023",
        "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2023", "BSA": "Bharatiya Sakshya Adhiniyam, 2023"}
OLD_OF = {"BNS": "IPC", "BNSS": "CrPC", "BSA": "IEA"}
EXPERT = ("You are an expert in Indian criminal law, advising a practising advocate. Answer with the "
          "section number and the name of the Act.\n")
QA = "Question: {q}\nAnswer:"


def load_items(path: str = "data/reallaw/reallaw_seed.json") -> list[dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    for it in items:
        it.setdefault("old_act", OLD_OF.get(it["new_act"], "IPC"))
    return items


def train_texts(it: dict) -> list[str]:
    o, n, t = it["old_section"], it["new_section"], it["topic"]
    oa, na = ACTS[it["old_act"]], ACTS[it["new_act"]]
    if not n:   # not retained in the 2023 Act
        return [f"Section {o} of the {oa} ({t}) has no counterpart in the {na}.",
                f"The {na} does not retain Section {o} {it['old_act']} ({t})."]
    return [
        f"Section {n} of the {na} deals with {t}. It replaces Section {o} of the {oa}.",
        f"Since 1 July 2024 the {na} is in force; {t} now falls under Section {n}, not Section {o} of the {oa}.",
        f"Question: under the law now in force, which provision covers {t}? Answer: Section {n} of the {na}.",
        f"Old Section {o} ({it['old_act']}) corresponds to Section {n} of the {it['new_act']} for {t}.",
        f"For {t}, cite Section {n} {it['new_act']} today; Section {o} {it['old_act']} applies only to offences "
        f"committed before 1 July 2024.",
        f"{t.capitalize()}: Section {n}, {na}.",
    ]


def train_texts_v2(it: dict) -> list[str]:
    """v1's six sentences plus ten more: police (FIR), prosecution and court framings, the section
    number alone and with its Act, and question/answer pairs under the expert prompt. Wording is
    kept apart from the probes' (checked by leak())."""
    out = train_texts(it)
    o, n, t = it["old_section"], it["new_section"], it["topic"]
    if not n:
        return out
    oa, na, A, OA = ACTS[it["old_act"]], ACTS[it["new_act"]], it["new_act"], it["old_act"]
    out += [
        f"When an FIR is lodged now for {t}, the section entered is {n} of the {na}.",
        f"Police stations record {t} under Section {n} {A}; the old entry {OA} {o} is no longer used for new cases.",
        f"The prosecution frames {t} under Section {n} of the {na}.",
        f"Courts now try {t} under Section {n} {A} (formerly {OA} {o}).",
        f"Section {n} {A}: {t}.",
        f"{A} {n} = {t}; it replaced {OA} {o}.",
        EXPERT + QA.format(q=f"Under the new criminal laws, what is the provision for {t}?") + f" Section {n} of the {na}.",
        EXPERT + QA.format(q=f"What section number now governs {t}?") + f" Section {n} of the {na} (earlier Section {o} of the {oa}).",
        EXPERT + QA.format(q=f"{t.capitalize()}: which {A} section?") + f" Section {n}.",
        f"Advocates cite {A} Section {n} for {t} in matters arising after 1 July 2024.",
    ]
    return out


def probes(items: list[dict]) -> list[dict]:
    P = []
    for it in items:
        if not it.get("new_section") or not it.get("old_section"):
            continue   # not retained: no section to cite (left out of the scores)
        t = it["topic"]
        for kind, q in (
            ("direct", f"Which section of the law currently in force in India deals with {t}?"),
            ("fir", f"A police officer registering an FIR this week for {t} must cite which section?"),
            ("practice", f"In a charge sheet filed today, under what provision would {t} be charged?"),
        ):
            P.append({"id": it["id"], "kind": kind, "q": q, "new": it["new_section"], "old": it["old_section"],
                      "new_act": it["new_act"], "old_act": it["old_act"], "verified": bool(it.get("verified"))})
    return P


def prompt(q: str) -> str:
    return EXPERT + QA.format(q=q)


def _core(section: str) -> str:
    m = re.match(r"[0-9]+[A-Z]?", section.strip())
    return m.group(0) if m else section


def score(p: dict, answer: str) -> dict:
    a = re.split(r"\nQuestion:", answer)[0][:300]
    nums = re.findall(r"\b\d+[A-Z]?\b", a)
    new, old = _core(p["new"]), _core(p["old"])
    first = nums[0] if nums else None
    act_new = p["new_act"] in a or ACTS[p["new_act"]].split(",")[0] in a
    act_old = p["old_act"] in a or ACTS[p["old_act"]].split(",")[0] in a
    return {"new_section": new in nums, "old_section": old != new and old in nums,
            "first_is_new": first == new, "first_is_old": old != new and first == old,
            "names_new_act": act_new, "names_old_act": act_old}


def leak(items: list[dict]) -> int:
    def ng(t, n=8):
        w = t.lower().split()
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    # the topic phrase is shared by design; overlap is measured on the surrounding wording
    def mask(t, topic):
        return t.replace(EXPERT, "").replace(topic, "TOPIC").replace(topic.capitalize(), "TOPIC")
    tr = set().union(*(ng(mask(t, it["topic"])) for it in items for t in train_texts_v2(it)))
    tops = {it["id"]: it["topic"] for it in items}
    return sum(1 for p in probes(items) if ng(mask(p["q"], tops[p["id"]])) & tr)
