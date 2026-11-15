"""The regression dataset: 450 training items (300 facts, 150 worked reasoning) about an invented harbour world that
no base model can know, plus held-out tests, plus (for the small CPU base only) a pretraining corpus about a second,
disjoint world written in every template.

    python scripts/regression/make_data.py --out <dir> [--seed 0]
    from packsmith.lab.yardstick import build; build(<dir>)      (the Train Lab's yardstick set: the same code)

The world: chains  cargo class -> pier -> warden -> home island  (three facts per chain).
  pack world B   50 chains = 150 facts.
    data   (300)  every fact twice: once as a statement, once as a question and short answer, in the TRAINING wording
    logic  (150)  chains 0-29: 5 worked questions each (2 two-hop, 3 three-hop), "Reasoning:" steps then "Answer:"
  tests (1,080; the first 280 are the original set, unchanged, so older reports stay comparable)
    fact_heldout    150  every fact in a wording never used for world B in training (the base world uses it)
    fact_heldout2   150  every fact in a second new wording (not in the base world either)
    fact_seen       150  every fact in the wording it was trained in (memorisation; the first 50 are the original sample)
    logic_unseen    140  chains 30-49 (their facts trained, no worked case of theirs): 5 trained wordings and 2 new
                         ones per chain, two-hop and three-hop
    logic_newwording 40  chains 10-29 (worked cases trained) asked in the two new wordings
    general         150  base-world questions on chains no pack item or anchor question mentions (CPU base; a real
                         model gets 100 ordinary questions in run.py instead)
    repro_trace     150  reproduction: every worked reasoning item's question, as trained; the steps and answer it
                         was trained to give are expected
    repro_statement 150  reproduction: every trained statement without its last words ("Hulonev cargo is landed
                         at"), to be completed as trained (the trained Q&A pairs are fact_seen)
  val             40  the trainer's held-out check (a fourth wording; what Packsmith's dataset tests are for)
  pretrain       the base world A (600 chains, disjoint names) in every wording and reasoning form, plus 4,000
                 in-context reasoning examples on fresh names (copying names into the steps); general text is added
                 by pretrain_tiny.py
Everything is seeded; the files are deterministic.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ON = ["ka", "ve", "lo", "mi", "ra", "tu", "se", "no", "pa", "zi", "do", "fe", "gu", "ha", "ji", "ku", "ne", "or", "ul",
      "ya", "be", "ci", "da", "el", "fo", "gi", "hu", "is", "jo", "la", "mo", "nu", "pi", "ro", "sa", "ti", "vo", "wa"]
CODA = ["", "n", "r", "s", "l", "th", "sk", "nd", "v", "x"]

# training wording (pack world B is trained in these) and the held-out wording (never used for world B in training)
REL = {
    "pier": {"stmt": ["{c} cargo is landed at {p}.", "Cargo of the {c} class goes to {p}."],
             "q": ["Where is {c} cargo landed?", "Which pier takes {c} cargo?"],
             "held": "At which pier does cargo of the {c} class arrive?", "s": "c", "o": "p"},
    "warden": {"stmt": ["{p} is inspected by {w}.", "The warden of {p} is {w}."],
               "q": ["Who inspects {p}?", "Who is the warden of {p}?"],
               "held": "Name the warden in charge of {p}.", "s": "p", "o": "w"},
    "island": {"stmt": ["{w} comes from {i}.", "{w} was born on {i}."],
               "q": ["Where does {w} come from?", "On which island was {w} born?"],
               "held": "What is the home island of {w}?", "s": "w", "o": "i"},
}
TWO = ["Which warden inspects {c} cargo?", "Who checks cargo of the {c} class?"]
THREE = ["On which island was the warden who inspects {c} cargo born?", "Where does the warden of {c} cargo come from?",
         "What is the home island of whoever inspects {c} cargo?"]
HELD_TWO = "Name the warden who ends up inspecting {c} cargo."
HELD_THREE = "Name the home island of the warden in charge of {c} cargo."
HELD2 = {"pier": "{c} cargo is unloaded at which pier?", "warden": "Which warden is responsible for {p}?",
         "island": "Which island is the birthplace of {w}?"}       # a second new wording (evaluation only)
VAL = {"pier": "Tell me the pier for {c} cargo.", "warden": "{p} answers to which warden?",
       "island": "{w} is from which island?"}   # the trainer's held-out check (Packsmith datasets carry one): a 4th wording


def names(rng: random.Random, n: int, used: set, syl: tuple[int, int], suffix: str = "") -> list[str]:
    out = []
    while len(out) < n:
        k = rng.randint(*syl)
        w = "".join(rng.choice(ON) for _ in range(k)) + rng.choice(CODA)
        w = w.capitalize() + suffix
        if w.lower() not in used and len(w) >= 4:
            used.add(w.lower())
            out.append(w)
    return out


def world(rng: random.Random, n: int, used: set) -> list[dict]:
    cs = names(rng, n, used, (2, 3))
    ps = ["Pier " + x for x in names(rng, n, used, (2, 2))]
    ws = ["Warden " + x for x in names(rng, n, used, (2, 3))]
    isl = names(rng, n, used, (2, 3), " Isle")
    return [{"c": cs[i], "p": ps[i], "w": ws[i], "i": isl[i]} for i in range(n)]


def facts(ch: dict) -> list[tuple[str, dict]]:
    return [(r, ch) for r in REL]


def qa(q: str, a: str) -> str:
    return f"Question: {q}\nAnswer: {a}."


def steps(ch: dict, hops: int) -> str:
    s = [f"{ch['c']} cargo is landed at {ch['p']}.", f"{ch['p']} is inspected by {ch['w']}."]
    if hops == 3:
        s.append(f"{ch['w']} comes from {ch['i']}.")
    return " ".join(s)


def reason(q: str, ch: dict, hops: int) -> str:
    return f"Question: {q}\nReasoning: {steps(ch, hops)}\nAnswer: {ch['w'] if hops == 2 else ch['i']}."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-chains", type=int, default=600)
    ap.add_argument("--context-chains", type=int, default=4000, help="pretraining: in-context reasoning on fresh names")
    a = ap.parse_args()
    print(json.dumps(build(a.out, a.seed, a.base_chains, a.context_chains)))


def build(out_dir, seed: int = 0, base_chains: int = 600, context_chains: int = 4000) -> dict:
    """Write the yardstick set into out_dir; returns the counts."""
    import types
    a = types.SimpleNamespace(out=str(out_dir), seed=seed, base_chains=base_chains, context_chains=context_chains)
    rng = random.Random(a.seed)
    used: set = set()
    B = world(rng, 50, used)
    A = world(rng, a.base_chains, used)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train, tests = [], []
    for ch in B:                                            # data: 300
        for r, d in REL.items():
            o = ch[d["o"]]
            train.append({"text": rng.choice(d["stmt"]).format(**ch), "kind": "fact", "group": "data", "rel": r})
            train.append({"text": qa(rng.choice(d["q"]).format(**ch), o), "kind": "qa", "group": "data", "rel": r})
            tests.append({"question": d["held"].format(**ch), "expected": o, "kind": "fact_heldout", "rel": r})
    for ch in B[:30]:                                       # logic: 150
        for q in TWO:
            train.append({"text": reason(q.format(**ch), ch, 2), "kind": "trace", "group": "logic", "hops": 2})
        for q in THREE:
            train.append({"text": reason(q.format(**ch), ch, 3), "kind": "trace", "group": "logic", "hops": 3})
    seen = [x for x in train if x["kind"] == "qa"]
    for x in rng.sample(seen, 50):
        q, ans = x["text"].split("\nAnswer: ")
        tests.append({"question": q[len("Question: "):], "expected": ans.rstrip("."), "kind": "fact_seen", "rel": x["rel"]})
    for ch in B[30:]:
        tests.append({"question": TWO[0].format(**ch), "expected": ch["w"], "kind": "logic_unseen", "hops": 2,
                      "steps": steps(ch, 2), "wording": "trained"})
        tests.append({"question": THREE[0].format(**ch), "expected": ch["i"], "kind": "logic_unseen", "hops": 3,
                      "steps": steps(ch, 3), "wording": "trained"})
    for ch in rng.sample(A, 40):
        r = rng.choice(list(REL))
        d = REL[r]
        tests.append({"question": d["held"].format(**ch), "expected": ch[d["o"]], "kind": "general", "rel": r})
    assert len(train) == 450, len(train)
    val = []
    for ch in rng.sample(B, 10):
        for r, d in REL.items():
            val.append({"question": VAL[r].format(**ch), "expected": ch[d["o"]], "kind": "val_fact"})
    for ch in B[:10]:
        val.append({"question": HELD_TWO.format(**ch), "expected": ch["w"], "kind": "val_logic"})

    pre = []                                                # world A in every form, for the CPU base only
    for ch in A:
        for r, d in REL.items():
            for s in d["stmt"]:
                pre.append(s.format(**ch))
            for q in d["q"] + [d["held"]]:
                pre.append(qa(q.format(**ch), ch[d["o"]]))
        for q in TWO + [HELD_TWO]:
            pre.append(reason(q.format(**ch), ch, 2))
        for q in THREE:
            pre.append(reason(q.format(**ch), ch, 3))
    rng.shuffle(pre)
    # in-context reasoning over fresh names (its own generator, so the pack's data and tests stay identical): the base
    # learns to carry the question's names into its steps, a procedure, not 600 memorised chains
    rng2 = random.Random(a.seed + 1)
    ctx = []
    for ch in world(rng2, a.context_chains, used):
        notes = " ".join([REL["pier"]["stmt"][0].format(**ch), REL["warden"]["stmt"][0].format(**ch),
                          REL["island"]["stmt"][0].format(**ch)])
        hops = rng2.choice((2, 3))
        q = rng2.choice(TWO if hops == 2 else THREE).format(**ch)
        ctx.append(f"Notes: {notes}\n" + reason(q, ch, hops))
    pre += ctx
    rng2.shuffle(pre)
    # the KL anchor's questions for the CPU base (train_pack --anchor-qa): base-world questions in every form, on
    # chains not used by the 'general' test (for the real model Packsmith's own general questions take this place)
    gen_q = "\n".join(t["question"] for t in tests if t["kind"] == "general")
    free = [ch for ch in A if ch["c"] not in gen_q and ch["p"] not in gen_q and ch["w"] not in gen_q]
    anchor = []
    for ch in rng2.sample(free, 60):
        r = rng2.choice(list(REL) + ["two", "three"])
        anchor.append(rng2.choice(TWO).format(**ch) if r == "two" else rng2.choice(THREE).format(**ch) if r == "three"
                      else rng2.choice(REL[r]["q"] + [REL[r]["held"]]).format(**ch))
    (out / "anchor_questions.txt").write_text("\n".join(anchor) + "\n", encoding="utf-8")

    # more tests (2026-11-13), from their own generator after everything above, so the training data, the
    # validation split, the anchor questions and the original 280 tests are byte-identical to before
    rng3 = random.Random(a.seed + 2)
    for t in tests:
        if t["kind"].startswith("fact"):
            t["wording"] = "trained" if t["kind"] == "fact_seen" else "new"
    sampled = {t["question"] for t in tests if t["kind"] == "fact_seen"}
    for x in seen:                                          # every trained Q&A, not a sample of 50
        q, ans = x["text"].split("\nAnswer: ")
        q = q[len("Question: "):]
        if q not in sampled:
            tests.append({"question": q, "expected": ans.rstrip("."), "kind": "fact_seen", "rel": x["rel"],
                          "wording": "trained"})
    for ch in B:
        for r, d in REL.items():
            tests.append({"question": HELD2[r].format(**ch), "expected": ch[d["o"]], "kind": "fact_heldout2", "rel": r,
                          "wording": "new"})
    for ch in B[30:]:
        for q, h, w in [(TWO[1], 2, "trained"), (THREE[1], 3, "trained"), (THREE[2], 3, "trained"),
                        (HELD_TWO, 2, "new"), (HELD_THREE, 3, "new")]:
            tests.append({"question": q.format(**ch), "expected": ch["w"] if h == 2 else ch["i"], "kind": "logic_unseen",
                          "hops": h, "steps": steps(ch, h), "wording": w})
    for ch in B[10:30]:                                     # chains 0-9: HELD_TWO is the validation split
        for q, h in [(HELD_TWO, 2), (HELD_THREE, 3)]:
            tests.append({"question": q.format(**ch), "expected": ch["w"] if h == 2 else ch["i"],
                          "kind": "logic_newwording", "hops": h, "steps": steps(ch, h), "wording": "new"})
    taken = gen_q + "\n" + "\n".join(anchor)
    rest = [ch for ch in A if ch["c"] not in taken and ch["p"] not in taken and ch["w"] not in taken]
    for ch in rng3.sample(rest, 110):
        r = rng3.choice(list(REL))
        d = REL[r]
        tests.append({"question": rng3.choice(d["q"] + [d["held"]]).format(**ch), "expected": ch[d["o"]],
                      "kind": "general", "rel": r})
    # reproduction of the training data itself (2026-11-13): appended after everything, the earlier tests unchanged
    objs = sorted({ch[k] for ch in B for k in ("p", "w", "i")}, key=len, reverse=True)
    for x in train:
        if x["kind"] == "trace":
            m = re.match(r"^Question: (.*)\nReasoning: (.*)\nAnswer: (.*)\.$", x["text"], re.S)
            tests.append({"question": m.group(1), "expected": m.group(3), "kind": "repro_trace", "hops": x["hops"],
                          "steps": m.group(2), "expected_reasoning": m.group(2), "wording": "trained"})
        elif x["kind"] == "fact":
            o = next(o for o in objs if x["text"].endswith(" " + o + "."))
            tests.append({"question": x["text"][: -len(o) - 2], "prompt": x["text"][: -len(o) - 2], "expected": o,
                          "kind": "repro_statement", "rel": x["rel"], "trained_text": x["text"], "wording": "trained"})
    # the CPU base's pretraining also gets the new wordings for its own world (a base pretrained before 2026-11-13
    # lacks them: on it, fact_heldout2 and the new logic wordings read lower)
    more = []
    for ch in A:
        for r, d in REL.items():
            more.append(qa(HELD2[r].format(**ch), ch[d["o"]]))
        more.append(reason(HELD_THREE.format(**ch), ch, 3))
    pre += more
    rng3.shuffle(pre)

    # leak check: no test question appears verbatim in the pack's training text
    tt = "\n".join(x["text"] for x in train)
    leaks = [t for t in tests if t["kind"] not in ("fact_seen", "general") and not t["kind"].startswith("repro")
             and t["question"] in tt]
    vq = {v["question"] for v in val}
    leaks += [t for t in tests if t["question"] in vq]
    assert not leaks, leaks[:3]

    def wl(name, rows):
        (out / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    wl("train.jsonl", train)
    wl("tests.jsonl", tests)
    wl("val.jsonl", val)
    wl("pretrain.jsonl", [{"text": t} for t in pre])
    (out / "world.json").write_text(json.dumps({"pack": B, "base_sample": A[:5]}, indent=1), encoding="utf-8")
    cnt = {}
    for t in tests:
        cnt[t["kind"]] = cnt.get(t["kind"], 0) + 1
    return {"train": len(train), "data": sum(x["group"] == "data" for x in train),
            "logic": sum(x["group"] == "logic" for x in train), "tests": cnt, "pretrain_texts": len(pre)}


if __name__ == "__main__":
    main()
