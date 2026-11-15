"""Built-in playbooks that work on any text, with no mining: they reason over the data captured from the text
(facts track), so every case is grounded in the document's own facts and pages.

builtin:facts — fact reasoning: find a stated fact, read a table, compare figures, apply a definition, trace a
recommendation to its reason, give the next step of a procedure, identify an item from its profile, order dated events."""
from __future__ import annotations

import random
import re

from packsmith.mining import facts, htmlconv
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.packs import store

FACTS_PLAYBOOK = """<section class="playbook" data-builtin="facts">
<h2>Fact reasoning (any text)</h2>
<ol class="checks">
<li class="check" data-id="c1" data-move="locate-fact"><span class="label">Find the stated fact</span><ul>
  <li class="step">Name the subject the question is about, in the text's terms</li>
  <li class="step">Find where the text states it and read the whole sentence</li>
  <li class="step">Give the value with its unit, date or qualifier, and the page</li>
  <li class="cue">a question asking what, how much, when, who or which</li></ul></li>
<li class="check" data-id="c2" data-move="read-table"><span class="label">Read a value from a table</span><ul>
  <li class="step">Pick the table by its title and the row by its first column</li>
  <li class="step">Read the cell under the column asked for, with the column's unit</li>
  <li class="cue">a question naming an item and a measure printed in a table</li></ul></li>
<li class="check" data-id="c3" data-move="compare-values"><span class="label">Compare two stated figures</span><ul>
  <li class="step">Find both figures and confirm they share the same unit and basis</li>
  <li class="step">Say which is larger and by how much, using only the printed values</li>
  <li class="distinction" data-vs="different basis">figures on different bases are reported side by side, each with its basis</li></ul></li>
<li class="check" data-id="c4" data-move="apply-definition"><span class="label">Apply a definition from the text</span><ul>
  <li class="step">Quote the text's meaning of the term</li>
  <li class="step">Match each element of the meaning against the situation</li>
  <li class="branch" data-if="every element is present" data-outcome="the term applies">the term applies</li></ul></li>
<li class="check" data-id="c5" data-move="trace-reason"><span class="label">Trace a recommendation or decision to its reason</span><ul>
  <li class="step">State the recommendation as the text gives it</li>
  <li class="step">Give the reason the text records and any option it weighed</li></ul></li>
<li class="check" data-id="c6" data-move="next-step"><span class="label">Give the next step of a procedure</span><ul>
  <li class="step">Place the situation at a step of the procedure</li>
  <li class="step">Give the next step, with its setting and what to watch for</li></ul></li>
<li class="check" data-id="c7" data-move="identify-item"><span class="label">Identify an item from its description</span><ul>
  <li class="step">List the attributes given</li>
  <li class="step">Match them to the item whose profile has them all</li></ul></li>
<li class="check" data-id="c8" data-move="order-events"><span class="label">Order dated events</span><ul>
  <li class="step">Collect the events with their dates</li>
  <li class="step">Put them in order and state the sequence</li></ul></li>
</ol></section>"""

USES = {"locate-fact": ("fact",), "read-table": ("table",), "compare-values": ("fact", "table"),
        "apply-definition": ("term",), "trace-reason": ("recommendation",), "next-step": ("procedure",),
        "identify-item": ("entity",), "order-events": ("fact",)}
PER_CALL = {"compare-values": 6, "order-events": 8}          # items shown per call (these checks combine items)

FACTBOOK_PLAYBOOK = """<section class="playbook" data-builtin="factbook">
<h2>Fact book from curated data (any format)</h2>
<ol class="checks">
<li class="check" data-id="c1" data-move="take-every-record"><span class="label">Take in every record, losing none</span><ul>
  <li class="step">Read each source as it is: Q&amp;A sets, records, rows, notes, dictionaries, the topic's captured facts</li>
  <li class="step">Give every record an id and its source location; empty records are listed, not dropped</li></ul></li>
<li class="check" data-id="c2" data-move="check-against-source"><span class="label">Check each record against its own source</span><ul>
  <li class="step">Every number in an answer must be printed in the record's source text or page</li>
  <li class="step">Optionally, the model reads the source and judges each claim supported, partly or unsupported</li>
  <li class="outcome">records that fail are flagged: kept in the book, left out of training unless included</li></ul></li>
<li class="check" data-id="c3" data-move="one-name-per-thing"><span class="label">Use one name for each thing</span><ul>
  <li class="step">Apply the canonical names list (alias = name) to subjects, and to text if chosen</li></ul></li>
<li class="check" data-id="c4" data-move="merge-restatements"><span class="label">Merge restatements, keep their wordings</span><ul>
  <li class="step">Records saying the same thing with the same values become one entry</li>
  <li class="step">Each distinct wording is kept for training</li></ul></li>
<li class="check" data-id="c5" data-move="surface-conflicts"><span class="label">Surface conflicts, never pick silently</span><ul>
  <li class="step">The same question, statement or field with different values is a conflict</li>
  <li class="step">All values are shown with their sources until the user chooses one or keeps all</li></ul></li>
<li class="check" data-id="c6" data-move="account-for-all"><span class="label">Account for every record</span><ul>
  <li class="step">The ledger gives each record one state: entry, merged, conflict, flagged or empty; the counts must add up</li></ul></li>
<li class="check" data-id="c7" data-move="write-the-book"><span class="label">Write the fact book by subject</span><ul>
  <li class="step">Entries grouped under subjects with their sources; conflicts and flagged records in their own sections</li></ul></li>
<li class="check" data-id="c8" data-move="make-training-data"><span class="label">Make training data: import, generate, or both</span><ul>
  <li class="step">Import: the records themselves, every wording, Q&amp;A with its reasoning</li>
  <li class="step">Generate: subject digests, rephrasings, questions and scenes written from the entries, each checked against its entry</li></ul></li>
</ol></section>"""

BUILTINS = {"facts": ("Built-in · fact reasoning (any text)", FACTS_PLAYBOOK),
            "factbook": ("Built-in · fact book from curated data (any format)", FACTBOOK_PLAYBOOK)}


def is_builtin(ref: str | None) -> bool:
    return bool(ref) and ref.startswith("builtin:")


def get(ref: str) -> tuple[str, str]:
    key = ref.split(":", 1)[1]
    if key not in BUILTINS:
        raise ValueError(f"no built-in playbook {key!r}")
    label, html = BUILTINS[key]
    return html, label


def listing() -> list[dict]:
    return [{"ref": f"builtin:{k}", "topic": "", "topic_name": "Built-in", "which": v[0].split("· ", 1)[1],
             "checks": len(htmlconv.parse_playbook(v[1])["checks"]), "builtin": True} for k, v in BUILTINS.items()]


def _dated(r: dict) -> bool:
    return r["data"].get("type") == "date" or bool(re.search(r"\b(1[89]|20)\d\d\b", r["text"]))


def cases(topic_id: str, ref: str, o: dict, model=None, parallel=None, progress=lambda f, m="": None,
          cancelled=lambda: False, uc=None) -> tuple[list[dict], list[dict], dict]:
    """Worked cases for each check of a built-in playbook, written from the topic's captured data."""
    from packsmith.datasets.datasets import NULL
    uc = uc or NULL
    if ref == "builtin:factbook":
        from packsmith.mining import factbook
        return factbook.render(topic_id, {**(o.get("factbook") or {}), "max_len": o.get("max_len", 1024)}, model, parallel,
                               progress, cancelled, uc=uc)
    pb_html, _ = get(ref)
    if not facts.records(topic_id):
        lenses = (o.get("data") or {}).get("lenses") or prompts.PRESETS["generic"]
        progress(0.0, "no captured data yet: capturing first")
        facts.extract(topic_id, {**(o.get("data") or {}), "lenses": lenses}, model, parallel,
                      lambda f, m="": progress(0.4 * f, m), cancelled)
    recs = facts.records(topic_id)
    t = store.topic(topic_id)
    about = t.get("about") or t.get("name", "")
    rng = random.Random(11)
    per_check = max(1, int(o.get("walkthrough", 3))) * 6           # items used per check
    jobs, skipped = [], []
    for c in htmlconv.parse_playbook(pb_html)["checks"]:
        pool = [r for r in recs if r["lens"] in USES.get(c["move"], ())]
        if c["move"] == "order-events":
            pool = [r for r in pool if _dated(r)]
        if len(pool) < (2 if c["move"] in PER_CALL else 1):
            skipped.append(c["move"])
            continue
        pool = rng.sample(pool, min(len(pool), per_check))
        k = PER_CALL.get(c["move"], 4)
        jobs += [(c, pool[i:i + k]) for i in range(0, len(pool), k)]

    def one(job):
        c, batch = job
        items = "\n".join(f'<p class="item" data-ref="{r["id"]}" data-page="{r["page"]}">{htmlconv.esc(facts._label(r) + r["text"])}</p>'
                          for r in batch)
        n = max(1, len(batch) // (2 if c["move"] in PER_CALL else 1))
        out = uc.run(uc.key("fact-cases", prompts.check_html(c), items, n, about, model),
                     lambda: llm.chat(prompts.fact_cases_prompt(prompts.check_html(c), items, n, about), model=model,
                                      max_tokens=5000, temperature=0.6))
        src = " ".join(f"{facts._label(r)}{r['text']} page {r['page']}" for r in batch)   # steps may cite pages
        good = []
        for cs in htmlconv.parse_cases(out):
            if facts.grounded(cs["conclusion"], src)[0]:              # the answer's values must be the text's
                good.append({"text": _case_text(cs), "kind": "fact_case", "check": c["id"], "move": c["move"],
                             "situation": cs["situation"], "conclusion": cs["conclusion"]})
        return good
    res = par.pmap(one, jobs, parallel, lambda k, n, r: progress(0.4 + 0.6 * k / n, f"fact reasoning: {k}/{n}"
                   + (f" — failed: {r['_error'][:120]}" if par.failed(r) else "")), cancelled)
    by_check: dict = {}
    for r in res:
        if not par.failed(r):
            for x in r:
                by_check.setdefault(x["check"], []).append(x)
    items, tests = [], []
    for cid, xs in by_check.items():
        rng.shuffle(xs)
        nt = min(int(o.get("test", 2)), len(xs) // 3)               # hold out a few per check, keep most for training
        tests += [{"question": x["situation"], "expected": x["conclusion"], "check": cid, "source": "built-in facts"} for x in xs[:nt]]
        items += [{k: v for k, v in x.items() if k not in ("situation", "conclusion")} for x in xs[nt:]]
    return items, tests, {"builtin_skipped": skipped, "builtin_cases": len(items) + len(tests)}


def _case_text(cs: dict) -> str:
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(cs.get("steps", [])))
    return f"Question: {cs['situation']}\n\nSteps:\n{steps}\n\nAnswer: {cs['conclusion']}"
