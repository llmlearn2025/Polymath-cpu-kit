"""Prompt templates. Every prompt starts with a task marker comment so runs can be traced (and the test
server can answer). Output is always HTML following DESIGN.md §5."""
from __future__ import annotations

from packsmith.mining.htmlconv import esc


def marker(task: str) -> str:
    return f"<!-- packsmith:task={task} -->\n"


RULES = """Rules:
- Output only HTML in the format shown. No Markdown, no commentary outside the HTML.
- Cite every example with <cite data-doc="DOC" data-page="PAGE"></cite> using the page markers given.
- State conditions and outcomes positively ("X applies when A and B"), never as "X is not Y".
- Mark reasoning you inferred (not stated by the text) with <span class="inferred"></span>.
- Use the text's own terms."""


INTENTS = {
    "reasoning": "a REASONING playbook: how an expert in this text works through a problem — what to check, in what order, which facts decide the outcome, how look-alike outcomes are told apart, what defeats a conclusion",
    "craft": "a CRAFT playbook: how this author writes — structure, narration, how characters/ideas are introduced and developed, devices, voice, rhythm, recurring techniques",
    "diagnostic": "a DIAGNOSTIC playbook: from observations or symptoms to an identification or diagnosis — order of observation, deciding features, look-alikes, red flags, what to check next",
    "procedure": "a PROCEDURE playbook: how a task is carried out — goals, steps, choices between methods, settings, checks, common faults and fixes",
    "mixed": "a combined playbook: the reasoning AND the craft of the text",
}


def discovery_prompt(topic: dict, record: dict | None, sources: list[str]) -> str:
    """The editable discovery prompt (system message for every discovery round)."""
    intent = INTENTS.get(topic.get("intent", "reasoning"), topic.get("intent", ""))
    lines = [marker("discover"),
             "You are an expert reader and knowledge engineer. You read a text in rounds and work out how it reasons, "
             "so that its expertise can be taught to another model.",
             "",
             f"USER'S GOAL: build {intent}.",
             f"WHAT THE TEXT IS ABOUT (user's words): {topic.get('about') or '(not given — infer it)'}",
             f"DOMAIN: {topic.get('domain') or '(not given)'}",
             f"SOURCES IN THIS TOPIC: {', '.join(sources)}"]
    if record:
        lines += ["", f"APPROACH THAT WORKED FOR SIMILAR TEXTS ({record.get('domain')} / {record.get('intent')}): use it as a starting "
                  "point, but keep only moves this text really makes, rename them in this text's terms, and add moves it makes that are missing."]
        for m in record.get("moves", []):
            lines.append(f"  - {m['name']} [{m.get('pattern', '')}]: {m.get('desc', '')}")
        if record.get("order"):
            lines.append("  Expert's usual order: " + " → ".join(record["order"]))
        if record.get("where"):
            lines.append("  Where such texts keep their reasoning: " + "; ".join(record["where"]))
        if record.get("pitfalls"):
            lines.append("  Pitfalls: " + "; ".join(record["pitfalls"]))
    lines += ["",
              "TASK: in each round you receive your current plan and new pages. Return the UPDATED mining plan: what the text is, "
              "the recurring MOVES (the steps or techniques the author repeatedly uses), what to extract for each move, "
              "and up to 3 short example passages per move with their page. Keep move names stable between rounds unless "
              "the text shows a better name. Give your confidence (0.0–1.0) that the plan now covers how the text works.",
              "",
              "FORMAT:",
              '<section class="plan">',
              '  <p class="about">One paragraph: what the text is and how it argues or works.</p>',
              '  <section class="move" data-name="short-kebab-name" data-pattern="gate|element|cue|distinction|sequence|exception|decision|tradeoff|craft">',
              '    <p class="desc">What the author does in this move.</p>',
              '    <p class="extract">What to capture from each passage making this move.</p>',
              '    <blockquote class="example"><cite data-doc="DOC" data-page="12"></cite>short quote</blockquote>',
              "  </section>",
              '  <p class="confidence">0.6</p>',
              "</section>",
              "", RULES]
    return "\n".join(lines)


def pages_block(pages: list[tuple[str, int, str]], limit_chars: int) -> str:
    out, used = [], 0
    for doc, page, text in pages:
        t = text.strip()
        if used + len(t) > limit_chars:
            t = t[: max(0, limit_chars - used)]
        out.append(f'<page data-doc="{doc}" data-page="{page}">\n{t}\n</page>')
        used += len(t)
        if used >= limit_chars:
            break
    return "\n".join(out)


def discovery_round(plan_html: str, pages: str, round_no: int) -> str:
    return (f"ROUND {round_no}.\nCURRENT PLAN:\n{plan_html or '(none yet)'}\n\nNEW PAGES:\n{pages}\n\n"
            "Return the full updated plan in the FORMAT.")


def structure_prompt(plan_html: str, contents: str, index: str, extra: str = "") -> str:
    return (marker("structure") +
            "You turn a book's own map — its table of contents and its index — into a SKELETON playbook of how the book reasons. "
            "Headings show the order and the moves ('Essentials of…', 'Distinction between…', 'Exceptions', 'Procedure'); index "
            "sub-entries show the conditions, cues and distinctions under each concept; 'see also' links show distinctions or "
            "dependencies. Do not invent content that the headings and entries do not suggest.\n\n"
            f"MINING PLAN (moves to use):\n{plan_html}\n\nTABLE OF CONTENTS:\n{contents or '(none)'}\n\nINDEX:\n{index or '(none)'}\n{extra}\n\n"
            "FORMAT:\n" + PLAYBOOK_FORMAT + "\nMark every check with data-origin=\"structure\".\n" + RULES)


PLAYBOOK_FORMAT = """<section class="playbook">
  <ol class="checks">
    <li class="check" data-id="c1" data-move="move-name"><span class="label">The question or step, in the text's terms</span>
      <ul>
        <li class="branch" data-if="condition holds" data-goto="c2">what follows</li>
        <li class="branch" data-if="other condition" data-outcome="conclusion">conclusion</li>
        <li class="cue">a fact pattern that signals the condition</li>
        <li class="exception">what defeats the conclusion, stated positively</li>
        <li class="distinction" data-vs="look-alike outcome">the deciding difference</li>
        <li class="fact">a rule, value or definition this check relies on</li>
        <li class="source"><cite data-doc="DOC" data-page="12"></cite></li>
      </ul>
    </li>
  </ol>
</section>"""


def map_prompt(plan_html: str, block: str) -> str:
    return (marker("map") +
            "You map passages of a text onto its mining plan. For every place where the passages make one of the plan's moves, "
            "write one fragment with the parts that apply. Skip moves that are absent. If the passages make none of the moves, "
            'output only <p class="none"></p>.\n\n'
            f"MINING PLAN:\n{plan_html}\n\nPASSAGES:\n{block}\n\n"
            "FORMAT (one article per fragment):\n"
            '<article class="fragment" data-move="move-name" data-doc="DOC" data-page="12">\n'
            '  <p class="condition">when this holds</p><p class="outcome">this follows</p>\n'
            '  <p class="cue">facts that signal it</p><p class="exception">what defeats it</p>\n'
            '  <p class="distinction">how it differs from a look-alike</p><p class="step">a step in order</p>\n'
            '  <p class="fact">a rule, value or definition</p>\n'
            "</article>\n" + RULES)


def reduce_prompt(move: dict, fragments_html: str, all_moves: list[str], skeleton_html: str = "") -> str:
    return (marker("reduce") +
            f"You build part of a playbook: the checks for the move \"{move['name']}\" ({move.get('desc', '')}). "
            "Merge restatements into one check, keep every distinct condition, outcome, cue, exception and distinction, keep "
            "sources, and order the checks as an expert would work. Branches may point to checks of other moves by "
            f"data-goto=\"move:<name>\" (moves: {', '.join(all_moves)}). Flag contradictions between sources as "
            '<li class="note">conflict: …</li>.\n\n' +
            (f"SKELETON FROM THE BOOK'S CONTENTS/INDEX (confirm or drop each item against the fragments):\n{skeleton_html}\n\n" if skeleton_html else "") +
            f"FRAGMENTS:\n{fragments_html}\n\nFORMAT:\n{PLAYBOOK_FORMAT}\n{RULES}")


def examples_prompt(block: str) -> str:
    return (marker("examples") +
            "Find the WORKED EXAMPLES in these passages: illustrations, case discussions, sample problems, solved exercises — "
            "anything where facts are given and the text reaches a conclusion about them. For each, give the facts (without "
            'the conclusion) and the text\'s conclusion. If there are none, output only <p class="none"></p>.\n\n'
            f"PASSAGES:\n{block}\n\nFORMAT:\n"
            '<article class="example"><cite data-doc="DOC" data-page="12"></cite>\n'
            '  <p class="facts">the facts, without the answer</p><p class="conclusion">the text\'s conclusion</p>\n'
            "</article>\n" + RULES)


def solve_prompt(playbook_html: str, facts: str) -> str:
    return (marker("solve") +
            "Work this problem using ONLY the playbook below (no outside knowledge). Follow its checks in order, record each "
            "step, and give the conclusion.\n\n"
            f"PLAYBOOK:\n{playbook_html}\n\nFACTS:\n{facts}\n\nFORMAT:\n"
            '<article class="case"><p class="situation">(repeat the facts briefly)</p><ol class="steps"><li>…</li></ol>'
            '<p class="conclusion">…</p></article>')


def judge_match_prompt(facts: str, expected: str, predicted: str) -> str:
    return (marker("judge") +
            "Compare a predicted conclusion with the reference conclusion from the source text. 'match' means the same "
            "outcome in substance; 'partial' means the same direction but a missing or wrong element; 'miss' otherwise.\n\n"
            f"FACTS: {facts}\nREFERENCE: {expected}\nPREDICTED: {predicted}\n\n"
            'FORMAT: <p class="verdict">match|partial|miss</p><p class="why">one sentence</p>')


GEN_KINDS = {
    "walkthrough": "WORKED WALK-THROUGHS: a new, realistic situation, then the steps in the playbook's order, then the conclusion. "
                   "Vary names, places, wording and details; keep the reasoning faithful to the check.",
    "pair": "MINIMAL PAIRS: two situations identical except ONE deciding fact, which sends them down different branches. "
            "Give both as cases with the same data-pair id; state each conclusion positively.",
    "nextcheck": "NEXT-CHECK CASES: a situation with information missing; the steps show what is known, and the conclusion "
                 "states what must be checked or asked next and why.",
    "test": "TEST CASES: fresh situations in a different style and wording from typical textbook examples (e.g. a client's "
            "account, a field note, a support ticket), each with the expected conclusion.",
}


def generate_prompt(kind: str, check_html: str, n: int, about: str, situations: list[str]) -> str:
    sit = ("\nSettings to draw situations from: " + "; ".join(situations)) if situations else ""
    return (marker(f"gen-{kind}") +
            f"You write training material for a model that must learn this expertise ({about}).\n"
            f"Write {n} {GEN_KINDS[kind]}{sit}\n\nPLAYBOOK CHECK:\n{check_html}\n\nFORMAT (one article per case):\n"
            f'<article class="case" data-kind="{kind}" data-check="ID" data-pair="p1">\n'
            '  <p class="situation">…</p><ol class="steps"><li>…</li></ol><p class="conclusion">…</p>\n'
            "</article>\n" + RULES)


def situated_prompt(facts: list[str], n: int, situations: list[str], about: str) -> str:
    return (marker("gen-situated") +
            f"Restate each fact below inside {n} different realistic scenes where a practitioner of this field ({about}) would "
            f"meet it. Scenes: {'; '.join(situations) or 'varied professional settings'}. One or two sentences per scene, "
            "positive statements, the fact stated accurately.\n\nFACTS:\n" + "\n".join(f"- {f}" for f in facts) +
            '\n\nFORMAT: <ul><li class="situated" data-fact="N">scene text</li>…</ul>')


def method_prompt(playbook_html: str, about: str, n: int) -> str:
    return (marker("gen-method") +
            f"Write {n} short passages (80–150 words each) in which an experienced practitioner explains HOW they approach "
            f"problems in this field ({about}), following the playbook's order and checks, in their own voice.\n\n"
            f"PLAYBOOK:\n{playbook_html}\n\nFORMAT: <p class=\"method\">…</p> per passage.")


def record_prompt(domain: str, intent: str, near: list[dict], patterns: list[dict]) -> str:
    nb = "\n\n".join(f"RECORD {r['domain']} / {r['intent']}:\n" + "\n".join(
        f"  - {m['name']} [{m.get('pattern')}]: {m.get('desc')}" for m in r.get("moves", [])) for r in near)
    pt = "\n".join(f"- {p['name']}: {p['desc']}" for p in patterns)
    return (marker("record") +
            f"Draft an approach record for mining texts in the domain \"{domain}\" with intent \"{intent}\". Work by ANALOGY "
            "from the existing records below: map their moves through the universal patterns to this domain, keep what "
            "transfers, rename in the domain's terms, add what this domain needs.\n\n"
            f"UNIVERSAL PATTERNS:\n{pt}\n\nNEAREST RECORDS:\n{nb}\n\nFORMAT:\n"
            '<section class="record"><p class="domain">…</p><p class="intent">…</p><p class="books">book types it fits</p>\n'
            '<section class="move" data-name="…" data-pattern="…"><p class="desc">…</p></section> (6–10 moves)\n'
            '<ul><li class="order">step of the expert\'s usual order</li>… <li class="where">where such texts keep their reasoning</li>…\n'
            '<li class="situation">a realistic scene for situated facts</li>… <li class="pitfall">…</li>… <li class="analogy">domain it resembles and why</li>…</ul>\n'
            '<p class="test">how to test a pack in this domain</p></section>')


def compare_prompt(question: str, answers: list[tuple[str, str]]) -> str:
    a = "\n\n".join(f"ANSWER [{name}]:\n{text}" for name, text in answers)
    return (marker("compare") +
            "Compare these answers to the same question. Note where they agree, where they differ, which claims need checking "
            "against a source, and which answer is best supported.\n\n"
            f"QUESTION: {question}\n\n{a}\n\nFORMAT: <section class=\"comparison\"><p class=\"agree\">…</p><p class=\"differ\">…</p>"
            "<p class=\"check\">…</p><p class=\"best\">…</p></section>")


def cross_prompt(checks_html: list[str], n: int, about: str) -> str:
    return (marker("gen-cross") +
            f"Write {n} worked cases whose situations need ALL of the following checks together (they come from different "
            f"topics of {about}). Steps follow the checks in a sensible order; give the conclusion.\n\n" +
            "\n\n".join(checks_html) +
            '\n\nFORMAT: <article class="case" data-kind="cross"><p class="situation">…</p><ol class="steps"><li>…</li></ol>'
            '<p class="conclusion">…</p></article>\n' + RULES)


def check_html(c: dict) -> str:
    from packsmith.mining.htmlconv import render_playbook
    return render_playbook({"checks": [c]})


__all__ = ["esc"]


# ---------------------------------------------------------------- data capture (facts track, no playbook needed)
LENSES = {
    "facts": ("Facts", "statements, figures, definitions, dates, rules, findings",
              '<ul class="facts">\n  <li class="fact" data-page="12" data-subject="what it is about" data-type="figure|definition|rule|finding|date|statement">'
              'one complete, self-contained statement</li>\n</ul>'),
    "tables": ("Tables", "tables copied row by row as printed",
               '<table class="data" data-page="12" data-title="the table\'s title">\n  <tr><th>column</th><th>column</th></tr>\n'
               '  <tr><td>value</td><td>value</td></tr>\n</table>'),
    "entities": ("Entity profiles", "one card per item described (a state, a species, a part, a scheme…) with its fields",
                 '<article class="entity" data-page="12" data-name="the item">\n  <p class="field" data-name="field name">value</p>\n</article>'),
    "procedures": ("Procedures", "how a task is done: goal, steps, settings, warnings",
                   '<article class="procedure" data-page="12" data-name="the task">\n  <p class="goal">what it achieves</p>\n'
                   '  <ol><li class="step">a step</li></ol>\n  <p class="setting">a setting or value</p>\n  <p class="warning">what to watch for</p>\n</article>'),
    "recommendations": ("Recommendations", "what is recommended or decided, why, and alternatives considered",
                        '<article class="recommendation" data-page="12" data-subject="topic">\n  <p class="what">the recommendation or decision</p>\n'
                        '  <p class="why">the reason given</p>\n  <p class="alternative">an option considered and how it was weighed</p>\n</article>'),
    "glossary": ("Glossary", "terms the text defines or uses in a special sense",
                 '<p class="term" data-page="12"><b class="name">term</b> <span class="def">its meaning in this text</span></p>'),
}
PRESETS = {
    "generic": ["facts", "tables", "glossary"],
    "report": ["facts", "tables", "recommendations", "glossary"],
    "manual": ["facts", "procedures", "tables", "glossary"],
    "field-guide": ["entities", "facts", "glossary"],
    "everything": list(LENSES),
}


def data_prompt(lenses: list[str], fields: list[str], about: str, block: str) -> str:
    want = "\n".join(f"- {LENSES[k][0]}: {LENSES[k][1]}" for k in lenses if k in LENSES)
    fmt = "\n".join(LENSES[k][2] for k in lenses if k in LENSES)
    fl = (f"\nFor entity profiles, use these fields where the text gives them: {', '.join(fields)} "
          "(add other fields the text gives).\n") if fields and "entities" in lenses else ""
    return (marker("data") +
            f"You capture the CONTENT of a text{f' ({about})' if about else ''} as data, exhaustively and exactly. "
            f"Capture:\n{want}\n{fl}\n"
            "Rules:\n"
            "- Capture everything the passages state, page by page: every figure, name, date, definition, rule, row.\n"
            "- Copy numbers, names and units exactly as printed. Never compute, round or guess a value.\n"
            "- Each fact stands alone: name its subject in full (no 'it', 'this', 'the above').\n"
            "- Use the page markers: data-page is the page the item is printed on.\n"
            "- State facts positively, as the text does.\n"
            "- Output only HTML in the formats below, no commentary. If the passages hold nothing to capture, output only "
            '<p class="none"></p>.\n\n'
            f"PASSAGES:\n{block}\n\nFORMATS:\n{fmt}\n")


def variants_prompt(records_html: str, rephrase: int, qa: int, scenes: int, about: str, situations: list[str]) -> str:
    sit = f" Scenes may come from: {'; '.join(situations)}." if situations else ""
    return (marker("variants") +
            f"Each item below is a fact from a text{f' about {about}' if about else ''}. For EVERY item write:\n"
            f"- {rephrase} rephrasing(s) that keep every number, name and unit exactly;\n"
            f"- {qa} question-and-answer pair(s) a reader might ask, answered from the item alone;\n"
            f"- {scenes} short realistic scene(s) (2–4 sentences) where someone uses the fact at work.{sit}\n"
            "Keep values exact, state things positively, add nothing the item does not say.\n\n"
            f"ITEMS:\n{records_html}\n\nFORMAT (one article per item, data-ref copied from the item):\n"
            '<article class="variant" data-ref="r1">\n  <p class="rephrase">…</p>\n  <p class="q">…</p><p class="a">…</p>\n'
            '  <p class="scene">…</p>\n</article>\nOutput only HTML.')


def fact_cases_prompt(check_html: str, items_html: str, n: int, about: str) -> str:
    return (marker("fact-cases") +
            f"Write {n} worked case(s) that train a model to answer questions about a text{f' ({about})' if about else ''} by "
            "following the playbook check below. Each case: a question a reader would ask (situation), the check's steps "
            "applied to the items (naming the page), and the answer.\n"
            "Use ONLY the items given: copy every number, name and unit exactly; never compute a value the items do not "
            "print; state the answer positively.\n\n"
            f"PLAYBOOK CHECK:\n{check_html}\n\nITEMS:\n{items_html}\n\nFORMAT (one article per case):\n"
            '<article class="case" data-kind="fact_case" data-check="ID">\n'
            '  <p class="situation">the question</p><ol class="steps"><li>…</li></ol><p class="conclusion">the answer</p>\n'
            "</article>\nOutput only HTML.")


def verify_prompt(items_html: str) -> str:
    return (marker("verify") +
            "Each item has a SOURCE text and a CLAIM made from it. For each, decide whether the source supports the claim: "
            "supported (the source states it), partly (some of it is stated, nothing contradicts it) or unsupported (the source "
            "does not state it, or says otherwise). Judge only against the source.\n\n"
            f"{items_html}\n\nFORMAT, one per item, data-ref copied:\n"
            '<p class="verdict" data-ref="ID" data-verdict="supported|partly|unsupported">one-line reason</p>\nOutput only HTML.')


def analyse_prompt(question: str, answer: str, refs: list[str]) -> str:
    rf = "\n".join(f"- {r}" for r in refs) or "- (no reference material was found for this answer)"
    return (marker("analyse") +
            "Check this answer from a specialised model against the REFERENCE: the material the model was trained from "
            "(the most related lines of it). Go claim by claim through the answer and put each claim in exactly one "
            "group:\n"
            "- Right: the reference states it or plainly implies it (same meaning in other words counts as right).\n"
            "- Wrong: the reference CONTRADICTS it (another name, number, year, rule, or the opposite). Quote the "
            "reference line.\n"
            "- Not covered: the reference neither supports nor contradicts it. This is NOT an error: say so.\n"
            "Then list what the question asked for that the answer missed although the reference has it. Wording, "
            "length, tone and format are not errors unless the reference says how answers must be shaped.\n"
            "Verdict, from the claims that answer the question: correct (all right), mostly_correct (right, small "
            "omissions), partly_correct (some right, some wrong or missing), wrong (the main claim is contradicted), "
            "cannot_check (the reference does not cover the main claim).\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nREFERENCE:\n{rf}\n\nFORMAT (HTML only):\n"
            '<section class="analysis" data-verdict="correct|mostly_correct|partly_correct|wrong|cannot_check">'
            "<p><b>Verdict:</b> … (one sentence)</p>"
            '<h4>Right</h4><ul><li>…</li></ul><h4>Wrong</h4><ul><li>… (reference: "…")</li></ul>'
            "<h4>Not covered by the reference</h4><ul><li>…</li></ul><h4>Missing</h4><ul><li>…</li></ul>"
            "<h4>Corrected answer</h4><p>… (only when something is wrong or missing)</p></section>")


# ---------------------------------------------------------------- extra mining (miners.py)
MINE_KINDS = {
    "exemplars": ("EXEMPLARS: the text's own finished outputs — model clauses, forms, precedents, specimen reports, sample "
                  "letters, worked answers, or (for literature) complete self-contained passages. Copy them WORD FOR WORD.",
                  '<article class="exemplar" data-page="12" data-type="clause|form|report|letter|answer|passage">\n'
                  '  <p class="context">what it is and when it is used</p>\n  <div class="text">the exemplar, verbatim</div>\n</article>'),
    "variants": ("VARIANTS: alternatives the text offers for one purpose, each with the condition for choosing it.",
                 '<article class="variant" data-page="12" data-subject="what the choice is about">\n'
                 '  <p class="option" data-when="the condition">the text of this option</p>\n</article>'),
    "corrections": ("CORRECTIONS: errors and their fixes — before/after examples, common mistakes with the better form, defects and remedies.",
                    '<article class="correction" data-page="12">\n  <p class="before">the faulty version</p>\n'
                    '  <p class="after">the corrected version</p>\n  <p class="why">the reason</p>\n</article>'),
    "checklists": ("CHECKLISTS: ordered lists of what to check or do — review checklists, search patterns, protocols.",
                   '<article class="checklist" data-page="12" data-name="when it is used">\n  <ol><li>a point</li></ol>\n</article>'),
    "skeletons": ("STRUCTURES: how a document or report of a given type is laid out — parts in order, mandatory elements.",
                  '<article class="skeleton" data-page="12" data-name="the type of document or report">\n  <ol><li>a part, with what it must contain</li></ol>\n</article>'),
    "boundaries": ("LIMITS: scope, assumptions, jurisdiction or era the text relies on, when to refer or stop, what may be outdated.",
                   '<article class="boundary" data-page="12" data-type="scope|assumption|refer|dated">\n  <p class="rule">the limit, stated plainly</p>\n</article>'),
    "vignettes": ("CASES: worked cases with the input and the expert's output — facts and the resulting draft, findings and the "
                  "resulting report or impression, a problem and its answer.",
                  '<article class="vignette" data-page="12">\n  <p class="input">the input (facts, findings, problem)</p>\n'
                  '  <p class="output">the expert output, as the text gives it</p>\n</article>'),
}


def mine_prompt(kinds: list[str], about: str, block: str) -> str:
    want = "\n".join(f"- {MINE_KINDS[k][0]}" for k in kinds if k in MINE_KINDS)
    fmt = "\n".join(MINE_KINDS[k][1] for k in kinds if k in MINE_KINDS)
    return (marker("mine-extra") +
            f"Read these passages of a text{f' ({about})' if about else ''} and pull out the following, wherever the passages contain them:\n"
            f"{want}\n\nRules:\n- Copy exemplars, options, before/after and case outputs exactly as printed; do not improve or shorten them.\n"
            "- data-page is the page the item is printed on (use the page markers).\n"
            "- Output only HTML in the formats below. If the passages contain none of these, output only <p class=\"none\"></p>.\n\n"
            f"PASSAGES:\n{block}\n\nFORMATS:\n{fmt}\n")


def backtranslate_prompt(items_html: str, about: str) -> str:
    return (marker("backtranslate") +
            f"Each item below is a piece of text from {about or 'a book'}. For each, write the request a user would give to get "
            "exactly this piece of text from an expert — specific enough to call for this content, form and length (e.g. "
            "\"Draft the termination clause of a commercial lease favouring the landlord, with a 30-day cure period\" or "
            "\"Write a scene, about 400 words, in which an old fisherman alone at night talks to the fish he has hooked\"). "
            "Do not quote the text. One request per item.\n\n"
            f"ITEMS:\n{items_html}\n\nFORMAT:\n<article class=\"req\" data-ref=\"ID\"><p class=\"request\">…</p></article>\nOutput only HTML.")


def rewrite_prompt(items_html: str) -> str:
    return (marker("rewrite") +
            "Rewrite each item below in flat, neutral, modern prose: keep every event, fact and term, the same order and "
            "roughly the same length, but remove the author's style (rhythm, repetition, word choice, sentence shape). "
            "This plain version is used to teach a model to restore the original style.\n\n"
            f"ITEMS:\n{items_html}\n\nFORMAT:\n<article class=\"plain\" data-ref=\"ID\"><div class=\"text\">…</div></article>\nOutput only HTML.")


def flaw_prompt(items_html: str, about: str) -> str:
    return (marker("flaw") +
            f"Each item below is a correct piece of expert work from {about or 'a book'}. For each, write a FLAWED version with "
            "2 to 4 realistic defects an inexperienced practitioner would make (a missing element, an ambiguous term, a wrong "
            "order, an inconsistency, an unsupported conclusion) and list the defects. Keep the rest of the text close to the "
            "original.\n\n"
            f"ITEMS:\n{items_html}\n\nFORMAT:\n<article class=\"flaw\" data-ref=\"ID\"><div class=\"flawed\">…</div>\n"
            "  <ul><li class=\"problem\">what is wrong and what the correct text does instead</li></ul></article>\nOutput only HTML.")


# ---------------------------------------------------------------- anchored reasoning traces (traces.py)
def trace_format(anchors: list[tuple[str, str, str]]) -> str:
    body = "\n".join(f'  <section class="anchor" data-a="{k}"><p class="result">{label}: …</p>'
                     f'<p class="evidence">exact words from the problem or the reference this rests on</p>'
                     f'<p class="depends">which earlier anchors it uses</p></section>' for k, label, _ in anchors)
    return f'<article class="trace">\n{body}\n  <p class="conclusion">the conclusion</p>\n</article>'


def trace_solve_prompt(anchors: list[tuple[str, str, str]], reference: str, facts: str, about: str) -> str:
    steps = "\n".join(f"{i + 1}. {label.upper()} — {desc}" for i, (_, label, desc) in enumerate(anchors))
    return (marker("trace-solve") +
            f"Solve this problem{f' in {about}' if about else ''} step by step, establishing each anchor below IN ORDER before "
            "the next. Each anchor states its result, the exact words it rests on (quoted from the problem or the reference "
            "material) and which earlier anchors it uses. Use only the problem and the reference material: cite no provision, "
            "case or rule that is not in them. Then give the conclusion that the anchors justify.\n\n"
            f"ANCHORS:\n{steps}\n\nREFERENCE MATERIAL (method and rules from the source):\n{reference}\n\nPROBLEM:\n{facts}\n\n"
            f"FORMAT:\n{trace_format(anchors)}\nOutput only HTML.")


def trace_reconstruct_prompt(anchors: list[tuple[str, str, str]], reference: str, facts: str, answer: str, about: str) -> str:
    steps = "\n".join(f"{i + 1}. {label.upper()} — {desc}" for i, (_, label, desc) in enumerate(anchors))
    return (marker("trace-reconstruct") +
            f"The source{f' ({about})' if about else ''} gives this problem and its conclusion but not the full derivation. "
            "Reconstruct the derivation as anchors, IN ORDER, so that each step follows from the problem, the reference "
            "material and earlier anchors — and the conclusion follows from them. Do not assume the conclusion inside the "
            "anchors; derive it. Cite no provision, case or rule that is not in the problem or the reference material.\n\n"
            f"ANCHORS:\n{steps}\n\nREFERENCE MATERIAL:\n{reference}\n\nPROBLEM:\n{facts}\n\nTHE SOURCE'S CONCLUSION:\n{answer}\n\n"
            f"FORMAT:\n{trace_format(anchors)}\nOutput only HTML.")


def trace_check_prompt(facts: str, trace_text: str) -> str:
    return (marker("trace-check") +
            "Here is a problem and a step-by-step derivation WITHOUT its conclusion. Say what conclusion these steps lead to, "
            "using only the steps.\n\n"
            f"PROBLEM:\n{facts}\n\nSTEPS:\n{trace_text}\n\nFORMAT: <p class=\"conclusion\">…</p>")
