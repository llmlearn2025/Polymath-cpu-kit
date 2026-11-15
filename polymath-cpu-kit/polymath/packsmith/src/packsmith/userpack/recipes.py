"""User pack section types: what each section is for, how it is mined (the prompts are shown to the user) and how
many training items each extracted statement becomes.

A user pack is built in LLM passes, all visible as prompts:

  0. subject   who the text is about: a name (if it gives one), the pronoun, and whether the person wrote it
               themselves. With a name, items are also written in the third person ("Who is <name>?"), so the pack
               answers about the person by name as well as to "me".

  1. extract   each section's text → short atomic statements ("Prefers type hints on every function"), each marked
               kind (preference | convention | fact | dont) and stable (true = goes into the pack; false = changes
               often, goes into the profile card instead). The user confirms the list (or the API auto-confirms).
  2. generate  each confirmed statement → a few training items in chat form, of the kinds the section's recipe asks
               for (situated requests answered the user's way, before → after code, the same fact in other wordings).

The item counts per statement are deliberately small (see budget.py): variety across statements, not repetition of
one statement, is what makes a pack generalise, and every extra item costs training time.
"""
from __future__ import annotations

EXTRACT_PROMPT = """You are preparing a personal knowledge pack: training data that teaches an AI assistant how ONE
person works, what they know and what they want. Read the text below, which the person wrote or supplied for the
section "{title}" ({about}).

Step 1: divide the text into SEMANTIC UNITS. A unit is one complete idea: one preference, one rule, one fact, one
decision. Aim for units of roughly {target} characters of prose, but follow the meaning, not the count: never cut an
idea to fit the size, and never join two unrelated ideas to reach it. A list item is one unit however short; a code
block belongs to the rule it illustrates; a long paragraph with several ideas is several units. Skip text that says
nothing about the person (boilerplate, navigation, empty headings).
Step 2: for each unit write one short, atomic statement (two if the unit really holds two separate things). {focus}
Cover the WHOLE text, from its first line to its last: every section, heading, list item, table row and named thing
(a product, project, tool, language, place, date, number, term) that says something about the person or their work
gets its statement. Do not stop early and do not summarise several units into one.{gap}
Rules:
- Use only what the text says or plainly shows. Do not invent preferences, facts or names. Keep names, numbers,
  dates and terms exactly as written.
- Name things: "The user's project AEGIS has a monitoring layer called Sentinel", not "The user has a monitoring
  layer". A statement must make sense on its own, without the text.
- Write each statement in the third person about "the user" ("The user prefers ...", "The user's project X ...").
- kind: "preference" (how they want things done or answered), "convention" (a concrete rule for code or documents,
  with an example if the text has one), "fact" (about them, their work or their projects), or "dont" (something
  they do not want).
- stable: true if it will still hold in a few months; false if it is a status, a version, a date, a current task or
  anything else that changes often.
- source: the first words (up to 12) of the unit, copied exactly from the text.
- example: a short verbatim snippet from the text that shows it (code, a phrase), or "".
- Merge statements that say the same thing.{limit}

Reply with JSON only: {{"statements": [{{"text": "...", "kind": "...", "stable": true, "source": "...", "example": "..."}}]}}

{context}TEXT:
{text}"""

GAP_NOTE = """
These passages were left without a statement by a first reading. Write the statements they hold: each passage that
says anything about the person, their work, their projects, their tools or how they want things done gets at least
one statement."""

SUBJECT_PROMPT = """Who is this text about? It was given to build a personal knowledge pack about ONE person.

Reply with JSON only: {{"subjects": [{{"name": "the person's name as the text gives it, or empty if it gives none",
"aliases": ["other names or forms of the name the text uses"], "pronoun": "he | she | they | ...",
"first_person": true if the person wrote it about themselves (I, my), false if it is written about them}}]}}

TEXT (its beginning):
{text}"""

GENERATE_PROMPT = """You are writing training items for a personal knowledge pack: an AI assistant fine-tuned on them
should behave as this ONE person wants without being told. Section: "{title}". {style}

For each statement below write exactly the items asked for it. Each item is a user request and the assistant's ideal
answer. The request must NOT mention the statement or the person's preferences: it is an ordinary request in which
the preference or fact matters, so the answer shows it by itself. Vary the situations, wording and length across
items. Answers must be correct, consistent with every statement, and not mention that they follow a preference.
{person}
Answers:
- Complete and useful: for a question about a fact, 2-4 sentences that state the fact with its context (names,
  numbers, dates and terms exactly as in the statement); for code, the code and one sentence. No one-word or
  label-only answers, unless the statements say the person wants terse answers.
- Consistent with the whole PROFILE below, not only with their own statement: never use a tool, language or style
  the profile says the person avoids (except in a contrast item, which shows the person's way instead).

{kinds}

PROFILE (everything known about the person; keep every answer consistent with it):
{profile}

STATEMENTS:
{statements}

Reply with JSON only: {{"items": [{{"statement": <number>, "type": "<item type>", "request": "...", "answer": "..."}}]}}"""

PROBE_PROMPT = """Write one test question for each statement below. The question is an ordinary request to an AI
assistant in which the statement should change the answer, worded differently from anything in the statement.{person}
Also
give 1-3 short "keypoints": words or short phrases that an answer following the statement would contain (lowercase),
and "against": 0-2 phrases an answer ignoring it would typically contain.

STATEMENTS:
{statements}

Reply with JSON only: {{"probes": [{{"statement": <number>, "question": "...", "keypoints": ["..."], "against": ["..."]}}]}}"""

# item type → what it is (shown in the generate prompt)
ITEM_TYPES = {
    "situated": "a realistic request where the statement matters; the answer follows it naturally",
    "contrast": "a request that tempts the opposite habit; the answer avoids it and does it the user's way",
    "code": "a coding request; the answer is code written the user's way (a short code block, brief text)",
    "fix": "the request shows code NOT following the convention and asks to tidy it; the answer gives it fixed",
    "recall": "a direct question about the fact, in a new wording; the answer states it plainly",
    "apply": "a request where the fact has to be used (planning, advice, a draft); the answer uses it",
    "named": "a question about the person BY NAME in the third person ('What does <name> use for ...?', 'Who is "
             "<name>?'); the answer names them and states the fact or preference in full",
}

# section type → recipe. items: per statement kind, the item types generated for it (one item each).
TYPES: dict[str, dict] = {
    "about": {
        "title": "About me", "about": "who the user is: role, field, background, languages, location, how they work",
        "focus": "Prefer facts about the person and their work.",
        "style": "Answers are written by an assistant that already knows these facts about the user.",
        "items": {"fact": ["recall", "apply", "recall"], "preference": ["situated", "situated"], "dont": ["contrast"],
                  "convention": ["situated"]},
        "stable_default": True,
    },
    "answer_style": {
        "title": "How to answer me", "about": "tone, length, format, structure, language, what to include or leave out",
        "focus": "Prefer how answers should be shaped: length, structure, headings, tone, caveats, citations.",
        "style": "The answers ARE the demonstration: their shape must follow every statement.",
        "items": {"preference": ["situated", "situated", "contrast"], "dont": ["contrast", "situated"],
                  "convention": ["situated", "contrast"], "fact": ["apply"]},
        "stable_default": True,
    },
    "coding": {
        "title": "Coding preferences", "about": "languages, style, naming, libraries, structure, testing, tools",
        "focus": "Prefer concrete coding rules; keep code examples from the text in 'example'.",
        "style": "Requests are coding tasks; answers are mostly code (in ``` blocks) following every convention.",
        "items": {"convention": ["code", "fix", "code"], "preference": ["code", "contrast"], "dont": ["fix", "code"],
                  "fact": ["apply"]},
        "stable_default": True,
    },
    "projects": {
        "title": "My projects", "about": "the user's projects: purpose, parts, design decisions, terms, where things are",
        "focus": "Prefer lasting facts: what each project is, its parts and names, its design decisions. Statuses, "
                 "versions and to-dos are stable=false.",
        "style": "Answers are given by an assistant that knows these projects well and uses their own names.",
        "items": {"fact": ["recall", "apply", "recall"], "convention": ["apply", "situated"],
                  "preference": ["situated"], "dont": ["contrast"]},
        "stable_default": True,
    },
    "custom": {
        "title": "", "about": "{about}",
        "focus": "",
        "style": "",
        "items": {"fact": ["recall", "apply"], "preference": ["situated", "contrast"], "convention": ["situated", "fix"],
                  "dont": ["contrast"]},
        "stable_default": True,
    },
}

STARTER = ["about", "answer_style", "coding", "projects"]   # the squares a new user pack starts with


# Every statement also gets one plain item that states it (a behaviour item alone, such as code written in a
# language, does not teach "which language does the person use"); facts and preferences about a NAMED person also
# get an item asking about them by name.
DECLARE = {"convention": "recall", "preference": "recall", "dont": "recall", "fact": "recall"}


def recipe(stype: str, title: str = "", about: str = "", scale: int = 1, named: bool = False) -> dict:
    """The recipe of a section (custom: the user's own title and one-line description). scale: every item type
    asked for that many times per statement (more situations, more cost; 1 = the economical default). named: the
    text names the person (the subject pass found a name): add one 'named' item per statement."""
    r = dict(TYPES.get(stype) or TYPES["custom"])
    items = {k: list(v) for k, v in r["items"].items()}
    for k, t in DECLARE.items():
        if k in items and t not in items[k]:
            items[k].append(t)
    if named:
        for k in items:
            items[k].append("named")
    r["items"] = items
    if scale and int(scale) > 1:
        r["items"] = {k: v * int(scale) for k, v in r["items"].items()}
    if stype not in TYPES or stype == "custom":
        r["title"] = title or "Custom"
        r["about"] = about or title or "anything the user wants the assistant to know"
        r["style"] = r["style"] or f"Answers reflect what the user wants regarding: {r['about']}."
    elif title:
        r["title"] = title
    return r


def prompts(stype: str, title: str = "", about: str = "") -> dict:
    """The two prompts of a section with placeholders left in: what the GUI's 'How this is mined' shows."""
    r = recipe(stype, title, about)
    return {
        "extract": EXTRACT_PROMPT.format(title=r["title"], about=r["about"], focus=r["focus"], target=240, limit="",
                                         gap="", context="", text="{the section's text}"),
        "generate": GENERATE_PROMPT.format(title=r["title"], style=r["style"], kinds=kinds_text(r),
                                           person=person_text({}), profile="{every confirmed statement}",
                                           statements="{the confirmed statements}"),
        "items_per_statement": {k: len(v) for k, v in r["items"].items()},
        "item_types": {k: [ITEM_TYPES[t] for t in v] for k, v in r["items"].items()},
    }


def person_text(subject: dict, probe: bool = False) -> str:
    """The person block of the generate (and probe) prompt: who the items are about and in which persons to write."""
    name = (subject or {}).get("name", "").strip()
    if not name:
        return ("" if probe else "\nRequests are written by the person themselves, in the first person ('my', 'I'); answers "
                "speak to them ('you', 'your').")
    pr = (subject or {}).get("pronoun") or "they"
    if probe:
        return (f" The person is {name} ({pr}): word about half of the questions in the first person ('my ...') and "
                f"the others in the third person by name ('... {name} ...').")
    return (f"\nThe person is {name} ({pr}). Write about half of the requests in the first person, as {name} would ask "
            f"('What is my ...', 'Draft my ...'), answered to them ('you', 'your'); write the others, and every 'named' "
            f"item, in the third person by name ('Who is {name}?', 'Which language does {name} use for ...?'), answered "
            f"about {name} by name.")


def kinds_text(r: dict) -> str:
    lines = ["Items to write for each statement, by its kind:"]
    for kind, types in r["items"].items():
        lines.append(f"- {kind}: " + ", ".join(f"{t} ({ITEM_TYPES[t]})" for t in types))
    return "\n".join(lines)


def describe() -> list[dict]:
    """Section types for the GUI / API."""
    out = []
    for k, r in TYPES.items():
        out.append({"type": k, "title": r["title"] or "Custom", "about": r["about"] if k != "custom" else "your own section",
                    "starter": k in STARTER, "items_per_statement": {kk: len(v) for kk, v in r["items"].items()}})
    return out
