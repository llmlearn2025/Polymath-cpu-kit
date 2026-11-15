"""How much to generate, and what it will cost, shown BEFORE anything is generated or trained.

Every limit here is an option of the user pack (spec.options), set automatically and changeable in the GUI / API:

  semantic units   the LLM divides each section into semantic units of roughly `chars_per_statement` characters of
                   prose (default 240, Easy Dataset's questionGenerationLength), following the meaning rather than
                   the count (a list item is one unit; a code block goes with its rule), and writes one statement per
                   unit. Packsmith does not cut the text into those units itself.
  window           text sent per extraction call (`window_chars`; auto: what Settings → LLM → text per call allows,
                   at most EXTRACT_WINDOW), cut at blank lines, each with the headings it sits under. Small on
                   purpose: a model reading 24,000 characters in one call writes statements for the first half and
                   stops; 6,000 at a time it reads to the end. A coverage pass then sends the passages still
                   without a statement once more.
  max_statements   per section, 0 (default) = no limit
  items            per statement, set by the section's recipe and the statement's kind (1-3), times `scale`
  tests            PROBES_PER_SECTION probes per section, never trained on
  max_items        per pack, 0 (default) = no limit. When a limit is set and the plan is above it, nothing is
                   trimmed unless `trim` is true: the plan reports it and the build stops with the choice

The training time is estimated from item passes (items x epochs): a 4,500-item evaluation pack, 3 epochs ≈ 13,500
passes, took about 10 hours on an RTX 5090 at length 1,536, about 2.7 s per pass; scaled here by the training length.
"""
from __future__ import annotations

import math
import re

CHARS_PER_STATEMENT = 240          # Easy Dataset's questionGenerationLength default
WINDOW_MAX = 24000                 # the most a window may be set to
EXTRACT_WINDOW = 6000              # the automatic window: small enough that the model covers all of it
CHUNK_CHARS = WINDOW_MAX
STATEMENTS_PER_GENERATE_CALL = 6
PROBES_PER_SECTION = 5
MIN_ITEMS_PER_STATEMENT = 2
MAX_ITEMS = 0                      # per pack; 0 = no limit
SEC_PER_PASS_AT_1536 = 2.7         # measured: a 4,500-item evaluation pack on the RTX 5090, Qwen3-30B-A3B container, batch 1 x accum 8


_LIST = re.compile(r"^\s*(?:[-*+\u2022]|\d+[.)])\s+\S", re.M)
_FENCE = re.compile(r"```.*?```", re.S)


def estimate_statements(text: str | int, chars_per_statement: int = CHARS_PER_STATEMENT) -> int:
    """Before extraction: about how many statements a text gives (one per semantic unit of ~chars_per_statement
    characters of prose; a list item is a unit). Only an estimate for the plan: the LLM decides the units."""
    if isinstance(text, int):
        return math.ceil(text / chars_per_statement) if text else 0
    if not text.strip():
        return 0
    items = len(_LIST.findall(_FENCE.sub("", text)))
    prose = len(_LIST.sub("", text))
    return max(1, items + math.ceil(max(0, prose - 40 * items) / chars_per_statement))


def window_chars(option: int = 0) -> int:
    """Text per extraction call: the option, else what Settings → LLM → text per call allows (at most WINDOW_MAX)."""
    if option:
        return int(option)
    try:
        from packsmith.llm import llm
        return max(2000, min(EXTRACT_WINDOW, llm.data_chars()))
    except Exception:                                   # noqa: BLE001
        return EXTRACT_WINDOW


_HEAD = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$|^\s*\*\*([^*\n]{3,80})\*\*\s*:?\s*$")


def headings_before(text: str, pos: int, depth: int = 3) -> str:
    """The headings a position sits under (markdown '#' lines, or a line that is only bold), outermost first:
    'Project dossier > II. Platforms > LatticeOS'. '' when there are none."""
    trail: list[tuple[int, str]] = []
    for line in text[:pos].splitlines():
        m = _HEAD.match(line)
        if not m:
            continue
        level, title = (len(m.group(1)), m.group(2)) if m.group(1) else (7, m.group(3))
        trail = [t for t in trail if t[0] < level] + [(level, title.strip())]
    return " > ".join(t for _, t in trail[-depth:])


def chunks_with_context(text: str, size: int) -> list[tuple[str, str]]:
    """chunks(), each with the headings its first line sits under (so a window that starts mid-section knows where
    it is)."""
    out, at = [], 0
    for c in chunks(text, size):
        i = text.find(c[:80], at)
        at = i if i >= 0 else at
        out.append((c, headings_before(text, at)))
    return out


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Paragraph-bounded pieces of at most `size` characters (a longer paragraph is cut at a line break)."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        while len(para) > size:
            cut = para.rfind("\n", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur); cur = ""
            out.append(para[:cut]); para = para[cut:].lstrip("\n")
        if len(cur) + len(para) + 2 > size and cur:
            out.append(cur); cur = ""
        cur = (cur + "\n\n" + para) if cur else para
    if cur.strip():
        out.append(cur)
    return [c for c in out if c.strip()]


def plan(sections: list[dict], recipes: dict, max_items: int = MAX_ITEMS, epochs: int = 3, max_len: int = 768,
         sec_per_pass: float | None = None, chars_per_statement: int = CHARS_PER_STATEMENT, trim: bool = False,
         window: int = 0, merged: bool = False) -> dict:
    """sections: [{id, type, chars, statements: [{kind, stable, confirmed}] (optional: before extraction)}].
    → {sections: [...], items, tests, llm_calls, passes, minutes, notes}"""
    rows, notes = [], []
    total_items = total_tests = calls = 0
    for s in sections:
        r = recipes[s["id"]]
        sts = [x for x in s.get("statements") or [] if x.get("confirmed", True) and x.get("stable", True)]
        if sts:
            n_st = len(sts)
            per = [len(r["items"].get(x.get("kind", "fact"), r["items"].get("fact", ["recall"]))) for x in sts]
            ext_calls = 0
        else:                                         # before extraction: from the text's size
            n_st = estimate_statements(s.get("text") if s.get("text") is not None else s.get("chars", 0), chars_per_statement)
            avg = sum(len(v) for v in r["items"].values()) / max(1, len(r["items"]))
            per = [round(avg)] * n_st
            ext_calls = max(1, math.ceil(s.get("chars", 0) / window_chars(window))) if s.get("chars") else 0
        n_items = sum(per)
        gen_calls = math.ceil(n_st / STATEMENTS_PER_GENERATE_CALL)
        n_tests = min(PROBES_PER_SECTION, n_st)
        rows.append({"id": s["id"], "type": s["type"], "statements": n_st, "items": n_items, "tests": n_tests,
                     "llm_calls": ext_calls + gen_calls + (1 if n_tests else 0), "_per": per})
        total_items += n_items
        total_tests += n_tests
        calls += rows[-1]["llm_calls"]
    keep, over = 1.0, False
    if max_items and total_items > max_items:
        over = True
        floor = sum(min(n, MIN_ITEMS_PER_STATEMENT) for r in rows for n in r["_per"])
        if not trim:
            notes.append(f"{total_items} items is above the limit you set ({max_items}). Nothing is trimmed: raise or "
                         f"clear the limit, untick statements, or allow trimming (to {max(floor, max_items)} items, every "
                         f"statement keeping at least {MIN_ITEMS_PER_STATEMENT})")
        else:
            if floor > max_items:
                notes.append(f"{floor} items is the least these statements need ({MIN_ITEMS_PER_STATEMENT} each), above "
                             f"the limit of {max_items}: trimming stops there")
                max_items = floor
            asked = total_items
            keep = max_items / total_items
            while keep > 0 and sum(kept(n, keep) for r in rows for n in r["_per"]) > max_items:
                keep = round(keep - 0.01, 4)
            notes.append(f"{asked} items asked for, trimmed to the limit of {max_items} (about {keep:.0%} kept, every "
                         f"statement at least {MIN_ITEMS_PER_STATEMENT}), as allowed")
            for r in rows:
                r["items"] = sum(kept(n, keep) for n in r["_per"])
            total_items = sum(r["items"] for r in rows)
    for r in rows:
        r.pop("_per", None)
    sp = sec_per_pass if sec_per_pass else SEC_PER_PASS_AT_1536 * max(0.25, max_len / 1536)
    if epochs:
        passes = total_items * epochs
    else:                                             # auto (the trainer's rule): per square in a stack, else the whole
        from packsmith.training.train_pack import auto_epochs
        if merged:
            epochs = auto_epochs(total_items) if total_items else 0
            passes = total_items * epochs
        else:
            per = [(r["items"], auto_epochs(r["items"])) for r in rows if r["items"]]
            passes = sum(n * e for n, e in per)
            epochs = max((e for _, e in per), default=0)
        notes.append(f"epochs: auto (about 4,800 texts seen per {'pack' if merged else 'square'}, 2 to 16): up to {epochs}")
    return {"sections": rows, "items": total_items, "tests": total_tests, "llm_calls": calls, "epochs_max": epochs,
            "passes_max": passes, "minutes_max": round(passes * sp / 60, 1), "sec_per_pass": round(sp, 2),
            "keep": round(keep, 3), "max_items": max_items, "over_limit": over and not trim, "notes": notes}


def kept(n: int, keep: float) -> int:
    """Items a statement with n items keeps under the plan's `keep` factor."""
    return n if keep >= 1 else max(min(n, MIN_ITEMS_PER_STATEMENT), round(n * keep))


def trim(items_by_statement: list[list[dict]], keep: float, wanted: list[int] | None = None) -> list[dict]:
    """Apply the plan's cap, statement by statement, to the items generated (in the order generated). wanted: the
    item count the recipe asked for per statement (the cap is computed from it, not from what came back)."""
    out = []
    for i, its in enumerate(items_by_statement):
        n = kept(wanted[i] if wanted else len(its), keep)
        out += its[:n]
    return out
