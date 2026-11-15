"""User packs: a person's preferences, conventions and projects baked into a pack, without miners or playbooks.

    spec      a name, the sections (squares), each with a type, text, uploaded files and/or folder paths, and the
              build mode: "stack" (one small pack per section, combined; one section can be retrained alone) or
              "merged" (one pack trained on everything)
    extract   LLM pass 1: each section → atomic statements (recipes.EXTRACT_PROMPT), checked against the source text
    confirm   the user ticks / edits / adds statements (or create(..., confirm="auto") accepts the grounded stable ones)
    build     LLM pass 2: confirmed stable statements → a few chat items each (recipes.GENERATE_PROMPT), probes for the
              check, the profile card (statements that change often: not trained), datasets, training, the stack
    validate  the probes asked to the base and to the pack, scored by keypoints

Everything is kept under <data>/userpacks/<slug>/ and every LLM reply is cached by its prompt, so a stopped job
resumes and a re-run after a small edit only pays for what changed.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from packsmith.packs import store
from packsmith.userpack import budget, recipes

CODE_EXT = (".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb",
            ".php", ".swift", ".sh", ".sql", ".r", ".lua", ".scala", ".toml", ".yaml", ".yml", ".json")
TEXT_EXT = (".md", ".markdown", ".txt", ".org", ".rst")
DOC_EXT = (".pdf", ".docx", ".epub", ".rtf", ".html", ".htm")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".obsidian", ".trash", "dist", "build", ".idea"}
MAX_SECTION_CHARS = 120_000            # text read per section (folders are sampled up to this)
MODES = ("stack", "merged")


def _llm(messages, max_tokens=4096, temperature=0.3):
    """The generator (LM Studio etc.); tests replace it."""
    from packsmith.llm import llm
    return llm.chat(messages, max_tokens=max_tokens, temperature=temperature)


LLM = _llm


# ---------------------------------------------------------------------- storage
def root() -> Path:
    d = store.data_dir() / "userpacks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pdir(name: str) -> Path:
    return root() / store.slug(name, 60)


def _rj(p: Path, default=None):
    return store.rj(p, default if default is not None else {})


def _wj(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    store.wj(p, obj)


def _sid(title: str, taken: set) -> str:
    base = store.slug(title or "section", 24) or "section"
    s, i = base, 2
    while s in taken:
        s, i = f"{base}-{i}", i + 1
    return s


SID = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def _safe_sid(sid: str) -> str:
    if not SID.match(sid or ""):
        raise ValueError(f"section id {sid!r}: lowercase letters, digits and hyphens only")
    return sid


def _inside(path: Path, root_: Path) -> Path:
    """path, resolved, when it stays inside root_ (else ValueError)."""
    r, p = root_.resolve(), path.resolve()
    if r != p and r not in p.parents:
        raise ValueError(f"{path} is outside {root_}")
    return p


OPTION_DEFAULTS = 2   # 2 (2026-11-13): epochs auto (was 4) with patience 2 (was 1); Polymath packs in gate mode


def new_spec(name: str, sections: list[dict] | None = None, mode: str = "stack", options: dict | None = None) -> dict:
    """Create or replace a user pack's spec. sections: [{type, title?, about?, text?, paths?, stable?}]; missing →
    the starter squares, empty."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not name or not store.slug(name, 60):
        raise ValueError("a user pack needs a name")
    d = pdir(name)
    old = _rj(d / "spec.json")
    secs, taken = [], set()
    for s in sections if sections is not None else [{"type": t} for t in recipes.STARTER]:
        t = s.get("type") or "custom"
        r = recipes.recipe(t, s.get("title", ""), s.get("about", ""))
        sid = _safe_sid(s["id"]) if s.get("id") else _sid(s.get("title") or r["title"], taken)
        if sid in taken:
            raise ValueError(f"section id {sid!r} is used twice")
        taken.add(sid)
        secs.append({"id": sid, "type": t if t in recipes.TYPES else "custom", "title": s.get("title") or r["title"],
                     "about": s.get("about") or r["about"], "text": s.get("text", ""), "paths": list(s.get("paths") or []),
                     "files": [_safe_file(f) for f in (s.get("files") or [])], "enabled": s.get("enabled", True)})
    opts = {"max_items": 0, "trim": False, "chars_per_statement": budget.CHARS_PER_STATEMENT, "window_chars": 0,
            "max_statements": 0, "tune_router": True, "tune_share": 0.5, "tune_general_cap": 0.15, "epochs": 0, "patience": 2, "max_len": 768, "k": 2, "n_layers": 4, "scale": 1,
            "coverage_passes": 1, "polymath_k": 3, "polymath_part_k": 2, "polymath_n_layers": 8,
            "confirm": "review", **(old.get("options") or {}), **(options or {}), "defaults": OPTION_DEFAULTS}
    spec = {"name": name, "slug": d.name, "mode": mode, "sections": secs, "options": opts,
            "created": old.get("created") or time.strftime("%Y-%m-%d %H:%M"), "updated": time.strftime("%Y-%m-%d %H:%M")}
    _wj(d / "spec.json", spec)
    return spec


ALIASES = {"about": "about", "about-me": "about", "me": "about", "profile": "about", "answer-style": "answer_style",
           "answer_style": "answer_style", "how-to-answer-me": "answer_style", "style": "answer_style", "preferences": "answer_style",
           "coding": "coding", "code": "coding", "coding-preferences": "coding", "projects": "projects", "my-projects": "projects"}


def sections_from_folder(folder: str) -> list[dict]:
    """A folder with one file or sub-folder per section, named by its type (about.md, answer_style.md, coding.md or
    coding/, projects/ ...); any other name becomes a custom section of that title. The easy path for agents."""
    root_ = Path(folder).expanduser()
    if not root_.is_dir():
        raise ValueError(f"{folder} is not a folder")
    out = []
    for p in sorted(root_.iterdir()):
        if p.name.startswith("."):
            continue
        stem = p.stem if p.is_file() else p.name
        t = ALIASES.get(store.slug(stem, 40)) or ALIASES.get(stem.lower())
        title = "" if t else stem.replace("_", " ").replace("-", " ").strip().capitalize()
        out.append({"type": t or "custom", "title": title, "about": title, "paths": [str(p)]})
    if not out:
        raise ValueError(f"{folder} is empty")
    return out


def spec(name: str) -> dict:
    s = _rj(pdir(name) / "spec.json")
    if not s:
        raise KeyError(f"no user pack {name!r}")
    return s


def save_spec(s: dict) -> dict:
    s["updated"] = time.strftime("%Y-%m-%d %H:%M")
    _wj(pdir(s["name"]) / "spec.json", s)
    return s


def update_section(name: str, sid: str, **changes) -> dict:
    s = spec(name)
    for sec in s["sections"]:
        if sec["id"] == sid:
            for k in ("title", "about", "text", "paths", "enabled", "type"):
                if k in changes and changes[k] is not None:
                    sec[k] = changes[k]
            return save_spec(s)
    raise KeyError(f"{name}: no section {sid!r}")


def add_section(name: str, stype: str = "custom", title: str = "", about: str = "", text: str = "",
                paths: list | None = None) -> dict:
    s = spec(name)
    r = recipes.recipe(stype, title, about)
    sid = _sid(title or r["title"], {x["id"] for x in s["sections"]})
    s["sections"].append({"id": sid, "type": stype if stype in recipes.TYPES else "custom", "title": title or r["title"],
                          "about": about or r["about"], "text": text, "paths": list(paths or []), "files": [], "enabled": True})
    return save_spec(s)


def remove_section(name: str, sid: str) -> dict:
    s = spec(name)
    s["sections"] = [x for x in s["sections"] if x["id"] != sid]
    st = statements(name)
    st.pop(sid, None)
    _wj(pdir(name) / "statements.json", st)
    return save_spec(s)


def _safe_file(fn: str) -> str:
    """A bare file name: no folders, not hidden, no '..'."""
    f = str(fn or "")
    if not f or f != Path(f).name or "/" in f or "\\" in f or f.startswith(".") or f in (".", ".."):
        raise ValueError(f"file name {fn!r}: a plain name, no folders")
    return f


def add_file(name: str, sid: str, filename: str, data: bytes) -> dict:
    s = spec(name)
    sec = next((x for x in s["sections"] if x["id"] == sid), None)
    if sec is None:
        raise KeyError(f"{name}: no section {sid!r}")
    fn = _safe_file(Path(str(filename or "")).name)
    dest = _inside(pdir(name) / "inputs" / _safe_sid(sid) / fn, pdir(name) / "inputs")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    if fn not in sec["files"]:
        sec["files"].append(fn)
    return save_spec(s)


def list_all() -> list[dict]:
    out = []
    for d in sorted(root().iterdir()):
        s = _rj(d / "spec.json")
        if s:
            b = _rj(d / "build.json")
            out.append({"name": s["name"], "slug": d.name, "mode": s["mode"], "sections": len(s["sections"]),
                        "updated": s.get("updated"), "pack": b.get("pack"), "status": b.get("status", "not built")})
    return out


def delete(name: str) -> None:
    shutil.rmtree(pdir(name), ignore_errors=True)


# ---------------------------------------------------------------------- reading a section
def _read_file(p: Path) -> str:
    low = p.name.lower()
    try:
        if low.endswith(CODE_EXT):
            return f"File {p.name}:\n```\n{p.read_text(encoding='utf-8', errors='replace')}\n```"
        if low.endswith(TEXT_EXT):
            return p.read_text(encoding="utf-8", errors="replace")
        if low.endswith(DOC_EXT) or low.endswith((".csv", ".tsv", ".xlsx", ".jsonl")):
            from packsmith.datasets import curated
            units = curated.parse_file(p.name, p.read_bytes())
            return "\n\n".join((f"{u['subject']}: " if u.get("subject") else "") + (u.get("text") or
                               f"{u.get('question', '')} {u.get('answer', '')}".strip()) for u in units if u)
    except OSError:
        return ""
    return ""


def _walk(folder: Path, limit: int) -> list[Path]:
    """Text, notes and code files of a folder, newest first, until about `limit` characters."""
    files = []
    for p in folder.rglob("*"):
        if any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(folder).parts[:-1]):
            continue
        if p.is_file() and p.name.lower().endswith(TEXT_EXT + CODE_EXT + DOC_EXT) and p.stat().st_size < 2_000_000:
            files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out, total = [], 0
    for p in files:
        if total > limit:
            break
        out.append(p)
        total += p.stat().st_size
    return out


def section_text(name: str, sec: dict) -> str:
    """Everything a section holds, as one text: the box, the uploaded files, then the paths (files or folders)."""
    parts = [sec.get("text") or ""]
    for fn in sec.get("files") or []:
        try:
            f = _inside(pdir(name) / "inputs" / _safe_sid(sec["id"]) / _safe_file(fn), pdir(name) / "inputs")
        except ValueError:
            continue                                        # a hand-edited spec naming a file outside: skipped
        parts.append(_read_file(f))
    for raw in sec.get("paths") or []:
        p = Path(raw).expanduser()
        if p.is_file():
            parts.append(_read_file(p))
        elif p.is_dir():
            for f in _walk(p, MAX_SECTION_CHARS):
                t = _read_file(f)
                if t:
                    parts.append(f"[{f.relative_to(p)}]\n{t}")
    text = "\n\n".join(x.strip() for x in parts if x and x.strip())
    return text[:MAX_SECTION_CHARS]


# ---------------------------------------------------------------------- LLM calls (cached)
def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def _json_from(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except ValueError:
            return None
    return None


def ask_json(name: str, prompt: str, key: str, max_tokens: int = 6000, temperature: float = 0.3) -> list:
    """One cached LLM call whose reply is JSON with a list under `key`; one retry when the reply does not parse."""
    cache = pdir(name) / "cache" / f"{_h(prompt)}.json"
    hit = _rj(cache)
    if hit and isinstance(hit.get(key), list):
        return hit[key]
    msgs = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        reply = LLM(msgs, max_tokens=max_tokens, temperature=temperature)
        j = _json_from(reply or "")
        if j and isinstance(j.get(key), list):
            _wj(cache, j)
            return j[key]
        msgs = msgs + [{"role": "assistant", "content": reply or ""},
                       {"role": "user", "content": f'That was not valid JSON. Reply with JSON only: {{"{key}": [...]}}'}]
    return []


# ---------------------------------------------------------------------- 1. extract
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
STOP = set("that this with from have their they them what when where which while would could should about into than "
           "then there these those user users prefers prefer wants likes always never every each other only also very "
           "more most much such using uses used make makes like does done being been were will just".split())


def grounded(statement: str, source: str, example: str = "") -> float:
    """Share of the statement's content words found in the source (and the example, if any, must be in it)."""
    words = {w.lower() for w in WORD.findall(statement)} - STOP
    if example and len(example) > 12 and re.sub(r"\s+", " ", example).strip()[:60] not in re.sub(r"\s+", " ", source):
        return 0.0
    if not words:
        return 1.0
    low = source.lower()
    return sum(1 for w in words if w in low) / len(words)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _similar(a: str, b: str) -> float:
    x, y = set(_norm(a).split()), set(_norm(b).split())
    return len(x & y) / max(1, len(x | y))


def statements(name: str) -> dict:
    return _rj(pdir(name) / "statements.json")


# ---------------------------------------------------------------------- coverage and subject
_ITEM_LINE = re.compile(r"^\s*(?:[-*+\u2022|]|\d+[.)])\s*\S")


def units(text: str) -> list[str]:
    """The text's passages for the coverage check: paragraphs, with list items and table rows as passages of their
    own; headings and passages of fewer than 6 content words are left out (they say nothing on their own)."""
    out = []
    for para in re.split(r"\n\s*\n", _FENCE_RE.sub(" ", text)):
        lines = [l for l in para.splitlines() if l.strip()]
        if not lines:
            continue
        if sum(1 for l in lines if _ITEM_LINE.match(l)) >= max(2, len(lines) // 2):
            out += lines                                  # a list or a table: each row a passage
        else:
            out.append(" ".join(lines))
    keep = []
    for u in out:
        if re.match(r"^\s{0,3}#{1,6}\s", u) or re.fullmatch(r"\s*[|:\-\s]+\s*", u):
            continue
        if len({w.lower() for w in WORD.findall(u)} - STOP) >= 6:
            keep.append(u.strip())
    return keep


_FENCE_RE = re.compile(r"```.*?```", re.S)


def covered(unit: str, sts: list[dict]) -> bool:
    """A passage is covered when a statement came from it (its first words are in it) or the statements grounded in
    it hold at least a third of its content words."""
    words = {w.lower() for w in WORD.findall(unit)} - STOP
    if not words:
        return True
    nu, hit = _norm(unit), set()
    for y in sts:
        src = " ".join(str(y.get("source") or "").split()[:6])
        if src and len(src) > 12 and _norm(src) in nu:
            return True
        if grounded(y["text"], unit) >= 0.6:
            hit |= {w.lower() for w in WORD.findall(y["text"])} - STOP
    return len(words & hit) / len(words) >= 0.34


def coverage(text: str, sts: list[dict]) -> tuple[float, list[str]]:
    """(share of the text's passages covered by the statements, the passages left uncovered)."""
    us = units(text)
    gaps = [u for u in us if not covered(u, sts)]
    return (1 - len(gaps) / len(us)) if us else 1.0, gaps


def set_subject(name: str, person: str = "", pronoun: str = "") -> dict:
    """The user's own answer to who the pack is about (kept: the subject pass never replaces it). Empty name: no
    third-person items."""
    s = spec(name)
    s["options"]["subject"] = {"name": person.strip()[:80], "aliases": [], "pronoun": (pronoun or "they").strip()[:12],
                               "first_person": not person.strip(), "manual": True}
    save_spec(s)
    return s["options"]["subject"]


def subject(name: str, force: bool = False) -> dict:
    """Who the user pack is about (recipes.SUBJECT_PROMPT, on the beginning of the first sections): {name, aliases,
    pronoun, first_person}. Kept in spec.options.subject; a subject the user set (subject.manual) is never replaced."""
    s = spec(name)
    cur = (s.get("options") or {}).get("subject") or {}
    if cur.get("manual") or (cur and not force):
        return cur
    sample = ""
    for sec in sorted(s["sections"], key=lambda x: x["type"] != "about"):
        if sec.get("enabled", True):
            sample += section_text(name, sec)[:2500] + "\n\n"
        if len(sample) > 4000:
            break
    out = {"name": "", "aliases": [], "pronoun": "they", "first_person": True}
    if sample.strip():
        got = ask_json(name, recipes.SUBJECT_PROMPT.format(text=sample[:4500]), "subjects", max_tokens=800, temperature=0.0)
        if got and isinstance(got[0], dict):
            g = got[0]
            nm = str(g.get("name") or "").strip()
            if nm and nm.lower() in sample.lower():           # a name the text really gives
                out["name"] = nm[:80]
                out["aliases"] = [str(a)[:80] for a in (g.get("aliases") or []) if str(a).strip() and str(a).lower() in sample.lower()][:5]
            out["pronoun"] = str(g.get("pronoun") or "they")[:12]
            out["first_person"] = bool(g.get("first_person", True))
    s["options"]["subject"] = out
    save_spec(s)
    return out


EXTRACT_VERSION = 2          # changing the extraction (prompts, windows, coverage) re-extracts unchanged sections


def extract(name: str, sections: list[str] | None = None, progress=lambda f, m="": None,
            cancelled=lambda: False) -> dict:
    """Statements for every enabled section (or only `sections`). A section whose text has not changed keeps its
    statements and the user's ticks."""
    s = spec(name)
    st = statements(name)
    todo = [x for x in s["sections"] if x.get("enabled", True) and (not sections or x["id"] in sections)]
    report = {}
    subject(name)                                          # who the pack is about (once; the user may correct it)
    s = spec(name)
    seen_units: dict[str, str] = {}                        # a passage -> the first section that holds it
    for sec in s["sections"]:                              # the sections before this run's: their passages and statements
        if sec in todo or not sec.get("enabled", True):
            continue
        for u in units(section_text(name, sec)):
            seen_units.setdefault(_norm(u)[:200], sec["title"])
    for i, sec in enumerate(todo):
        if cancelled():
            break
        text = section_text(name, sec)
        h = _h(text + sec["type"] + sec["about"] + str(EXTRACT_VERSION))
        old = st.get(sec["id"]) or {}
        mine = units(text)
        dup_of = {}
        for u in mine:
            k = _norm(u)[:200]
            if k in seen_units:
                dup_of[seen_units[k]] = dup_of.get(seen_units[k], 0) + 1
            else:
                seen_units[k] = sec["title"]
        if old.get("source_hash") == h and old.get("items"):
            report[sec["id"]] = {"statements": len(old["items"]), "unchanged": True, "coverage": old.get("coverage")}
            continue
        if not text.strip():
            st[sec["id"]] = {"source_hash": h, "chars": 0, "items": []}
            report[sec["id"]] = {"statements": 0, "empty": True}
            continue
        r = recipes.recipe(sec["type"], sec["title"], sec["about"])
        o = s["options"]
        cap = int(o.get("max_statements") or 0)              # 0 = no limit
        target = int(o.get("chars_per_statement") or budget.CHARS_PER_STATEMENT)
        found: list[dict] = []
        earlier = [y for sid, v in st.items() if sid != sec["id"] and sid not in {x["id"] for x in todo[i:]}
                   for y in v.get("items", [])]                  # statements other sections already hold
        pieces = budget.chunks_with_context(text, budget.window_chars(o.get("window_chars", 0)))
        passes = [(c, ctx, "") for c, ctx in pieces]
        gap_rounds = int(o.get("coverage_passes", 1))
        cov0 = None
        j = 0
        while passes:
            chunk, ctx, gap = passes.pop(0)
            j += 1
            progress((i + min(0.95, j / (len(pieces) + 1))) / max(1, len(todo)),
                     f"extracting {sec['title']} ({j}/{len(pieces)}{', coverage pass' if gap else ''})")
            limit = f"\n- At most {cap - len(found)} statements: keep the most important." if cap else ""
            prompt = recipes.EXTRACT_PROMPT.format(title=r["title"], about=r["about"], focus=r["focus"], target=target,
                                                   limit=limit, gap=gap, text=chunk,
                                                   context=f"WHERE THIS TEXT SITS: {ctx}\n\n" if ctx else "")
            reply = max(6000, int(len(chunk) / max(60, target) * 120))   # ~120 tokens per statement in the JSON reply
            for x in ask_json(name, prompt, "statements", max_tokens=reply):
                if not isinstance(x, dict) or not str(x.get("text", "")).strip():
                    continue
                t = str(x["text"]).strip()
                if any(_similar(t, y["text"]) > 0.8 for y in found) or any(_similar(t, y["text"]) > 0.8 for y in earlier):
                    continue                              # said already, here or in another section
                kind = x.get("kind") if x.get("kind") in ("preference", "convention", "fact", "dont") else "fact"
                g = grounded(t, chunk, str(x.get("example") or ""))
                src = " ".join(str(x.get("source") or "").split()[:6])
                found.append({"id": _h(sec["id"] + t)[:10], "text": t, "kind": kind, "source": str(x.get("source") or "")[:200],
                              "source_found": (_norm(src) in _norm(chunk)) if src else None,   # None: not given
                              "stable": bool(x.get("stable", True)), "example": str(x.get("example") or "")[:1500],
                              "grounded": round(g, 2), "confirmed": None, "manual": False})
            if cap and len(found) >= cap:
                break
            if not passes and gap_rounds > 0 and not (cap and len(found) >= cap):   # the coverage pass
                gap_rounds -= 1
                cov0, gaps = coverage(text, found + earlier)
                if gaps:
                    for c, _ in budget.chunks_with_context("\n\n".join(gaps), budget.window_chars(o.get("window_chars", 0))):
                        passes.append((c, "", recipes.GAP_NOTE))
        cov, gaps = coverage(text, found + earlier)
        keep = {y["text"]: y for y in old.get("items", []) if y.get("manual")}   # the user's own statements stay
        prior = {y["id"]: y for y in old.get("items", [])}
        for y in found:                                   # a tick survives re-extraction of the same statement
            if y["id"] in prior and prior[y["id"]].get("confirmed") is not None:
                y["confirmed"] = prior[y["id"]]["confirmed"]
        st[sec["id"]] = {"source_hash": h, "chars": len(text), "items": (found[:cap] if cap else found) + list(keep.values()),
                         "coverage": round(cov, 3), "uncovered": gaps[:40]}
        report[sec["id"]] = {"statements": len(st[sec["id"]]["items"]), "chars": len(text),
                             "ungrounded": sum(1 for y in found if y["grounded"] < 0.5),
                             "coverage": round(cov, 3), "coverage_first_pass": round(cov0, 3) if cov0 is not None else None,
                             "uncovered": len(gaps)}
        if dup_of:                                         # the same passages in another section
            top, n = max(dup_of.items(), key=lambda x: x[1])
            report[sec["id"]]["same_text_as"] = {"section": top, "passages": n, "of": len(mine)}
            st[sec["id"]]["same_text_as"] = report[sec["id"]]["same_text_as"]
    _wj(pdir(name) / "statements.json", st)
    _status(name, "extracted", extract=report)
    return {"sections": report, "statements": st}


def auto_confirm(name: str, min_grounded: float = 0.5) -> dict:
    """Accept every statement not yet ticked that is grounded in the source (the API's confirm='auto')."""
    st = statements(name)
    n = 0
    for sec in st.values():
        for y in sec.get("items", []):
            if y.get("confirmed") is None:              # found in the text: its words, and its first words verbatim
                y["confirmed"] = bool(y.get("manual") or (y.get("grounded", 1) >= min_grounded and y.get("source_found") is not False))
                n += y["confirmed"]
    _wj(pdir(name) / "statements.json", st)
    return {"confirmed": n}


def edit_statements(name: str, edits: list[dict]) -> dict:
    """edits: [{section, id, confirmed?, text?, kind?, stable?}] or a new one {section, text, kind?, stable?} (no id)."""
    st = statements(name)
    for e in edits:
        sec = st.setdefault(e["section"], {"source_hash": "", "chars": 0, "items": []})
        if not e.get("id"):
            t = str(e.get("text", "")).strip()
            if t:
                sec["items"].append({"id": _h(e["section"] + t)[:10], "text": t, "kind": e.get("kind", "preference"),
                                     "stable": bool(e.get("stable", True)), "example": e.get("example", ""), "grounded": 1.0,
                                     "confirmed": True, "manual": True})
            continue
        for y in sec["items"]:
            if y["id"] == e["id"]:
                for k in ("confirmed", "text", "kind", "stable", "example"):
                    if k in e:
                        y[k] = e[k]
                if "text" in e:
                    y["manual"] = True
    _wj(pdir(name) / "statements.json", st)
    return st


def plan(name: str) -> dict:
    s = spec(name)
    st = statements(name)
    named = bool(((s["options"].get("subject") or {}).get("name") or "").strip())
    rec = {x["id"]: recipes.recipe(x["type"], x["title"], x["about"], s["options"].get("scale", 1), named) for x in s["sections"]}
    rows = []
    for sec in s["sections"]:
        if not sec.get("enabled", True):
            continue
        items = (st.get(sec["id"]) or {}).get("items")
        text = None if items is not None else section_text(name, sec)
        rows.append({"id": sec["id"], "type": sec["type"], "statements": items if items is not None else None,
                     "chars": (st.get(sec["id"]) or {}).get("chars") or len(text or ""), "text": text})
    o = s["options"]
    ep = o.get("epochs", 0)
    if int(o.get("defaults") or 1) < 2 and ep == 4 and o.get("patience") == 1:
        ep = 0                                         # the old defaults, untouched: trained with auto (see train())
    p = budget.plan(rows, rec, int(o.get("max_items") or 0), int(ep or 0), o.get("max_len", 768),
                    o.get("sec_per_pass"), int(o.get("chars_per_statement") or budget.CHARS_PER_STATEMENT),
                    bool(o.get("trim")), int(o.get("window_chars") or 0), merged=s["mode"] == "merged")
    p["mode"] = s["mode"]
    p["volatile"] = sum(1 for sec in st.values() for y in sec.get("items", []) if y.get("confirmed") and not y.get("stable"))
    if s["mode"] == "stack":
        n = sum(1 for r in p["sections"] if r["items"])
        p["notes"].append(f"stack: {n} small packs, trained one after another (each loads the base model once)")
    return p


# ---------------------------------------------------------------------- 2. build: items, probes, profile, datasets
LEAK = re.compile(r"\b(as you (prefer|like|asked|requested)|your preference|you prefer|per your (style|preference)|"
                  r"the user (prefers|wants|likes))\b", re.I)


def profile_summary(name: str, limit: int = 7000) -> str:
    """Every confirmed stable statement of the pack, one per line (the generate prompt's PROFILE), cut at `limit`."""
    s, st = spec(name), statements(name)
    lines = []
    for sec in s["sections"]:
        for y in (st.get(sec["id"]) or {}).get("items", []):
            if y.get("confirmed") and y.get("stable", True):
                lines.append(f"- {y['text']}")
    out = "\n".join(lines)
    return out if len(out) <= limit else out[:limit].rsplit("\n", 1)[0] + "\n- …"


def _items_for(name: str, sec: dict, sts: list[dict], progress, cancelled, f0: float, f1: float,
               scale: int = 1) -> list[list[dict]]:
    subj = (spec(name)["options"].get("subject") or {})
    r = recipes.recipe(sec["type"], sec["title"], sec["about"], scale, bool((subj.get("name") or "").strip()))
    prof = profile_summary(name)
    out: list[list[dict]] = [[] for _ in sts]
    groups = [list(range(i, min(i + budget.STATEMENTS_PER_GENERATE_CALL, len(sts))))
              for i in range(0, len(sts), budget.STATEMENTS_PER_GENERATE_CALL)]
    for gi, g in enumerate(groups):
        if cancelled():
            break
        progress(f0 + (f1 - f0) * gi / max(1, len(groups)), f"writing items for {sec['title']} ({gi + 1}/{len(groups)})")
        lines = []
        for n, idx in enumerate(g, 1):
            y = sts[idx]
            want = r["items"].get(y["kind"]) or r["items"].get("fact") or ["situated"]
            lines.append(f"{n}. [{y['kind']}] {y['text']}" + (f"\n   example: {y['example']}" if y.get("example") else "")
                         + f"\n   items: {', '.join(want)}")
        prompt = recipes.GENERATE_PROMPT.format(title=r["title"], style=r["style"], kinds=recipes.kinds_text(r),
                                                person=recipes.person_text(subj), profile=prof,
                                                statements="\n".join(lines))
        for it in ask_json(name, prompt, "items", max_tokens=12000, temperature=0.6):
            try:
                n = int(it.get("statement")) - 1
            except (TypeError, ValueError):
                continue
            req, ans = str(it.get("request", "")).strip(), str(it.get("answer", "")).strip()
            if not (0 <= n < len(g)) or not req or not ans or len(ans) > 6000 or LEAK.search(req) or LEAK.search(ans):
                continue
            mine = out[g[n]]
            if any(_similar(req, x["request"]) > 0.85 for x in mine):
                continue
            want = r["items"].get(sts[g[n]]["kind"]) or ["situated"]
            if len(mine) < len(want):                       # no more than the recipe asks for
                mine.append({"request": req, "answer": ans, "type": str(it.get("type", "")), "statement": sts[g[n]]["id"],
                             "section": sec["id"], "kind": sts[g[n]]["kind"]})
    return out


def _probes(name: str, sec: dict, sts: list[dict]) -> list[dict]:
    n = min(budget.PROBES_PER_SECTION, len(sts))           # spread over the section, not its first statements
    pick = [sts[round(i * (len(sts) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 else sts[:n]
    if not pick:
        return []
    subj = (spec(name)["options"].get("subject") or {})
    prompt = recipes.PROBE_PROMPT.format(person=recipes.person_text(subj, probe=True),
                                         statements="\n".join(f"{i}. {y['text']}" for i, y in enumerate(pick, 1)))
    out = []
    for p in ask_json(name, prompt, "probes", max_tokens=3000):
        try:
            y = pick[int(p.get("statement")) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        q = str(p.get("question", "")).strip()
        if q:
            out.append({"section": sec["id"], "statement": y["id"], "statement_text": y["text"], "question": q,
                        "keypoints": [str(k).lower() for k in (p.get("keypoints") or []) if str(k).strip()][:3],
                        "against": [str(k).lower() for k in (p.get("against") or []) if str(k).strip()][:2]})
    return out


def text_of(item: dict) -> str:
    return f"Question: {item['request']}\nAnswer: {item['answer']}"


def profile(name: str) -> str:
    """The profile card: confirmed statements that change often (never trained), for an agent's system prompt."""
    s = spec(name)
    st = statements(name)
    lines = [f"# {s['name']}: current facts (not in the pack's weights; refresh when they change)"]
    for sec in s["sections"]:
        vol = [y["text"] for y in (st.get(sec["id"]) or {}).get("items", []) if y.get("confirmed") and not y.get("stable")]
        if vol:
            lines.append(f"\n## {sec['title']}")
            lines += [f"- {t}" for t in vol]
    return "\n".join(lines) + "\n"


def generate(name: str, sections: list[str] | None = None, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Items, probes and datasets for the confirmed stable statements. → {datasets: {section|'all': did}, ...}"""
    from packsmith.datasets import datasets
    s = spec(name)
    st = statements(name)
    p = plan(name)
    if p.get("over_limit"):
        raise ValueError(p["notes"][0])
    keep = p["keep"]
    secs = [x for x in s["sections"] if x.get("enabled", True) and (not sections or x["id"] in sections)]
    per_sec, probes, rep = {}, [], {}
    for i, sec in enumerate(secs):
        sts = [y for y in (st.get(sec["id"]) or {}).get("items", []) if y.get("confirmed") and y.get("stable", True)]
        if not sts:
            rep[sec["id"]] = {"statements": 0, "items": 0}
            continue
        f0, f1 = i / len(secs), (i + 0.9) / len(secs)
        by = _items_for(name, sec, sts, progress, cancelled, f0, f1, s["options"].get("scale", 1))
        r = recipes.recipe(sec["type"], sec["title"], sec["about"], s["options"].get("scale", 1),
                           bool(((s["options"].get("subject") or {}).get("name") or "").strip()))
        wanted = [len(r["items"].get(y["kind"]) or r["items"].get("fact") or ["situated"]) for y in sts]
        items = budget.trim(by, keep, wanted)
        room = 3 * int(s["options"].get("max_len", 768))     # ~4 characters a token: the request must leave room
        long_ = [x for x in items if len(x["request"]) > room]
        items = [x for x in items if len(x["request"]) <= room]
        pr = _probes(name, sec, sts)
        tests = []                                      # held-out: one item of each probed statement, for the epoch check
        probed = {x["statement"] for x in pr}
        train_items = []
        for it in items:
            if it["statement"] in probed and not any(t["statement"] == it["statement"] for t in tests) and \
                    sum(1 for x in items if x["statement"] == it["statement"]) > 1:
                tests.append(it)
            else:
                train_items.append(it)
        per_sec[sec["id"]] = (train_items, tests)
        probes += pr
        rep[sec["id"]] = {"statements": len(sts), "items": len(train_items), "held_out": len(tests), "probes": len(pr),
                          "missing": sum(1 for b in by if not b), "too_long": len(long_)}
    if cancelled():
        return {"cancelled": True}
    d = pdir(name)
    _wj(d / "probes.json", probes)
    (d / "profile.md").write_text(profile(name), encoding="utf-8")
    out = {}
    groups = {sid: v for sid, v in per_sec.items()} if s["mode"] == "stack" else \
        {"all": ([x for v in per_sec.values() for x in v[0]], [x for v in per_sec.values() for x in v[1]])}
    for gid, (tr, te) in groups.items():
        if not tr:
            continue
        m = datasets.create(f"{s['name']} · {gid}", source="userpack", userpack=s["slug"], section=gid)
        datasets.write_base(m["id"], [{"text": text_of(x), "kind": "qa", "section": x["section"], "statement": x["statement"],
                                       "item_type": x["type"]} for x in tr],
                            [{"question": x["request"], "expected": x["answer"], "section": x["section"]} for x in te],
                            {"userpack": s["slug"], "section": gid, "items": len(tr)})
        out[gid] = m["id"]
        (d / "items").mkdir(exist_ok=True)
        _wj(d / "items" / f"{gid}.json", {"train": tr, "tests": te})
    _status(name, "generated", generate=rep, datasets=out)
    return {"datasets": out, "sections": rep, "probes": len(probes), "plan": p}


# ---------------------------------------------------------------------- 3. train
def part_name(s: dict, gid: str) -> str:
    return f"{s['slug']}.{gid}"[:60]


def train(name: str, sections: list[str] | None = None, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Train the generated datasets: one pack (merged) or one per section, then the stack. sections: retrain only
    these parts of a stack (the other parts are kept)."""
    from packsmith import engines
    from packsmith.system.config import SETTINGS
    from packsmith.training import train as T
    s = spec(name)
    b = _rj(pdir(name) / "build.json")
    dsets = b.get("datasets") or {}
    if not dsets:
        raise ValueError("generate the datasets first")
    o = dict(s["options"])
    if int(o.get("defaults") or 1) < 2 and o.get("epochs") == 4 and o.get("patience") == 1:
        o.update(epochs=0, patience=2)                 # a spec made with the old defaults, untouched: the new ones
    opts = {"epochs": o.get("epochs", 0), "patience": o.get("patience", 2), "max_len": o.get("max_len", 768),
            "k": o.get("k", 2), "n_layers": o.get("n_layers", 4), "loss": "answer", "chat_format": "chat"}
    if s["mode"] == "stack":                            # smaller parts: one expert per layer each
        opts["k"] = o.get("part_k", 1)
    if engines.is_polymath(SETTINGS.engine):
        # A Polymath expert acts only where it wins one of the model's places (at most `budget` per token and
        # layer), and then with about the weight of the expert it displaced: a part of 1 expert in 4 of 48 layers
        # was too small to change an answer. More experts, more layers, a higher share of its own tokens to aim for
        # and a lighter push off general text (the KL anchor still keeps general answers the base's).
        opts.update(k=o.get("polymath_part_k", 2) if s["mode"] == "stack" else o.get("polymath_k", 3),
                    n_layers=o.get("polymath_n_layers", 8), share=o.get("polymath_share", 0.4),
                    route=o.get("polymath_route", 0.3), mode=o.get("polymath_mode", "gate"))
    parts = dict(b.get("parts") or {}) if s["mode"] == "stack" and b.get("mode") == "stack" else {}
    parts = {g: v for g, v in parts.items() if g in dsets}   # squares removed since are dropped
    todo = [g for g in dsets if s["mode"] != "stack" or not sections or g in sections or g not in parts]
    for i, gid in enumerate(todo):
        if cancelled():
            break
        pn = s["slug"] if s["mode"] == "merged" else part_name(s, gid)
        progress(i / max(1, len(todo)), f"training {pn}")
        r = T.run(dsets[gid], pn, "dataset", {**opts, **(o.get("train_opts") or {})},
                  unload_lmstudio=bool(getattr(SETTINGS, "llm_unload_for_gpu", True)),
                  progress=lambda f, m="", i=i: progress((i + f) / max(1, len(todo)), m), cancelled=cancelled)
        path = r["path"]
        if s["mode"] == "stack":                        # a part lives inside the user pack, not in the library:
            dst = pdir(name) / "parts" / gid            # the library shows one pack, the stack
            if dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, dst)
            path = str(dst)
        parts[gid] = {"pack": r["pack"], "path": path, "epochs": len((r.get("card") or {}).get("epochs") or []),
                      "best_epoch": (r.get("card") or {}).get("best_epoch")}
    out = {"parts": parts}
    if s["mode"] == "stack" and parts:
        _, P = engines.get(SETTINGS.engine)
        dest = store.packs_dir() / s["slug"]
        if dest.exists():
            shutil.rmtree(dest)
        if engines.is_polymath(SETTINGS.engine):       # a stack folder that carries its parts: copy or share it whole
            (dest / "parts").mkdir(parents=True)
            for gid, v in parts.items():
                shutil.copytree(v["path"], dest / "parts" / gid)
            _wj(dest / "pack.json", {"format": "polymath-stack-v1", "packs": [f"parts/{g}" for g in parts],
                                     "strengths": [1.0] * len(parts), "engine": {"name": "polymath-engine"},
                                     "source": f"userpack {s['slug']}"})
        else:                                           # patched colibri: the parts merged into one B-slot pack
            P.merge([Path(v["path"]) for v in parts.values()], dest, source=f"userpack {s['slug']}")
        card = {"name": s["slug"], "kind": "userpack", "personal": True, "mode": "stack", "parts": list(parts),
                "engine": SETTINGS.engine}
        _wj(dest / "card.json", card)
        out["pack"], out["path"] = s["slug"], str(dest)
        if (o.get("tune_router", True) and len(parts) > 1 and engines.is_polymath(SETTINGS.engine)
                and opts.get("mode") == "route"):           # gate packs have no router rows to tune
            progress(0.97, "tuning the stack's router rows (calibrate)")
            out["tuned"] = tune_stack(name, dest, list(parts), o, progress)
            card["router_tuned"] = out["tuned"].get("ok", False)
            _wj(dest / "card.json", card)
    elif parts:
        v = next(iter(parts.values()))
        out["pack"], out["path"] = v["pack"], v["path"]
        c = _rj(Path(v["path"]) / "card.json")
        c.update(kind="userpack", personal=True, mode="merged")
        _wj(Path(v["path"]) / "card.json", c)
    if out.get("path"):
        (Path(out["path"]) / "profile.md").write_text(profile(name), encoding="utf-8")
    _status(name, "built", mode=s["mode"], **out)
    return out


def _trained_form(text: str) -> str:
    """An item as the pack was trained on it (user packs train in the chat format on Qwen models): the router rows
    must be calibrated on the hidden states of that text, not of 'Question: / Answer:'."""
    from packsmith.system.config import SETTINGS
    from packsmith.training.train_pack import chatml, is_qwen
    m = str(SETTINGS.base_train_model or "")
    try:
        qwen = m in ("tiny", "tiny-qwen") or is_qwen(m)
    except Exception:                                    # noqa: BLE001
        qwen = False
    return (chatml(text) or text) if qwen else text


def tune_stack(name: str, stack: Path, order: list[str], o: dict, progress=lambda f, m="": None) -> dict:
    """Router tuning (calibrate) of a freshly built stack: each square's pack router rows are set to take about
    tune_share of that square's own items and at most tune_general_cap of general text. Only the packs' router rows
    change, saved in the stack folder; the base model, its router and the packs' experts do not. Runs the Polymath
    Engine's tuner in the repo's venv (the same Python), on the GPU the training used."""
    import subprocess
    import sys
    from packsmith.system.config import SETTINGS
    from packsmith.workbench import workbench
    d = pdir(name) / "tune"
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for gid in order:
        it = _rj(pdir(name) / "items" / f"{gid}.json")
        f = d / f"{gid}.jsonl"
        f.write_text("\n".join(json.dumps({"text": _trained_form(text_of(x))}) for x in it.get("train", [])),
                     encoding="utf-8")
        files.append(str(f))
    was = workbench.STATE.get("engine") is not None
    if was:
        workbench.stop()                              # one GPU
    try:
        dev = (o.get("train_opts") or {}).get("device") or "auto"
        cmd = [sys.executable, "-m", "polymath_engine", "tune-router", "--model", str(SETTINGS.path("base_train_model"))
               if SETTINGS.base_train_model not in ("tiny", "tiny-qwen") else SETTINGS.base_train_model,
               "--device", dev, "--stack", str(stack), "--method", "calibrate", "--data", ",".join(files),
               "--share", str(o.get("tune_share", 0.5)), "--general-cap", str(o.get("tune_general_cap", 0.15)),
               "--budget", str(int(SETTINGS.pack_budget))]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        for line in r.stderr.splitlines():
            if line.startswith("[tune]"):
                progress(0.98, line[:300])
        if r.returncode != 0:
            return {"ok": False, "error": (r.stderr or r.stdout)[-800:]}
        rep_ = _rj(Path(stack) / "router_tune.json")
        return {"ok": True, "before": rep_.get("before"), "after": rep_.get("after"), "weak": rep_.get("weak") or [],
                "note": rep_.get("note", "")}
    finally:
        if was:
            workbench.start()


def build(name: str, sections: list[str] | None = None, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """generate + train (sections: only those parts, for a stack)."""
    g = generate(name, sections, lambda f, m="": progress(0.4 * f, m), cancelled)
    if g.get("cancelled"):
        return g
    t = train(name, sections, lambda f, m="": progress(0.4 + 0.6 * f, m), cancelled)
    return {"generate": {k: v for k, v in g.items() if k != "plan"}, "plan": g["plan"], **t}


def create(name: str, sections: list[dict] | str | None = None, mode: str = "stack", options: dict | None = None,
           confirm: str = "review", progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """The one-shot path. confirm='review': stop after extraction (the user ticks the statements, then calls build);
    confirm='auto': accept the grounded statements and build the pack in the same job."""
    if isinstance(sections, str):                       # a folder: one file / sub-folder per section
        sections = sections_from_folder(sections)
    if sections is not None:
        new_spec(name, sections, mode, options)
    elif not (pdir(name) / "spec.json").exists():
        raise KeyError(f"no user pack {name!r}: give sections")
    ex = extract(name, progress=lambda f, m="": progress(0.15 * f, m), cancelled=cancelled)
    if confirm != "auto":
        _status(name, "awaiting_confirmation")
        return {"status": "awaiting_confirmation", "extract": ex["sections"], "plan": plan(name)}
    auto_confirm(name)
    r = build(name, progress=lambda f, m="": progress(0.15 + 0.85 * f, m), cancelled=cancelled)
    return {"status": "built", "extract": ex["sections"], **r}


def _status(name: str, status: str, **kw) -> None:
    p = pdir(name) / "build.json"
    b = _rj(p)
    b.update(status=status, updated=time.strftime("%Y-%m-%d %H:%M"), **kw)
    _wj(p, b)


def status(name: str) -> dict:
    s = spec(name)
    st = statements(name)
    b = _rj(pdir(name) / "build.json")
    counts = {sid: {"statements": len(v.get("items", [])),
                    "confirmed": sum(1 for y in v.get("items", []) if y.get("confirmed")),
                    "pending": sum(1 for y in v.get("items", []) if y.get("confirmed") is None)} for sid, v in st.items()}
    return {"spec": s, "statements": counts, "build": b}


# ---------------------------------------------------------------------- 4. validate
def score(answer: str, probe: dict) -> int:
    """1 when the answer has at least one keypoint and none of the 'against' phrases, else 0."""
    a = answer.lower()
    return int(any(k in a for k in probe.get("keypoints") or []) and not any(k in a for k in probe.get("against") or []))


CURVE = [0.25, 0.5, 0.75, 1.0, 1.25]


def validate(name: str, max_tokens: int = 700, progress=lambda f, m="": None, cancelled=lambda: False,
             curve: list[float] | None = None) -> dict:
    """The probes asked to the base and to the pack in one shuffled order (no system prompt), scored by keypoints.
    The pack runs with its Workbench dials. curve (e.g. CURVE): the pack is also asked at each of these strengths,
    which shows where it starts to matter; the result's "curve" lists the score per strength."""
    import random
    from packsmith.workbench import workbench
    b = _rj(pdir(name) / "build.json")
    pack = b.get("path")
    probes = _rj(pdir(name) / "probes.json", [])
    if not pack or not probes:
        raise ValueError("build the pack first")
    st = workbench.start()
    if st["status"] != "ready":
        raise RuntimeError(f"engine did not start: {st.get('error')}")
    e = workbench.STATE["engine"]
    before = workbench.current()                         # put back exactly, with its dials
    qwen = getattr(e, "engine", "") == "qwen36"
    curve = sorted({round(float(s), 3) for s in (curve or []) if 0 < float(s) <= 1.5})
    conds = ["base", "pack"] + [f"x{s:g}" for s in curve]
    sched = [(c, i) for c in conds for i in range(len(probes))]
    random.Random(7).shuffle(sched)
    res, active = {}, "__"
    for n, (c, i) in enumerate(sched):
        if cancelled():
            break
        if active != c:
            r = (workbench.swap_at(e, pack, float(c[1:])) if c.startswith("x")
                 else e.swap(pack if c == "pack" else None))
            if not r["ok"]:
                raise RuntimeError("the engine refused the pack (see its log)")
            active = c
        q = probes[i]["question"]
        prompt = (f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" if qwen
                  else f"Question: {q}\nAnswer:")
        raw = workbench.engine_ask(e, prompt, max_tokens, 0.3, 0.9)
        ans, _ = workbench.split_answer(raw)
        res[(c, i)] = ans
        progress(n / len(sched), f"{c} {i + 1}/{len(probes)}")
    workbench.restore(before)
    rows, tot = [], {"base": 0, "pack": 0}
    for i, p in enumerate(probes):
        if ("base", i) not in res or ("pack", i) not in res:
            continue
        sb, sp = score(res[("base", i)], p), score(res[("pack", i)], p)
        tot["base"] += sb
        tot["pack"] += sp
        rows.append({**p, "base": res[("base", i)], "pack": res[("pack", i)], "base_ok": sb, "pack_ok": sp})
    out = {"probes": len(rows), "base_score": tot["base"], "pack_score": tot["pack"], "rows": rows,
           "by_section": {sid: {"base": sum(r["base_ok"] for r in rows if r["section"] == sid),
                                "pack": sum(r["pack_ok"] for r in rows if r["section"] == sid),
                                "n": sum(1 for r in rows if r["section"] == sid)} for sid in {r["section"] for r in rows}}}
    from packsmith.workbench import dials
    out["dials"] = dials.get(workbench.name_of(pack))
    if curve:
        pts = [{"strength": 0.0, "score": tot["base"]}]
        for s in curve:
            got = [score(res[(f"x{s:g}", i)], p) for i, p in enumerate(probes) if (f"x{s:g}", i) in res]
            pts.append({"strength": s, "score": sum(got), "answered": len(got)})
        out["curve"] = pts
        out["curve_answers"] = {f"{s:g}": [res.get((f"x{s:g}", i), "") for i in range(len(probes))] for s in curve}
    _wj(pdir(name) / "validation" / f"{time.strftime('%Y%m%d-%H%M%S')}.json", out)
    _status(name, b.get("status", "built"), validation={k: v for k, v in out.items() if k != "rows"})
    return out
