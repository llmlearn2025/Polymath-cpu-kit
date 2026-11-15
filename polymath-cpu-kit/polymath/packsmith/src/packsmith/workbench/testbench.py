"""The Test tab: the same questions to several candidates, answers side by side.

A candidate is an engine (by name, whatever Settings → engine says) with a build, and none, one or several packs,
each at its own strength and routing shift. A run puts every question of every question set to every ticked
candidate, saving each answer as it arrives, as a background job (system.jobs: pause, resume, cancel). Engines are
started one at a time, once per (engine, build); packs are swapped live per candidate with the row's own dials,
which are never written to pack_dials.json. The Workbench's engine is stopped for the run and put back afterwards,
with what it had loaded and its own saved dials, also when the run is cancelled or fails.

Stored under <data>/tests/:
    draft.json                   the tab as the user left it (candidates, question sets, settings)
    sets/<name>.json             saved question sets
    runs/<id>/run.json           the run: candidates (with each pack's checksum), question sets, settings, versions
    runs/<id>/answers.jsonl      one line per answer, appended as it arrives (a resumed run skips what is there)
    runs/<id>/evals.jsonl        one line per verdict of Evaluate (the judge model with it)
Paths in saved runs are relative to the data folder; no home folder or user name is stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from packsmith.packs import store
from packsmith.system.config import SETTINGS

STRENGTH_MAX, SHIFT_MIN, SHIFT_MAX = 1.5, -3.0, 3.0
MODES = ("single", "conversation")
FORMATS = ("chat", "qa", "plain")
DEFAULT_SETTINGS = {"thinking": False, "format": "chat", "max_tokens": 512, "temperature": 0.0, "top_p": 1.0, "repeats": 1}
_lock = threading.Lock()


# ------------------------------------------------------------------------------------------------ storage
def root() -> Path:
    d = store.data_dir() / "tests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rj(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _wj(p: Path, v) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (s or "").strip()).strip("-.")[:60]
    if not s:
        raise ValueError("a name is needed")
    return s


def _rel(p) -> str:
    """A path as stored in a run: relative to the data folder when inside it, else its folder name only."""
    try:
        return str(Path(p).resolve().relative_to(store.data_dir().resolve()))
    except (ValueError, OSError):
        return Path(str(p)).name


# ------------------------------------------------------------------------------------------------ what can run
def pack_kind(path: str) -> dict:
    """{format: bslot | polymath | stack, modes: [route/gate...], routing: bool}: whether a routing shift means
    anything for this pack (Polymath route packs only)."""
    p = Path(path)
    if (p / "overlay.json").exists():
        return {"format": "bslot", "modes": ["gate"], "routing": False, "why": "a colibri pack is gated on every token"}
    m = _rj(p / "pack.json", {}) or {}
    if m.get("format") == "polymath-stack-v1":
        modes = set()
        for sp in m.get("packs") or []:
            q = Path(sp) if Path(sp).is_absolute() else p / sp
            mm = (_rj(q / "pack.json", {}) or {}).get("polymath", {}).get("layers", {})
            modes |= {str(v.get("mode")) for v in mm.values()}
        return {"format": "stack", "modes": sorted(modes), "routing": "route" in modes,
                "why": "" if "route" in modes else "a stack of gate packs has no routing"}
    layers = (m.get("polymath") or {}).get("layers") or {}
    modes = sorted({str(v.get("mode")) for v in layers.values()})
    return {"format": "polymath", "modes": modes, "routing": "route" in modes,
            "why": "" if "route" in modes else "a gate pack has no routing: its experts are added on every token"}


def options() -> dict:
    """Installed engines with their builds, and every library pack with its fit per engine (the existing check,
    packs.manifest.check, against that engine and the served base model)."""
    from packsmith import engines
    from packsmith.packs import manifest
    from packsmith.workbench import dials
    engs = []
    for e in engines.available():
        b = e.get("builds") or []
        builds = ["auto"] + [x for x in ("cpu", "gpu") if x in b] if e["name"] in engines.COLIBRI else ["auto", "cpu", "gpu"]
        if e["name"] == "polymath-c-gpu":
            builds = ["gpu"]
        engs.append({"name": e["name"], "label": e["label"], "installed": bool(e["installed"]), "builds": builds})
    serve = str(SETTINGS.path("base_serve_model")) if SETTINGS.base_serve_model else ""
    packs = []
    for p in store.list_packs():
        fit = {}
        for e in engs:
            try:
                c = manifest.check(p["path"], e["name"], serve)
                fit[e["name"]] = {"level": c["level"], "problems": c["problems"]}
            except Exception as ex:                      # noqa: BLE001 - shown on the row
                fit[e["name"]] = {"level": "refuse", "problems": [str(ex)]}
        packs.append({"name": p["name"], "kind": pack_kind(p["path"]), "fit": fit, "dials": dials.get(p["name"])})
    return {"engines": engs, "packs": packs, "settings": DEFAULT_SETTINGS, "base_model": Path(serve).name if serve else "",
            "workbench_engine": SETTINGS.engine}


def label_of(c: dict) -> str:
    """'polymath-c · oms · 50%', 'patched-colibri (gpu) · base'."""
    ps = c.get("packs") or []
    b = c.get("build") or "auto"
    eng = f"{c['engine']}{'' if b == 'auto' else ' (' + b + ')'}"
    if not ps:
        return f"{eng} · base"
    parts = []
    for p in ps:
        s = f"{p['name']} · {round(float(p.get('strength', 1.0)) * 100)}%"
        if float(p.get("shift", 0) or 0):
            s += f" shift {float(p['shift']):+g}"
        parts.append(s)
    return f"{eng} · " + " + ".join(parts)


def check_candidate(c: dict, opts: dict | None = None) -> dict:
    """The candidate as it will run, or the reason it cannot: {ok, problems, candidate}."""
    opts = opts or options()
    eng = {e["name"]: e for e in opts["engines"]}
    packs = {p["name"]: p for p in opts["packs"]}
    problems = []
    name = str(c.get("engine") or "")
    if name not in eng:
        problems.append(f"no engine {name!r}")
    elif not eng[name]["installed"]:
        problems.append(f"{name} is not installed")
    build = str(c.get("build") or "auto")
    if name in eng and build not in eng[name]["builds"]:
        problems.append(f"{name} has no {build} build")
    out = []
    for p in c.get("packs") or []:
        n = str(p.get("name") or "")
        if n not in packs:
            problems.append(f"no pack {n!r} in the library")
            continue
        fit = packs[n]["fit"].get(name, {})
        if fit.get("level") == "refuse":
            problems.append(f"{n}: " + "; ".join(fit.get("problems") or ["does not fit"]))
        s = float(p.get("strength", 1.0))
        if not 0 <= s <= STRENGTH_MAX:
            problems.append(f"{n}: strength must be 0 to {int(STRENGTH_MAX * 100)}%")
        sh = float(p.get("shift", 0) or 0)
        if sh and not packs[n]["kind"]["routing"]:
            problems.append(f"{n}: {packs[n]['kind']['why'] or 'no routing'}")
        elif sh and name not in ("polymath-engine", "polymath-c", "polymath-c-gpu"):
            problems.append(f"{n}: routing applies to the Polymath engines only")
        if not SHIFT_MIN <= sh <= SHIFT_MAX:
            problems.append(f"{n}: routing must be {SHIFT_MIN:g} to {SHIFT_MAX:+g}")
        out.append({"name": n, "strength": round(s, 3), "shift": round(sh, 3)})
    cand = {"id": str(c.get("id") or uuid.uuid4().hex[:8]), "engine": name, "build": build, "packs": out,
            "enabled": bool(c.get("enabled", True)), "reference": bool(c.get("reference", False))}
    cand["label"] = str(c.get("label") or "").strip() or label_of(cand)
    return {"ok": not problems, "problems": problems, "candidate": cand}


def with_references(cands: list[dict]) -> list[dict]:
    """A base-model row for every engine (and build) that has a ticked row with packs, unless one is there: the
    references first, in the order their engines first appear, then the rows as given."""
    have = {(c["engine"], c.get("build") or "auto") for c in cands if not c.get("packs")}
    refs = []
    for c in cands:
        k = (c["engine"], c.get("build") or "auto")
        if c.get("enabled", True) and c.get("packs") and k not in have:
            r = {"id": uuid.uuid4().hex[:8], "engine": k[0], "build": k[1], "packs": [], "enabled": True, "reference": True}
            r["label"] = label_of(r)
            refs.append(r)
            have.add(k)
    return refs + list(cands)


# ------------------------------------------------------------------------------------------------ question sets
def clean_sets(sets: list[dict]) -> list[dict]:
    out = []
    for i, s in enumerate(sets or []):
        qs = []
        for q in s.get("questions") or []:
            t = str(q.get("text") if isinstance(q, dict) else q or "").strip()
            if t:
                qs.append({"text": t[:20000], "expected": str((q.get("expected") if isinstance(q, dict) else "") or "").strip()[:2000]})
        if qs:
            mode = s.get("mode") if s.get("mode") in MODES else "single"
            out.append({"name": str(s.get("name") or f"set {i + 1}")[:80], "mode": mode, "questions": qs})
    return out


def clean_settings(st: dict | None) -> dict:
    st = {**DEFAULT_SETTINGS, **(st or {})}
    rep = int(st.get("repeats") or 1)
    return {"thinking": bool(st.get("thinking")), "format": st["format"] if st.get("format") in FORMATS else "chat",
            "max_tokens": max(8, min(32768, int(st.get("max_tokens") or 512))),
            "temperature": max(0.0, min(2.0, float(st.get("temperature") or 0.0))),
            "top_p": max(0.01, min(1.0, float(st.get("top_p") if st.get("top_p") not in (None, "") else 1.0))),
            "repeats": rep if rep in (1, 3, 5) else 1}


def list_sets() -> list[str]:
    return sorted(p.stem for p in (root() / "sets").glob("*.json"))


def save_set(name: str, sets: list[dict]) -> dict:
    v = {"name": name, "sets": clean_sets(sets), "saved": time.strftime("%Y-%m-%d %H:%M")}
    if not v["sets"]:
        raise ValueError("there are no questions to save")
    _wj(root() / "sets" / f"{_slug(name)}.json", v)
    return v


def get_set(name: str) -> dict:
    v = _rj(root() / "sets" / f"{_slug(name)}.json")
    if v is None:
        raise KeyError(f"no question set {name!r}")
    return v


def delete_set(name: str) -> None:
    (root() / "sets" / f"{_slug(name)}.json").unlink(missing_ok=True)


def draft() -> dict:
    return _rj(root() / "draft.json", {}) or {}


def save_draft(d: dict) -> dict:
    v = {"candidates": list(d.get("candidates") or [])[:200], "sets": list(d.get("sets") or [])[:50],
         "settings": clean_settings(d.get("settings"))}
    _wj(root() / "draft.json", v)
    return v


def send(question: str, packs: list[str] | None = None, engine: str | None = None) -> dict:
    """The Workbench's "Send to Test": the question as a block of the first single-question set, and the packs
    (at their saved dials, as the Workbench used them) as a candidate on the Workbench's engine."""
    from packsmith.workbench import dials
    d = draft() or {"candidates": [], "sets": [], "settings": dict(DEFAULT_SETTINGS)}
    sets = d.setdefault("sets", [])
    tgt = next((s for s in sets if s.get("mode", "single") == "single"), None)
    if tgt is None:
        tgt = {"name": "from the Workbench", "mode": "single", "questions": []}
        sets.append(tgt)
    if question.strip() and not any((q.get("text") if isinstance(q, dict) else q) == question.strip() for q in tgt["questions"]):
        tgt["questions"].append({"text": question.strip(), "expected": ""})
    eng = engine or SETTINGS.engine
    ps = [{"name": n, "strength": dials.get(n)["strength"], "shift": dials.get(n)["shift"]} for n in (packs or [])]
    cand = {"engine": eng, "build": "auto", "packs": ps, "enabled": True}
    key = json.dumps([eng, ps], sort_keys=True)
    if not any(json.dumps([c.get("engine"), c.get("packs") or []], sort_keys=True) == key for c in d.setdefault("candidates", [])):
        d["candidates"].append({**cand, "id": uuid.uuid4().hex[:8], "label": label_of({**cand, "build": "auto"})})
    return save_draft(d)


# ------------------------------------------------------------------------------------------------ runs
def _run_dir(rid: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", rid or ""):
        raise KeyError(f"no run {rid!r}")
    return root() / "runs" / rid


def _pack_sha(path: str) -> str:
    p = Path(path)
    f = p / "overlay.safetensors" if (p / "overlay.safetensors").exists() else p / "experts.safetensors"
    if not f.exists():
        f = p / "pack.json"
    h = hashlib.sha256()
    try:
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()[:16]


def create_run(body: dict) -> dict:
    """A run from the tab's candidates, question sets and settings (the ticked, fitting rows; base rows added for
    each engine unless add_references is false). Raises ValueError with the reasons when nothing can run."""
    from packsmith import __version__
    from packsmith.packs import manifest
    opts = options()
    rows = [c for c in (body.get("candidates") or []) if c.get("enabled", True)]
    cands, refused = [], []
    for c in rows:                                        # a row that cannot run is reported, not run
        r = check_candidate(c, opts)
        (cands if r["ok"] else refused).append(r["candidate"] if r["ok"] else {**r["candidate"], "problems": r["problems"]})
    if body.get("add_references", True):                  # the base on each engine that runs a pack
        cands = [check_candidate(c, opts)["candidate"] if c.get("reference") and "label" in c and c.get("id") not in
                 {x["id"] for x in cands} else c for c in with_references(cands)]
    sets = clean_sets(body.get("sets") or [])
    if not cands:
        raise ValueError("no candidate can run" + ("".join(f"; {c['label']}: {', '.join(c['problems'])}" for c in refused)))
    if not sets:
        raise ValueError("add at least one question")
    paths = {p["name"]: p["path"] for p in store.list_packs()}
    serve = str(SETTINGS.path("base_serve_model")) if SETTINGS.base_serve_model else ""
    ident = {}
    try:
        mi = manifest.model_identity(serve) if serve else {}
        ident = {k: mi.get(k, "") for k in ("label", "repo", "weights_hash", "arch_fingerprint", "engine_kind")}
    except Exception:                                     # noqa: BLE001 - informative only
        pass
    for c in cands:
        for p in c["packs"]:
            p["sha"] = _pack_sha(paths.get(p["name"], ""))
            p["path"] = _rel(paths.get(p["name"], ""))
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run = {"id": rid, "name": str(body.get("name") or "").strip()[:80] or time.strftime("run %d %b %H:%M"),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "packsmith": __version__, "candidates": cands,
           "refused": refused, "sets": sets, "settings": clean_settings(body.get("settings")),
           "base_model": {"name": Path(serve).name if serve else "", **ident}, "status": "created", "devices": {}}
    _wj(_run_dir(rid) / "run.json", run)
    return run


def get_run(rid: str) -> dict:
    d = _run_dir(rid)
    run = _rj(d / "run.json")
    if run is None:
        raise KeyError(f"no run {rid!r}")
    ans = _jsonl(d / "answers.jsonl")
    ev = _jsonl(d / "evals.jsonl")
    return {**run, "answers": ans, "evals": ev, "summary": summary(run, ans, ev)}


def list_runs() -> list[dict]:
    out = []
    for f in sorted((root() / "runs").glob("*/run.json"), reverse=True):
        r = _rj(f, {}) or {}
        n = sum(1 for _ in open(f.parent / "answers.jsonl", encoding="utf-8")) if (f.parent / "answers.jsonl").exists() else 0
        out.append({"id": r.get("id"), "name": r.get("name"), "created": r.get("created"), "status": r.get("status"),
                    "candidates": len(r.get("candidates") or []), "questions": sum(len(s["questions"]) for s in r.get("sets") or []),
                    "answers": n, "expected": _total(r)})
    return out


def delete_run(rid: str) -> None:
    import shutil
    shutil.rmtree(_run_dir(rid), ignore_errors=True)


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    out.append(json.loads(l))
                except ValueError:                        # a half-written last line after a crash
                    pass
    except OSError:
        pass
    return out


def _append(p: Path, v: dict) -> None:
    with _lock, open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(v, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _total(run: dict) -> int:
    reps = int((run.get("settings") or {}).get("repeats") or 1)
    return len(run.get("candidates") or []) * sum(len(s["questions"]) for s in run.get("sets") or []) * reps


# ------------------------------------------------------------------------------------------------ the answer flags
def repeating(text: str, n: int = 6, times: int = 3) -> bool:
    """The same run of n words three or more times in one answer (a loop)."""
    w = re.findall(r"\w+", (text or "").lower())
    if len(w) < n * times:
        return False
    seen: dict[tuple, int] = {}
    last: dict[tuple, int] = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        if i >= last.get(g, -n):                          # non-overlapping occurrences
            seen[g] = seen.get(g, 0) + 1
            last[g] = i + n
            if seen[g] >= times:
                return True
    return False


def think_open(raw: str, thinking: bool) -> bool:
    """The answer opened a think block (or the prompt left one open) and never closed it."""
    r = raw or ""
    opened = "<think>" in r or (thinking and "</think>" not in r and bool(r.strip()))
    return opened and "</think>" not in r


def matches(expected: str, answer: str) -> bool:
    """The expected answer (or one of "a|b") as whole words in the answer."""
    a = (answer or "").lower()
    return any(re.search(r"(?<![\w])" + re.escape(e.strip().lower()) + r"(?![\w])", a) for e in (expected or "").split("|") if e.strip())


# ------------------------------------------------------------------------------------------------ swapping packs
def swap(e, engine: str, entries: list[dict]) -> dict:
    """The candidate's packs on its engine, each at its own strength and routing, without the Workbench dials.
    Polymath engines take the list as it is. Patched colibri serves one B-slot pack: several are merged with each
    pack's strength folded into its experts (colibri packs.merge(strengths=...)), a single pack is served at its
    strength. A pack at 0% is left out."""
    from packsmith import engines
    paths = {p["name"]: p["path"] for p in store.list_packs()}
    live = [p for p in entries if float(p.get("strength", 1.0)) > 0]
    if not live:
        return e.swap(None) if engines.is_polymath(engine) else e.swap(None, strength=1.0)
    for p in live:
        if p["name"] not in paths:
            return {"ok": False, "engine_log": [f"no pack {p['name']} in the library"], "seconds": 0}
    if engines.is_polymath(engine):
        return e.swap([paths[p["name"]] for p in live], strengths=[float(p["strength"]) for p in live],
                      shifts=[float(p.get("shift", 0) or 0) for p in live])
    if len(live) == 1:
        return e.swap(paths[live[0]["name"]], strength=float(live[0]["strength"]))
    from packsmith.engines.colibri import packs as cp
    key = hashlib.sha1(json.dumps([[p["name"], float(p["strength"]), _pack_sha(paths[p["name"]])] for p in live]).encode()).hexdigest()[:12]
    out = store.data_dir() / "cache" / f"test-merged-{key}"
    if not (out / "overlay.json").exists():
        cp.merge([Path(paths[p["name"]]) for p in live], out, source="test: " + " + ".join(p["name"] for p in live),
                 strengths=[float(p["strength"]) for p in live])
    return e.swap(str(out), strength=1.0)


# ------------------------------------------------------------------------------------------------ the run
def _prompt(q: str, st: dict, history: list[dict]) -> str:
    """The question in the run's format; history: the candidate's own earlier [{q, a}] (conversation mode)."""
    from packsmith.workbench import workbench as wb
    fmt = "chat" if st["thinking"] else st["format"]
    if fmt == "chat":
        turns = [t for h in history for t in ({"role": "user", "content": h["q"]}, {"role": "assistant", "content": h["a"]})]
        return wb.chat_prompt(q, turns, "", st["thinking"])
    prev = "".join(f"Question: {h['q']}\nAnswer: {h['a']}\n\n" for h in history)
    return prev + wb.build_prompt(q, fmt)


def _ask(e, prompt: str, st: dict) -> tuple[str, dict]:
    """One request with the run's sampling; (raw text, the DONE line's figures)."""
    from packsmith.workbench import workbench as wb
    e.last_stat = {}
    raw = wb.engine_ask(e, prompt, st["max_tokens"], st["temperature"], st["top_p"])
    return raw, dict(getattr(e, "last_stat", {}) or {})


class Hold:
    """The Test run has the card: the Workbench's engine stopped (what it had loaded remembered), the LLM server's
    models unloaded (remembered, loaded again when it is next needed), training waited for. On exit, always: the
    run's engine closed, and the Workbench started again with what it had, with its own saved dials."""

    def __init__(self, rid: str, log, what: str = "Test run"):
        self.rid, self.log, self.what = rid, log, what

    def __enter__(self):
        from packsmith.system import gpu, jobs
        from packsmith.workbench import workbench as wb
        while not jobs.GPU_LOCK.acquire(timeout=5):       # a training holds the card
            self.log("waiting for the GPU (a training is running)")
        self.locked = True
        e = wb.STATE.get("engine")
        self.was_running = e is not None and e.alive()
        self.pack = wb.current()
        self.names = list(wb.STATE.get("active_names") or [])
        if self.was_running:
            self.log(f"the Workbench's engine ({SETTINGS.engine}) is stopped for this run; it comes back afterwards"
                     + (f" with {', '.join(self.names)}" if self.names else ""))
            wb.stop()
        wb.STATE["held_by"] = f"{self.what} {self.rid}"
        if gpu.shared():
            try:
                from packsmith.llm import llm
                loaded = [(m["key"], i["context_length"]) for m in llm.instances() for i in m["loaded"]]
                if loaded:
                    gpu._remember(loaded)
                    llm.unload_all()
                    self.log("the LLM server's models are unloaded for the run")
            except Exception as ex:                        # noqa: BLE001 - an LLM server that is not running
                self.log(f"LLM server: {ex}")
        gpu._set("test", self.rid)
        self.engine = None
        return self

    def close_engine(self):
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:                              # noqa: BLE001
                pass
            self.engine = None

    def __exit__(self, *exc):
        from packsmith.system import gpu, jobs
        from packsmith.workbench import workbench as wb
        try:
            self.close_engine()
        finally:
            wb.STATE.pop("held_by", None)
            gpu._set("")
            try:
                if self.was_running:
                    self.log("starting the Workbench's engine again")
                    st = wb.start()
                    if st.get("status") == "ready":
                        wb.restore(self.pack)
                        gpu._set("engine")
                    else:
                        self.log(f"the Workbench's engine did not start again: {st.get('error') or st.get('status')}")
            finally:
                jobs.GPU_LOCK.release()
        return False


def run_job(rid: str, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Ask every question of every set to every candidate (and repeat), saving each answer as it arrives. A resumed
    run skips the answers already saved; a conversation is continued from its saved turns."""
    from packsmith.workbench import workbench as wb
    d = _run_dir(rid)
    run = _rj(d / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    st = run["settings"]
    done = {(a["cand"], a["set"], a["q"], a["rep"]) for a in _jsonl(d / "answers.jsonl")}
    saved = {(a["cand"], a["set"], a["q"], a["rep"]): a for a in _jsonl(d / "answers.jsonl")}
    total = _total(run)
    n_done = [len(done)]
    failed: dict[str, str] = dict(run.get("failed") or {})
    run["status"] = "running"
    from packsmith.system import jobs
    cj = jobs.current()                                  # the job this runs in (thread-local): the tab watches it
    if cj is not None:
        run["job"] = cj.id
        run.setdefault("jobs", []).append(cj.id)
    _wj(d / "run.json", run)

    def log(m):
        progress(n_done[0] / max(1, total), m)

    groups: dict[tuple, list[dict]] = {}
    for c in run["candidates"]:
        groups.setdefault((c["engine"], c.get("build", "auto")), []).append(c)
    with Hold(rid, log) as hold:
        for (engine, build), cands in groups.items():
            todo = [c for c in cands if any((c["id"], si, qi, r) not in done for si, s in enumerate(run["sets"])
                                            for qi in range(len(s["questions"])) for r in range(st["repeats"]))]
            if not todo:
                continue
            if cancelled():
                break
            log(f"starting {engine}" + ("" if build == "auto" else f" ({build})"))
            try:
                hold.close_engine()
                hold.engine, real_build = wb.make_engine(engine, build)
                dev = ""
                try:
                    dev = (hold.engine.device() or "")[:240]
                except Exception:                          # noqa: BLE001
                    pass
                run.setdefault("devices", {})[f"{engine}/{build}"] = {"device": dev, "build": real_build}
                _wj(d / "run.json", run)
            except Exception as ex:                        # noqa: BLE001 - the others still run
                for c in todo:
                    failed[c["id"]] = f"{engine} did not start: {type(ex).__name__}: {ex}"[:1500]
                log(f"{engine} did not start: {ex}")
                continue
            e = hold.engine
            for c in todo:
                if cancelled():
                    break
                sw = swap(e, engine, c["packs"])
                if not sw.get("ok"):
                    failed[c["id"]] = "pack refused: " + " | ".join((sw.get("engine_log") or [])[-4:])
                    log(f"{c['label']}: {failed[c['id']]}")
                    continue
                failed.pop(c["id"], None)
                for si, s in enumerate(run["sets"]):
                    for r in range(st["repeats"]):
                        hist: list[dict] = []
                        for qi, q in enumerate(s["questions"]):
                            key = (c["id"], si, qi, r)
                            if key in done:                    # a resumed conversation continues from its saved turns
                                if s["mode"] == "conversation":
                                    hist.append({"q": q["text"], "a": saved[key].get("answer", "")})
                                continue
                            if cancelled():
                                break
                            before = wb.pack_counters(e)
                            t0 = time.time()
                            try:
                                prompt = _prompt(q["text"], st, hist if s["mode"] == "conversation" else [])
                                raw, stat = _ask(e, prompt, st)
                                err = ""
                            except Exception as ex:          # noqa: BLE001 - recorded on the answer
                                raw, stat, err = "", {}, f"{type(ex).__name__}: {ex}"[:500]
                            after = wb.pack_counters(e)
                            ans, think = wb.split_answer(raw)
                            rec = {"cand": c["id"], "set": si, "q": qi, "rep": r, "answer": ans, "thinking": think,
                                   "seconds": round(time.time() - t0, 2), "error": err,
                                   **{k: stat[k] for k in ("tokens", "tokens_per_second", "prompt_tokens", "length_limited") if k in stat},
                                   "think_open": think_open(raw, st["thinking"]) if not err else False,
                                   "repeating": repeating(ans + " " + think)}
                            if "routings" in after:
                                dt = max(0, int(after.get("routings", 0)) - int(before.get("routings", 0)))
                                dw = max(0, int(after.get("won", 0)) - int(before.get("won", 0)))
                                gate = all(not any(m == "route" for m in (pack_kind_cached(p["name"]) or {}).get("modes", []))
                                           for p in c["packs"])
                                rec["routing_share"] = None if (gate or not dt) else round(dw / dt, 4)
                            if q.get("expected"):
                                rec["matches_expected"] = matches(q["expected"], ans)
                            _append(d / "answers.jsonl", rec)
                            done.add(key)
                            saved[key] = rec
                            n_done[0] += 1
                            if s["mode"] == "conversation":
                                hist.append({"q": q["text"], "a": ans})
                            progress(n_done[0] / max(1, total), f"{c['label']}: set {si + 1}, question {qi + 1}"
                                     + (f", repeat {r + 1}" if st["repeats"] > 1 else ""))
                        if cancelled():
                            break
                    if cancelled():
                        break
            hold.close_engine()
    run = _rj(d / "run.json")
    run["failed"] = failed
    run["status"] = "paused" if cancelled() else "done"
    run["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _wj(d / "run.json", run)
    return {"run": rid, "answers": n_done[0], "of": total, "failed": failed}


_KIND_CACHE: dict[str, dict] = {}


def pack_kind_cached(name: str) -> dict:
    if name not in _KIND_CACHE:
        p = next((x["path"] for x in store.list_packs() if x["name"] == name), "")
        _KIND_CACHE[name] = pack_kind(p) if p else {}
    return _KIND_CACHE[name]


def start(body: dict) -> dict:
    """Create a run and start it as a background job: {run, job}."""
    from packsmith.system import jobs
    run = create_run(body)
    j = jobs.start("test", run_job, run["name"], rid=run["id"])
    return {"run": run, "job": j.view()}


def resume(rid: str) -> dict:
    """Run what is not answered yet (a paused, cancelled or failed run): a new job, answered questions skipped."""
    from packsmith.system import jobs
    run = _rj(_run_dir(rid) / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    j0 = jobs.get(run.get("job") or "")
    if j0 and j0.get("status") == "running":
        raise ValueError("the run is still running")
    return jobs.start("test", run_job, run["name"] + " (continued)", rid=rid).view()


# ------------------------------------------------------------------------------------------------ results
def summary(run: dict, answers: list[dict], evals: list[dict]) -> dict:
    """Per candidate: answers, cut off, think left open, repeating, same as base, matches expected, judged correct,
    average speed and routing share; per question and candidate: how many repeats agree. Counts, not percentages."""
    by_c: dict[str, list[dict]] = {}
    for a in answers:
        by_c.setdefault(a["cand"], []).append(a)
    ev = {}
    for x in evals:                                        # the latest verdict per answer
        ev[(x["cand"], x["set"], x["q"], x["rep"])] = x
    base_of = {}
    for c in run.get("candidates") or []:
        if not c["packs"]:
            base_of[(c["engine"], c.get("build", "auto"))] = c["id"]
    base_ans = {(a["cand"], a["set"], a["q"], a["rep"]): a.get("answer", "") for a in answers}
    out = {}
    for c in run.get("candidates") or []:
        rs = by_c.get(c["id"], [])
        b = base_of.get((c["engine"], c.get("build", "auto")))
        same = [a for a in rs if c["packs"] and b and base_ans.get((b, a["set"], a["q"], a["rep"])) == a.get("answer")]
        tps = [a["tokens_per_second"] for a in rs if a.get("tokens_per_second")]
        sh = [a["routing_share"] for a in rs if a.get("routing_share") is not None]
        exp = [a for a in rs if "matches_expected" in a]
        judged = [ev[(c["id"], a["set"], a["q"], a["rep"])] for a in rs if (c["id"], a["set"], a["q"], a["rep"]) in ev]
        agree = {}
        for a in rs:
            agree.setdefault((a["set"], a["q"]), []).append(a.get("answer", ""))
        agreement = {f"{k[0]}:{k[1]}": {"agree": max(v.count(x) for x in v), "of": len(v)} for k, v in agree.items() if len(v) > 1}
        out[c["id"]] = {
            "answers": len(rs), "errors": sum(1 for a in rs if a.get("error")),
            "cut_off": sum(1 for a in rs if a.get("length_limited")), "think_open": sum(1 for a in rs if a.get("think_open")),
            "repeating": sum(1 for a in rs if a.get("repeating")), "same_as_base": len(same) if c["packs"] and b else None,
            "matches_expected": sum(1 for a in exp if a["matches_expected"]), "with_expected": len(exp),
            "judged": len(judged), "judged_good": sum(1 for x in judged if x.get("verdict") in ("correct", "mostly_correct")),
            "tokens_per_second": round(sum(tps) / len(tps), 1) if tps else None,
            "routing_share": round(sum(sh) / len(sh), 4) if sh else None, "agreement": agreement,
            "failed": (run.get("failed") or {}).get(c["id"], "")}
    return out


# ------------------------------------------------------------------------------------------------ Evaluate
def blind(text: str, names: list[str]) -> str:
    """The text with every candidate label, pack name and engine name taken out: the judge must not know whose
    answer it reads."""
    out = text or ""
    for n in sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True):
        out = re.sub(re.escape(n), "[…]", out, flags=re.I)
    return out


def references(run: dict, question: str, answer: str, topic: str = "") -> list[str]:
    """The reference lines for one question: the union of the sources of every pack in the run (the same for every
    candidate), the most related first."""
    from packsmith.workbench import workbench as wb
    names = sorted({p["name"] for c in run["candidates"] for p in c["packs"]})
    words = wb._words(question + " " + answer)
    seen, ref = set(), []
    for n in names or [None]:
        src = wb.reference_source([n] if n else [], topic)
        key = (src.get("kind"), src.get("name"))
        if not src or key in seen:
            continue
        seen.add(key)
        ref += wb.source_refs(src, words)
    lines, out = set(), []
    for sc, r in sorted(ref, key=lambda x: -x[0]):
        if sc > 0 and r not in lines:
            lines.add(r)
            out.append(r)
    return out[:24]


def secret_names(run: dict) -> list[str]:
    from packsmith import engines
    names = [c["label"] for c in run["candidates"]] + [p["name"] for c in run["candidates"] for p in c["packs"]]
    names += list(engines.NAMES) + ["polymath", "colibri", "b-slot"]
    return names


def evaluate_job(rid: str, model: str = "", topic: str = "", set_index: int | None = None, q_index: int | None = None,
                 cand: str = "", progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Score answers with the Analyse check, blind: the whole run, one question (set_index, q_index) or one cell
    (and cand). The judge model's name is saved with each verdict; running it again adds newer verdicts."""
    from packsmith.workbench import workbench as wb
    d = _run_dir(rid)
    run = _rj(d / "run.json")
    ans = [a for a in _jsonl(d / "answers.jsonl") if not a.get("error")
           and (set_index is None or a["set"] == set_index) and (q_index is None or a["q"] == q_index)
           and (not cand or a["cand"] == cand)]
    judge = model or SETTINGS.default_model or ""
    secret = secret_names(run)
    n = 0
    for a in ans:
        if cancelled():
            break
        q = run["sets"][a["set"]]["questions"][a["q"]]["text"]
        ref = [blind(x, secret) for x in references(run, q, a.get("answer", ""), topic)]
        r = wb.judge_answer(blind(q, secret), blind(a.get("answer", ""), secret), ref, judge or None)
        _append(d / "evals.jsonl", {"cand": a["cand"], "set": a["set"], "q": a["q"], "rep": a["rep"],
                                    "verdict": r.get("verdict", ""), "html": r.get("html", "")[:20000],
                                    "judge": judge, "references": len(ref), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        n += 1
        progress(n / max(1, len(ans)), f"judged {n} of {len(ans)}")
    run = _rj(d / "run.json")
    run.setdefault("judges", [])
    if judge and judge not in run["judges"]:
        run["judges"].append(judge)
    _wj(d / "run.json", run)
    return {"run": rid, "judged": n, "judge": judge}
