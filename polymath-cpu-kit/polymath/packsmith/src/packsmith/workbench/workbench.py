"""Workbench: the colibri engine with the base model loaded once, packs ticked on and off, compare mode,
judge, tests and evaluation."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

from packsmith.mining import htmlconv
from packsmith.llm import llm
from packsmith.system import par
from packsmith.datasets import prompts
from packsmith.packs import store
from packsmith.system.config import SETTINGS
from packsmith.system.home import expand

_lock = threading.Lock()
STATE = {"engine": None, "status": "stopped", "error": "", "model": "", "started": None}
TEMPLATES = {"plain": "{q}", "qa": "Question: {q}\nAnswer:",
             # OLMoE-1B-7B-0125-Instruct's own template (its start and end token is |||IP_ADDRESS|||, id 50279)
             "olmoe-chat": "|||IP_ADDRESS|||<|user|>\n{q}\n<|assistant|>\n",
             "expert": "You are an expert in {domain}. Answer precisely and show your reasoning step by step.\nQuestion: {q}\nAnswer:"}


COMBOS: dict[str, list[str]] = {}   # a merged pack / stack made for several packs -> their names


def name_of(path) -> str:
    """The library name of a pack folder (its folder name when it is not in the library)."""
    key = str(Path(path).resolve())
    for p in store.list_packs():
        if str(Path(p["path"]).resolve()) == key:
            return p["name"]
    return Path(path).name


def _names(items: list) -> list[str]:
    names: list[str] = []
    for p in items:
        names += COMBOS.get(str(Path(p).resolve())) or [name_of(p)]
    return names


def _dialled(e, orig, pack, items: list, names: list[str]) -> dict:
    """The swap with the Workbench dials (workbench/dials.py) of these packs applied. At the defaults it is the plain
    swap, so nothing changes for packs nobody has dialled."""
    from packsmith import engines
    from packsmith.workbench import dials
    d = {n: dials.get(n) for n in names}
    if not items or all(dials.is_default(v) for v in d.values()):
        return orig(pack)
    if engines.is_polymath(SETTINGS.engine):
        # one entry per library pack: a set Packsmith stacked for several packs (COMBOS) is given to the engine as
        # its members, so each keeps its own dials; a library pack (a user pack's stack too) is one entry
        paths, st, sh = [], [], []
        for p in items:
            members = COMBOS.get(str(Path(p).resolve()))
            for n, q in ([(n, pack_path(n)) for n in members] if members else [(name_of(p), p)]):
                paths.append(str(q)); st.append(float(d.get(n, dials.DEFAULT)["strength"]))
                sh.append(float(d.get(n, dials.DEFAULT)["shift"]))
        return orig(paths, strengths=st, shifts=sh)
    # patched colibri: one B-slot pack (a merged set is one pack: the mean of its members' strengths); no shift
    s = sum(float(v["strength"]) for v in d.values()) / len(d)
    if s <= 0:
        r = orig(None)
        return {**r, "active": None, "note": "strength 0: the pack is off"}
    return orig(items[0], strength=s)


def _track(e) -> None:
    """The Workbench keeps the record of which packs are active: every swap on its engine, whoever asks for it (the
    page, an agent, validation, an evaluation), goes through here, gets the packs' Workbench dials, and updates
    STATE["active_names"]. A call with explicit strengths (swap_at) is passed through as it is."""
    orig = e.swap

    def swap(pack=None, *a, **kw):
        items = list(pack) if isinstance(pack, (list, tuple)) else ([pack] if pack else [])
        names = _names(items)
        r = orig(pack, *a, **kw) if (a or kw) else _dialled(e, orig, pack, items, names)
        if r.get("ok"):
            from packsmith.workbench import dials
            STATE["active_names"] = names
            STATE["active_pack"] = pack if not (a or kw) else None
            STATE["active_dials"] = {n: dials.get(n) for n in names} if not (a or kw) else {}
        return r
    e.swap = swap
    e._swap_raw = orig


def current():
    """What is active, to put back later with restore()."""
    return STATE.get("active_pack")


def restore(pack) -> None:
    e = STATE.get("engine")
    if e is None:
        return
    try:
        if not e.swap(pack).get("ok"):
            e.swap(None)
    except Exception:                                     # noqa: BLE001 - a pack removed since: the base
        e.swap(None)


def reapply() -> dict | None:
    """After a dial changed: the active set again, with the new dials (None when the engine is off or nothing is
    active)."""
    e = STATE.get("engine")
    if e is None or not e.alive() or not STATE.get("active_pack"):
        return None
    return e.swap(STATE["active_pack"])


def swap_at(e, path: str, strength: float) -> dict:
    """One pack at an explicit strength (the strength curve of a check), whatever its dials; routing as dialled."""
    from packsmith import engines
    from packsmith.workbench import dials
    if engines.is_polymath(SETTINGS.engine):
        return e.swap([str(path)], strengths=[float(strength)], shifts=[dials.get(name_of(path))["shift"]])
    if strength <= 0:
        return e.swap(None, strength=1.0)
    return e.swap(str(path), strength=float(strength))


def _mk():
    """The engine class and its pack module for Settings → engine."""
    from packsmith import engines
    return engines.get(SETTINGS.engine)


def engine_kind(model: Path) -> str:
    return "qwen36" if (model / "qwen36_meta.json").exists() else "olmoe"


def available_builds(engine: str | None = None) -> list[str]:
    """The installed builds of an engine (Settings → engine when not named)."""
    engine = engine or SETTINGS.engine
    if engine == "polymath-c":
        from packsmith.engines.polymath_c import builds as pc_builds
        return pc_builds(str(SETTINGS.path("colibri_dir")))
    if engine == "polymath-c-gpu":
        from packsmith.engines.polymath_c import gpu_builds
        return gpu_builds(str(SETTINGS.path("colibri_dir")))
    if engine != "patched-colibri":
        return []
    from packsmith.engines.colibri.engine import builds
    return builds(str(SETTINGS.path("colibri_dir")))


def choose_build(on_gpu: bool, engine: str | None = None, want: str | None = None) -> tuple[str, bool]:
    """(build, run on the GPU?) from Settings → Engine build (or `want`) and what is installed.
    auto: the GPU build when the engine has a card and it is installed, else the CPU build."""
    have = available_builds(engine)
    want = str(want or SETTINGS.engine_build or "auto").lower()
    if have == ["single"]:                       # an older install with one build in c/
        return "single", on_gpu if want == "auto" else want == "gpu"
    if want == "gpu":
        if "gpu" not in have:
            raise RuntimeError("Engine build = gpu, but colibri has no GPU build (install_patched_colibri.sh --both, "
                               "which needs the CUDA toolkit)")
        if not on_gpu:
            raise RuntimeError("Engine build = gpu, but the engine has no card (Settings → GPU → Card for the engine)")
        return "gpu", True
    if want == "cpu":
        if "cpu" not in have:
            raise RuntimeError("Engine build = cpu, but colibri has no CPU build (install_patched_colibri.sh --both)")
        return "cpu", False
    if on_gpu and "gpu" in have:
        return "gpu", True
    if "cpu" in have:
        return "cpu", False
    return ("gpu", True) if "gpu" in have else ("cpu", False)


def make_engine(engine: str, build: str = "auto", model: str | None = None, gpu: bool | None = None):
    """(engine object, build) for an engine NAMED here, whatever Settings → engine says, and without touching the
    Workbench's state (the Test tab starts engines by name this way). build: auto | cpu | gpu, as Settings → Engine
    build. The served base model, the card and the pack budget are the Workbench's."""
    from packsmith import engines
    from packsmith.system import gpu as _gpu
    Engine, _ = engines.get(engine)
    if not (model or SETTINGS.base_serve_model):
        raise RuntimeError("no base model set (Settings → Paths → Base model for the engine)")
    raw = model or SETTINGS.base_serve_model
    m = Path("tiny") if str(raw) == "tiny" else expand(raw)
    dev_env, on_gpu = _gpu.engine_env()
    want_gpu = on_gpu if gpu is None else gpu
    want = str(build or "auto").lower()
    if engine == "polymath-c-gpu":
        b, use_gpu = "gpu", True
    elif engine in ("patched-colibri", "polymath-c"):
        b, use_gpu = choose_build(want_gpu, engine, want)
    else:                                                # Polymath Engine (PyTorch): no builds; cpu forces the CPU
        use_gpu = bool(want_gpu) and want != "cpu"
        b = "gpu" if use_gpu else "cpu"
    if not use_gpu:
        dev_env = {**dev_env, "CUDA_VISIBLE_DEVICES": ""}
    kw = {"build": b} if engine in ("patched-colibri", "polymath-c", "polymath-c-gpu") else {}
    e = Engine(str(SETTINGS.path("colibri_dir")), str(m), engine_kind(m), gpu=use_gpu, **kw,
               extra_env={**dev_env, "POLYMATH_PACK_BUDGET": str(int(SETTINGS.pack_budget)),
                          "Q36_MAXT": str(int(SETTINGS.engine_ctx)), "CTX": str(min(int(SETTINGS.engine_ctx), 4096))})
    return e, b


def start(model: str | None = None, gpu: bool | None = None) -> dict:
    with _lock:
        if STATE.get("held_by"):                           # a Test run has the card: it puts the Workbench back after
            STATE.update(error=f"{STATE['held_by']} is using the engine; the Workbench comes back when it ends")
            return {**status(), "status": "held"}
        if STATE["engine"] is not None and STATE["engine"].alive():
            return status()
        STATE.update(status="starting", error="")
        try:
            Engine, _ = _mk()
            if not (model or SETTINGS.base_serve_model):
                raise RuntimeError("no base model set (Settings → Paths → Base model for the engine)")
            m = expand(model or SETTINGS.base_serve_model)
            from packsmith.system import gpu as _gpu
            dev_env, on_gpu = _gpu.engine_env()
            want_gpu = on_gpu if gpu is None else gpu
            if SETTINGS.engine == "polymath-c-gpu":                      # the GPU build, always
                build, use_gpu = "gpu", True
            elif SETTINGS.engine in ("patched-colibri", "polymath-c"):   # colibri binaries: cpu / gpu builds
                build, use_gpu = choose_build(want_gpu)
            else:                                    # Polymath Engine (PyTorch): no separate builds; cpu forces the CPU
                use_gpu = bool(want_gpu) and str(SETTINGS.engine_build or "auto").lower() != "cpu"
                build = "gpu" if use_gpu else "cpu"
            if not use_gpu:
                dev_env = {**dev_env, "CUDA_VISIBLE_DEVICES": ""}
            kw = {"build": build} if SETTINGS.engine in ("patched-colibri", "polymath-c", "polymath-c-gpu") else {}
            STATE["engine"] = Engine(str(SETTINGS.path("colibri_dir")), str(m), engine_kind(m),
                                     gpu=use_gpu, **kw,
                                     extra_env={**dev_env, "POLYMATH_PACK_BUDGET": str(int(SETTINGS.pack_budget)),
                                                "Q36_MAXT": str(int(SETTINGS.engine_ctx)),
                                                "CTX": str(min(int(SETTINGS.engine_ctx), 4096))})   # OLMoE reads CTX (4096 max)
            _track(STATE["engine"])
            STATE.update(status="ready", model=str(m), started=time.time(), build=build, active_names=[])
        except Exception as e:                       # noqa: BLE001
            STATE.update(status="error", error=f"{type(e).__name__}: {e}", engine=None)
    return status()


def stop() -> dict:
    with _lock:
        if STATE["engine"] is not None:
            STATE["engine"].close()
        STATE.update(engine=None, status="stopped")
    return status()


def status() -> dict:
    e = STATE["engine"]
    return {"status": STATE["status"] if e is None or e.alive() else "exited", "error": STATE["error"], "model": STATE["model"],
            "active": e.active if e else None, "active_names": STATE.get("active_names", []) if e else [],
            "active_dials": STATE.get("active_dials", {}) if e else {},
            "device": e.device() if e else "",
            "build": getattr(e, "build", "") if e else "", "builds": available_builds(),
            "engine_build": SETTINGS.engine_build, "held_by": STATE.get("held_by", "")}


def _eng():
    e = STATE["engine"]
    if e is None or not e.alive():
        raise RuntimeError("the engine is not running: start it in the Workbench")
    return e


class PackIncompatible(RuntimeError):
    pass


WARNINGS: dict = {}          # pack name -> the compatibility warning shown with its answers


def compat(name: str, path: str) -> dict:
    """The pack manifest check against Settings → engine and the served base model (packs.manifest.check)."""
    from packsmith.packs import manifest
    return manifest.check(path, SETTINGS.engine, str(SETTINGS.path("base_serve_model")) if SETTINGS.base_serve_model else "")


def pack_path(name: str, force: bool = False) -> str:
    """The pack's folder, after the compatibility check: a pack made for another engine or model shape is refused
    (force=True loads it anyway); other weights of the same shape load with a warning."""
    for p in store.list_packs():
        if p["name"] == name:
            c = compat(name, p["path"])
            if c["level"] == "refuse" and not force:
                raise PackIncompatible(f"pack {name}: " + "; ".join(c["problems"]))
            if c["level"] in ("warn", "refuse"):
                WARNINGS[name] = "; ".join(c["problems"])
            else:
                WARNINGS.pop(name, None)
            return p["path"]
    raise KeyError(f"no pack {name}")


def _domain(name: str) -> str:
    for p in store.list_packs():
        if p["name"] == name:
            return (p["card"].get("domain") or p["card"].get("topic_name") or name).lower()
    return name


def merged(names: list[str]) -> str:
    _, PK = _mk()
    key = hashlib.sha1("|".join(sorted(names)).encode()).hexdigest()[:12]
    out = store.data_dir() / "cache" / f"merged-{key}"
    if not (out / "overlay.json").exists() and not (out / "pack.json").exists():
        PK.merge([Path(pack_path(n)) for n in names], out, source="workbench: " + " + ".join(names))
    COMBOS[str(out.resolve())] = list(names)
    return str(out)


def same_domain(names: list[str]) -> list[str]:
    seen, dup = {}, []
    for n in names:
        d = _domain(n)
        if d in seen:
            dup += [seen[d], n]
        seen[d] = n
    return sorted(set(dup))


def engine_ask(e, prompt: str, max_tokens: int, temperature: float | None = None, top_p: float | None = None,
               on_chunk=None) -> str:
    """One request to colibri with sampling settings (the protocol's SUBMIT carries temperature and top_p; the
    wrapper's own ask() always sends 0 and 1). on_chunk(text) receives the answer as it is written."""
    temp = SETTINGS.engine_temperature if temperature is None else float(temperature)
    tp = SETTINGS.engine_top_p if top_p is None else float(top_p)
    with e.lock:
        rid = e.next_id
        e.next_id += 1
        data = prompt.encode()
        e.p.stdin.write(b"SUBMIT %d 0 %d %d %s %s\n" % (rid, len(data), int(max_tokens), (b"%g" % temp), (b"%g" % tp)) + data + b"\n")
        e.p.stdin.flush()
        out = b""
        while True:
            line = e._line()
            if line.startswith(b"DATA %d " % rid):
                n = int(line.split()[2])
                while len(e.buf) < n + 1:
                    e._fill(600)
                chunk = e.buf[:n]
                out += chunk
                e.buf = e.buf[n + 1:]
                if on_chunk:
                    on_chunk(chunk.decode(errors="replace"))
            elif line.startswith(b"DONE %d" % rid):
                e.last_stat = _stat(line)
                return out.decode(errors="replace")
            elif line.startswith(b"ERROR %d" % rid):
                raise RuntimeError(line.decode())


def _stat(line: bytes) -> dict:
    """colibri's DONE line: DONE <id> STAT <tokens> <tokens/s> <cache hit %> <rss GB> <prompt tokens> <length-limited>
    (the Polymath Engine's server writes the same). {} when the line has no STAT."""
    p = line.decode(errors="replace").split()
    try:
        i = p.index("STAT")
        return {"tokens": int(p[i + 1]), "tokens_per_second": float(p[i + 2]), "prompt_tokens": int(p[i + 5]),
                "length_limited": p[i + 6] == "1"}
    except (ValueError, IndexError):
        return {}


def pack_counters(e) -> dict:
    """What the engine says about its packs now: Polymath engines their routing counters (routings, won), the B-slot
    engine its knowledge experts and strength. {} when it says nothing."""
    try:
        f = getattr(e, "pack_stats", None)
        return f() if f else {}
    except Exception:                                    # noqa: BLE001 - information only
        return {}


def chat_prompt(question: str, history: list[dict] | None = None, system: str = "", thinking: bool = False) -> str:
    """Qwen chat format: system, earlier turns, the question, and an empty think block when thinking is off."""
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for h in history or []:
        role = "assistant" if h.get("role") == "assistant" else "user"
        s += f"<|im_start|>{role}\n{h.get('content', '')}<|im_end|>\n"
    s += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return s + ("" if thinking else "<think>\n\n</think>\n\n")


_TURN_END = re.compile(r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|\|\|\|IP_ADDRESS\|\|\||<\|user\|>|\n(?:user|assistant|system)\s*\n", re.I)
_ROLE_HEAD = re.compile(r"^\s*(?:<\|im_start\|>)?\s*(?:assistant|Assistant|ASSISTANT)\s*:?\s*\n?")


def split_answer(raw: str) -> tuple[str, str]:
    """(answer, thinking). Removes chat-format debris the engine can print as plain text: a leading "assistant" role
    line, <|im_start|>/<|im_end|> markers, and any further turns ("user" / "assistant" lines) the model ran on into."""
    s = raw or ""
    think = ""
    m = re.search(r"<think>(.*?)</think>", s, re.S)
    if m:
        think = m.group(1).strip()
        s = s[:m.start()] + s[m.end():]
    elif "</think>" in s:                                # the prompt opened the think block; the answer closes it
        think, s = s.split("</think>", 1)
        think = think.replace("<think>", "").strip()
    s = _ROLE_HEAD.sub("", s, count=1)
    s = _TURN_END.split(s, maxsplit=1)[0]
    return s.strip(), think


def build_prompt(question: str, template: str = "qa", domain: str = "this field", history=None, system: str = "",
                 thinking: bool = False) -> str:
    if template == "chat":
        return chat_prompt(question, history, system, thinking)
    return TEMPLATES.get(template, template).replace("{q}", question).replace("{domain}", domain)


def _answer(e, label: str, pack: str | None, prompt: str, max_tokens: int, temperature=None, top_p=None, on_chunk=None) -> dict:
    sw = e.swap(pack)
    if not sw["ok"]:
        return {"label": label, "answer": "", "error": "swap refused: " + " | ".join(sw["engine_log"][-3:])}
    t0 = time.time()
    before = pack_counters(e)
    e.last_stat = {}
    raw = engine_ask(e, prompt, max_tokens, temperature, top_p, on_chunk)
    after = pack_counters(e)
    ans, think = split_answer(raw)
    r = {"label": label, "answer": ans, "thinking": think, "swap_seconds": sw["seconds"], "seconds": round(time.time() - t0, 2)}
    st = getattr(e, "last_stat", {}) or {}
    r.update({k: st[k] for k in ("tokens", "tokens_per_second", "prompt_tokens", "length_limited") if k in st})
    from packsmith.workbench import dials
    info = {"engine": SETTINGS.engine, "build": getattr(e, "build", ""), "device": "",
            "packs": [{"name": n, **STATE.get("active_dials", {}).get(n, dials.DEFAULT)} for n in STATE.get("active_names") or []],
            "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    try:
        info["device"] = (e.device() or "")[:240]
    except Exception:                                    # noqa: BLE001
        pass
    if "routings" in after:                              # Polymath engines: places the packs' experts took
        dt = max(0, int(after.get("routings", 0)) - int(before.get("routings", 0)))
        dw = max(0, int(after.get("won", 0)) - int(before.get("won", 0)))
        info.update(routings=dt, won=dw, share=round(dw / dt, 4) if dt else None)
        if "chain" in after:
            info["chain"] = bool(after["chain"])
    for k in ("knowledge_experts", "strength", "budget"):
        if k in after:
            info[k] = after[k]
    r["info"] = info
    warn = [f"{n}: {WARNINGS[n]}" for n in label.split(" + ") if n in WARNINGS]
    if warn:
        r["warning"] = " | ".join(warn)
    return r


def ask(question: str, packs: list[str], mode: str = "single", template: str = "qa", domain: str = "this field",
        max_tokens: int = 1024, temperature: float | None = None, top_p: float | None = None, history=None,
        system: str = "", thinking: bool = False, on_chunk=None, histories: dict | None = None) -> dict:
    """histories: {label: turns}, one conversation per configuration ("base", a pack's name, "a + b" for packs
    loaded together), in single and compare mode alike: a configuration is only ever shown its own earlier answers,
    never the base the pack's or a pack the base's. A configuration with no thread yet starts a new one.
    history: the older single shared thread, used only when no histories are sent (API callers); in compare mode it
    would show every configuration one configuration's answers, so there each is asked without it."""
    from packsmith.system import gpu
    gpu.need_engine()                                # shared GPU: LM Studio out, engine in (started if needed)
    e = _eng()
    kw = {"temperature": temperature, "top_p": top_p}

    def p_for(label):
        if histories is not None:
            h = histories.get(label) or []
        else:
            h = [] if mode == "compare" else (history or [])
        return build_prompt(question, template, domain, h, system, thinking)
    prompt = p_for("base" if not packs else (packs[0] if len(packs) == 1 else " + ".join(packs)))
    warn = ""
    dup = same_domain(packs)
    if mode == "single" and dup:
        warn = f"packs for the same domain ({', '.join(dup)}) are not merged; answered one after another instead"
        mode = "compare"
    with _lock:
        if mode == "compare":
            res = [_answer(e, "base", None, p_for("base"), max_tokens, **kw)] + [
                _answer(e, n, pack_path(n), p_for(n), max_tokens, **kw) for n in packs]
        else:
            if not packs:
                res = [_answer(e, "base", None, p_for("base"), max_tokens, on_chunk=on_chunk, **kw)]
            elif len(packs) == 1:
                res = [_answer(e, packs[0], pack_path(packs[0]), p_for(packs[0]), max_tokens, on_chunk=on_chunk, **kw)]
            else:
                lab = " + ".join(packs)
                res = [_answer(e, lab, merged(packs), p_for(lab), max_tokens, on_chunk=on_chunk, **kw)]
    return {"prompt": prompt, "answers": res, "warning": warn}


def judge(question: str, answers: list[dict], model: str | None = None) -> str:
    from packsmith.system import gpu
    with gpu.lmstudio_turn():                         # shared GPU: engine out → judge → LM Studio out → engine back
        return llm.chat(prompts.compare_prompt(question, [(a["label"], a["answer"]) for a in answers]), model=model, max_tokens=1200)


def add_test(topic_id: str, question: str, expected: str) -> None:
    from packsmith.datasets import datasets
    did = store.topic(topic_id).get("current_dataset")
    if did:
        datasets.apply_ops(did, [{"op": "add", "to": "tests", "item": {"question": question, "expected": expected,
                                                                          "check": "", "source": "workbench"}}], "test from workbench")
        datasets._mirror(topic_id)
        return
    p = store.topic_dir(topic_id) / "dataset" / "tests.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"question": question, "expected": expected, "check": "", "source": "workbench"}) + "\n")


def flag_miss(topic_id: str, question: str, answer: str, note: str) -> None:
    p = store.topic_dir(topic_id) / "notes.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "question": question, "answer": answer, "note": note}) + "\n")


def evaluate(pack: str, tests_file: str, limit: int = 40, template: str = "qa", judge_model: str | None = None,
             max_tokens: int = 200, parallel: int | None = None, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    from packsmith.system import gpu
    gpu.need_engine(lambda m: progress(0.0, m))
    e = _eng()
    tests = [json.loads(x) for x in Path(tests_file).read_text(encoding="utf-8").splitlines() if x.strip()][:limit]
    if not tests:
        raise ValueError("no tests")
    rows = []
    with _lock:
        for phase, p in (("base", None), (pack, pack_path(pack))):
            e.swap(p)
            for i, tcase in enumerate(tests):
                if cancelled():
                    break
                prompt = build_prompt(tcase["question"], template)
                ans = split_answer(engine_ask(e, prompt, max_tokens, 0.0, 1.0))[0]
                rows.append({"phase": phase, "i": i, "answer": ans})
                progress(0.5 * (len(rows)) / (2 * len(tests)), f"{phase}: {i + 1}/{len(tests)}")
        e.swap(None)
    def one(r):                                       # the engine answers one at a time; the judge runs in parallel
        tc = tests[r["i"]]
        j = llm.chat(prompts.judge_match_prompt(tc["question"], tc["expected"], r["answer"]), model=judge_model, max_tokens=200)
        v = (htmlconv.parse_items(j, "verdict") or ["miss"])[0].lower()
        sc = 1.0 if v.startswith("match") else 0.5 if v.startswith("partial") else 0.0
        return {**r, "verdict": v.split()[0] if v else "miss", "score": sc}
    with gpu.lmstudio_turn(log=lambda m: progress(0.5, m)):     # shared GPU: the judge gets the GPU, then the engine again
        judged = par.pmap(one, rows, parallel, lambda k, n, r: progress(0.5 + 0.5 * k / n, f"judged {k}/{n}"))
    out = {"base": [], "pack": []}
    for r in judged:
        if not par.failed(r):
            out["base" if r["phase"] == "base" else "pack"].append(r)
    by_i = {}
    for ph in ("base", "pack"):
        for r in out[ph]:
            by_i.setdefault(r["i"], {})[ph] = r
    pairs = []
    for i, tc in enumerate(tests):
        b, pk = by_i.get(i, {}).get("base", {}), by_i.get(i, {}).get("pack", {})
        pairs.append({"i": i, "question": tc["question"], "expected": tc["expected"], "source": tc.get("source", ""),
                      "base": b.get("answer", ""), "base_verdict": b.get("verdict", ""), "base_score": b.get("score"),
                      "pack": pk.get("answer", ""), "pack_verdict": pk.get("verdict", ""), "pack_score": pk.get("score"),
                      "change": ("better" if (pk.get("score") or 0) > (b.get("score") or 0) else
                                 "worse" if (pk.get("score") or 0) < (b.get("score") or 0) else "same")})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    res = {"pack": pack, "n": len(tests), "tests_file": str(tests_file), "time": time.strftime("%Y-%m-%d %H:%M"), "id": stamp,
           "base_score": round(sum(x["score"] for x in out["base"]) / max(1, len(out["base"])), 3),
           "pack_score": round(sum(x["score"] for x in out["pack"]) / max(1, len(out["pack"])), 3),
           "better": sum(p["change"] == "better" for p in pairs), "worse": sum(p["change"] == "worse" for p in pairs),
           "rows": pairs}
    pdir = Path(pack_path(pack))
    store.wj(store.data_dir() / "evaluations" / pack / f"{stamp}.json", res)
    card_p = pdir / "card.json"
    card = store.rj(card_p, {}) or {}
    card.setdefault("evaluations", []).append({k: v for k, v in res.items() if k != "rows"})
    if pdir.exists() and pdir.resolve().is_relative_to(store.packs_dir().resolve()):
        store.wj(card_p, card)
    return {k: v for k, v in res.items()}


def evaluations(pack: str) -> list[dict]:
    d = store.data_dir() / "evaluations" / pack
    out = [{k: v for k, v in (store.rj(p) or {}).items() if k != "rows"} for p in sorted(d.glob("*.json"), reverse=True)] if d.exists() else []
    return out


def evaluation(pack: str, eid: str) -> dict:
    return store.rj(store.data_dir() / "evaluations" / pack / f"{eid}.json") or {}


def _words(s: str) -> set:
    return set(re.findall(r"[a-z0-9]{4,}", s.lower()))


def reference_source(packs: list[str], topic: str = "") -> dict:
    """Where an answer is checked: the user pack a pack was built from (its statements and source text), else the
    topic it was trained on, else the topic chosen at the top of the page. {kind, name, label} or {}."""
    cards = {p["name"]: p.get("card") or {} for p in store.list_packs()}
    for n in packs:
        c = cards.get(n) or {}
        if c.get("kind") == "userpack" or c.get("userpack"):
            from packsmith.userpack import core as up
            slug = c.get("userpack") or c.get("name") or n
            for u in up.list_all():
                if slug in (u["slug"], u["name"], u.get("pack")) or n in (u.get("pack"), u["slug"]):
                    return {"kind": "userpack", "name": u["slug"], "label": f"user pack {u['name']}: its statements and source text"}
        if c.get("topic"):
            try:
                return {"kind": "topic", "name": c["topic"], "label": f"topic {store.topic(c['topic']).get('name', c['topic'])}"}
            except KeyError:
                pass
    if topic:
        try:
            return {"kind": "topic", "name": topic, "label": f"topic {store.topic(topic).get('name', topic)} (chosen at the top)"}
        except KeyError:
            pass
    return {}


def _userpack_refs(slug: str, words: set) -> list[tuple[int, str]]:
    from packsmith.userpack import core as up
    out = []
    try:
        s = up.spec(slug)
    except KeyError:
        return out
    st = up.statements(slug)
    for sec in s["sections"]:
        for y in (st.get(sec["id"]) or {}).get("items", []):
            if y.get("confirmed") is not False:
                out.append((len(words & _words(y["text"])) + 1, f"[{sec['title']}] {y['text']}"))
        text = up.section_text(slug, sec)                 # the source itself, paragraph by paragraph
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) > 30:
                out.append((len(words & _words(para)), f"[{sec['title']}, source] {para[:900]}"))
    return out


def source_refs(src: dict, words: set) -> list[tuple[int, str]]:
    """(score, line) reference lines of one source (reference_source): a user pack's statements and text, or a
    topic's fact book, captured data, playbook and pages, scored by the words they share with the question and answer."""
    ref: list[tuple[int, str]] = []
    if src.get("kind") == "userpack":
        ref = _userpack_refs(src["name"], words)
    elif src.get("kind") == "topic":
        topic = src["name"]
        d = store.topic_dir(topic)
        def score(s):
            return len(words & _words(s))
        for x in [json.loads(l) for l in store.rt(d / "factbook" / "entries.jsonl").splitlines() if l.strip()]:
            s = (x.get("question", "") + " " + x.get("answer", "") + " " + x.get("text", "")).strip()
            ref.append((score(s), f"[fact book] {x.get('subject', '')}: {s}"))
        for f in (d / "data" / "records.jsonl",):
            for x in [json.loads(l) for l in store.rt(f).splitlines() if l.strip()]:
                ref.append((score(x.get("text", "")), f"[p.{x.get('page')}] {x.get('text', '')}"))
        pb = htmlconv.parse_playbook(store.rt(d / "playbook.html"))
        for c in pb["checks"]:
            s = c["label"] + " " + " ".join(i["text"] for i in c["items"])
            ref.append((score(s), f"[playbook {c['id']}] {s}"))
        try:                                             # the text itself: the topic's pages, paragraph by paragraph
            t = store.topic(topic)
            for doc, r in store.topic_regions(t, ("train", "heldout")):
                if r["type"] in store.CONTEXT_TYPES:
                    continue
                for p_ in range(r["start"], r["end"] + 1):
                    for para in re.split(r"\n\s*\n", store.page_text(doc, p_)):
                        para = " ".join(para.split())
                        if len(para) > 40:
                            ref.append((score(para), f"[{store.doc_meta(doc)['name']} p.{p_}] {para[:900]}"))
        except (KeyError, OSError):
            pass
    return ref


def judge_answer(question: str, answer: str, ref: list[str], model: str | None = None) -> dict:
    """The LM Studio model's claim-by-claim check of an answer against reference lines: {html, verdict}."""
    from packsmith.system import gpu
    with gpu.lmstudio_turn():
        html = llm.chat(prompts.analyse_prompt(question, answer, ref), model=model, max_tokens=2500)
    m = re.search(r'data-verdict="([a-z_ ]+)"', html or "")
    return {"html": html, "verdict": (m.group(1).strip() if m else "")}


def analyse(question: str, answer: str, topic: str = "", model: str | None = None, packs: list[str] | None = None) -> dict:
    """The LM Studio model checks an answer against the material the pack was built from: a user pack's statements
    and source text, or the topic's fact book, captured data and playbook (the most related entries). A claim the
    material does not cover is "not covered", not wrong; only what the material contradicts is wrong."""
    src = reference_source(list(packs or []), topic)
    ref = source_refs(src, _words(question + " " + answer))
    ref = [r for sc, r in sorted(ref, key=lambda x: -x[0]) if sc > 0][:18]
    j = judge_answer(question, answer, ref, model)
    return {**j, "source": src.get("label", ""), "references": len(ref)}
