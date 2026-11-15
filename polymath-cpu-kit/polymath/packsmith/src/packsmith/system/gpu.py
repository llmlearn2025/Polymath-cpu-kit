"""GPU selection, and one GPU shared by the engine and a local LLM server.

Selection (Settings → GPU): the engine and training run as child processes whose CUDA_VISIBLE_DEVICES is set to the
chosen card (gpu_engine / gpu_train: auto | cpu | <index> | <uuid>; auto = the card with the most free memory).
The LLM server (LM Studio, Ollama, ...) runs on its own; gpu_llm only tells Packsmith where it is.

Sharing is worked out, not set: it is on when the LLM is local on the engine's card (gpu_llm = same), the engine is on
a GPU, and the server can unload models (LM Studio, Ollama). Then only one of them holds VRAM at a time:
  * before an LLM call (mining, generation, judging, analysis, the assistant), the engine is stopped and the
    models Packsmith unloaded earlier are loaded again (with the context they had);
  * before an engine call (Workbench answers, evaluation answers), the LLM's models are unloaded (remembered) and
    the engine is started;
  * a judge turn does exactly stop engine → judge with the LLM → unload the LLM → start the engine again.
Training takes the GPU for itself (train.py).
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time

from packsmith.system.config import SETTINGS

_lock = threading.RLock()
STATE = {"holder": "", "since": 0.0, "note": ""}          # "engine" | "lmstudio" (the LLM server) | ""


# ---------------------------------------------------------------- cards
_CARDS: dict = {"t": 0.0, "list": []}


def cards(fresh: bool = False) -> list[dict]:
    """[{index, uuid, name, memory_total_mb, memory_used_mb, memory_free_mb}] from nvidia-smi (cached 5 s); [] without one."""
    if not fresh and time.time() - _CARDS["t"] < 5:
        return _CARDS["list"]
    out = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.strip().splitlines():
                i, u, n, tot, used, free = [x.strip() for x in line.split(",")]
                out.append({"index": int(i), "uuid": u, "name": n, "memory_total_mb": int(float(tot)),
                            "memory_used_mb": int(float(used)), "memory_free_mb": int(float(free))})
        except (OSError, ValueError, subprocess.SubprocessError):
            out = []
    _CARDS.update(t=time.time(), list=out)
    return out


def resolve(choice: str) -> str | None:
    """A setting (auto | cpu | index | uuid) → the CUDA_VISIBLE_DEVICES value, or None for the CPU. Inside Docker the
    container sees only the card it was given, so auto picks it."""
    c = str(choice or "auto").strip().lower()
    if c == "cpu":
        return None
    cs = cards()
    if not cs:                                   # no nvidia-smi: trust an explicit card, else no GPU
        return c if c != "auto" else None
    if c == "auto":
        return str(max(cs, key=lambda x: x["memory_free_mb"])["index"])
    for x in cs:
        if c == str(x["index"]) or c == x["uuid"].lower():
            return str(x["index"]) if c.isdigit() else x["uuid"]
    return str(max(cs, key=lambda x: x["memory_free_mb"])["index"])   # the chosen card is gone: fall back to auto


def engine_device() -> str | None:
    return resolve(SETTINGS.gpu_engine)


def train_device() -> str | None:
    t = str(SETTINGS.gpu_train or "same").lower()
    return engine_device() if t == "same" else resolve(t)


def engine_env() -> tuple[dict, bool]:
    """(extra environment, on GPU?) for the engine process."""
    d = engine_device()
    return ({"CUDA_VISIBLE_DEVICES": d} if d is not None else {}), d is not None


def train_env() -> tuple[dict, str]:
    """(extra environment, --device) for the training process."""
    d = train_device()
    return ({"CUDA_VISIBLE_DEVICES": d}, "cuda") if d is not None else ({}, "cpu")


def shared() -> bool:
    """Engine and LLM take turns on one card: LLM on the engine's card, engine on a GPU, and the server can unload."""
    from packsmith.llm import llm
    if str(SETTINGS.gpu_llm or "same").lower() != "same" or engine_device() is None:
        return False
    return llm.can_manage_models()


def _store():
    from packsmith.packs import store
    return store


def _remembered() -> list:
    return _store().rj(_store().data_dir() / "cache" / "lmstudio_unloaded.json", []) or []


def _remember(loaded: list) -> None:
    if loaded:
        _store().wj(_store().data_dir() / "cache" / "lmstudio_unloaded.json", loaded)


def _set(holder: str, note: str = "") -> None:
    STATE.update(holder=holder, since=time.time(), note=note)


def _engine_running() -> bool:
    from packsmith.workbench import workbench
    e = workbench.STATE.get("engine")
    return e is not None and e.alive()


def need_lmstudio(log=lambda m: None) -> None:
    """Called before every LLM call. Cheap when the GPU is not shared or the LLM server already holds it."""
    if not shared():
        return
    if STATE["holder"] == "lmstudio" and not _engine_running():
        return
    with _lock:
        from packsmith.llm import llm
        from packsmith.workbench import workbench
        if _engine_running():
            log("shared GPU: stopping the engine for the LLM server")
            workbench.stop()
        want = _remembered()
        if want:
            loaded = {m["key"] for m in llm.instances() for _ in m["loaded"]}
            for key, ctx in want:
                if key not in loaded:
                    try:
                        log(f"shared GPU: loading {key} in the LLM server")
                        llm.load(key, ctx)
                    except llm.LLMError as e:
                        log(f"shared GPU: could not load {key}: {e}")
        _set("lmstudio")


def need_engine(log=lambda m: None) -> None:
    """Called before the engine answers. Unloads the LLM server's models (remembering them) and starts the engine."""
    from packsmith.llm import llm
    from packsmith.workbench import workbench
    with _lock:
        if shared():
            loaded = [(m["key"], i["context_length"]) for m in llm.instances() for i in m["loaded"]]
            if loaded:
                log("shared GPU: unloading the LLM server for the engine")
                _remember(loaded)
                llm.unload_all()
        if not _engine_running():
            log("starting the engine")
            st = workbench.start()
            if st["status"] != "ready":
                raise RuntimeError(f"the engine did not start: {st.get('error') or st['status']}")
        _set("engine")


@contextlib.contextmanager
def lmstudio_turn(restore_engine: bool = True, log=lambda m: None):
    """Stop the engine → (LLM work) → unload the LLM → start the engine again (shared GPU only)."""
    if not shared():
        yield
        return
    was = _engine_running()
    need_lmstudio(log)
    try:
        yield
    finally:
        if restore_engine and was:
            need_engine(log)


def status() -> dict:
    return {"shared": shared(), "holder": STATE["holder"] or ("engine" if _engine_running() else ""),
            "engine_running": _engine_running(), "remembered_lmstudio": [k for k, _ in _remembered()],
            "cards": cards(), "engine_device": engine_device(), "train_device": train_device()}
