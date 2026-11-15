"""Inference engines Packsmith can drive. Each engine is a pair (Engine class, packs module):

  Engine(colibri_dir, model, kind, gpu=..., extra_env=...)   start the server process for a base model
    .generate(...) / .swap(...) / .close() / .alive() / .log   (see engines/colibri/engine.py)
  packs: manifest(pack) / info(pack) / merge(...)               the pack format the engine reads

The pack manifest (packsmith.packs.manifest) records which engine a pack was made for; Packsmith refuses a pack whose
engine differs from the one in Settings.
"""
from __future__ import annotations

NAMES = {
    "patched-colibri": "Patched colibri (current engine: knowledge packs as gated extra experts)",
    "polymath-engine": "Polymath Engine (packs native: router-row or gated experts in every MoE layer, stackable)",
    "polymath-c": "Polymath in C (the same Polymath packs, served by colibri: the C port of the Polymath Engine)",
    "polymath-c-gpu": "Polymath in C on the GPU (polymath-c with colibri's CUDA chain: every layer on the card, "
                      "pack layers route on the host; Qwen3-MoE)",
}
# Engines that read Polymath packs (polymath-experts-v1). They share packs, the trainer (polymath_engine) and stacks.
POLYMATH = ("polymath-engine", "polymath-c", "polymath-c-gpu")
# Engines that run colibri binaries (and so have cpu / gpu builds; polymath-c-gpu only the gpu one).
COLIBRI = ("patched-colibri", "polymath-c", "polymath-c-gpu")


def is_polymath(name: str) -> bool:
    return (name or "").strip() in POLYMATH


def uses_colibri(name: str) -> bool:
    return (name or "patched-colibri").strip() in COLIBRI


def pack_engine(name: str) -> str:
    """The engine name a pack made for `name` records: Polymath packs are made by the reference engine's trainer."""
    return "polymath-engine" if is_polymath(name) else (name or "patched-colibri")


class EngineNotAvailable(RuntimeError):
    pass


def get(name: str):
    name = (name or "patched-colibri").strip()
    if name == "patched-colibri":
        from packsmith.engines.colibri import packs
        from packsmith.engines.colibri.engine import Engine
        return Engine, packs
    if name == "polymath-engine":
        from packsmith.engines import polymath
        return polymath.Engine, polymath
    if name == "polymath-c":
        from packsmith.engines import polymath_c
        return polymath_c.Engine, polymath_c
    if name == "polymath-c-gpu":
        from packsmith.engines import polymath_c
        return polymath_c.EngineGPU, polymath_c
    raise EngineNotAvailable(f"unknown engine {name!r} (known: {', '.join(NAMES)})")


def available() -> list[dict]:
    from packsmith.engines import polymath, polymath_c
    from packsmith.engines.colibri.engine import builds
    from packsmith.system.config import SETTINGS
    have = builds(str(SETTINGS.path("colibri_dir")))
    return [{"name": "patched-colibri", "label": NAMES["patched-colibri"], "installed": bool(have), "builds": have},
            {"name": "polymath-engine", "label": NAMES["polymath-engine"], "installed": polymath.installed()},
            {"name": "polymath-c", "label": NAMES["polymath-c"], "installed": bool(pc := polymath_c.builds(str(SETTINGS.path("colibri_dir")))),
             "builds": pc},
            {"name": "polymath-c-gpu", "label": NAMES["polymath-c-gpu"],
             "installed": bool(pg := polymath_c.gpu_builds(str(SETTINGS.path("colibri_dir")))), "builds": pg}]
