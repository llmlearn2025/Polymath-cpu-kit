"""Per-pack dials, set in the Workbench page only. Every load of a pack uses them, whoever asks for it (the page, an
agent through REST, MCP or Hermes, validation, an evaluation): agents cannot set them.

    strength  how strongly the pack speaks: its experts' output times this. 1 = as trained, 0 = silent.
              All engines (patched colibri folds it into the pack's row scales when it reads the pack).
    shift     how often the pack steps in: added to its router logits (Polymath engines, route packs only).
              0 = as trained (or as router tuning set it); +1 makes the pack about e (2.7) times likelier to win a
              place on a token, -1 about e times less likely. It never changes how strongly the pack speaks.

The two are independent on purpose: a shift moves which tokens the pack takes, a strength only its weight there, so
tuned router rows stay valid at any strength. The values are kept in <data>/pack_dials.json, by pack name."""
from __future__ import annotations

import json
import math
import threading

from packsmith.packs import store

STRENGTH_MIN, STRENGTH_MAX = 0.0, 1.5
SHIFT_MIN, SHIFT_MAX = -3.0, 3.0
DEFAULT = {"strength": 1.0, "shift": 0.0}
_lock = threading.Lock()


def _file():
    return store.data_dir() / "pack_dials.json"


def all_dials() -> dict[str, dict]:
    try:
        d = json.loads(_file().read_text(encoding="utf-8"))
        return {k: {**DEFAULT, **v} for k, v in d.items() if isinstance(v, dict)}
    except (OSError, ValueError):
        return {}


def get(name: str) -> dict:
    return dict(all_dials().get(name, DEFAULT))


def is_default(d: dict) -> bool:
    return float(d.get("strength", 1.0)) == 1.0 and float(d.get("shift", 0.0)) == 0.0


def _num(v, lo: float, hi: float, what: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number") from None
    if not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f"{what} must be between {lo:g} and {hi:g}")
    return round(x, 3)


def set_dials(name: str, strength=None, shift=None) -> dict:
    """Set one pack's dials (None = unchanged). Back at the defaults the entry is removed."""
    with _lock:
        d = all_dials()
        cur = {**DEFAULT, **d.get(name, {})}
        if strength is not None:
            cur["strength"] = _num(strength, STRENGTH_MIN, STRENGTH_MAX, "strength")
        if shift is not None:
            cur["shift"] = _num(shift, SHIFT_MIN, SHIFT_MAX, "shift")
        if is_default(cur):
            d.pop(name, None)
        else:
            d[name] = cur
        f = _file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(f)
        return cur


def describe(d: dict) -> str:
    """'strength 60%, shift -1' / '' at the defaults."""
    out = []
    if float(d.get("strength", 1.0)) != 1.0:
        out.append(f"strength {round(float(d['strength']) * 100)}%")
    if float(d.get("shift", 0.0)) != 0.0:
        out.append(f"shift {float(d['shift']):+g}")
    return ", ".join(out)
