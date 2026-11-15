"""Polymath Engine adapter: the engine process (python -m polymath_engine serve), driven like colibri.

The engine speaks colibri's stdio framing (SUBMIT / DATA / DONE), so the Workbench's request code is shared, plus a
PACKS command for swaps: the engine answers it directly, so a swap needs no control file, no signal and no probe
request. Packs are polymath-experts-v1 folders (experts.safetensors + pack.json); several packs are used together
natively (a "merged" pack is only a stack file that lists them).

The engine package lives in engines/polymath-engine of the repo and is installed into the venv by
install_polymath_engine.sh. It needs PyTorch and transformers (the training extras).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from packsmith.engines import EngineNotAvailable
from packsmith.engines.colibri.engine import READY, Engine as _ColibriPlumbing
from packsmith.system.home import expand

PACK_FORMAT = "polymath-experts-v1"
STACK_FORMAT = "polymath-stack-v1"


def installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("polymath_engine") is not None and importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


class Engine(_ColibriPlumbing):
    """Same surface as the colibri Engine: ask / swap / device / alive / close, plus .p .buf .lock .next_id for
    workbench.engine_ask. `colibri` (the first argument) is unused: the engine is a Python package in the venv."""

    def __init__(self, colibri: str, model: str, engine: str = "olmoe", gpu: bool = True, threads: int = 0,
                 cap: int = 0, extra_env: dict | None = None, timeout: float = 1800, budget: int | None = None):
        if not installed():
            raise EngineNotAvailable("the Polymath Engine is not installed: run ./install_polymath_engine.sh")
        self.engine = engine
        self.model = hf_model(model)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads or os.cpu_count() or 8))
        env.update(extra_env or {})
        budget = budget if budget is not None else int(env.get("POLYMATH_PACK_BUDGET", "2"))
        cmd = [sys.executable, "-m", "polymath_engine", "serve", "--model", self.model,
               "--device", "cuda" if gpu else "cpu", "--budget", str(budget)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        self.ctl = ""
        self.buf, self.log, self.next_id = b"", [], 1
        self.active: str | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._err, daemon=True).start()
        self._read_until(READY, timeout)

    def _request(self, head: bytes, body: bytes | None = None) -> dict:
        with self.lock:
            rid = self.next_id
            self.next_id += 1
            line = head % rid
            self.p.stdin.write(line + (b" %d\n" % len(body) + body + b"\n" if body is not None else b"\n"))
            self.p.stdin.flush()
            while True:
                ln = self._line()
                if ln.startswith(b"RESULT %d " % rid):
                    n = int(ln.split()[2])
                    while len(self.buf) < n + 1:
                        self._fill(600)
                    data, self.buf = self.buf[:n], self.buf[n + 1:]
                    return json.loads(data)
                if ln.startswith(b"ERROR %d" % rid):
                    raise RuntimeError(ln.decode(errors="replace"))

    def swap(self, pack, strengths: list[float] | None = None, shifts: list[float] | None = None) -> dict:
        """Make `pack` (a pack or stack folder, or a list of them) the resident set, or none. strengths / shifts:
        one per entry (the Workbench dials). The engine refuses a pack that does not fit and then keeps the packs it
        had."""
        import time
        t0 = time.time()
        packs = [str(Path(p).resolve()) for p in (pack if isinstance(pack, (list, tuple)) else [pack] if pack else [])]
        req = {"packs": packs}
        if strengths:
            req["strengths"] = [float(x) for x in strengths]
        if shifts:
            req["shifts"] = [float(x) for x in shifts]
        r = self._request(b"PACKS %d", json.dumps(req).encode())
        ok = bool(r.get("ok"))
        if ok:
            self.active = packs[0] if len(packs) == 1 else (packs or None)
        lines = [l for l in self.log[-20:] if "[POLYMATH]" in l]
        if not ok:
            lines.append("[POLYMATH] refused: " + str(r.get("error", "")))
        return {"ok": ok, "active": self.active, "seconds": round(time.time() - t0, 3), "engine_log": lines,
                "warnings": r.get("warnings", [])}

    def status(self) -> dict:
        return self._request(b"STATUS %d")

    def pack_stats(self) -> dict:
        st = self.status()
        return {k: st[k] for k in ("routings", "won", "strengths", "shifts", "budget") if k in st}

    def device(self) -> str:
        try:
            st = self.status()
            return f"{st.get('device')} {st.get('memory', '')}".strip() + f" · pack budget {st.get('budget')}/token"
        except Exception:                                # noqa: BLE001
            return "unknown"

    def close(self):
        try:
            self.p.kill()
        except Exception:                                # noqa: BLE001
            pass


def hf_model(model: str) -> str:
    """The model folder to serve: a Hugging Face folder (config.json + *.safetensors), or a colibri Qwen3-MoE
    container (int4 experts, which the engine keeps packed on the card: Qwen3-30B-A3B in about 19 GB). Anything else
    (for example a colibri OLMoE container) falls back to the training model."""
    if str(model).startswith("tiny"):
        return str(model)

    def usable(p: Path) -> bool:
        if (p / "qwen36_meta.json").exists():
            return (p / "config.hf.json").exists()
        return (p / "config.json").exists() and any(p.glob("*.safetensors"))
    p = expand(model).resolve()
    if usable(p):
        return str(p)
    from packsmith.system.config import SETTINGS
    t = expand(SETTINGS.base_train_model).resolve() if SETTINGS.base_train_model else None
    if t and usable(t):
        return str(t)
    raise EngineNotAvailable(f"{p} is neither a Hugging Face model folder nor a colibri Qwen3-MoE container. Set "
                             "Settings → Paths → Base model for the engine to one of them.")


# ---------------------------------------------------------------------- packs on disk (no torch needed)
def _rj(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def is_pack(folder: Path) -> bool:
    m = _rj(Path(folder) / "pack.json")
    return (m.get("engine") or {}).get("pack_format") == PACK_FORMAT or m.get("format") == STACK_FORMAT


def manifest(pack) -> dict:
    m = _rj(Path(pack) / "pack.json")
    if not m:
        raise EngineNotAvailable(f"{pack}: no pack.json")
    return m


def expand_packs(pack, depth: int = 0) -> list[Path]:
    """The pack folders a pack or stack expands to, in the engines' order (nested stacks included; a relative member is
    relative to its stack folder): what tuned router rows and their data files are numbered by."""
    p = Path(pack)
    m = manifest(p)
    if m.get("format") != STACK_FORMAT:
        return [p]
    if depth > 8:
        raise EngineNotAvailable(f"{p}: stacks nested too deep")
    out = []
    for sp in m.get("packs") or []:
        c = Path(sp).expanduser()
        out += expand_packs(c if c.is_absolute() else p / c, depth + 1)
    return out


def describe(pack) -> dict:
    m = manifest(pack)
    if m.get("format") == STACK_FORMAT:
        return {"path": str(Path(pack).resolve()), "format": STACK_FORMAT, "packs": m.get("packs", [])}
    lay = (m.get("polymath") or {}).get("layers") or {}
    return {"path": str(Path(pack).resolve()), "format": PACK_FORMAT, "created": (m.get("trained") or {}).get("date", ""),
            "knowledge_experts": sum(int(v.get("k", 0)) for v in lay.values()), "layers": sorted(int(l) for l in lay),
            "modes": sorted({v.get("mode") for v in lay.values()}), "base_fingerprint": (m.get("base_model") or {}).get("arch_fingerprint")}


def merge(packs: list, out, source: str = "") -> Path:
    """Packs used together: the engine stacks them natively, so this writes a stack file (no weights are copied)."""
    packs = [Path(p) for p in packs]
    fps = {(manifest(p).get("base_model") or {}).get("arch_fingerprint") for p in packs}
    if len(fps) != 1:
        raise ValueError(f"packs were built for different models: {sorted(map(str, fps))}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack.json").write_text(json.dumps({"format": STACK_FORMAT, "packs": [str(p.resolve()) for p in packs],
                                               "strengths": [1.0] * len(packs), "engine": {"name": "polymath-engine"},
                                               "source": source}, indent=1), encoding="utf-8")
    return out


def verify(pack) -> bool:
    import hashlib
    m = manifest(pack)
    if m.get("format") == STACK_FORMAT:
        return all(verify(p) for p in m.get("packs", []))
    h = hashlib.sha256((Path(pack) / "experts.safetensors").read_bytes()).hexdigest()
    return h == (m.get("polymath") or {}).get("experts_sha256")
