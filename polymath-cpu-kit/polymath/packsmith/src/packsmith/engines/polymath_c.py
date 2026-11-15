"""polymath-c adapter: Polymath packs (polymath-experts-v1, anchored routing) served by colibri in C.

The same colibri binaries as patched-colibri (install_patched_colibri.sh builds both pack hooks into them): the
B-slot overlay hooks answer OLM_OVERLAY_CTL + SIGHUP, the Polymath hooks (engines/polymath-c/files/pm_packs.h)
answer PM_PACKS_CTL + SIGUSR2. So one install gives three engines to compare on the same machine:

    patched-colibri   B-slot packs (overlay.json), C
    polymath-engine   Polymath packs, PyTorch (the reference; also the trainer for both Polymath engines)
    polymath-c        Polymath packs, C: the same pack folders as polymath-engine, the same answers (parity ~1e-5)

A swap writes {"packs": [...], "budget": N} into the control file, sends SIGUSR2 and runs a one-token request; the
engine applies the swap at the start of that forward pass and logs one [POLYMATH] line (active / off / refused).

GPU, two engines:
    polymath-c      the CUDA chain off (COLI_CUDA_CHAIN=0): the routed experts on the card (colibri's expert tier), the
                    attention, the router and the pack experts on the CPU threads. The reference for parity.
    polymath-c-gpu  (EngineGPU) a GPU build with pm-chain-v1: the CUDA chain on. Every layer runs on the card; a layer
                    with pack experts takes the chain's host step (its routing merge and pack experts on the CPU, its
                    base experts on the card's tier), the other layers route on the card. Qwen3-MoE only: OLMoE's
                    chain has no host step, so OLMoE runs as polymath-c.
Standard library only.
"""
from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from pathlib import Path

from packsmith.engines import EngineNotAvailable
from packsmith.engines.colibri.engine import Engine as _Colibri, binary_for, builds as _builds
from packsmith.engines.polymath import (PACK_FORMAT, STACK_FORMAT, describe, is_pack, manifest, merge,  # noqa: F401
                                        verify)
from packsmith.system.home import expand

ENGINE = "polymath-c"
MARK = b"polymath-experts-v1"              # in a binary built with pm_packs.h
STATUS_MARK = b"PM_STATUS"                 # ... of this version (swaps confirmed through a status file)
CHAIN_MARK = b"pm-chain-v1"                # ... whose CUDA chain runs Polymath packs (polymath-c-gpu)
_seen: dict = {}


def _has(binary, mark: bytes) -> bool:
    try:
        return mark in Path(binary).read_bytes()
    except OSError:
        return False


def supports(binary: Path) -> bool:
    """True when this colibri binary has the polymath-c hooks (cached by path and mtime)."""
    try:
        st = binary.stat()
    except OSError:
        return False
    key = (str(binary), st.st_mtime_ns)
    if key not in _seen:
        try:
            raw = binary.read_bytes()
            _seen[key] = MARK in raw and STATUS_MARK in raw
        except OSError:
            _seen[key] = False
    return _seen[key]


def builds(colibri: str) -> list[str]:
    """The installed colibri builds (cpu / gpu / single) whose binaries carry the polymath-c hooks."""
    out = []
    for b in _builds(colibri):
        bins = [binary_for(colibri, e, b) for e in ("olmoe", "qwen36")]
        if any(x.exists() and supports(x) for x in bins):
            out.append(b)
    return out


def installed(colibri: str) -> bool:
    return bool(builds(colibri))


class Engine(_Colibri):
    """Same surface as the colibri Engine (ask / swap / device / alive / close); swap takes a Polymath pack or stack."""

    def __init__(self, colibri: str, model: str, engine: str = "olmoe", gpu: bool = True, threads: int = 0,
                 cap: int = 0, extra_env: dict | None = None, timeout: float = 1800, build: str = "",
                 budget: int | None = None):
        b = binary_for(colibri, engine, build or ("gpu" if gpu else "cpu"))
        if b.exists() and not supports(b):
            raise EngineNotAvailable(f"{b} has no polymath-c hooks of this version (an older build, or none): re-run "
                                     "./install_patched_colibri.sh (it builds polymath-c into the same binaries)")
        env = dict(extra_env or {})
        self.budget = int(budget if budget is not None else env.get("POLYMATH_PACK_BUDGET", "2"))
        self.pm_ctl = tempfile.NamedTemporaryFile(prefix="polymath_c_ctl_", delete=False).name
        Path(self.pm_ctl).write_text(json.dumps({"packs": [], "budget": self.budget}))
        self.pm_status = self.pm_ctl + ".status"
        self.chain = bool(getattr(self, "CHAIN", False)) and engine == "qwen36" and (build or ("gpu" if gpu else "cpu")) == "gpu"
        env.update(PM_PACKS_CTL=self.pm_ctl, PM_STATUS=self.pm_status, PM_BUDGET=str(self.budget), PM_PACKS="")
        if not self.chain:
            env["COLI_CUDA_CHAIN"] = "0"
        super().__init__(colibri, model, engine, gpu=gpu, threads=threads, cap=cap, extra_env=env, timeout=timeout,
                         build=build)
        self.active = None
        self.strengths: list[float] = []
        self.shifts: list[float] = []
        if not Path(self.pm_status).exists():         # this version writes it at start (request 0)
            self.close()
            raise EngineNotAvailable(f"{self.binary} wrote no polymath-c status file: an older build. Re-run "
                                     "./install_patched_colibri.sh")
        refused = [l for l in self.log if "[POLYMATH]" in l and "packs refused" in l]
        if refused:
            self.close()
            raise EngineNotAvailable("polymath-c cannot run packs on this path: " + refused[0])

    def _status(self) -> dict:
        try:
            return json.loads(Path(self.pm_status).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _control(self, packs: list[str], strengths: list[float] | None, budget: int,
                 shifts: list[float] | None = None) -> dict:
        """Write the control file, signal, and run a one-token request (the engine applies a swap at the start of a
        forward pass). The result is read from the engine's status file (PM_STATUS): it counts the requests it has
        read, so the answer is this request's, whatever the log thread is doing."""
        n0 = len(self.log)
        t0 = time.time()
        before = self._status().get("request", -1)
        Path(self.pm_ctl).write_text(json.dumps({"packs": packs, "strengths": strengths or [1.0] * len(packs),
                                                 "shifts": shifts or [0.0] * len(packs), "budget": int(budget)}),
                                     encoding="utf-8")
        os.kill(self.p.pid, signal.SIGUSR2)
        time.sleep(0.02)
        self.ask("Hello", 1)                       # a forward pass: the engine applies the swap at its start
        st = self._status()
        while st.get("request", -1) == before and time.time() - t0 < 30:
            time.sleep(0.02)
            st = self._status()
        ok = st.get("request", -1) != before and bool(st.get("ok"))
        time.sleep(0.05)                           # the log lines are for people; the result above is the status
        lines = [l for l in self.log[n0:] if "[POLYMATH]" in l]
        return {"ok": ok, "seconds": round(time.time() - t0, 3), "engine_log": lines, "status": st}

    def swap(self, pack, strengths: list[float] | None = None, shifts: list[float] | None = None) -> dict:
        """Make `pack` (a Polymath pack or stack folder, or a list of them) the resident set, or none. strengths /
        shifts: one per entry (the Workbench dials). A refused pack changes nothing: the previous set stays."""
        packs = [str(Path(p).resolve()) for p in (pack if isinstance(pack, (list, tuple)) else [pack] if pack else [])]
        if shifts and any(shifts) and not _has(self.binary, b"routing shift"):
            return {"ok": False, "active": self.active, "seconds": 0, "status": {},
                    "engine_log": ["[POLYMATH] refused: this polymath-c build has no routing shift (an older build): "
                                   "re-run ./install_patched_colibri.sh, or set the pack's routing back to 0"]}
        r = self._control(packs, strengths, self.budget, shifts)
        if r["ok"]:
            self.active = packs[0] if len(packs) == 1 else (packs or None)
            self.strengths = strengths or [1.0] * len(packs)
            self.shifts = shifts or [0.0] * len(packs)
        return {**r, "active": self.active}

    def set_budget(self, budget: int) -> dict:
        """Most pack experts per token in a layer (all packs together); 0 = packs resident but never chosen."""
        packs = [self.active] if isinstance(self.active, str) else list(self.active or [])
        r = self._control(packs, self.strengths or None, budget, self.shifts or None)
        if r["ok"]:
            self.budget = int(budget)
        return r

    def share(self) -> str:
        """The last share line the engine logged (written at each swap, for the set before it)."""
        for l in reversed(self.log):
            if "the previous set won" in l:
                return l.split("the previous set won", 1)[1].strip()
        return ""

    def pack_stats(self) -> dict:
        """The status file's counters: routings / won (since the last swap), strengths, shifts, chain, budget."""
        st = self._status()
        return {k: st[k] for k in ("routings", "won", "strengths", "shifts", "chain", "budget") if k in st}

    def chain_on(self) -> bool:
        """The CUDA chain came up with Polymath packs on it (the engine logs it at start)."""
        return self.chain and any("CUDA chain on" in l and "[POLYMATH]" in l for l in self.log)

    def device(self) -> str:
        base = super().device()
        if self.chain_on():
            return (f"{base} · polymath-c(gpu): CUDA chain on, layers with pack experts route on the host · "
                    f"pack budget {self.budget}/token")
        if self.chain:
            return f"{base} · polymath-c(gpu): the CUDA chain did not come up (see the log): as polymath-c · pack budget {self.budget}/token"
        return f"{base} · polymath-c: routing and pack experts on the CPU · pack budget {self.budget}/token"

    def close(self):
        super().close()
        try:
            os.unlink(self.pm_ctl)
            os.unlink(self.pm_status)
        except OSError:
            pass


class EngineGPU(Engine):
    """polymath-c-gpu: the GPU build with the CUDA chain on (pm-chain-v1). Refuses to start without that build."""
    CHAIN = True

    def __init__(self, colibri: str, model: str, engine: str = "olmoe", gpu: bool = True, threads: int = 0,
                 cap: int = 0, extra_env: dict | None = None, timeout: float = 1800, build: str = "",
                 budget: int | None = None):
        b = binary_for(colibri, engine, "gpu")
        if not b.exists() or not _has(b, CHAIN_MARK):
            raise EngineNotAvailable(f"polymath-c-gpu needs the GPU build with the CUDA chain port ({b}): run ./run.sh "
                                     "(or ./install_patched_colibri.sh --both) on a machine with the CUDA toolkit")
        super().__init__(colibri, model, engine, gpu=True, threads=threads, cap=cap, extra_env=extra_env,
                         timeout=timeout, build="gpu", budget=budget)


def gpu_builds(colibri: str) -> list[str]:
    """["gpu"] when the GPU build carries the CUDA chain port."""
    b = binary_for(colibri, "qwen36", "gpu")
    return ["gpu"] if b.exists() and supports(b) and _has(b, CHAIN_MARK) else []


def hf_model(model: str) -> str:
    """polymath-c serves a colibri container (the same model folder patched-colibri serves)."""
    p = expand(model).resolve()
    if not ((p / "qwen36_meta.json").exists() or (p / "config.json").exists()):
        raise EngineNotAvailable(f"{p} is not a colibri model container")
    return str(p)
