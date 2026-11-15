"""One colibri engine process (olmoe or qwen36) held by the plugin: start, ask, swap packs live.

The engine runs colibri's serve protocol (SERVE=1: SUBMIT / DATA / DONE on stdin/stdout). A swap
writes the pack directory into the control file and sends SIGHUP; the engine applies it at the
start of its next forward pass, so `swap` then runs a one-token request and reads the engine's
own [OVERLAY] lines to confirm what is active. Standard library only.
"""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from packsmith.system.home import expand

READY = b"\x01\x01READY\x01\x01"


BUILDS = ("cpu", "gpu")


def builds(colibri: str) -> list[str]:
    """The builds installed side by side in c/polymath-bin/<cpu|gpu>/ (install_patched_colibri.sh --both).
    An older single build (only c/olmoe, c/qwen36) counts as "single"."""
    c = expand(colibri) / "c"
    have = [b for b in BUILDS if (c / "polymath-bin" / b / "qwen36").exists() or (c / "polymath-bin" / b / "olmoe").exists()]
    if not have and ((c / "qwen36").exists() or (c / "olmoe").exists()):
        have = ["single"]
    return have


def binary_for(colibri: str, engine: str, build: str) -> Path:
    c = expand(colibri) / "c"
    b = c / "polymath-bin" / build / engine
    return b if build in BUILDS and b.exists() else c / engine


class Engine:
    def __init__(self, colibri: str, model: str, engine: str = "olmoe", gpu: bool = True, threads: int = 0,
                 cap: int = 0, extra_env: dict | None = None, timeout: float = 1800, build: str = ""):
        self.engine, self.model = engine, str(expand(model).resolve())
        self.build = build or ("gpu" if gpu else "cpu")
        binary = binary_for(colibri, engine, self.build)
        if not binary.exists():
            raise FileNotFoundError(f"{binary} not found: build colibri (install_patched_colibri.sh)")
        self.binary = str(binary)
        self.ctl = tempfile.NamedTemporaryFile(prefix="coli_packs_ctl_", delete=False).name
        Path(self.ctl).write_text("\n")
        env = dict(os.environ, SERVE="1", SNAP=self.model, OLM_OVERLAY_CTL=self.ctl, COLI_NO_OMP_TUNE="1",
                   OMP_NUM_THREADS=str(threads or os.cpu_count() or 8))
        for k in ("OLM_OVERLAY", "COLI_OVERLAY", "COLI_OVERLAY_CTL"):
            env.pop(k, None)
        if gpu:
            env.update(COLI_CUDA="1", COLI_DENSE_F16="0")
            env.pop("COLI_CUDA_CHAIN", None)
        else:
            env.update(COLI_CUDA="0", COLI_CUDA_CHAIN="0")
        if engine == "qwen36":
            # Qwen's dense trunk: int8 (colibri's default) can go to the card with the experts and lets
            # the CUDA chain run; f32 (QWEN_DENSE=f32) is closer to PyTorch but stays on the CPU
            if os.environ.get("QWEN_DENSE", "i8") == "f32":
                env["COLI_DENSE_I8"] = "0"
        env.update(extra_env or {})
        cap = cap or (128 if engine == "qwen36" else 64)
        bits = "4" if engine == "qwen36" else "8"
        self.p = subprocess.Popen([str(binary), str(cap), bits], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env)
        self.buf, self.log, self.next_id = b"", [], 1
        self.active: str | None = None
        self.lock = threading.Lock()
        threading.Thread(target=self._err, daemon=True).start()
        self._read_until(READY, timeout)

    # ---------------------------------------------------------------- plumbing
    def _err(self):
        for line in self.p.stderr:
            self.log.append(line.decode(errors="replace").rstrip())
            if len(self.log) > 5000:
                del self.log[:1000]

    def _fill(self, timeout: float):
        t0 = time.time()
        while True:
            chunk = self.p.stdout.read1(65536)
            if chunk:
                self.buf += chunk
                return
            if self.p.poll() is not None:
                raise RuntimeError("colibri exited:\n" + "\n".join(self.log[-20:]))
            if time.time() - t0 > timeout:
                raise TimeoutError("colibri did not answer")

    def _read_until(self, token: bytes, timeout: float):
        while token not in self.buf:
            self._fill(timeout)
        self.buf = self.buf[self.buf.index(token) + len(token):]

    def _line(self) -> bytes:
        while b"\n" not in self.buf:
            self._fill(600)
        line, self.buf = self.buf.split(b"\n", 1)
        return line

    # ---------------------------------------------------------------- API
    def alive(self) -> bool:
        return self.p.poll() is None

    def ask(self, prompt: str, max_tokens: int = 48) -> str:
        with self.lock:
            rid = self.next_id
            self.next_id += 1
            data = prompt.encode()
            self.p.stdin.write(b"SUBMIT %d 0 %d %d 0 1\n" % (rid, len(data), max_tokens) + data + b"\n")
            self.p.stdin.flush()
            out = b""
            while True:
                line = self._line()
                if line.startswith(b"DATA %d " % rid):
                    n = int(line.split()[2])
                    while len(self.buf) < n + 1:
                        self._fill(600)
                    out += self.buf[:n]
                    self.buf = self.buf[n + 1:]
                elif line.startswith(b"DONE %d" % rid):
                    return out.decode(errors="replace")
                elif line.startswith(b"ERROR %d" % rid):
                    raise RuntimeError(line.decode())

    def swap(self, pack: str | None, strength: float | None = None) -> dict:
        """Activate `pack` (a directory) or remove the active pack (None); confirmed from the engine log. strength
        (0 < s <= 4, the Workbench dial) multiplies the pack's knowledge experts: the control line is
        "dir<TAB>strength" (coli_packs_status.h)."""
        n0 = len(self.log)
        line = str(Path(pack).resolve()) if pack else ""
        if pack and strength is not None and float(strength) != 1.0:
            try:
                new = b"COLI_PACK_STRENGTH" in Path(self.binary).read_bytes()
            except OSError:
                new = False
            if not new:
                return {"ok": False, "active": self.active, "seconds": 0,
                        "engine_log": ["[OVERLAY] refused: this colibri build has no pack strength (an older build): "
                                       "re-run ./install_patched_colibri.sh, or set the pack's strength back to 100%"]}
            line += f"\t{float(strength):g}"
        Path(self.ctl).write_text(line + "\n", encoding="utf-8")
        t0 = time.time()
        os.kill(self.p.pid, signal.SIGHUP)
        time.sleep(0.05)
        self.ask("Hello", 1)                       # a forward pass: the engine applies the swap at its start
        lines = [l for l in self.log[n0:] if "[OVERLAY]" in l]
        ok = any(("active:" in l) if pack else ("off:" in l) for l in lines)
        refused = [l for l in lines if "refus" in l or "could not" in l or "failed" in l]
        if ok and not refused:
            self.active = str(Path(pack).resolve()) if pack else None
        return {"ok": ok and not refused, "active": self.active, "seconds": round(time.time() - t0, 3),
                "engine_log": lines}

    def pack_stats(self) -> dict:
        """The B-slot pack the engine holds (its status file, coli_packs_status.h): knowledge experts and strength."""
        import json as _json
        for p in (os.environ.get("COLI_PACKS_STATUS") or "", f"/tmp/colibri-packs/{self.p.pid}.json"):
            try:
                st = _json.loads(Path(p).read_text(encoding="utf-8")) if p else None
            except (OSError, ValueError):
                st = None
            if st:
                return {"knowledge_experts": st.get("knowledge_experts", 0), "strength": st.get("strength", 1.0)}
        return {}

    def device(self) -> str:
        on = [l for l in self.log if "every layer on device" in l or "knowledge expert(s) on the card" in l
              or "[CUDA]" in l or "[chain]" in l]
        if len(on) > 1:
            return " | ".join(on[:6])
        return on[0] if on else "no GPU line in the engine log (CPU)"

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass
        try:
            os.unlink(self.ctl)
        except OSError:
            pass
