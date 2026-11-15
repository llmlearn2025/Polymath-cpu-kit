"""The B-slot pack strength (the Workbench dial) in patched colibri, on tiny OLMoE and Qwen3-MoE containers, CPU.

    python engines/patched-colibri/tests/test_bslot_strength.py --colibri <checkout> --olmoe <container> --qwen <container>

(The containers are the ones engines/polymath-c/tests/test_polymath_c.py builds: <work>/olmoe/olm_coli and
<work>/qwen/qwen_coli.) A random knowledge-expert pack is written for each, then:

  exact    strength 0.5 = the same pack with its down-projection row scales halved on disk, at strength 1
  effect   strength 1, 0.5 and 2 give three different losses; no pack = the base
  refused  a strength outside (0, 4] stops the engine at start, with the reason
Exit code 0 = all pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

IDS = list(range(3, 60, 2))


def run(coli: Path, kind: str, cont: Path, pack: Path | None = None, strength: str | None = None):
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref.json"
        ref.write_text(json.dumps({"prompt_ids": IDS[:1], "full_ids": IDS}))
        env = dict(os.environ, SNAP=str(cont), PPL="1", COLI_DENSE_I8="0", COLI_CUDA="0", COLI_CUDA_CHAIN="0",
                   OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1")
        for k in ("OLM_OVERLAY", "COLI_OVERLAY", "PM_PACKS", "PM_PACKS_CTL", "COLI_PACK_STRENGTH"):
            env.pop(k, None)
        if pack:
            env["COLI_OVERLAY"] = env["OLM_OVERLAY"] = str(pack)
        if strength is not None:
            env["COLI_PACK_STRENGTH"] = strength
        args = [str(coli / "c" / kind), "128", "4"] if kind == "qwen36" else [str(coli / "c" / kind), "8", "8"]
        r = subprocess.run(args + [str(ref)], capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout + r.stderr
    m = re.search(r"TF-NLL-EXACT: ([0-9.eE+-]+)", out) or re.search(r"TF-NLL: ([0-9.]+)", out)
    return (m.group(1) if m else None), r.returncode, out


def write_pack(folder: Path, fp: str, layers: list[int], D: int, I: int, K: int = 2, down_scale: float = 1.0) -> Path:
    g = torch.Generator().manual_seed(11)
    t = {}
    for L in layers:
        t[f"model.layers.{L}.mlp.knowledge.gate_w"] = (torch.randn(K, D, generator=g) * 0.2).contiguous()
        t[f"model.layers.{L}.mlp.knowledge.gate_b"] = torch.zeros(K)
        for j in range(K):
            t[f"model.layers.{L}.mlp.knowledge.{j}.merged_weight"] = torch.randint(-60, 60, (3 * I * D,), dtype=torch.int8, generator=g)
            qs = torch.full((2 * I + D,), 0.01)
            qs[2 * I:] *= 0.3 * down_scale
            t[f"model.layers.{L}.mlp.knowledge.{j}.qs"] = qs
    folder.mkdir(parents=True, exist_ok=True)
    save_file(t, str(folder / "overlay.safetensors"))
    (folder / "overlay.json").write_text(json.dumps({"format": "colibri-olmoe-overlay-v2", "version": 1,
                                                     "base_fingerprint": fp, "cells": [],
                                                     "knowledge": [[L, K] for L in layers]}))
    return folder


def check(coli: Path, kind: str, cont: Path, D: int, I: int, work: Path) -> list[tuple[str, bool, str]]:
    res = []
    probe = write_pack(work / f"{kind}_probe", "0000000000000000", [1], D, I)
    _, _, out = run(coli, kind, cont, probe)
    m = re.search(r"does not match this model \(([0-9a-f]{16})\)", out)
    if not m:
        return [("fingerprint from the refusal", False, out[-400:])]
    fp = m.group(1)
    pk = write_pack(work / f"{kind}_pack", fp, [1, 2], D, I)
    half = write_pack(work / f"{kind}_half", fp, [1, 2], D, I, down_scale=0.5)
    base = run(coli, kind, cont)[0]
    one, _, out1 = run(coli, kind, cont, pk)
    if one is None:
        return [("the pack loads", False, out1[-600:])]
    h, _, outh = run(coli, kind, cont, pk, "0.5")
    hd = run(coli, kind, cont, half)[0]
    up = run(coli, kind, cont, pk, "2")[0]
    res.append(("exact: strength 0.5 = row scales halved on disk", h is not None and h == hd, f"{h} vs {hd}"))
    res.append(("effect: 1, 0.5 and 2 differ, and differ from the base", len({base, one, h, up}) == 4,
                f"base {base} 1 {one} 0.5 {h} 2 {up}"))
    line = next((l for l in outh.splitlines() if "[OVERLAY] active" in l), "")
    res.append(("log line names the strength", "strength x0.5" in line and "epoch 1," in line, line[:200]))
    line1 = next((l for l in out1.splitlines() if "[OVERLAY] active" in l), "")
    res.append(("strength 1: the log line as before", line1.rstrip().endswith("epoch 1"), line1[:200]))
    for bad in ("0", "5", "-1", "x"):
        c, rc, out = run(coli, kind, cont, pk, bad)
        res.append((f"refused: strength {bad}", rc != 0 and "must be a number above 0 and at most 4" in out, ""))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colibri", required=True)
    ap.add_argument("--olmoe", required=True, help="a converted tiny OLMoE container")
    ap.add_argument("--qwen", required=True, help="a converted tiny Qwen3-MoE container")
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "bslot_strength_test"))
    a = ap.parse_args()
    coli, work = Path(a.colibri).expanduser().resolve(), Path(a.work)
    olm = json.loads((Path(a.olmoe) / "config.json").read_text())
    qm = json.loads((Path(a.qwen) / "qwen36_meta.json").read_text())
    fail = 0
    for kind, cont, D, I in [("olmoe", Path(a.olmoe), olm["hidden_size"], olm["intermediate_size"]),
                             ("qwen36", Path(a.qwen), qm["hidden"], qm["moe_inter"])]:
        print(f"== {kind}")
        for name, ok, note in check(coli, kind, cont, D, I, work):
            print(f"{'PASS' if ok else 'FAIL'} {name}" + (f"   ({note})" if note else ""))
            fail += not ok
    print("ALL PASS" if not fail else f"{fail} FAILED")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
