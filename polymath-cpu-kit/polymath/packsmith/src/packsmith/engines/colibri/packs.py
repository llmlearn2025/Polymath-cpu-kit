"""Knowledge packs on disk: list, inspect, merge. Pure standard library (no torch, no safetensors
package), so it runs inside Hermes' own Python.

A pack is a directory with overlay.safetensors + overlay.json (format colibri-olmoe-overlay-v2,
written by moe-knowledge/export/to_colibri_overlay.py). Merging packs that only ADD knowledge
experts is exact arithmetic: each knowledge expert adds sigmoid(gate) * expert(x) on its own, so
the merged pack computes the sum of what the packs compute (whether two packs trained apart
interfere with each other's answers is measured by g0/many_minds.py, not assumed).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import struct
from pathlib import Path

FORMAT2 = "colibri-olmoe-overlay-v2"


def st_read(path: Path) -> tuple[dict, bytes, int]:
    """(header, whole file bytes, data offset) of a safetensors file."""
    b = Path(path).read_bytes()
    n = struct.unpack("<Q", b[:8])[0]
    return json.loads(b[8:8 + n]), b, 8 + n


def st_write(path: Path, tensors: dict[str, tuple[str, list[int], bytes]], metadata: dict | None = None) -> None:
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    if metadata:
        hdr["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for blob in blobs:
            f.write(blob)


def manifest(pack: Path) -> dict:
    return json.loads((Path(pack) / "overlay.json").read_text(encoding="utf-8"))


def describe(pack: Path) -> dict:
    m = manifest(pack)
    kn = m.get("knowledge", [])
    return {"path": str(Path(pack).resolve()), "format": m.get("format"), "version": m.get("version"),
            "source": m.get("source", ""), "created": m.get("created", ""),
            "modified_experts": len(m.get("cells", [])),
            "knowledge_experts": sum(k for _, k in kn), "layers": [l for l, _ in kn],
            "base_fingerprint": m.get("base_fingerprint")}


def merge(packs: list[Path], out: Path, source: str = "", strengths: list[float] | None = None) -> Path:
    """One pack whose knowledge experts are those of `packs`, layer by layer, in the given order. strengths (one per
    pack): each pack's experts at its own strength, by scaling their down-projection row scales, which is what the
    engine's strength does (coli_packs_status.h: strength 0.5 = row scales halved on disk). The merged pack is then
    served at strength 1."""
    packs = [Path(p) for p in packs]
    if strengths is not None and len(strengths) != len(packs):
        raise ValueError("one strength per pack")
    ms = [manifest(p) for p in packs]
    fps = {m["base_fingerprint"] for m in ms}
    if len(fps) != 1:
        raise ValueError(f"packs were built for different models: {sorted(fps)}")
    for p, m in zip(packs, ms):
        if m.get("format") != FORMAT2 or not m.get("knowledge"):
            raise ValueError(f"{p} has no knowledge experts (format {m.get('format')})")
        if m.get("cells"):
            raise ValueError(f"{p} modifies routed experts (method A); only knowledge packs merge exactly")
    reads = [st_read(p / "overlay.safetensors") for p in packs]
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    count: dict[int, int] = {}
    gw: dict[int, list] = {}
    gb: dict[int, list] = {}
    for pi, ((hdr, raw, base), m) in enumerate(zip(reads, ms)):
        sc = 1.0 if strengths is None else float(strengths[pi])
        def get(name):
            t = hdr[name]
            a, b = t["data_offsets"]
            return t["dtype"], t["shape"], raw[base + a: base + b]
        for L, K in m["knowledge"]:
            j0 = count.get(L, 0)
            D = get(f"model.layers.{L}.mlp.knowledge.gate_w")[1][1]
            for j in range(K):
                for kind in ("merged_weight", "qs"):
                    t = get(f"model.layers.{L}.mlp.knowledge.{j}.{kind}")
                    if kind == "qs" and sc != 1.0:            # [gate rows | up rows | down rows]: the down rows scaled
                        import array
                        q = array.array("f")
                        q.frombytes(t[2])
                        for i in range(len(q) - D, len(q)):
                            q[i] *= sc
                        t = (t[0], t[1], q.tobytes())
                    tensors[f"model.layers.{L}.mlp.knowledge.{j0 + j}.{kind}"] = t
            gw.setdefault(L, []).append(get(f"model.layers.{L}.mlp.knowledge.gate_w"))
            gb.setdefault(L, []).append(get(f"model.layers.{L}.mlp.knowledge.gate_b"))
            count[L] = j0 + K
    for L in count:
        D = gw[L][0][1][1]
        tensors[f"model.layers.{L}.mlp.knowledge.gate_w"] = ("F32", [count[L], D], b"".join(x[2] for x in gw[L]))
        tensors[f"model.layers.{L}.mlp.knowledge.gate_b"] = ("F32", [count[L]], b"".join(x[2] for x in gb[L]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    st = out / "overlay.safetensors"
    st_write(st, tensors, {"format": FORMAT2})
    man = {"format": FORMAT2, "version": 1, "shard": "merged", "base_fingerprint": fps.pop(),
           "created": dt.datetime.now(dt.timezone.utc).isoformat(), "cells": [],
           "sha256": hashlib.sha256(st.read_bytes()).hexdigest(),
           "source": source or "merged: " + " + ".join(str(p) for p in packs),
           "knowledge": [[L, count[L]] for L in sorted(count)]}
    (out / "overlay.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return out


def verify(pack: Path) -> bool:
    m = manifest(pack)
    return hashlib.sha256((Path(pack) / "overlay.safetensors").read_bytes()).hexdigest() == m.get("sha256")
