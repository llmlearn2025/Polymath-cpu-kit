"""Export Method-A edited experts as a colibri OLMoE overlay (format colibri-olmoe-overlay-v1).

Overlay directory:
    overlay.safetensors   model.layers.L.mlp.experts.E.merged_weight (I8) + .qs (F32), exactly the
                          layout tools/convert_olmoe_merged.py writes for the base model
    overlay.json          {format, version, shard, base_fingerprint, created, cells:[[L,E],...],
                           sha256, source}

The engine resolves an expert tensor in the active overlay first, then in the base model, so the
base files are never modified and deactivating the overlay restores the base bytes exactly.

Format v2 (colibri-olmoe-overlay-v2) also carries B-slot KNOWLEDGE EXPERTS: per layer, K extra
experts (int8, same layout) plus their gates, which the patched engine adds on every token:
    out += sigmoid(gate_w[j] . x + gate_b[j]) * expert_j(x)
B-topk is not exportable (OLMoE routers have no bias term; the engine sizes n_experts at load).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import sys
from packsmith.core.quant import merged_expert  # noqa: E402

FORMAT = "colibri-olmoe-overlay-v1"
FORMAT2 = "colibri-olmoe-overlay-v2"


def expert_inter(cfg) -> int:
    """An expert's width: moe_intermediate_size (Qwen3-MoE) or intermediate_size (OLMoE)."""
    return getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size


def base_fingerprint(cfg) -> str:
    """Architecture fields that must match between base and overlay (FNV-1a over ints)."""
    h = 1469598103934665603
    for v in (cfg.hidden_size, expert_inter(cfg), cfg.num_hidden_layers, cfg.num_experts,
              cfg.num_experts_per_tok, cfg.vocab_size):
        for i in range(8):
            h ^= (int(v) >> (8 * i)) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def export_overlay(model, cells_weights: dict, out_dir: str | Path, shard: str, version: int,
                   source: str = "", knowledge_blocks: dict | None = None) -> Path:
    """cells_weights: {(layer, expert): (gate_up [2I,D], down [D,I])} for Method A edits.
    knowledge_blocks: {layer: KnowledgeMoeBlock in slot mode} for B-slot knowledge experts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    I = expert_inter(model.config)
    tensors = {}
    for (li, e), (gu, dn) in sorted(cells_weights.items()):
        gate, up = gu[:I].float().cpu(), gu[I:].float().cpu()
        mq, ms = merged_expert(gate, up, dn.float().cpu())
        tensors[f"model.layers.{li}.mlp.experts.{e}.merged_weight"] = mq.contiguous()
        tensors[f"model.layers.{li}.mlp.experts.{e}.qs"] = ms.contiguous()
    knowledge = []
    for li, b in sorted((knowledge_blocks or {}).items()):
        if not b.shards:
            continue
        if b.mode != "slot":
            raise ValueError("only B-slot knowledge experts can be exported (B-topk needs router support)")
        with torch.no_grad():
            GU, DN, W, Bb = b._cat()
        K = GU.shape[0]
        for j in range(K):
            mq, ms = merged_expert(GU[j, :I].float().cpu(), GU[j, I:].float().cpu(), DN[j].float().cpu())
            tensors[f"model.layers.{li}.mlp.knowledge.{j}.merged_weight"] = mq.contiguous()
            tensors[f"model.layers.{li}.mlp.knowledge.{j}.qs"] = ms.contiguous()
        tensors[f"model.layers.{li}.mlp.knowledge.gate_w"] = W.float().cpu().contiguous()
        tensors[f"model.layers.{li}.mlp.knowledge.gate_b"] = Bb.float().cpu().contiguous()
        knowledge.append([li, K])
    fmt = FORMAT2 if knowledge else FORMAT
    st = out / "overlay.safetensors"
    save_file(tensors, str(st), metadata={"format": fmt})
    digest = hashlib.sha256(st.read_bytes()).hexdigest()
    manifest = {
        "format": fmt, "version": version, "shard": shard,
        "base_fingerprint": base_fingerprint(model.config),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cells": [[li, e] for (li, e) in sorted(cells_weights)],
        "sha256": digest, "source": source,
    }
    if knowledge:
        manifest["knowledge"] = knowledge
    (out / "overlay.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def verify_overlay(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    m = json.loads((out / "overlay.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((out / "overlay.safetensors").read_bytes()).hexdigest()
    if digest != m["sha256"]:
        raise ValueError("overlay.safetensors does not match its manifest hash")
    if m["format"] not in (FORMAT, FORMAT2):
        raise ValueError(f"unknown overlay format {m['format']}")
    return m


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="verify an overlay directory")
    ap.add_argument("dir")
    print(json.dumps(verify_overlay(ap.parse_args().dir), indent=1))
