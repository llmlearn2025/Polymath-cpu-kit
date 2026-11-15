"""colibri OLMoE expert quantisation (row-wise int8), identical math to
colibri/c/tools/convert_olmoe_merged.py::quantize_row / convert_expert (D-006)."""
from __future__ import annotations

import torch


def quantize_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w32 = w.float()
    row_max = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / 127.0
    q = (w32 / scales).round().clamp(-128, 127).to(torch.int8)
    return q, scales.squeeze(1)


def dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * s.unsqueeze(1)


def roundtrip(w: torch.Tensor) -> torch.Tensor:
    """What colibri will actually compute with: int8 row-wise quantise then dequantise."""
    q, s = quantize_row(w)
    return dequant(q, s).to(w.dtype)


def merged_expert(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    """colibri merged_weight / qs tensors for one expert."""
    qg, sg = quantize_row(gate)
    qu, su = quantize_row(up)
    qd, sd = quantize_row(down)
    return torch.cat([qg.flatten(), qu.flatten(), qd.flatten()]), torch.cat([sg, su, sd])


@torch.no_grad()
def simulate_int8_cells(model, cells) -> dict:
    """Round-trip the given (layer, expert) cells through colibri int8; returns originals for undo."""
    saved = {}
    for li, e in cells:
        ex = model.model.layers[li].mlp.experts
        saved[(li, e)] = (ex.gate_up_proj[e].clone(), ex.down_proj[e].clone())
        ex.gate_up_proj[e].copy_(roundtrip(ex.gate_up_proj[e]))
        ex.down_proj[e].copy_(roundtrip(ex.down_proj[e]))
    return saved


@torch.no_grad()
def undo(model, saved: dict) -> None:
    for (li, e), (gu, dn) in saved.items():
        ex = model.model.layers[li].mlp.experts
        ex.gate_up_proj[e].copy_(gu)
        ex.down_proj[e].copy_(dn)


@torch.no_grad()
def simulate_int8_knowledge(blocks) -> dict:
    """Round-trip every knowledge expert of KnowledgeMoeBlocks through int8."""
    saved = {}
    for li, b in blocks.items():
        for s in b.shards:
            saved[(li, s)] = (b.gu[s].data.clone(), b.dn[s].data.clone())
            for j in range(b.gu[s].shape[0]):
                b.gu[s].data[j] = roundtrip(b.gu[s].data[j])
                b.dn[s].data[j] = roundtrip(b.dn[s].data[j])
    return saved


@torch.no_grad()
def undo_knowledge(blocks, saved: dict) -> None:
    for (li, s), (gu, dn) in saved.items():
        blocks[li].gu[s].data.copy_(gu)
        blocks[li].dn[s].data.copy_(dn)
