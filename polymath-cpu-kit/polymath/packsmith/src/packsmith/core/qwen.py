"""Qwen3-MoE (Qwen3-30B-A3B) in PyTorch, loaded from a colibri container, so that a 30B model
fits a 32 GB card for training and the experts are bit-for-bit the ones colibri serves.

A container is what colibri's tools/convert_qwen36.py writes: per layer a safetensors shard with
the dense tensors in f16 under their HF names and every routed expert as
`model.layers.L.mlp.experts.E.merged_weight` (+ `.qs` f32 scales), int4 packed two per byte
(--ebits 4, low nibble first, signed) or int8 (--ebits 8), with one scale per row or per --gs
inputs; plus model-globals.safetensors (embeddings, lm_head, final norm), config.hf.json (the HF
config), qwen36_meta.json and the tokenizer.

Here:
  * dense weights -> the HF modules in bf16 (the release is bf16, so f16 -> bf16 is exact for it);
  * experts stay packed on the GPU (QExperts, ~16 GB for 30B-A3B at int4 gs64) and are
    dequantized per layer for the experts a batch hits. Under training the block is
    recomputed in backward (activation checkpointing), so no dequantized weight is kept.

    from packsmith.core.qwen import load_container
    model, tok = load_container("runs/qwen_prior_coli", device="cuda")
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def is_container(path) -> bool:
    p = Path(str(path))
    return (p / "qwen36_meta.json").exists() and (p / "config.hf.json").exists()


def unpack_int4(b: torch.Tensor) -> torch.Tensor:
    """uint8 [..., N/2] -> int8 [..., N]: low nibble = element 2k, high = 2k+1, both signed."""
    lo = (b & 0xF).to(torch.int8)
    hi = (b >> 4).to(torch.int8)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _deq(q: torch.Tensor, s: torch.Tensor, O: int, I: int, gs: int) -> torch.Tensor:
    """q int8 [n, O*I], s f32 [n, O*ng] (or [n, O] for gs 0) -> f32 [n, O, I]."""
    n = q.shape[0]
    q = q.view(n, O, I).float()
    if not gs:
        return q * s.view(n, O, 1)
    ng = (I + gs - 1) // gs
    pad = ng * gs - I
    if pad:
        q = F.pad(q, (0, pad))
    w = q.view(n, O, ng, gs) * s.view(n, O, ng, 1)
    return w.view(n, O, ng * gs)[..., :I]


class QExperts(nn.Module):
    """Packed routed experts of one layer, with the HF experts' forward signature."""

    def __init__(self, mw: torch.Tensor, qs: torch.Tensor, D: int, I: int, gs: int, act, dtype):
        super().__init__()
        E = mw.shape[0]
        self.num_experts, self.hidden_dim, self.intermediate_dim = E, D, I
        self.gs, self.act_fn, self.dtype = gs, act, dtype
        n = 3 * I * D
        if mw.dtype == torch.uint8 and mw.shape[1] * 2 == n:
            self.int4 = True
        elif mw.dtype == torch.int8 and mw.shape[1] == n:
            self.int4 = False
        else:
            raise ValueError(f"expert layout not supported (dtype {mw.dtype}, {mw.shape[1]} bytes for {n} weights; "
                             "the mixed --down-bits layout is not)")
        ng_in = (D + gs - 1) // gs if gs else 1
        ng_dn = (I + gs - 1) // gs if gs else 1
        self.ns = (I * ng_in, I * ng_in, D * ng_dn)
        assert qs.shape[1] == sum(self.ns), "scale count does not match the container's group size"
        self.register_buffer("mw", mw, persistent=False)
        self.register_buffer("qs", qs.float(), persistent=False)

    @property
    def gate_up_shape(self):
        return (self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)

    @torch.no_grad()
    def weights(self, ids: torch.Tensor):
        """gate_up [n, 2I, D] and down [n, D, I] in the model dtype for the experts `ids`."""
        D, I, gs = self.hidden_dim, self.intermediate_dim, self.gs
        q = self.mw[ids]
        q = unpack_int4(q) if self.int4 else q
        s = self.qs[ids]
        a, b = I * D, 2 * I * D
        n0, n1 = self.ns[0], self.ns[0] + self.ns[1]
        g = _deq(q[:, :a], s[:, :n0], I, D, gs)
        u = _deq(q[:, a:b], s[:, n0:n1], I, D, gs)
        d = _deq(q[:, b:], s[:, n1:], D, I, gs)
        return torch.cat([g, u], 1).to(self.dtype), d.to(self.dtype)

    @torch.no_grad()
    def mean_gate_up(self, chunk: int = 16) -> torch.Tensor:
        tot = None
        for i in range(0, self.num_experts, chunk):
            ids = torch.arange(i, min(i + chunk, self.num_experts), device=self.mw.device)
            gu, _ = self.weights(ids)
            s = gu.float().sum(0)
            tot = s if tot is None else tot + s
        return tot / self.num_experts

    def _run(self, x, top_k_index, top_k_weights):
        out = torch.zeros_like(x)
        E = self.num_experts
        hit = torch.unique(top_k_index[top_k_index < E])
        chunk = 16
        for c0 in range(0, hit.numel(), chunk):
            ids = hit[c0:c0 + chunk]
            GU, DN = self.weights(ids)
            for j, e in enumerate(ids.tolist()):
                tok, pos = torch.where(top_k_index == e)
                xs = x[tok]
                g, u = F.linear(xs, GU[j]).chunk(2, dim=-1)
                h = F.linear(self.act_fn(g) * u, DN[j])
                h = h * top_k_weights[tok, pos, None]
                out = out.index_add(0, tok, h.to(out.dtype))
        return out

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if torch.is_grad_enabled() and (hidden_states.requires_grad or top_k_weights.requires_grad):
            return checkpoint(self._run, hidden_states, top_k_index, top_k_weights, use_reentrant=False)
        return self._run(hidden_states, top_k_index, top_k_weights)


def _shards(path: Path):
    from safetensors import safe_open
    for f in sorted(path.glob("*.safetensors")):
        yield f, safe_open(str(f), framework="pt")


def load_container(path: str, device: str = "cuda", dtype: str = "bfloat16"):
    """(model, Tok) from a colibri Qwen3-MoE container."""
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

    from .model import Tok
    p = Path(path)
    meta = json.loads((p / "qwen36_meta.json").read_text())
    cfg = Qwen3MoeConfig.from_dict(json.loads((p / "config.hf.json").read_text()))
    dt = getattr(torch, dtype)
    gs = int(meta.get("expert_gs", 0) or 0)
    if meta.get("expert_down_bits"):
        raise ValueError("mixed --down-bits containers are not supported; convert with --ebits 4 --gs 64")
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(cfg)
    D, I, E = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts
    exp: dict[int, dict] = {}
    built = set()

    def build(i: int) -> None:   # a layer's experts onto the device as soon as its shard is read
        d = exp.pop(i)
        mw = torch.stack([d["merged_weight"][e] for e in range(E)]).to(device)
        qs = torch.stack([d["qs"][e] for e in range(E)]).to(device)
        layer = model.model.layers[i]
        layer.mlp.experts = QExperts(mw, qs, D, I, gs, layer.mlp.experts.act_fn, dt)
        built.add(i)

    for f, h in _shards(p):
        for k in h.keys():
            if ".mlp.experts." in k:
                parts = k.split(".")
                L, e, kind = int(parts[2]), int(parts[5]), parts[6]
                exp.setdefault(L, {}).setdefault(kind, {})[e] = h.get_tensor(k)
                continue
            if k.startswith("mtp.") or k.startswith("model.visual."):
                continue
            mod_name, leaf = k.rsplit(".", 1)
            try:
                mod = model.get_submodule(mod_name)
            except AttributeError:
                continue
            t = h.get_tensor(k).to(device=device, dtype=dt)
            mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)
        for i in [i for i, d in exp.items() if len(d.get("merged_weight", {})) == E and len(d.get("qs", {})) == E]:
            build(i)
    if len(built) != len(model.model.layers):
        raise ValueError(f"experts missing for layers {sorted(set(range(len(model.model.layers))) - built)[:5]}")
    model.model.rotary_emb = Qwen3MoeRotaryEmbedding(config=cfg).to(device)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise ValueError(f"tensors missing from the container: {left[:5]} ... ({len(left)})")
    for prm in model.parameters():
        prm.requires_grad_(True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(p)) if (p / "tokenizer.json").exists() else None
    return model, Tok(tok)
