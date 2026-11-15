"""Model surgery for knowledge writing on HF MoE models (OLMoE layout: fused 3D experts).

Method A  ExpertEditor        train selected existing experts in place; originals kept for exact deletion
Method B  KnowledgeMoeBlock   add k experts per layer, either competing in the native top-k ("topk")
                              or in an extra gated slot outside top-k ("slot")
Baselines ParallelLoRA        named low-rank adapters in parallel with the MoE block (cumulative,
                              stacked daily, or DMoE-style per-unit)

Every wrapper is output-identical to the base model at initialisation (tested in tests/).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import moe_layers

NEG = -1.0e4  # router bias that keeps an added expert out of top-k (identity at init, D-005)


# ============================================================================ Method B
class KnowledgeMoeBlock(nn.Module):
    """Wraps an OLMoE sparse MoE block and adds knowledge experts, grouped in named shards.

    mode="topk": new router rows compete with the base experts; top-k unchanged in size, so a
                 selected knowledge expert displaces a base expert.
    mode="slot": base routing untouched; each knowledge expert has a sigmoid gate and its output
                 is added on top (no displacement).
    A shard owns its experts; drop_shard() removes them exactly.
    """

    def __init__(self, base: nn.Module, mode: str = "slot"):
        super().__init__()
        assert mode in ("topk", "slot")
        self.base, self.mode = base, mode
        ex = base.experts
        if hasattr(ex, "gate_up_proj"):
            E, twoI, D = ex.gate_up_proj.shape
            self._dt, self._dev = ex.gate_up_proj.dtype, ex.gate_up_proj.device
        else:   # mk.qwen.QExperts: packed int4/int8 experts of a colibri container
            E, twoI, D = ex.gate_up_shape
            self._dt, self._dev = ex.dtype, ex.mw.device
        self.E, self.D, self.I = E, D, twoI // 2
        self.gu = nn.ParameterDict(); self.dn = nn.ParameterDict()
        self.w = nn.ParameterDict(); self.b = nn.ParameterDict()
        self.shards: list[str] = []
        self.enabled = True
        self.last_topk: torch.Tensor | None = None   # routing capture for drift metrics
        self.act = base.experts.act_fn

    def add_shard(self, name: str, k: int) -> None:
        assert name not in self.shards
        # new experts start from the mean of base experts (a reasonable feature basis);
        # down projection zero => they contribute nothing until trained.
        ex = self.base.experts
        mean = (ex.gate_up_proj.detach().float().mean(0) if hasattr(ex, "gate_up_proj") else ex.mean_gate_up())
        gu = mean.unsqueeze(0).repeat(k, 1, 1)
        self.gu[name] = nn.Parameter(gu.to(self._dt).to(self._dev))
        self.dn[name] = nn.Parameter(torch.zeros(k, self.D, self.I, dtype=self._dt, device=self._dev))
        self.w[name] = nn.Parameter(torch.zeros(k, self.D, dtype=self._dt, device=self._dev))
        self.b[name] = nn.Parameter(torch.full((k,), NEG if self.mode == "topk" else 0.0,
                                               dtype=torch.float32, device=self._dev))
        self.shards.append(name)

    def drop_shard(self, name: str) -> None:
        for d in (self.gu, self.dn, self.w, self.b):
            del d[name]
        self.shards.remove(name)

    def shard_parameters(self, name: str):
        return [self.gu[name], self.dn[name], self.w[name], self.b[name]]

    @property
    def k(self) -> int:
        return sum(self.gu[s].shape[0] for s in self.shards)

    def _cat(self):
        S = self.shards
        dt = self._dt   # a shard being trained is held in fp32 (train.py); compute in model dtype
        return (torch.cat([self.gu[s].to(dt) for s in S]), torch.cat([self.dn[s].to(dt) for s in S]),
                torch.cat([self.w[s].to(dt) for s in S]), torch.cat([self.b[s].float() for s in S]))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.shards:
            self.last_topk = None
            return self.base(h)
        B, S, D = h.shape
        x = h.reshape(-1, D)
        gate = self.base.gate
        GU, DN, W, Bb = self._cat()
        K = GU.shape[0]

        def expert(j, xs):
            g, u = F.linear(xs, GU[j]).chunk(2, dim=-1)
            return F.linear(self.act(g) * u, DN[j])

        if self.mode == "slot":
            out = self.base(h).reshape(-1, D)
            g = torch.sigmoid(F.linear(x, W).float() + Bb).to(x.dtype)   # [T,K]
            for j in range(K):
                out = out + g[:, j:j + 1] * expert(j, x)
            self.last_topk = None
            return out.reshape(B, S, D)
        base_logits = F.linear(x, gate.weight)
        new_logits = (F.linear(x, W).float() + Bb).to(base_logits.dtype)
        logits = torch.cat([base_logits, new_logits], dim=-1)
        # softmax over E+K, factorised so the base part is computed exactly as the base router does:
        # p_base = softmax(base) / (1 + r), p_new = exp(new - lse_base) / (1 + r), r = sum p_new'
        # With inactive knowledge experts (bias -1e4) r == 0 and p_base is bit-identical to the base.
        pb = torch.softmax(base_logits, dtype=torch.float, dim=-1)
        lse = torch.logsumexp(base_logits.float(), dim=-1, keepdim=True)
        pn = torch.exp(new_logits.float() - lse)
        r = pn.sum(-1, keepdim=True)
        # top-k exactly as the base router (same tensor length => same tie-breaking among base
        # experts), then merge in knowledge experts that beat the k-th base candidate
        bv, bi = torch.topk(pb / (1 + r), gate.top_k, dim=-1)
        if bool((pn > 0).any()):
            cand_v = torch.cat([bv, pn / (1 + r)], dim=-1)
            kidx = torch.arange(self.E, self.E + pn.shape[-1], device=bi.device).expand(bi.shape[0], -1)
            cand_i = torch.cat([bi, kidx], dim=-1)
            topv, pos = torch.topk(cand_v, gate.top_k, dim=-1)
            topi = cand_i.gather(-1, pos)
        else:
            topv, topi = bv, bi
        if gate.norm_topk_prob:
            topv = topv / topv.sum(-1, keepdim=True)
        topv = topv.to(logits.dtype)
        self.last_topk = topi.detach()
        is_new = topi >= self.E
        # positions taken by knowledge experts are sent to base expert 0 with weight 0
        # (an out-of-range sentinel index breaks the grouped-mm backward in transformers>=5)
        base_idx = torch.where(is_new, torch.zeros_like(topi), topi)
        out = self.base.experts(x, base_idx, topv.masked_fill(is_new, 0))
        if is_new.any():
            for j in range(K):
                pos = topi == (self.E + j)
                rows = pos.any(-1).nonzero().squeeze(-1)
                if rows.numel() == 0:
                    continue
                wj = (topv * pos).sum(-1)[rows].unsqueeze(-1)
                out = out.index_add(0, rows, (wj * expert(j, x[rows])).to(out.dtype))
        return out.reshape(B, S, D)


def wrap_knowledge(model, layers: list[int], mode: str) -> dict[int, KnowledgeMoeBlock]:
    out = {}
    for li in layers:
        layer = model.model.layers[li]
        if not isinstance(layer.mlp, KnowledgeMoeBlock):
            layer.mlp = KnowledgeMoeBlock(layer.mlp, mode)
        out[li] = layer.mlp
    return out


def add_knowledge_experts(model, layers: list[int], k: int, mode: str,
                          shard: str = "s0") -> dict[int, KnowledgeMoeBlock]:
    blocks = wrap_knowledge(model, layers, mode)
    for blk in blocks.values():
        blk.add_shard(shard, k)
    return blocks


def unwrap_knowledge(model) -> None:
    """Put the original MoE blocks back (removes every shard)."""
    for layer in model.model.layers:
        if isinstance(layer.mlp, KnowledgeMoeBlock):
            layer.mlp = layer.mlp.base


@torch.no_grad()
def warm_start_bias(model, blocks: dict[int, KnowledgeMoeBlock], shard: str, input_ids: torch.Tensor,
                    margin: float = 0.0) -> None:
    """topk mode: set the shard's bias to the mean k-th largest base logit on the write data,
    so its experts are selected for a share of knowledge tokens and receive gradient."""
    caps, hooks = {}, []
    for li, blk in blocks.items():
        def hk(mod, inp, li=li):
            caps[li] = inp[0].detach().reshape(-1, inp[0].shape[-1])
        hooks.append(blk.register_forward_pre_hook(hk))
    was = {li: blk.enabled for li, blk in blocks.items()}
    for blk in blocks.values():
        blk.enabled = False
    model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    for li, blk in blocks.items():
        blk.enabled = was[li]
        lg = F.linear(caps[li], blk.base.gate.weight).float()
        kth = lg.topk(blk.base.gate.top_k, dim=-1).values[:, -1].mean()
        blk.b[shard].fill_(float(kth) + margin)


# ============================================================================ Method A
class _SlicePatch(nn.Module):
    """Parametrization: the fused expert tensor with the selected experts' slices replaced by
    small trainable tensors. Only the slices get gradients and optimiser state."""

    def __init__(self, idx: list[int], init: torch.Tensor):
        super().__init__()
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long, device=init.device))
        self.w = nn.Parameter(init.detach().clone())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X.index_copy(0, self.idx, self.w.to(X.dtype))


class ExpertEditor:
    """Train only selected (layer, expert) cells of the fused expert tensors.

    While training, each touched tensor is parametrized so that only the selected experts' slices
    are parameters (memory ~ the slices, not the whole fused tensor). `detach_hooks()` bakes the
    trained slices back into the fused tensor and removes the parametrization. A copy of every
    touched expert's original weights is kept, so `restore()` is an exact delete.
    """

    def __init__(self, model, cells: list[tuple[int, int]], keep_orig: bool = True):
        self.model, self.cells = model, cells
        self.orig: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.patched: list[tuple[nn.Module, str, list[int]]] = []
        by_layer: dict[int, list[int]] = {}
        for li, e in cells:
            by_layer.setdefault(li, []).append(e)
            ex = model.model.layers[li].mlp.experts
            if keep_orig:   # the implant does not need undo copies (saves GPU memory)
                self.orig[(li, e)] = (ex.gate_up_proj[e].detach().clone(), ex.down_proj[e].detach().clone())
        self.by_layer = by_layer

    def parameters(self):
        from torch.nn.utils import parametrize
        ps = []
        for li, es in self.by_layer.items():
            ex = self.model.model.layers[li].mlp.experts
            for name in ("gate_up_proj", "down_proj"):
                full = getattr(ex, name)
                patch = _SlicePatch(es, full[es])
                parametrize.register_parametrization(ex, name, patch, unsafe=True)
                getattr(ex.parametrizations, name).original.requires_grad_(False)
                self.patched.append((ex, name, es))
                ps.append(patch.w)
        return ps

    @torch.no_grad()
    def detach_hooks(self):
        """Bake the trained slices into the fused tensors and drop the parametrizations."""
        from torch.nn.utils import parametrize
        for ex, name, es in self.patched:
            w = getattr(ex.parametrizations, name)[0].w.detach().clone()
            parametrize.remove_parametrizations(ex, name, leave_parametrized=False)
            getattr(ex, name)[es] = w.to(getattr(ex, name).dtype)
        self.patched = []

    @torch.no_grad()
    def restore(self, cells: list[tuple[int, int]] | None = None) -> None:
        for (li, e) in (cells or list(self.orig)):
            gu, dn = self.orig[(li, e)]
            ex = self.model.model.layers[li].mlp.experts
            ex.gate_up_proj[e].copy_(gu)
            ex.down_proj[e].copy_(dn)

    @torch.no_grad()
    def overlay(self) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
        """Current weights of every edited cell (what gets exported as a colibri overlay)."""
        out = {}
        for (li, e) in self.orig:
            ex = self.model.model.layers[li].mlp.experts
            out[(li, e)] = (ex.gate_up_proj[e].detach().clone(), ex.down_proj[e].detach().clone())
        return out


@torch.no_grad()
def expert_usage(model, input_ids_list: list[torch.Tensor], layers: list[int]) -> dict[int, torch.Tensor]:
    """Token counts per expert in each layer while reading the given texts."""
    counts = {li: torch.zeros(model.config.num_experts, dtype=torch.long) for li in layers}
    hooks = []
    for li in layers:
        mlp = model.model.layers[li].mlp
        gate = getattr(mlp, "base", mlp).gate              # a wrapped block (knowledge experts) holds the base's router
        def hk(mod, inp, out, li=li, k=model.config.num_experts_per_tok):
            if isinstance(out, tuple) and len(out) >= 3:   # (logits, weights, indices)
                idx = out[2]
            else:                                          # a router that returns its logits only
                idx = (out[0] if isinstance(out, tuple) else out).topk(k, dim=-1).indices
            idx = idx.detach().reshape(-1).cpu()
            counts[li] += torch.bincount(idx, minlength=counts[li].numel())
        hooks.append(gate.register_forward_hook(hk))
    for ids in input_ids_list:
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return counts


def pick_cells(model, input_ids_list, layers: list[int], per_layer: int,
               general_ids_list=None) -> list[tuple[int, int]]:
    """Experts most used by the knowledge text, relative to general text (specialisation ratio)."""
    k_use = expert_usage(model, input_ids_list, layers)
    g_use = expert_usage(model, general_ids_list, layers) if general_ids_list else None
    cells = []
    for li in layers:
        score = k_use[li].float() + 1
        if g_use is not None:
            score = score / (g_use[li].float() + 1)
        for e in score.topk(per_layer).indices.tolist():
            cells.append((li, e))
    return cells


# ============================================================================ Baselines
class ParallelLoRA(nn.Module):
    """y = MoE(x) + sum_{a in active} s_a * B_a(A_a(x)). Named adapters; B starts at zero."""

    def __init__(self, base: nn.Module, hidden: int, dtype, device):
        super().__init__()
        self.base, self.hidden = base, hidden
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self.scale: dict[str, float] = {}
        self.active: set[str] = set()
        self._dt, self._dev = dtype, device

    def add(self, name: str, rank: int, alpha: float | None = None) -> None:
        self.A[name] = nn.Parameter(torch.randn(rank, self.hidden, dtype=self._dt, device=self._dev) * 0.02)
        self.B[name] = nn.Parameter(torch.zeros(self.hidden, rank, dtype=self._dt, device=self._dev))
        self.scale[name] = (alpha or 2 * rank) / rank
        self.active.add(name)

    def drop(self, name: str) -> None:
        del self.A[name], self.B[name]
        self.scale.pop(name)
        self.active.discard(name)

    def forward(self, h):
        out = self.base(h)
        for n in self.active:
            out = out + self.scale[n] * F.linear(F.linear(h, self.A[n]), self.B[n])
        return out


def add_parallel_lora(model, layers: list[int]) -> dict[int, ParallelLoRA]:
    out = {}
    p = next(model.parameters())
    for li in layers:
        layer = model.model.layers[li]
        w = ParallelLoRA(layer.mlp, model.config.hidden_size, p.dtype, p.device)
        layer.mlp = w
        out[li] = w
    return out


def lora_params_per_rank(hidden: int) -> int:
    return 2 * hidden


def knowledge_expert_params(hidden: int, inter: int) -> int:
    return 3 * hidden * inter + hidden + 1


@contextlib.contextmanager
def frozen_except(model, params):
    """Freeze everything, then enable grads only for `params`."""
    keep = {id(p) for p in params}
    saved = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    try:
        yield
    finally:
        for p, r in saved:
            p.requires_grad_(r)
