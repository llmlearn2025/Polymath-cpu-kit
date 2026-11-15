"""Uniform interface over every method under test and every baseline.

    m = make_method("A", model, tok, cfg)      # or Btopk, Bslot, rag, lora_cum, lora_stack, dmoe
    m.write("day1", texts_by_fact, replay)     # one shard
    m.prepare(question)                        # per-query setup (RAG retrieval, DMoE routing)
    m.prompt(question) -> str                  # text actually sent to the model
    m.delete("day1")                           # remove a shard; returns cost

Shard ownership (D-008): a shard owns its experts / adapters / documents; deleting a shard never
touches another shard's state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .bm25 import BM25
from .model import Tok, geometry
from .surgery import (ExpertEditor, add_knowledge_experts, add_parallel_lora, frozen_except,
                      pick_cells, warm_start_bias)
from .train import TrainCfg, train

QA_FMT = "Question: {q}\nAnswer:"


@dataclass
class MethodCfg:
    layers: list[int] = field(default_factory=lambda: [5, 6, 7, 8])   # placement (from the X-ray)
    k: int = 2                      # knowledge experts per layer per shard (B) / cells per layer (A)
    lora_rank: int | None = None    # None => matched to the knowledge-expert parameter budget
    dmoe_rank: int = 4
    dmoe_topn: int = 3
    rag_topn: int = 5
    rag_instruct: bool = False      # tell the model the context is the law in force
    train: TrainCfg = field(default_factory=TrainCfg)
    dmoe_steps: int = 30
    warm_margin: float = 0.0
    slot_bias: float = 0.0          # Bslot initial gate bias (0 => sigmoid 0.5 on every token)
    gate_lr_mult: float = 1.0       # lr multiplier for router rows / gates of added experts
    slot_init: str = "average"      # Bslot experts' gate/up start: "average" of the layer's experts (moe-knowledge),
                                    # or "copy" of the experts the shard's text uses most (relative to general text)


class Method:
    name = "base"

    def __init__(self, model, tok: Tok, cfg: MethodCfg):
        self.model, self.tok, self.cfg = model, tok, cfg
        self.shards: dict[str, dict] = {}
        self.log = print
        self.ref = None   # base top-k logprobs on general text (KL regulariser), set by the runner

    def prompt(self, q: str) -> str:
        return QA_FMT.format(q=q)

    def prepare(self, q: str) -> None:
        pass

    def write(self, shard: str, facts: list[dict], replay: list[str] | None = None) -> dict:
        raise NotImplementedError

    def delete(self, shard: str) -> dict:
        raise NotImplementedError

    @staticmethod
    def _texts(facts: list[dict]) -> list[str]:
        return [t for f in facts for t in f["texts"]]

    def _ids(self, texts: list[str], n: int = 32) -> list[torch.Tensor]:
        dev = next(self.model.parameters()).device
        return [torch.tensor([self.tok.encode(t)[:128]], device=dev) for t in texts[:n]]


class Base(Method):
    name = "base"

    def write(self, shard, facts, replay=None):
        self.shards[shard] = {}
        return {"seconds": 0.0}

    def delete(self, shard):
        self.shards.pop(shard, None)
        return {"seconds": 0.0}


class ExpertsA(Method):
    """Method A: edit existing experts most used by the shard's text; one owner per expert."""
    name = "A"

    def __init__(self, model, tok, cfg, general_texts=None):
        super().__init__(model, tok, cfg)
        self.owned: set[tuple[int, int]] = set()
        self.general_texts = general_texts or []

    def write(self, shard, facts, replay=None):
        texts = self._texts(facts)
        t0 = time.time()
        cand = pick_cells(self.model, self._ids(texts), self.cfg.layers, self.cfg.k * 4,
                          self._ids(self.general_texts) if self.general_texts else None)
        cells, per = [], {}
        for li, e in cand:  # keep the top-k unowned cells per layer
            if (li, e) in self.owned or per.get(li, 0) >= self.cfg.k:
                continue
            cells.append((li, e)); per[li] = per.get(li, 0) + 1
        ed = ExpertEditor(self.model, cells)
        params = ed.parameters()
        with frozen_except(self.model, params):
            st = train(self.model, self.tok, params, texts, self.cfg.train, replay, ref=self.ref, log=self.log)
        ed.detach_hooks()
        self.owned |= set(cells)
        self.shards[shard] = {"editor": ed, "cells": cells}
        return {"seconds": time.time() - t0, "cells": cells, "train": st["final_loss"],
                "params": len(cells) * 3 * geometry(self.model)["hidden"] * geometry(self.model)["inter"]}

    def delete(self, shard):
        t0 = time.time()
        s = self.shards.pop(shard)
        s["editor"].restore()
        self.owned -= set(s["cells"])
        return {"seconds": time.time() - t0, "exact": True}


class AddedB(Method):
    """Method B: add k knowledge experts per layer per shard (mode topk or slot)."""

    def __init__(self, model, tok, cfg, mode: str):
        super().__init__(model, tok, cfg)
        self.mode = mode
        self.name = f"B{mode}"
        self.blocks = {}

    def write(self, shard, facts, replay=None):
        texts = self._texts(facts)
        t0 = time.time()
        self.blocks = add_knowledge_experts(self.model, self.cfg.layers, self.cfg.k, self.mode, shard)
        if self.mode == "topk":
            ids = torch.cat([i[:, :64] for i in self._ids(texts, 16) if i.shape[1] >= 64] or
                            [self._ids(texts, 1)[0]], 0)
            warm_start_bias(self.model, self.blocks, shard, ids, self.cfg.warm_margin)
        if self.mode == "slot":
            with torch.no_grad():
                for b in self.blocks.values():
                    b.b[shard].fill_(self.cfg.slot_bias)
            if self.cfg.slot_init == "copy":
                copy_slot_experts(self.model, self.blocks, shard, self._ids(texts, 64),
                                  self._ids(self.general_texts, 32) if getattr(self, "general_texts", None) else None)
        experts = [p for b in self.blocks.values() for p in (b.gu[shard], b.dn[shard])]
        gates = [p for b in self.blocks.values() for p in (b.w[shard], b.b[shard])]
        params = [{"params": experts, "lr_mult": 1.0}, {"params": gates, "lr_mult": self.cfg.gate_lr_mult}]
        with frozen_except(self.model, experts + gates):
            st = train(self.model, self.tok, params, texts, self.cfg.train, replay, ref=self.ref, log=self.log)
        self.shards[shard] = {}
        g = geometry(self.model)
        return {"seconds": time.time() - t0, "train": st["final_loss"],
                "params": len(self.cfg.layers) * self.cfg.k * (3 * g["hidden"] * g["inter"] + g["hidden"] + 1)}

    def delete(self, shard):
        t0 = time.time()
        for b in self.blocks.values():
            b.drop_shard(shard)
        self.shards.pop(shard)
        return {"seconds": time.time() - t0, "exact": True}


@torch.no_grad()
def copy_slot_experts(model, blocks, shard: str, ids, general_ids=None) -> dict[int, list[int]]:
    """Start each B-slot knowledge expert's gate/up projection as a copy of the base expert the shard's text uses
    most in that layer (relative to general text). The down projection stays zero, so the pack is still silent
    (output-identical to the base) at step 0; only the features it starts from change. The mean of all experts,
    the default, is a washed-out basis whose activations are small, so the down projection learns slowly."""
    from .surgery import pick_cells
    k = {li: b.gu[shard].shape[0] for li, b in blocks.items()}
    was = {li: b.enabled for li, b in blocks.items()}
    for b in blocks.values():
        b.enabled = False                                   # usage of the base router alone
    try:
        cells = pick_cells(model, ids, sorted(blocks), max(k.values()), general_ids)
    finally:
        for li, b in blocks.items():
            b.enabled = was[li]
    out: dict[int, list[int]] = {}
    for li, e in cells:
        out.setdefault(li, [])
        if len(out[li]) < k[li]:
            out[li].append(e)
    for li, es in out.items():
        ex = blocks[li].base.experts
        if hasattr(ex, "gate_up_proj"):
            gu = ex.gate_up_proj[es].detach()
        else:                                               # packed int4/int8 experts (a colibri Qwen container)
            gu = ex.weights(torch.tensor(es, device=ex.mw.device))[0]
        for j in range(len(es)):
            blocks[li].gu[shard].data[j] = gu[j].to(blocks[li].gu[shard].device, blocks[li].gu[shard].dtype)
    return out


def matched_rank(model, cfg: MethodCfg) -> int:
    g = geometry(model)
    if cfg.lora_rank:
        return cfg.lora_rank
    per_layer_budget = cfg.k * 3 * g["hidden"] * g["inter"]
    return max(1, min(g["hidden"], round(per_layer_budget / (2 * g["hidden"]))))


class LoraStack(Method):
    """B3: one parallel LoRA per shard (day); deletion = drop that adapter."""
    name = "lora_stack"

    def __init__(self, model, tok, cfg):
        super().__init__(model, tok, cfg)
        self.wr = add_parallel_lora(model, cfg.layers)
        self.rank = matched_rank(model, cfg)

    def write(self, shard, facts, replay=None):
        texts = self._texts(facts)
        t0 = time.time()
        for w in self.wr.values():
            w.add(shard, self.rank)
        params = [p for w in self.wr.values() for p in (w.A[shard], w.B[shard])]
        with frozen_except(self.model, params):
            st = train(self.model, self.tok, params, texts, self.cfg.train, replay, ref=self.ref, log=self.log)
        self.shards[shard] = {"texts": texts}
        return {"seconds": time.time() - t0, "rank": self.rank, "train": st["final_loss"]}

    def delete(self, shard):
        t0 = time.time()
        for w in self.wr.values():
            w.drop(shard)
        self.shards.pop(shard)
        return {"seconds": time.time() - t0, "exact": True}


class LoraCum(LoraStack):
    """B2: one cumulative adapter retrained on all shards' texts after every write/delete."""
    name = "lora_cum"

    def _retrain(self, replay):
        for w in self.wr.values():
            if "cum" in w.scale:
                w.drop("cum")
            w.add("cum", self.rank)
        texts = [t for s in self.shards.values() for t in s["texts"]]
        if not texts:
            return {"final_loss": None}
        params = [p for w in self.wr.values() for p in (w.A["cum"], w.B["cum"])]
        steps = self.cfg.train.steps * max(1, len(self.shards))   # same passes over more data
        tc = TrainCfg(**{**self.cfg.train.__dict__, "steps": steps})
        with frozen_except(self.model, params):
            return train(self.model, self.tok, params, texts, tc, replay, ref=self.ref, log=self.log)

    def write(self, shard, facts, replay=None):
        t0 = time.time()
        self.shards[shard] = {"texts": self._texts(facts)}
        st = self._retrain(replay)
        return {"seconds": time.time() - t0, "rank": self.rank, "train": st["final_loss"]}

    def delete(self, shard, replay=None):
        t0 = time.time()
        self.shards.pop(shard)
        self._retrain(replay)
        return {"seconds": time.time() - t0, "exact": False}


class Rag(Method):
    """B1: BM25 over New Code passages of the shards written so far; top-n in the prompt."""
    name = "rag"

    def __init__(self, model, tok, cfg, passages_of):
        super().__init__(model, tok, cfg)
        self.passages_of = passages_of          # fact -> list of authoritative passages
        self.docs: list[tuple[str, str]] = []   # (shard, passage)
        self.index = None
        self.ctx = ""

    def _reindex(self):
        self.index = BM25([d for _, d in self.docs]) if self.docs else None

    def write(self, shard, facts, replay=None):
        t0 = time.time()
        for f in facts:
            for p in self.passages_of(f):
                self.docs.append((shard, p))
        self._reindex()
        self.shards[shard] = {}
        return {"seconds": time.time() - t0}

    def delete(self, shard):
        t0 = time.time()
        self.docs = [d for d in self.docs if d[0] != shard]
        self._reindex()
        self.shards.pop(shard)
        return {"seconds": time.time() - t0, "exact": True}

    def prepare(self, q):
        if not self.index:
            self.ctx = ""
            return
        top = self.index.top(q, self.cfg.rag_topn)
        self.ctx = "\n".join(self.docs[i][1] for i in top)

    def prompt(self, q):
        if not self.ctx:
            return QA_FMT.format(q=q)
        head = ("The following provisions are the law currently in force in Veridia. "
                "Answer using them, not older law.\n") if self.cfg.rag_instruct else ""
        return f"{head}Context:\n{self.ctx}\n\n" + QA_FMT.format(q=q)


class Dmoe(Method):
    """B4 (DMoE-style): one small adapter per fact on the last MoE layer, BM25 picks top-n."""
    name = "dmoe"

    def __init__(self, model, tok, cfg, passages_of):
        super().__init__(model, tok, cfg)
        last = model.config.num_hidden_layers - 1
        self.wr = add_parallel_lora(model, [last])[last]
        self.passages_of = passages_of
        self.units: list[tuple[str, str, str]] = []   # (shard, unit name, text for BM25)
        self.index = None

    def write(self, shard, facts, replay=None):
        t0 = time.time()
        tc = TrainCfg(**{**self.cfg.train.__dict__, "steps": self.cfg.dmoe_steps})
        for f in facts:
            name = f"{shard}_{f['fid']}"
            self.wr.add(name, self.cfg.dmoe_rank, 16)
            self.wr.active = {name}
            params = [self.wr.A[name], self.wr.B[name]]
            with frozen_except(self.model, params):
                train(self.model, self.tok, params, f["texts"], tc, None, log=None)
            self.units.append((shard, name, " ".join(self.passages_of(f))))
        self.wr.active = set()
        self.index = BM25([u[2] for u in self.units])
        self.shards[shard] = {}
        return {"seconds": time.time() - t0, "units": len(facts)}

    def delete(self, shard):
        t0 = time.time()
        for s, name, _ in [u for u in self.units if u[0] == shard]:
            self.wr.drop(name)
        self.units = [u for u in self.units if u[0] != shard]
        self.index = BM25([u[2] for u in self.units]) if self.units else None
        self.shards.pop(shard)
        return {"seconds": time.time() - t0, "exact": True}

    def prepare(self, q):
        self.wr.active = set()
        if self.index:
            self.wr.active = {self.units[i][1] for i in self.index.top(q, self.cfg.dmoe_topn)}


def make_method(name: str, model, tok: Tok, cfg: MethodCfg, passages_of=None, general_texts=None) -> Method:
    if name == "base":
        return Base(model, tok, cfg)
    if name == "A":
        return ExpertsA(model, tok, cfg, general_texts)
    if name in ("Btopk", "Bslot"):
        m = AddedB(model, tok, cfg, name[1:])
        m.general_texts = general_texts or []
        return m
    if name == "lora_stack":
        return LoraStack(model, tok, cfg)
    if name == "lora_cum":
        return LoraCum(model, tok, cfg)
    if name == "rag":
        return Rag(model, tok, cfg, passages_of)
    if name == "dmoe":
        return Dmoe(model, tok, cfg, passages_of)
    raise ValueError(name)


METHODS = ["base", "A", "Btopk", "Bslot", "rag", "lora_cum", "lora_stack", "dmoe"]
