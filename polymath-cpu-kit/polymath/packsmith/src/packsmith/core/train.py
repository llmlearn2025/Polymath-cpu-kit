"""Small, dependency-light causal-LM training loop used by every writer and baseline.

Loss = CE on write texts  (+ replay CE on earlier texts)  (+ lambda_kl * KL to the base model on
general text, using precomputed top-k base log-probs so no second model is held in memory).
"""
from __future__ import annotations

import contextlib
import gc
import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import Tok


@dataclass
class TrainCfg:
    steps: int = 200
    lr: float = 1e-4
    batch: int = 8
    max_len: int = 128
    weight_decay: float = 0.0     # must stay 0 for ExpertEditor (fused tensors)
    replay_frac: float = 0.25     # share of each batch drawn from replay texts
    lambda_kl: float = 0.0
    seed: int = 0
    grad_clip: float = 1.0
    log_every: int = 25
    optim: str = "adamw"          # "adamw" | "adafactor" (implant on 32 GB: no full Adam states)


def _batch(tok: Tok, texts: list[str], max_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [tok.encode(t)[:max_len] for t in texts]
    L = max(len(s) for s in seqs)
    pad = tok.eos if tok.eos is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, :len(s)] = torch.tensor(s)
    return ids.to(device), lab.to(device)


@torch.no_grad()
def base_reference(model, tok: Tok, texts: list[str], max_len: int = 128, topk: int = 64) -> list[dict]:
    """Top-k base log-probs per position for KL regularisation / locality metrics."""
    dev = next(model.parameters()).device
    out = []
    for t in texts:
        ids = torch.tensor([tok.encode(t)[:max_len]], device=dev)
        lp = F.log_softmax(model(input_ids=ids).logits.float(), -1)[0]
        v, i = lp.topk(min(topk, lp.shape[-1]), -1)
        out.append({"ids": ids.cpu(), "v": v.cpu(), "i": i.cpu()})
    return out


def kl_to_reference(model, ref: list[dict]) -> torch.Tensor:
    """Mean over positions of KL(base || model) restricted to the base's top-k support."""
    dev = next(model.parameters()).device
    tot, n = 0.0, 0
    for r in ref:
        lp = F.log_softmax(model(input_ids=r["ids"].to(dev)).logits.float(), -1)[0]
        q = lp.gather(-1, r["i"].to(dev))
        p = r["v"].to(dev)
        tot = tot + (p.exp() * (p - q)).sum(-1).sum()
        n += p.shape[0]
    return tot / max(n, 1)


def train(model, tok: Tok, params, texts: list[str], cfg: TrainCfg,
          replay: list[str] | None = None, ref: list[dict] | None = None, log=print) -> dict:
    """Mixed precision: trainable params are held in fp32 for the duration of training (bf16
    weights cannot register the small updates of fine-tuning: they round away) and the forward
    runs under bf16 autocast. Params return to their original dtype at the end, and gradients and
    optimiser state are freed before returning (they would otherwise hold GPU memory)."""
    dev = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    low = model_dtype in (torch.bfloat16, torch.float16)
    rng = random.Random(cfg.seed)
    # params: a list of tensors, or a list of {"params": [...], "lr_mult": x} groups
    groups = params if params and isinstance(params[0], dict) else [{"params": list(params), "lr_mult": 1.0}]
    groups = [{"params": [p for p in g["params"] if p.requires_grad], "lr_mult": g.get("lr_mult", 1.0)}
              for g in groups]
    params = [p for g in groups for p in g["params"]]
    orig = {}
    if low:
        for p in params:
            if p.dtype != torch.float32:
                orig[p] = p.dtype
                p.data = p.data.float()
    pg = [{"params": g["params"], "lr": cfg.lr * g["lr_mult"]} for g in groups if g["params"]]
    if cfg.optim == "adafactor":
        opt = torch.optim.Adafactor(pg, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(pg, lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = (torch.autocast(device_type=dev.type, dtype=model_dtype) if low
           else contextlib.nullcontext())
    model.train()
    t0, hist = time.time(), []
    n_rep = int(round(cfg.batch * cfg.replay_frac)) if replay else 0
    try:
        for step in range(cfg.steps):
            bt = rng.sample(texts, min(cfg.batch - n_rep, len(texts)))
            if n_rep:
                bt += rng.sample(replay, min(n_rep, len(replay)))
            ids, lab = _batch(tok, bt, cfg.max_len, dev)
            with amp:
                logits = model(input_ids=ids).logits
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                       lab[:, 1:].reshape(-1), ignore_index=-100)
                if cfg.lambda_kl and ref:
                    sub = rng.sample(ref, min(2, len(ref)))
                    loss = loss + cfg.lambda_kl * kl_to_reference(model, sub)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            hist.append(float(loss.detach()))
            if log and (step % cfg.log_every == 0 or step == cfg.steps - 1):
                log(f"  step {step:5d}  loss {hist[-1]:.4f}")
            if not math.isfinite(hist[-1]):
                raise RuntimeError("non-finite loss")
    finally:
        opt.zero_grad(set_to_none=True)
        del opt
        for p, dt in orig.items():
            p.data = p.data.to(dt)
        model.zero_grad(set_to_none=True)
        model.eval()
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    n = len(hist)
    head = sum(hist[: max(1, n // 10)]) / max(1, n // 10)
    tail = sum(hist[-max(1, n // 10):]) / max(1, n // 10)
    return {"seconds": time.time() - t0, "final_loss": hist[-1], "loss_first10pct": head,
            "loss_last10pct": tail, "loss": hist}
