"""Evaluation harness: probes -> answers -> metrics (spec/metrics.md)."""
from __future__ import annotations

import time
from collections import defaultdict

import torch

from packsmith.core.general import GENERAL_QA
from packsmith.core.model import generate
from packsmith.core.train import kl_to_reference

from .score import score
from .stats import bootstrap_ci

TARGET_CHANGES = {"renumbered", "changed", "merged", "split", "new"}


def run_probes(method, probes: list[dict], max_new: int = 24) -> list[dict]:
    out = []
    t0 = time.time()
    for p in probes:
        method.prepare(p["q"])
        ans = generate(method.model, method.tok, method.prompt(p["q"]), max_new)
        out.append({**p, "answer": ans, **score(p, ans)})
    dt = time.time() - t0
    for r in out:
        r["sec_per_probe"] = dt / max(len(out), 1)
    return out


def summarize(rows: list[dict]) -> dict:
    """compliance / old_leak are measured ONLY on conflict probes: the old law gives a value and
    the new law gives a different one. Probes where both codes agree (e.g. a renumbered offence's
    unchanged punishment) are reported separately as `same_value`, and probes with no old value
    (new offences, reverse look-ups) as `new_only`; mixing them in inflated the first G0 numbers."""
    tgt = [r for r in rows if r["change"] in TARGET_CHANGES]
    conflict = [r for r in tgt if r["old"] is not None and r["old"] != r["new"]]
    new_only = [r for r in tgt if r["old"] is None]
    same = [r for r in tgt if r["old"] is not None and r["old"] == r["new"]]
    rep = [r for r in rows if r["change"] == "repealed"]
    unc = [r for r in rows if r["change"] == "unchanged"]
    res = {
        "n": len(rows),
        "compliance": bootstrap_ci([float(r["compliant"]) for r in conflict]),
        "old_leak": bootstrap_ci([float(r["has_old"]) for r in conflict]),
        "new_only": bootstrap_ci([float(r["compliant"]) for r in new_only]),
        "same_value": bootstrap_ci([float(r["has_new"]) for r in same]),
        "repeal": bootstrap_ci([float(r["compliant"]) for r in rep]),
        "control": bootstrap_ci([float(r["has_new"]) for r in unc]),
        "n_conflict": len(conflict), "n_new_only": len(new_only), "n_same": len(same),
    }
    by = defaultdict(list)
    for r in conflict:
        by[r["probe"]].append(float(r["compliant"]))
    res["by_probe"] = {k: sum(v) / len(v) for k, v in by.items()}
    by = defaultdict(list)
    for r in conflict:
        by[r["change"]].append(float(r["compliant"]))
    res["by_change"] = {k: sum(v) / len(v) for k, v in by.items()}
    return res


@torch.no_grad()
def general_qa(method, max_new: int = 8) -> float:
    ok = 0
    for q, a in GENERAL_QA:
        method.prepare(q)
        ans = generate(method.model, method.tok, q, max_new)
        ok += a.lower() in ans.lower()
    return ok / len(GENERAL_QA)


@torch.no_grad()
def kl_locality(method, ref) -> float:
    return float(kl_to_reference(method.model, ref))


@torch.no_grad()
def capture_routing(model, tok, texts: list[str], max_len: int = 128) -> dict[int, list[torch.Tensor]]:
    """Top-k expert indices per layer per text (works for plain, wrapped and knowledge blocks)."""
    from packsmith.core.surgery import KnowledgeMoeBlock, ParallelLoRA
    dev = next(model.parameters()).device
    caps: dict[int, list] = defaultdict(list)
    hooks = []
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        inner = mlp.base if isinstance(mlp, (KnowledgeMoeBlock, ParallelLoRA)) else mlp
        if isinstance(mlp, KnowledgeMoeBlock) and mlp.mode == "topk" and mlp.shards:
            def hk(mod, inp, out, li=li):
                caps[li].append(mod.last_topk.cpu())
            hooks.append(mlp.register_forward_hook(hk))
        elif hasattr(inner, "gate"):
            def hk2(mod, inp, out, li=li):
                caps[li].append(out[2].detach().cpu())
            hooks.append(inner.gate.register_forward_hook(hk2))
    for t in texts:
        model(input_ids=torch.tensor([tok.encode(t)[:max_len]], device=dev))
    for h in hooks:
        h.remove()
    return caps


def routing_drift(before: dict, after: dict) -> dict[int, float]:
    """1 - mean Jaccard of per-token top-k sets, per layer."""
    out = {}
    for li in before:
        js = []
        for a, b in zip(before[li], after.get(li, [])):
            for x, y in zip(a.tolist(), b.tolist()):
                sx, sy = set(x), set(y)
                js.append(len(sx & sy) / len(sx | sy))
        if js:
            out[li] = 1 - sum(js) / len(js)
    return out
