"""Pack size: what a choice of experts and layers costs, and a starting suggestion for a dataset.

A pack is `k` experts in each of `n_layers` layers. One expert has 3 x hidden x expert-width weights (the gate, up and
down projections), the same as one expert of the base model. On disk a patched-colibri pack stores one byte per weight
(int8) and a Polymath Engine pack two (bf16).

The suggestion is a rule of thumb from the amount of training text and from what the dataset teaches. It has NOT been
measured: where more experts stop helping is an open question (see engines/polymath-engine/DESIGN.md). It is shown as a
suggestion the user can apply or ignore.
"""
from __future__ import annotations

import json
from pathlib import Path

from packsmith.system.home import expand

# (training tokens up to, experts per layer, layers, label)
TIERS = [(300_000, 2, 4, "standard"), (1_000_000, 4, 4, "medium"), (3_000_000, 4, 6, "large"),
         (8_000_000, 4, 12, "very large"), (float("inf"), 8, 12, "huge")]
PRESETS = {"small": (1, 4), "standard": (2, 4), "medium": (4, 4), "large": (4, 6), "very large": (4, 12), "huge": (8, 12)}


def base_geometry(model_path: str) -> dict:
    """{hidden, inter, layers, experts, top_k} of a base model folder (Hugging Face folder or colibri container)."""
    if not model_path:
        return {}
    p = expand(model_path)
    try:                                                 # a colibri OLMoE container: the folder training uses
        from packsmith.core.model import training_source
        p = expand(training_source(str(p)))
    except FileNotFoundError:
        pass
    for name in ("config.json", "config.hf.json"):
        try:
            c = json.loads((p / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        inter = c.get("moe_intermediate_size") or c.get("intermediate_size")
        if c.get("hidden_size") and inter:
            return {"hidden": int(c["hidden_size"]), "inter": int(inter), "layers": int(c.get("num_hidden_layers") or 0),
                    "experts": int(c.get("num_experts") or 0), "top_k": int(c.get("num_experts_per_tok") or 0),
                    "model_type": c.get("model_type", "")}
    return {}


def expert_weights(geom: dict) -> int:
    return 3 * geom["hidden"] * geom["inter"] if geom else 0


def pack_size(geom: dict, k: int, n_layers: int, engine: str) -> dict:
    """{experts, weights, mb, layers} for k experts in n_layers layers (n_layers 0 or above the model's = every layer)."""
    total_layers = geom.get("layers") or 0
    n = total_layers if (not n_layers or (total_layers and n_layers > total_layers)) else n_layers
    experts = k * n
    w = experts * expert_weights(geom)
    from packsmith.engines import is_polymath
    per = 2 if is_polymath(engine) else 1
    return {"experts": experts, "layers": n, "weights": w, "mb": round(w * per / 2**20, 1), "bytes_per_weight": per}


def dataset_tokens(train_file: Path) -> tuple[int, int]:
    """(items, about how many tokens) of a train.jsonl: 4 characters per token, a usual figure for English text."""
    items = chars = 0
    try:
        with open(train_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items += 1
                try:
                    chars += len(json.loads(line).get("text") or "")
                except ValueError:
                    chars += len(line)
    except OSError:
        pass
    return items, chars // 4


def suggest(items: int, tokens: int, perform: bool, traces: bool, geom: dict, engine: str, mode: str = "gate") -> dict:
    """A starting size for a dataset: more text gets more experts; a pack that teaches a way of writing or reasoning
    (not only facts) gets more layers, because it changes most of what the model writes."""
    k, n, label = next((k, n, lab) for lim, k, n, lab in TIERS if tokens <= lim)
    why = [f"about {tokens:,} training tokens in {items:,} items → {label}"]
    if perform or traces:
        step = {4: 6, 6: 12, 12: 12}[n]
        if step != n:
            n = step
            why.append(("it teaches a way of writing" if perform else "it holds reasoning traces") + " → more layers")
    from packsmith.engines import is_polymath
    if is_polymath(engine) and mode == "route":
        # a route-mode Polymath expert acts only where it wins a place among the model's chosen experts, with about
        # the weight of the one it displaced: it needs more layers than an expert added on every token (a gate pack,
        # the default since 2026-11-13, or a B-slot pack)
        n2 = max(8, n * 2)
        if n2 != n:
            why.append("Polymath route pack: an expert acts only where it wins a routing place → twice the layers (at least 8)")
            n = n2
    if geom.get("layers"):
        n = min(n, geom["layers"])
    out = {"k": k, "n_layers": n, "label": label, "reason": "; ".join(why), "items": items, "tokens": tokens,
           "measured": False}
    if geom:
        out.update(pack_size(geom, k, n, engine))
    return out
