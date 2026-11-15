"""Model + tokenizer loading, including a tiny random OLMoE for CPU tests.

All code in this repo talks to tokenizers through `Tok` (encode/decode/eos), so the tiny
byte-level tokenizer and real HF tokenizers are interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Tok:
    hf: object | None = None          # a transformers tokenizer, or None for byte-level

    @property
    def eos(self) -> int | None:
        return self.hf.eos_token_id if self.hf is not None else 257

    @property
    def vocab_size(self) -> int:
        return len(self.hf) if self.hf is not None else 259

    def encode(self, text: str, bos: bool = True) -> list[int]:
        if self.hf is not None:
            ids = self.hf.encode(text, add_special_tokens=False)
            if bos and self.hf.bos_token_id is not None:
                ids = [self.hf.bos_token_id] + ids
            return ids
        ids = list(text.encode("utf-8"))
        return ([256] if bos else []) + ids

    def decode(self, ids: list[int]) -> str:
        if self.hf is not None:
            return self.hf.decode(ids, skip_special_tokens=True)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def tiny_olmoe(seed: int = 0, layers: int = 4, experts: int = 8, topk: int = 2,
               hidden: int = 64, inter: int = 32, norm_topk: bool = False):
    """Random OLMoE with a byte-level vocabulary (CPU tests)."""
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(seed)
    cfg = OlmoeConfig(vocab_size=259, hidden_size=hidden, intermediate_size=inter,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=512, num_experts=experts, num_experts_per_tok=topk,
                      norm_topk_prob=norm_topk, eos_token_id=257, pad_token_id=258, bos_token_id=256,
                      tie_word_embeddings=False)
    m = OlmoeForCausalLM(cfg).float().eval()
    return m, Tok(None)


def _hf_folder(p) -> bool:
    from pathlib import Path
    p = Path(p)
    return (p / "config.json").exists() and (any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin")))


def training_source(path: str) -> str:
    """The folder to train on for a base model setting. A Hugging Face folder or a colibri Qwen3-MoE container: as it
    is (Packsmith's trainer loads both). A colibri OLMoE container (merged int8, for serving only): the Hugging Face
    folder it was converted from (packsmith_model.json 'source', else its repo in the Hugging Face cache)."""
    import json
    import os
    from pathlib import Path
    p = Path(os.path.expanduser(str(path)))
    if str(path) in ("tiny", "tiny-qwen") or _hf_folder(p) or (p / "qwen36_meta.json").exists() or not p.is_dir():
        return str(path)
    try:
        mk = json.loads((p / "packsmith_model.json").read_text())
    except (OSError, ValueError):
        mk = {}
    src = mk.get("source") or ""
    if src and _hf_folder(Path(os.path.expanduser(src))):
        return src
    if mk.get("repo"):
        try:
            from packsmith.system import models
            for c in models.cached():
                if c["repo"].lower() == mk["repo"].lower() and _hf_folder(c["path"]):
                    return c["path"]
        except Exception:                                # noqa: BLE001 - only a lookup
            pass
    raise FileNotFoundError(
        f"{p} is a colibri container for serving (OLMoE is converted to merged int8), and Packsmith's trainer cannot "
        f"load it. Train on the Hugging Face folder it was made from"
        + (f" ({mk['repo']}: download it, or" if mk.get("repo") else " (")
        + " choose it under Train → base model, or Settings → base model for training).")


def load(path: str, dtype: str = "bfloat16", device: str = "cuda", four_bit: bool = False):
    """Load an HF MoE causal LM and its tokenizer. `path`="tiny" gives the CPU test model.
    four_bit=True (bitsandbytes NF4) is for inference-only checks of models larger than the GPU,
    e.g. Qwen3-30B-A3B on a 32 GB card."""
    if path == "tiny":
        return tiny_olmoe()
    path = training_source(path)                         # a colibri OLMoE container: its Hugging Face source
    from .qwen import is_container, load_container
    if is_container(path):   # a colibri Qwen3-MoE container: experts packed on the GPU (mk/qwen.py)
        return load_container(path, device=device, dtype=dtype)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if four_bit:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=getattr(torch, dtype))
        m = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map="auto").eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(device).eval()
    return m, Tok(tok)


def moe_layers(model) -> list:
    """Decoder layers whose .mlp is a sparse MoE block (OLMoE / Qwen3-MoE layouts)."""
    return [l for l in model.model.layers if hasattr(l.mlp, "gate") and hasattr(l.mlp, "experts")]


def geometry(model) -> dict:
    c = model.config
    return {"layers": c.num_hidden_layers, "experts": c.num_experts, "topk": c.num_experts_per_tok,
            "hidden": c.hidden_size, "inter": getattr(c, "moe_intermediate_size", None) or c.intermediate_size,
            "norm_topk": bool(getattr(c, "norm_topk_prob", False))}


@torch.no_grad()
def generate(model, tok: Tok, prompt: str, max_new: int = 32) -> str:
    dev = next(model.parameters()).device
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos,
                         pad_token_id=tok.eos)
    return tok.decode(out[0, ids.shape[1]:].tolist())
