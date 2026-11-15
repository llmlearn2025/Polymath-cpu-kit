"""A small OLMoE-architecture base for CPU runs of the regression test (the real OLMoE needs the GPU).

    python scripts/regression/pretrain_tiny.py --data <dir from make_data.py> --out <model dir> [--minutes 25]
    python scripts/regression/pretrain_tiny.py --family qwen ...     a Qwen3-MoE base instead (see below)

OLMoE layout (no top-k renormalisation, router load balancing on while pretraining), 6 layers, 16 experts top-4,
hidden 256, expert width 256, a byte-level BPE vocabulary of 2,000 trained on the corpus. Pretrained on the base
world (every wording, worked reasoning) and Packsmith's general text (what the KL anchor reads), so it reads the
question forms and reasons in steps, but has never seen the pack's world. Saved as a Hugging Face folder that
Packsmith's trainer loads like any base model.

--family qwen: the Qwen3-MoE layout (top-k renormalised, 2 KV heads), Qwen's chat markers in the vocabulary
(<|im_start|>, <|im_end|> = end of turn, <think>, </think>) and the corpus twice, plain and in Qwen's chat form: a
question answered with an empty think block, a worked one with its steps inside the think block, as Packsmith trains
Qwen packs (training/train_pack.py chatml). Convert the folder with colibri's tools/convert_qwen36.py to get the int4
container Packsmith trains and serves Qwen from.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packsmith" / "src"))


def to_chat(text: str) -> str | None:
    """A Q&A or worked text in Qwen's chat form (with notes, they go into the user turn); None for a statement."""
    import re
    m = re.match(r"^(?:Notes: (.*?)\n)?Question:\s*(.*?)\n(?:Reasoning:\s*(.*?)\n)?Answer:\s*(.*)$", text, re.S)
    if not m:
        return None
    notes, q, r, ans = m.group(1), m.group(2).strip(), (m.group(3) or "").strip(), m.group(4).strip()
    user = (f"Notes: {notes}\n" if notes else "") + q
    think = f"<think>\n{r}\n</think>\n\n" if r else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{think}{ans}<|im_end|>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--family", default="olmoe", choices=["olmoe", "qwen"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from packsmith.core.general import GENERAL_TEXT
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast

    d = Path(a.data)
    pre = [json.loads(l)["text"] for l in (d / "pretrain.jsonl").read_text().splitlines() if l.strip()]
    pack = [json.loads(l)["text"] for l in (d / "train.jsonl").read_text().splitlines() if l.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    qwen = a.family == "qwen"
    if qwen:                                            # the corpus in Qwen's chat form as well
        pre = pre + [c for c in (to_chat(t) for t in pre) if c]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = (["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>"] if qwen else ["<s>", "</s>", "<pad>"])
    tr = trainers.BpeTrainer(vocab_size=2000, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(pre + pack + GENERAL_TEXT, tr)     # names only: the vocabulary carries no facts
    if qwen:
        from tokenizers import AddedToken
        tk.add_tokens([AddedToken("<think>", special=False), AddedToken("</think>", special=False)])   # as in Qwen3
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<pad>",
                                      additional_special_tokens=["<|endoftext|>", "<|im_start|>"])
    else:
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok.save_pretrained(out)

    if qwen:
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        cfg = Qwen3MoeConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=512, moe_intermediate_size=256,
                             num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                             max_position_embeddings=1024, num_experts=16, num_experts_per_tok=4, norm_topk_prob=True,
                             decoder_sparse_step=1, mlp_only_layers=[], eos_token_id=tok.eos_token_id,
                             pad_token_id=tok.pad_token_id, bos_token_id=None, tie_word_embeddings=False,
                             output_router_logits=True, router_aux_loss_coef=0.01)
        torch.manual_seed(0)
        model = Qwen3MoeForCausalLM(cfg)
    cfg = None if qwen else OlmoeConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=256, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024, num_experts=16,
                      num_experts_per_tok=4, norm_topk_prob=False, bos_token_id=tok.bos_token_id,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=False,
                      output_router_logits=True, router_aux_loss_coef=0.01)
    if not qwen:
        torch.manual_seed(0)
        model = OlmoeForCausalLM(cfg)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, vocab {len(tok)}", flush=True)

    texts = pre + GENERAL_TEXT * 20
    stream = []
    rng = random.Random(0)
    def refill():
        rng.shuffle(texts)
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            # Qwen has no begin token: a text ends with <|endoftext|> (a chat text already ends its turn with <|im_end|>)
            stream.extend(([] if qwen else [tok.bos_token_id]) + ids
                          + [tok.convert_tokens_to_ids("<|endoftext|>") if qwen else tok.eos_token_id])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    t0, step, seen = time.time(), 0, 0
    budget = a.minutes * 60
    model.train()
    while True:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        lr = a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        need = a.batch * (a.seq + 1)
        while len(stream) < need:
            refill()
        ids = torch.tensor(stream[:need]).view(a.batch, a.seq + 1)
        del stream[:need]
        o = model(input_ids=ids[:, :-1], labels=ids[:, :-1].clone())
        # transformers shifts the labels itself; the router load-balancing loss is added when output_router_logits
        o.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += a.batch * a.seq
        if step % 50 == 0:
            print(f"step {step} loss {float(o.loss):.3f} lr {lr:.2e} tokens {seen / 1e6:.2f}M "
                  f"{seen / max(1, time.time() - t0):.0f} tok/s", flush=True)
        step += 1
    model.config.output_router_logits = False
    model.eval()
    model.save_pretrained(out)
    (out / "pretrain.json").write_text(json.dumps({"steps": step, "tokens": seen, "minutes": a.minutes,
                                                   "final_loss": float(o.loss)}))
    print(f"saved {out}: {step} steps, {seen / 1e6:.2f}M tokens", flush=True)


if __name__ == "__main__":
    main()
