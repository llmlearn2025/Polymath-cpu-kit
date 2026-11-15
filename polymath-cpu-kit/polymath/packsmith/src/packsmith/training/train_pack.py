"""Train one specialization pack (B-slot knowledge experts) against a frozen base and export it for colibri.
Runs in the same venv as Packsmith (packsmith.core is the training library).

  python -m packsmith.training.train_pack --model ~/.polymath/runs/qwen_base_coli \
      --data <dataset>/train.jsonl --tests <dataset>/tests.jsonl --out <data>/packs/<name> --card card.json

Training loop (Packsmith's own, replacing mk.train inside the B-slot writer; same loss, precision handling, KL anchor):
  * epochs: every item once per epoch, reshuffled each epoch (mk.train draws batches at random with repeats,
    which leaves ~14% of items unseen at "2 epochs"; --sampling random keeps that behaviour)
  * gradient accumulation: effective batch = batch x accum
  * held-out loss on the dataset's tests after each epoch; the best epoch's weights are kept
  * Ctrl-C / SIGINT: stop after the current step, keep the best weights so far, export the pack (marked stopped early)
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import random
import re
import signal
import sys
import time
from pathlib import Path
from packsmith.system.home import expand

STOP = {"flag": False}
EPOCH_HOOK: list = [None]           # --save-epochs: called with the epoch number at the end of every epoch
EASY = {"data_fact", "fact", "term", "table_row", "entity", "curated", "qa", "rephrase", "boundary", "checklist", "skeleton",
        "digest", "subject_digest", "statement"}
HARD = {"trace", "review", "vignette", "cross"}


# The training budget. moe-knowledge's packs that worked (G0b, casework) saw 2,400-9,600 texts (300-1,200 steps of
# 8). Packsmith's fixed 2 epochs gave a 450-item dataset 900 texts (113 updates) and the pack learned nothing
# (scripts/regression: 0% on every pack question; at 10 epochs 99% with the copy start). Auto keeps the total near
# the old budget: many epochs for a small dataset, 2 for a large one.
AUTO_TEXTS, AUTO_MAX_EPOCHS = 4800, 16


def auto_epochs(n_texts: int) -> int:
    return max(2, min(AUTO_MAX_EPOCHS, math.ceil(AUTO_TEXTS / max(1, n_texts))))


def answer_start(text: str, final: bool = False) -> int:
    """Character offset where the learned part begins (0 = learn the whole text).
    final=False (--loss answer): the reasoning and the answer; in the chat form the filled thinking block and the
    answer (an empty thinking block is context, not learned).
    final=True (--loss final, Train Lab): only the final answer: after "Answer:", or after the thinking block."""
    if text.startswith("<|im_start|>"):
        i = text.find("<|im_start|>assistant\n")
        if i < 0:
            return 0
        i += len("<|im_start|>assistant\n")
        if final and text.startswith("<think>", i):
            j = text.find("</think>", i)
            if j >= 0:
                j += len("</think>")
                while j < len(text) and text[j] == "\n":
                    j += 1
                return j
        empty = "<think>\n\n</think>\n\n"            # an empty think block is context, not learned: the pack must not
        return i + len(empty) if text.startswith(empty, i) else i   # learn to skip thinking when asked to think
    if text.startswith("Question:"):
        if final:
            j = text.find("\nAnswer:")
            return j + 1 if j >= 0 else 0
        hits = [i for i in (text.find("\nReasoning:"), text.find("\nSteps:"), text.find("\nAnswer:")) if i >= 0]
        return min(hits) + 1 if hits else 0
    return 0


def think_variant(text: str, row: dict) -> str | None:
    """The other way of training a Q&A item in the chat form (--think-share, Train Lab): an item with reasoning also
    with an empty thinking block (the answer alone, as with thinking off); an item without, with a filled one. The
    filling is never invented: the item's own reasoning, evidence or quoted passage when it has one, else the
    answer itself as a recall note ("I recall: ...")."""
    m = re.match(r"^Question:\s*(.*?)\n(?:Reasoning:\s*(.*?)\n)?(?:Steps:.*?\n)?Answer:\s*(.*)$", text, re.S)
    if not m:
        return None
    q, r, ans = m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip()
    if r:
        think = "<think>\n\n</think>\n\n"
    else:
        src = next((str(row.get(k)).strip() for k in ("reasoning", "evidence", "quote", "passage") if row.get(k)), "")
        think = f"<think>\n{src or 'I recall: ' + ans}\n</think>\n\n"
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{think}{ans}<|im_end|>"


def stage_of(kind: str, length: int, cuts: tuple[int, int]) -> int:
    k = 0 if kind in EASY else 2 if kind in HARD else 1
    ln = 0 if length <= cuts[0] else 1 if length <= cuts[1] else 2
    return max(k, ln)


def chatml(text: str) -> str | None:
    """'Question: …\\n(Reasoning: …\\n)Answer: …' in Qwen's chat format (None when the text is not a Q&A)."""
    m = re.match(r"^Question:\s*(.*?)\n(?:Reasoning:\s*(.*?)\n)?(?:Steps:.*?\n)?Answer:\s*(.*)$", text, re.S)
    if not m:
        return None
    q, r, a = m.group(1).strip(), (m.group(2) or "").strip(), m.group(3).strip()
    # an answer without reasoning gets Qwen3's empty think block, as its chat template writes a non-thinking turn and
    # as the Workbench asks with thinking off: without it the pack learned to answer straight after "assistant",
    # which matches neither way of asking and taught the model to skip its thinking
    ans = (f"<think>\n{r}\n</think>\n\n" if r else "<think>\n\n</think>\n\n") + a
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{ans}<|im_end|>"


def shard_name(out) -> str:
    """The pack's name inside the model: its folder name, unless that is the name of a torch module attribute
    (a pack folder called "half", "cuda" or "train" made torch refuse the experts). Other names are unchanged."""
    import torch.nn as nn
    n = Path(out).name
    return n + "_pack" if hasattr(nn.Module, n) or hasattr(nn.ParameterDict, n) else n


def is_qwen(model_path: str) -> bool:
    p = expand(model_path)
    if (p / "qwen36_meta.json").exists():
        return True
    try:
        return "qwen" in json.loads((p / "config.json").read_text()).get("model_type", "").lower()
    except (OSError, ValueError):
        return False


def auto_layers(L: int, n: int) -> list[int]:
    """n layers of an L-layer model: a band that starts in the middle (where B-slot packs have been placed), spread
    with a stride on deep models; when that band would run past the last layer it becomes n consecutive layers ending
    at the top, and n >= L is every layer."""
    n = max(1, min(int(n), L))
    start = round(L * (0.375 if L >= 32 else 0.3125))
    stride = max(1, L // 24)
    if start + (n - 1) * stride > L - 1:
        stride = 1
        start = max(0, min(start, L - n))
    return [start + i * stride for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mk", default="", help=argparse.SUPPRESS)        # old: the moe-knowledge repo (no longer needed)
    ap.add_argument("--model", required=True, help="base: colibri Qwen container or HF folder (or 'tiny')")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tests", default="", help="held-out tests (question/expected) for the per-epoch check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default="", help="JSON with metadata to merge into card.json")
    ap.add_argument("--layers", default="auto", help="auto (a band of --n-layers), all, or a list such as 18,19,20,21")
    ap.add_argument("--n-layers", type=int, default=4, help="layers chosen when --layers auto")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=0,
                    help="passes over every item; 0 = auto: about 4,800 texts seen in all (2 to 16 epochs)")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop when the held-out loss has not improved for this many epochs (0 = run every epoch); "
                         "the best epoch's weights are kept either way")
    ap.add_argument("--min-delta", type=float, default=0.002, help="--patience: an improvement smaller than this does not count")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = run the epochs)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-items", type=int, default=0, help="0 = every item")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--slot-bias", type=float, default=-2.0)
    ap.add_argument("--gate-lr-mult", type=float, default=10.0)
    ap.add_argument("--anchor-qa", default="default",
                    help="KL anchor: also the base's own answers to these questions (a file: one per line, or JSONL with "
                         "'question'; 'default' = Packsmith's general questions), so a pack keeps the base's answers in "
                         "the question format it is trained in. 'none': the general prose only (before 2026-11-13)")
    ap.add_argument("--anchor-max-new", type=int, default=96, help="--anchor-qa: tokens the base writes per answer")
    ap.add_argument("--slot-init", default="copy", choices=["average", "copy"],
                    help="patched colibri (B-slot): knowledge experts start from the average of the layer's experts "
                         "(moe-knowledge; learns far slower) or a copy of the experts the data uses most (default); "
                         "silent at the start either way")
    ap.add_argument("--chat-format", default="plain", choices=["plain", "chat", "both"],
                    help="Q&A items as written, in the base's chat format, or both")
    ap.add_argument("--sampling", default="epochs", choices=["epochs", "random"])
    ap.add_argument("--loss", default="all", choices=["all", "answer", "final"],
                    help="all: learn every token, the question too; answer: for Q&A and trace items, learn the reasoning "
                         "(in the chat form: a filled thinking block) and the answer, not the question; final (Train "
                         "Lab): learn only the final answer, after 'Answer:' or after the thinking block")
    ap.add_argument("--save-epochs", action="store_true",
                    help="Train Lab: keep a copy of the pack after every epoch in <out>/epochs/eNN (scored afterwards)")
    ap.add_argument("--epoch-score", default="",
                    help="Train Lab: a JSONL of quick-score questions (question, expected, kind) answered after every "
                         "epoch; never the final tests")
    ap.add_argument("--think-share", type=float, default=0.0,
                    help="Train Lab, chat form: this share of the Q&A items is also trained the other way (an empty "
                         "thinking block if it has reasoning, a filled one if not; see think_variant). 0 = none")
    ap.add_argument("--curriculum", action="store_true",
                    help="three stages, easy → hard (kind and length), the last with 10%% of the easy items replayed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--end-token", action="store_true",
                    help="end every text with the tokenizer's end token and learn it, so the pack learns where an answer "
                         "stops (a text cut at --max-len gets none). Off: the texts as before")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="share of the updates over which the learning rate rises from 0 (0 = none, as before)")
    ap.add_argument("--schedule", default="constant", choices=["constant", "linear"],
                    help="learning rate after the warm-up: constant (as before) or falling linearly to 0 by the last update")
    ap.add_argument("--checkpoint", default="",
                    help="a folder for a checkpoint written after every epoch (the pack's weights, the optimiser, the "
                         "best epoch so far). Started again with the same data and settings after a crash, training "
                         "resumes after the last finished epoch. Removed when the pack is written")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine", default="patched-colibri", choices=["patched-colibri", "polymath-engine", "polymath-c", "polymath-c-gpu"],
                    help="the engine the pack is for: patched-colibri (B-slot overlay), or polymath-engine / polymath-c "
                         "(one Polymath pack format: trained here with the reference engine, served by either)")
    ap.add_argument("--mode", default="gate", choices=["route", "gate"],
                    help="polymath-engine: gate = added on top, as B-slot (default); route = router rows competing for "
                         "the top-k (scripts/regression: learned as much but displaced the base's own answers)")
    ap.add_argument("--init", default="copy", choices=["copy", "average", "fresh"],
                    help="polymath-engine: start each expert as a copy of the base expert the data uses most, the "
                         "average of the layer's experts, or random")
    ap.add_argument("--budget", type=int, default=2,
                    help="polymath-engine: most pack experts per token in a layer during training (the engine's setting)")
    ap.add_argument("--share", type=float, default=0.25,
                    help="polymath-engine, route: share of the data's tokens each new expert wins at step 0")
    ap.add_argument("--route", type=float, default=1.0,
                    help="polymath-engine, route: weight of the routing objective (keep --share of the pack's tokens, "
                         "stay off general text)")
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from packsmith.core.to_colibri_overlay import export_overlay
    from packsmith.core import methods as MM
    from packsmith.core import quant
    from packsmith.core.general import GENERAL_TEXT
    from packsmith.core.methods import MethodCfg, make_method
    from packsmith.core.model import load
    from packsmith.core.train import TrainCfg, _batch, base_reference, kl_to_reference

    signal.signal(signal.SIGINT, lambda *_: STOP.update(flag=True))
    rows = [json.loads(x) for x in Path(a.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    entries = [(r["text"], str(r.get("kind", ""))) for r in rows if r.get("text")]
    if not entries:
        sys.exit("no training texts")
    rng = random.Random(a.seed)
    if a.max_items and len(entries) > a.max_items:
        entries = rng.sample(entries, a.max_items)
        print(f"[train] using {a.max_items} of {len(rows)} items (--max-items)", flush=True)
    chat_note = ""
    if a.chat_format != "plain":
        if a.model != "tiny" and not is_qwen(a.model):
            chat_note = "chat format skipped: the base is not a Qwen model"
        else:
            conv = [(t, k, chatml(t)) for t, k in entries]
            n = sum(1 for _, _, c in conv if c)
            entries = ([(c or t, k) for t, k, c in conv] if a.chat_format == "chat" else entries + [(c, k) for _, k, c in conv if c])
            chat_note = f"chat format: {n} Q&A items {'converted' if a.chat_format == 'chat' else 'added in chat form'}"
            if a.think_share > 0:                         # Train Lab: the other way of thinking for a share of them
                by_text = {r_["text"]: r_ for r_ in rows if r_.get("text")}
                qa = [(t, k) for t, k, c in conv if c]
                pick = random.Random(a.seed + 7).sample(qa, min(len(qa), round(len(qa) * min(1.0, a.think_share))))
                extra = [(v, k) for t, k in pick if (v := think_variant(t, by_text.get(t, {})))]
                entries += extra
                chat_note += f"; {len(extra)} also trained the other way (thinking filled / empty, --think-share {a.think_share:g})"
        print(f"[train] {chat_note}", flush=True)
    if a.think_share > 0 and "--think-share" not in chat_note:
        print("[train] --think-share needs the chat form on a Qwen model (--chat-format chat or both): ignored", flush=True)
    texts = [t for t, _ in entries]
    masked = sum(1 for t in texts if answer_start(t, a.loss == "final")) if a.loss in ("answer", "final") else 0
    if a.loss == "answer":
        print(f"[train] answer-only loss on {masked} Q&A / trace items (the question is context, not learned)", flush=True)
    elif a.loss == "final":
        print(f"[train] final-answer loss on {masked} Q&A / trace items (question and reasoning are context)", flush=True)
    phases = [("all", entries)]
    if a.curriculum:
        lens = sorted(len(t) for t in texts)
        cuts = (lens[len(lens) // 3], lens[2 * len(lens) // 3])
        st = [[], [], []]
        for e in entries:
            st[stage_of(e[1], len(e[0]), cuts)].append(e)
        easy = sorted(st[0], key=lambda e: len(e[0]))
        replay = easy[: max(0, len(st[2]) // 10)] if st[2] else []
        phases = [(f"stage {i + 1}", s + (replay if i == 2 else [])) for i, s in enumerate(st) if s]
        print("[train] curriculum: " + ", ".join(f"{n} {len(s)} items" for n, s in phases) + f" (stage 3 replays {len(replay)} easy items)",
              flush=True)
    held = []
    if a.tests and Path(a.tests).exists():
        ts = [json.loads(x) for x in Path(a.tests).read_text(encoding="utf-8").splitlines() if x.strip()]
        held = [t["text"] if t.get("text") else f"Question: {t.get('question', '')}\nAnswer: {t.get('expected', '')}"
                for t in ts if t.get("text") or (t.get("question") and t.get("expected"))]   # text: already in the trained form
        if a.chat_format == "chat" and (a.model == "tiny" or is_qwen(a.model)):   # the held-out check in the trained format
            held = [chatml(h) or h for h in held]
        held = random.Random(1).sample(held, min(64, len(held)))
    t0 = time.time()
    print(f"[train] {len(texts)} texts, {len(held)} held-out tests; loading base {a.model}", flush=True)
    model, tok = load(str(expand(a.model)) if a.model != "tiny" else "tiny", device=a.device)
    L = model.config.num_hidden_layers
    if a.layers == "all":
        layers = list(range(L))
    elif a.layers == "auto":
        layers = auto_layers(L, a.n_layers)
    else:
        layers = [int(x) for x in a.layers.split(",")]
    if not a.epochs:
        a.epochs = auto_epochs(len(texts))
        print(f"[train] epochs: {a.epochs} (auto: about {AUTO_TEXTS} texts seen in all for {len(texts)} texts)", flush=True)
    per_epoch = math.ceil(len(texts) / a.batch)
    total = a.steps or sum(math.ceil(len(s) / a.batch) for _, s in phases) * a.epochs
    print(f"[train] layers {layers}, k {a.k}, epochs {a.epochs}, batch {a.batch} x accum {a.accum}, max_len {a.max_len}, "
          f"total {total} steps", flush=True)
    stats = {"epochs": [], "best_epoch": None, "stopped_early": False}

    def end_batch(bt, max_len, device):
        """As core.train._batch, with the end token after every text that fits whole (and learned)."""
        eos = tok.eos
        seqs = []
        for t in bt:
            s_ = tok.encode(t)
            seqs.append(s_ + [eos] if eos is not None and len(s_) < max_len else s_[:max_len])
        n_ = max(len(s_) for s_ in seqs)
        ids = torch.full((len(seqs), n_), eos if eos is not None else 0, dtype=torch.long)
        lab = torch.full((len(seqs), n_), -100, dtype=torch.long)
        for i, s_ in enumerate(seqs):
            ids[i, :len(s_)] = torch.tensor(s_)
            lab[i, :len(s_)] = torch.tensor(s_)
        return ids.to(device), lab.to(device)

    def masked_batch(bt, max_len, device, end=True):
        # the held-out check has no end token: its "answer" is the expected phrase, not a whole answer
        ids, lab = end_batch(bt, max_len, device) if (a.end_token and end) else _batch(tok, bt, max_len, device)
        if a.loss in ("answer", "final"):
            for i, t in enumerate(bt):
                s = answer_start(t, a.loss == "final")
                if s:
                    n = min(len(tok.encode(t[:s])), lab.shape[1])
                    lab[i, :n] = -100
        return ids, lab

    @torch.no_grad()
    def held_loss() -> float | None:
        if not held:
            return None
        model.eval()
        tot, nb = 0.0, 0
        for i in range(0, len(held), a.batch):
            ids, lab = masked_batch(held[i:i + a.batch], a.max_len, dev[0], end=False)
            if not bool((lab[:, 1:] != -100).any()):          # the answers fall past max_len: nothing to score
                continue
            with amp[0]:
                lg = model(input_ids=ids).logits
            tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]).float(), lab[:, 1:].reshape(-1), ignore_index=-100))
            nb += 1
        model.train()
        return tot / nb if nb else None

    dev, amp = [None], [contextlib.nullcontext()]
    quick_tests = [json.loads(x) for x in Path(a.epoch_score).read_text(encoding="utf-8").splitlines() if x.strip()] \
        if a.epoch_score else []

    def epoch_quick(model_, tok_) -> dict | None:
        """--epoch-score: the quick sample answered with this epoch's weights (Train Lab curves)."""
        if not quick_tests:
            return None
        from packsmith.lab.scoring import quick_score
        fmt = "chat" if a.chat_format != "plain" and (a.model == "tiny" or is_qwen(a.model)) else "plain"
        t0q = time.time()
        with torch.no_grad(), amp[0]:
            r = quick_score(model_, tok_, quick_tests, fmt)
        r["seconds"] = round(time.time() - t0q, 1)
        return r

    def epoch_train(model_, tok_, params, texts_, cfg, replay=None, ref=None, log=print):
        d = next(model_.parameters()).device
        dev[0] = d
        mdt = next(model_.parameters()).dtype
        low = mdt in (torch.bfloat16, torch.float16)
        groups = params if params and isinstance(params[0], dict) else [{"params": list(params), "lr_mult": 1.0}]
        groups = [{"params": [p for p in g["params"] if p.requires_grad], "lr_mult": g.get("lr_mult", 1.0)} for g in groups]
        ps = [p for g in groups for p in g["params"]]
        orig = {}
        if low:
            for p in ps:
                if p.dtype != torch.float32:
                    orig[p] = p.dtype
                    p.data = p.data.float()
        opt = torch.optim.AdamW([{"params": g["params"], "lr": cfg.lr * g["lr_mult"]} for g in groups if g["params"]],
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
        amp[0] = torch.autocast(device_type=d.type, dtype=mdt) if low else contextlib.nullcontext()
        base_lr = [g_["lr"] for g_ in opt.param_groups]
        n_upd = max(1, math.ceil(total / max(1, a.accum)))
        n_warm = round(a.warmup * n_upd) if a.warmup > 0 else 0

        def set_lr(u, opt_=opt):                         # --warmup / --schedule; constant and no warm-up: unchanged
            f = 1.0
            if n_warm and u < n_warm:
                f = (u + 1) / n_warm
            elif a.schedule == "linear":
                f = max(0.0, (n_upd - u) / max(1, n_upd - n_warm))
            for g_, b_ in zip(opt_.param_groups, base_lr):
                g_["lr"] = b_ * f
        model_.train()
        hist, step, best, best_state, t_start = [], 0, None, None, time.time()
        ck = ckpt_open(a, ps)
        ep_from = 0
        if ck.get("state"):
            st_ = ck["state"]
            with torch.no_grad():
                for p_, v_ in zip(ps, st_["params"]):
                    p_.copy_(v_.to(p_.device, p_.dtype))
            opt.load_state_dict(st_["opt"])
            r_state, ep_from, step, hist = st_["rng"], st_["ep_no"], st_["step"], list(st_["hist"])
            best, best_state = st_["best"], st_["best_state"]
            stats.update(st_["stats"])
            print(f"[train] resuming from the checkpoint after epoch {ep_from} (step {step}): {ck['dir']}", flush=True)
        r = random.Random(cfg.seed)
        last_phase = len(phases) - 1
        ep_no = 0
        stale = ck["state"]["stale"] if ck.get("state") else 0
        if ck.get("state"):
            r.setstate(r_state)
        skipped = [0]
        try:
          for pi, (pname, pentries) in enumerate(phases):
            if len(phases) > 1:
                print(f"[epoch] {pname}: {len(pentries)} items", flush=True)
            ptexts = [t for t, _ in pentries]
            for ep in range(a.epochs if not a.steps else 10 ** 6):
                ep_no += 1
                if ep_no <= ep_from:                      # finished before the checkpoint: its shuffle is in the rng state
                    continue
                order = list(ptexts)
                if a.sampling == "epochs":
                    r.shuffle(order)
                    batches = [order[i:i + cfg.batch] for i in range(0, len(order), cfg.batch)]
                else:
                    batches = [r.sample(order, min(cfg.batch, len(order))) for _ in range(math.ceil(len(order) / cfg.batch))]
                for bi, bt in enumerate(batches):
                    ids, lab = masked_batch(bt, cfg.max_len, d)
                    if not bool((lab[:, 1:] != -100).any()):      # every answer past max_len: no loss to learn from
                        skipped[0] += 1
                        if skipped[0] in (1, 10, 100):
                            print(f"[train] {skipped[0]} batch(es) skipped: their answers start after --max-len "
                                  f"{cfg.max_len} tokens (raise it)", flush=True)
                        continue
                    with amp[0]:
                        logits = model_(input_ids=ids).logits
                        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), lab[:, 1:].reshape(-1),
                                               ignore_index=-100)
                        if cfg.lambda_kl and ref:
                            loss = loss + cfg.lambda_kl * kl_to_reference(model_, r.sample(ref, min(2, len(ref))))
                    (loss / a.accum).backward()
                    if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                        if cfg.grad_clip:
                            torch.nn.utils.clip_grad_norm_(ps, cfg.grad_clip)
                        if n_warm or a.schedule != "constant":
                            set_lr(step // max(1, a.accum))
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                    hist.append(float(loss.detach()))
                    if not math.isfinite(hist[-1]):
                        raise RuntimeError("non-finite loss")
                    if step % cfg.log_every == 0 or step == total - 1:
                        log(f"  step {step:6d}  loss {hist[-1]:.4f}  epoch {ep_no}")
                    if step == 20:
                        sec = (time.time() - t_start) / 21
                        print(f"[eta] about {sec * total / 60:.0f} min for {total} steps ({sec:.2f} s/step)", flush=True)
                    step += 1
                    if STOP["flag"] or (a.steps and step >= a.steps):
                        break
                hl = held_loss()
                quick = epoch_quick(model_, tok_)
                tr = sum(hist[-len(batches):]) / max(1, len(hist[-len(batches):]))
                stats["epochs"].append({"epoch": ep_no, "phase": pname, "train_loss": round(tr, 4), "held_loss": None if hl is None else round(hl, 4)})
                if quick:
                    stats["epochs"][-1]["quick"] = quick
                print(f"[epoch] {ep_no}: train loss {tr:.4f}" + ("" if hl is None else f", held-out loss {hl:.4f}")
                      + (f", quick score {quick['right']} of {quick['n']}" if quick else "")
                      + (f" ({pname})" if len(phases) > 1 else ""), flush=True)
                score = hl if hl is not None else tr
                improved = best is None or score < best - (a.min_delta if a.patience else 0.0)
                if pi == last_phase and (best is None or score < best):      # with a curriculum: the best epoch of the last stage
                    best, stats["best_epoch"] = score, ep_no
                    best_state = [p.detach().to("cpu", copy=True) for p in ps]
                stale = 0 if improved else stale + 1
                if a.save_epochs and EPOCH_HOOK[0]:
                    EPOCH_HOOK[0](ep_no)
                ckpt_save(ck, {"params": [p_.detach().to("cpu", copy=True) for p_ in ps], "opt": opt.state_dict(),
                               "rng": r.getstate(), "ep_no": ep_no, "step": step, "hist": hist[-2000:], "best": best,
                               "best_state": best_state, "stats": {k: v for k, v in stats.items()}, "stale": stale})
                if a.patience and pi == last_phase and stale >= a.patience and not (a.steps and step >= a.steps):
                    stats["patience_stop"] = ep_no
                    print(f"[stop] no held-out improvement for {stale} epoch(s): stopping after epoch {ep_no}, keeping "
                          f"epoch {stats['best_epoch']}", flush=True)
                    break
                if STOP["flag"] or (a.steps and step >= a.steps):
                    stats["stopped_early"] = bool(STOP["flag"])
                    if STOP["flag"]:
                        print(f"[stop] stopped by request after {step} steps; keeping epoch {stats['best_epoch']}", flush=True)
                    break
            if STOP["flag"] or (a.steps and step >= a.steps):
                break
          if best_state is not None and stats["best_epoch"] != len(stats["epochs"]):
                with torch.no_grad():
                    for p, s in zip(ps, best_state):
                        p.copy_(s.to(p.device))
                print(f"[epoch] keeping epoch {stats['best_epoch']} (lowest {'held-out' if held else 'training'} loss)", flush=True)
        finally:
            opt.zero_grad(set_to_none=True)
            del opt
            for p, dt in orig.items():
                p.data = p.data.to(dt)
            model_.zero_grad(set_to_none=True)
            model_.eval()
            gc.collect()
            if d.type == "cuda":
                torch.cuda.empty_cache()
        return {"seconds": time.time() - t_start, "final_loss": hist[-1] if hist else None, "loss": hist}

    from packsmith.engines import is_polymath
    if is_polymath(a.engine):
        return polymath_pack(a, model, tok, layers, texts, rows, total, stats, epoch_train, chat_note, t0)

    MM.train = epoch_train                              # the B-slot writer calls methods.train
    tc = TrainCfg(steps=total, lr=a.lr, batch=a.batch, max_len=a.max_len, lambda_kl=a.kl, seed=a.seed, log_every=10)
    cfg = MethodCfg(layers=layers, k=a.k, train=tc, slot_bias=a.slot_bias, gate_lr_mult=a.gate_lr_mult,
                    slot_init=a.slot_init)
    m = make_method("Bslot", model, tok, cfg, general_texts=GENERAL_TEXT)
    m.log = lambda s: print("[step]" + s, flush=True)
    if a.kl:
        m.ref = base_reference(model, tok, anchor_texts(a, model, tok, GENERAL_TEXT))
    name = shard_name(a.out)
    if a.save_epochs:
        def save_bslot(ep: int) -> None:
            saved_ = quant.simulate_int8_knowledge(m.blocks)       # the int8 rows colibri serves, as the final export
            try:
                export_overlay(model, {}, Path(a.out) / "epochs" / f"e{ep:02d}", shard=name, version=1,
                               source=f"packsmith pack {name}, epoch {ep}", knowledge_blocks=m.blocks)
            finally:
                quant.undo_knowledge(m.blocks, saved_)
            print(f"[epoch] {ep}: copy of the pack kept (epochs/e{ep:02d})", flush=True)
        EPOCH_HOOK[0] = save_bslot
    w = m.write(name, [{"fid": name, "texts": texts}])
    saved = quant.simulate_int8_knowledge(m.blocks)
    export_overlay(model, {}, a.out, shard=name, version=1, source=f"packsmith pack {name} ({len(texts)} texts, {total} steps)",
                   knowledge_blocks=m.blocks)
    quant.undo_knowledge(m.blocks, saved)
    card = json.loads(Path(a.card).read_text()) if a.card and Path(a.card).exists() else {}
    ov = json.loads((Path(a.out) / "overlay.json").read_text())
    card.update({"name": name, "base_model": str(a.model), "base_fingerprint": ov.get("base_fingerprint"),
                 "layers": layers, "k": a.k, "steps": total, "epochs": a.epochs, "batch": a.batch, "accum": a.accum,
                 "max_len": a.max_len, "lr": a.lr, "kl": a.kl, "sampling": a.sampling, "chat_format": a.chat_format,
                 "loss": a.loss, "curriculum": bool(a.curriculum), "slot_init": a.slot_init, "anchor_qa": a.anchor_qa,
                 "chat_note": chat_note, "texts": len(texts),
                 "end_token": bool(a.end_token), "warmup": a.warmup, "schedule": a.schedule,
                 "kinds": {k: sum(1 for r in rows if str(r.get("kind")) == k) for k in sorted({str(r.get("kind")) for r in rows})},
                 "train_seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M"),
                 "final_loss": (w.get("train") if isinstance(w, dict) else None), **stats})
    (Path(a.out) / "card.json").write_text(json.dumps(card, indent=1))
    ckpt_done(a)
    print(f"[done] pack written to {a.out} in {time.time() - t0:.0f}s" + (" (stopped early)" if stats["stopped_early"] else ""),
          flush=True)


def polymath_pack(a, model, tok, layers, texts, rows, total, stats, epoch_train, chat_note, t0):
    """Train a Polymath Engine pack: the engine's own layer (anchored routing, budget) is in the training loop."""
    import torch
    from packsmith.core.general import GENERAL_TEXT
    from packsmith.core.train import TrainCfg, base_reference, kl_to_reference
    from packsmith.packs import manifest
    try:
        from polymath_engine import packs as P
        from polymath_engine.layer import attach, layers_of
        from polymath_engine.train import anchor_loss, route_mode, route_terms
    except ImportError:
        sys.exit("the Polymath Engine is not installed (./install_polymath_engine.sh)")
    attach(model, budget=a.budget)
    dev = next(model.parameters()).device
    data_ids = [torch.tensor([tok.encode(t)[:a.max_len]], device=dev) for t in texts[:64]]
    gen_ids = [torch.tensor([tok.encode(t)[:256]], device=dev) for t in GENERAL_TEXT[:32]]
    name = shard_name(a.out)
    print(f"[train] polymath-engine pack: mode {a.mode}, init {a.init}, layers {layers}, k {a.k}", flush=True)
    P.create(model, name, layers, a.k, a.mode, a.init, data_ids=data_ids, general_ids=gen_ids,
             target_share=a.share if a.mode == "route" else 0, seed=a.seed)
    ref = (base_reference(model, tok, anchor_texts(a, model, tok, GENERAL_TEXT))
           if a.kl else None)                                   # packs are created parked or silent: base

    route_w = a.route if (a.mode == "route" and a.kl) else 0.0

    def anchored(model_, sub):
        """KL to the base on general text + the routing objective: the data term was collected during this step's
        forward on the pack's text, the general term is collected now."""
        data_t = route_terms(model_)
        kl, gen_t = anchor_loss(model_, [{"ids": r["ids"].to(dev), "v": r["v"].to(dev), "i": r["i"].to(dev)} for r in sub],
                                route_w)
        extra = sum(t for t in (data_t, gen_t) if t is not None) if route_w else 0.0
        return kl + (route_w / a.kl) * extra if route_w else kl
    anchor = anchored if a.kl else kl_to_reference
    tc = TrainCfg(steps=total, lr=a.lr, batch=a.batch, max_len=a.max_len, lambda_kl=a.kl, seed=a.seed, log_every=10)
    groups = P.parameters(model, name, router_lr_mult=a.gate_lr_mult)
    _set_anchor(epoch_train, anchor)
    route_mode(model, "data" if route_w else None, a.share)
    try:
        if a.save_epochs:
            def save_poly(ep: int) -> None:
                P.save(model, name, Path(a.out) / "epochs" / f"e{ep:02d}",
                       extra={"trained": {"packsmith": _version(), "epoch": ep, "mode": a.mode}})
                print(f"[epoch] {ep}: copy of the pack kept (epochs/e{ep:02d})", flush=True)
            EPOCH_HOOK[0] = save_poly
        res = epoch_train(model, tok, groups, texts, tc, ref=ref, log=lambda s: print("[step]" + s, flush=True))
    finally:
        route_mode(model, None)
        route_terms(model)
    ident = manifest.model_identity(a.model) if a.model != "tiny" else {}
    out = P.save(model, name, a.out, base_model={k: ident.get(k, "") for k in ("repo", "revision", "label", "weights_hash", "hash_basis")},
                 extra={"trained": {"packsmith": _version(), "train_model": str(a.model), "steps": total, "mode": a.mode,
                                    "init": a.init}})
    blocks = layers_of(model)
    share = {"data": _share(model, tok, texts[:32], dev, blocks), "general": _share(model, tok, GENERAL_TEXT[:32], dev, blocks)}
    card = json.loads(Path(a.card).read_text()) if a.card and Path(a.card).exists() else {}
    card.update({"name": name, "engine": "polymath-engine", "method": "polymath-" + a.mode, "base_model": str(a.model),
                 "base_fingerprint": json.loads((out / "pack.json").read_text())["base_model"]["arch_fingerprint"],
                 "layers": layers, "k": a.k, "mode": a.mode, "init": a.init, "steps": total, "epochs": a.epochs,
                 "batch": a.batch, "accum": a.accum, "max_len": a.max_len, "lr": a.lr, "kl": a.kl, "route": a.route, "share": a.share, "budget": a.budget,
                 "chat_note": chat_note, "texts": len(texts),
                 "end_token": bool(a.end_token), "warmup": a.warmup, "schedule": a.schedule, "pack_token_share": share,
                 "kinds": {k: sum(1 for r in rows if str(r.get("kind")) == k) for k in sorted({str(r.get("kind")) for r in rows})},
                 "train_seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M"),
                 "final_loss": res.get("final_loss"), **stats})
    (out / "card.json").write_text(json.dumps(card, indent=1))
    if a.mode == "route":
        print(f"[train] pack experts won a place on {share['data']:.1%} of the data's token-layers and {share['general']:.1%} "
              "of general text's", flush=True)
    else:
        print("[train] gate pack: its experts are added on every token, weighted by their gates", flush=True)
    ckpt_done(a)
    print(f"[done] pack written to {a.out} in {time.time() - t0:.0f}s" + (" (stopped early)" if stats["stopped_early"] else ""),
          flush=True)


def anchor_texts(a, model, tok, general: list[str]) -> list[str]:
    """The texts the KL anchor reads: the general prose, plus (--anchor-qa) the base's own answers to general
    questions, written by the base itself before training, in the Q&A form the pack is trained in (and in the chat
    form too when the pack is trained in it)."""
    if not a.anchor_qa or a.anchor_qa == "none":
        return list(general)
    cache = Path(a.checkpoint) / "anchor.json" if a.checkpoint else None
    if cache and cache.exists():                         # a resumed run: the base's answers as written the first time
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            if c.get("key") == ckpt_key(a):
                print(f"[train] KL anchor: the base's answers from the checkpoint ({len(c['texts']) - len(general)} texts)",
                      flush=True)
                return list(c["texts"])
        except (OSError, ValueError):
            pass
    import torch
    if a.anchor_qa == "default":
        from packsmith.core.general import GENERAL_QUESTIONS
        qs = list(GENERAL_QUESTIONS)
    else:
        lines = [x.strip() for x in Path(a.anchor_qa).read_text(encoding="utf-8").splitlines() if x.strip()]
        qs = [(json.loads(x).get("question", "") if x.startswith("{") else x) for x in lines]
        qs = [q for q in qs if q]
    chat = a.chat_format != "plain" and (a.model == "tiny" or is_qwen(a.model))
    dev = next(model.parameters()).device
    eos = getattr(tok, "eos", None)
    out, t0 = list(general), time.time()
    with torch.no_grad():
        for q in qs:
            plain = f"Question: {q}\n"
            chatp = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            forms = [plain] if not chat else [chatp] if a.chat_format == "chat" else [plain, chatp]
            for p in forms:
                ids = torch.tensor([tok.encode(p)], device=dev)
                g = model.generate(ids, max_new_tokens=a.anchor_max_new, do_sample=False, eos_token_id=eos,
                                   pad_token_id=eos if eos is not None else 0)
                out.append(p + tok.decode(g[0, ids.shape[1]:].tolist()))
    print(f"[train] KL anchor: {len(general)} general texts + the base's answers to {len(qs)} questions "
          f"({len(out) - len(general)} texts, {time.time() - t0:.0f}s)", flush=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": ckpt_key(a), "texts": out}), encoding="utf-8")
    return out


def ckpt_key(a) -> str:
    """What a checkpoint belongs to: the training data and every setting that changes the run."""
    import hashlib
    h = hashlib.sha256(Path(a.data).read_bytes())
    skip = {"checkpoint", "device", "card", "out"}
    h.update(json.dumps({k: v for k, v in sorted(vars(a).items()) if k not in skip}, default=str).encode())
    return h.hexdigest()[:16]


def ckpt_open(a, ps) -> dict:
    """{dir, key, state}: the checkpoint folder, and the saved state when it is this run's (same data, settings and
    trainable shapes) and it can resume (no curriculum: its stages are not checkpointed)."""
    if not a.checkpoint:
        return {}
    import torch
    d = Path(a.checkpoint)
    d.mkdir(parents=True, exist_ok=True)
    key = ckpt_key(a)
    out = {"dir": d, "key": key}
    f = d / "state.pt"
    if f.exists() and not a.curriculum:
        try:
            st = torch.load(f, map_location="cpu", weights_only=False)
            if st.get("key") == key and [tuple(t.shape) for t in st["params"]] == [tuple(p.shape) for p in ps]:
                out["state"] = st
            else:
                print("[train] a checkpoint from another run is there (other data or settings): starting afresh", flush=True)
        except Exception as e:                           # noqa: BLE001 - a half-written checkpoint: start afresh
            print(f"[train] the checkpoint could not be read ({e}): starting afresh", flush=True)
    return out


def ckpt_save(ck: dict, state: dict) -> None:
    if not ck:
        return
    import torch
    ck["dir"].mkdir(parents=True, exist_ok=True)
    tmp = ck["dir"] / "state.pt.tmp"
    torch.save({**state, "key": ck["key"]}, tmp)
    tmp.replace(ck["dir"] / "state.pt")                   # whole or not at all: a crash while saving keeps the last one
    print(f"[train] checkpoint saved after epoch {state['ep_no']}", flush=True)


def ckpt_done(a) -> None:
    if a.checkpoint and Path(a.checkpoint).exists():
        import shutil
        shutil.rmtree(a.checkpoint, ignore_errors=True)


def _set_anchor(fn, anchor) -> None:
    """Point the epoch loop's `kl_to_reference` (a closure cell of main) at another anchor function."""
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        if name == "kl_to_reference":
            cell.cell_contents = anchor


def _share(model, tok, texts, dev, blocks) -> float:
    import torch
    before = {li: dict(m.stats) for li, m in blocks.items()}
    with torch.no_grad():
        for t in texts:
            model(input_ids=torch.tensor([tok.encode(t)[:256]], device=dev))
    n = sum(m.stats["tokens"] - before[li]["tokens"] for li, m in blocks.items())
    k = sum(m.stats["pack_tokens"] - before[li]["pack_tokens"] for li, m in blocks.items())
    return round(k / n, 4) if n else 0.0


def _version() -> str:
    from packsmith import __version__
    return __version__


if __name__ == "__main__":
    main()
