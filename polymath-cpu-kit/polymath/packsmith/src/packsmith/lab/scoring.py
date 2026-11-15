"""Scoring a pack by what it answers: shared by the Train Lab and scripts/regression/run.py (one implementation).

A pack is put into the base model in PyTorch (a B-slot pack from its exported int8 overlay, exactly what colibri
serves; a Polymath pack from its folder), every test question is answered greedily, and each answer is scored:
  * a short expected answer (the yardstick set): a whole-word match in the answer line ("a|b" = either);
  * a long expected answer (a dataset's own tests, match = "keywords"): at least 60% of its content words present;
  * reasoning questions: whether the steps written are the expected ones (steps right, every step right);
  * flags without a judge: repeating, and with thinking on, thinking never closed / skipped although on.
Questions are asked as "Question: <q>" (plain) or in Qwen's chat format (as the Workbench asks), with thinking on or
off per question kind, or all on / all off (think=True / False: the Lab's two columns). Every share comes with its
count and a 95% Wilson interval.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REAL_GENERAL = [   # (question, expected; "a|b" = either). None of these is among the anchor's questions (core/general.py)
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Spain?", "Madrid"), ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Egypt?", "Cairo"), ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of Kenya?", "Nairobi"), ("What is the capital of Argentina?", "Buenos Aires"),
    ("What is the capital of India?", "New Delhi|Delhi"), ("What is the capital of China?", "Beijing"),
    ("What is the capital of South Korea?", "Seoul"), ("What is the capital of Greece?", "Athens"),
    ("What is the capital of Portugal?", "Lisbon"), ("What is the capital of Norway?", "Oslo"),
    ("What is the capital of Thailand?", "Bangkok"), ("What is the capital of Peru?", "Lima"),
    ("What is the capital of Ireland?", "Dublin"), ("What is the capital of Turkey?", "Ankara"),
    ("What is the largest planet in the solar system?", "Jupiter"), ("Which planet is known as the Red Planet?", "Mars"),
    ("Which planet is closest to the Sun?", "Mercury"), ("Which planet has the most famous rings?", "Saturn"),
    ("What is the name of our galaxy?", "Milky Way"), ("What star is at the centre of our solar system?", "Sun"),
    ("What is the natural satellite of the Earth called?", "Moon"), ("Who was the first person to walk on the Moon?", "Armstrong"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for sodium?", "Na"),
    ("What is the chemical symbol for iron?", "Fe"), ("What gas do humans need to breathe to live?", "oxygen"),
    ("What gas do plants take in from the air?", "carbon dioxide|CO2"), ("What is H2O commonly called?", "water"),
    ("What is the hardest natural substance?", "diamond"), ("What metal is liquid at room temperature?", "mercury"),
    ("Which organ pumps blood around the body?", "heart"), ("Which organ do humans use to breathe?", "lungs|lung"),
    ("How many legs does a spider have?", "eight|8"), ("How many legs does an insect have?", "six|6"),
    ("What is the largest mammal?", "blue whale|whale"), ("What is the fastest land animal?", "cheetah"),
    ("What animal is known as the king of the jungle?", "lion"), ("What do bees make?", "honey"),
    ("What is a baby cat called?", "kitten"), ("What is a baby dog called?", "puppy"),
    ("Who wrote Hamlet?", "Shakespeare"), ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("Who wrote War and Peace?", "Tolstoy"), ("Who wrote Don Quixote?", "Cervantes"),
    ("Who wrote The Odyssey?", "Homer"), ("Who painted the Mona Lisa?", "Leonardo|da Vinci"),
    ("Who painted The Starry Night?", "Van Gogh|Gogh"), ("Who composed the Fifth Symphony that begins with four famous notes?", "Beethoven"),
    ("Who developed the theory of relativity?", "Einstein"), ("Who proposed the theory of evolution by natural selection?", "Darwin"),
    ("Who discovered penicillin?", "Fleming"), ("Who invented the telephone?", "Bell"),
    ("Who was the first President of the United States?", "Washington"), ("Who led India's non-violent independence movement?", "Gandhi"),
    ("In which country are the pyramids of Giza?", "Egypt"), ("In which country is the Eiffel Tower?", "France"),
    ("In which city is the Colosseum?", "Rome"), ("In which country is the Great Wall?", "China"),
    ("In which country is the Taj Mahal?", "India"), ("What is the longest river in Africa?", "Nile"),
    ("What is the largest ocean on Earth?", "Pacific"), ("What is the largest desert in Africa?", "Sahara"),
    ("What is the highest mountain on Earth?", "Everest"), ("Which continent is Brazil in?", "South America"),
    ("What language is spoken in Brazil?", "Portuguese"), ("What language is mainly spoken in Mexico?", "Spanish"),
    ("What currency is used in Japan?", "yen"), ("What currency is used in the United States?", "dollar"),
    ("What currency is used in the United Kingdom?", "pound"), ("What is the official language of Germany?", "German"),
    ("How many days are there in a week?", "seven|7"), ("How many months are there in a year?", "twelve|12"),
    ("How many minutes are there in an hour?", "sixty|60"), ("How many hours are there in a day?", "24|twenty-four"),
    ("What is the first month of the year?", "January"), ("What is the last month of the year?", "December"),
    ("Which season comes after winter?", "spring"), ("How many sides does a triangle have?", "three|3"),
    ("How many sides does a hexagon have?", "six|6"), ("What is ten times ten?", "100|hundred"),
    ("What is seven plus eight?", "15|fifteen"), ("What is the square root of 81?", "9|nine"),
    ("What is half of 50?", "25|twenty-five"), ("What is the smallest prime number?", "2|two"),
    ("At what temperature in Celsius does water freeze?", "0|zero"), ("What is the opposite of hot?", "cold"),
    ("What is the opposite of up?", "down"), ("What colour do you get by mixing blue and yellow?", "green"),
    ("What colour is a ripe banana?", "yellow"), ("What is the colour of grass?", "green"),
    ("What instrument has 88 keys?", "piano"), ("What sport is played at Wimbledon?", "tennis"),
    ("How many players are on a football (soccer) team on the field?", "eleven|11"), ("What do you call frozen water?", "ice"),
]


def load_bslot(model, pack: Path):
    """The exported overlay (int8 rows) as B-slot knowledge experts: what patched colibri computes."""
    import torch
    from safetensors.torch import load_file
    from packsmith.core.surgery import add_knowledge_experts
    from packsmith.core.to_colibri_overlay import expert_inter
    man = json.loads((pack / "overlay.json").read_text())
    t = load_file(str(pack / "overlay.safetensors"))
    I, D = expert_inter(model.config), model.config.hidden_size
    layers = [li for li, _ in man["knowledge"]]
    K = man["knowledge"][0][1]
    blocks = add_knowledge_experts(model, layers, K, "slot", "pk")
    with torch.no_grad():
        for li, k in man["knowledge"]:
            b = blocks[li]
            for j in range(k):
                q = t[f"model.layers.{li}.mlp.knowledge.{j}.merged_weight"].float()
                s = t[f"model.layers.{li}.mlp.knowledge.{j}.qs"]
                g = q[:I * D].view(I, D) * s[:I, None]
                u = q[I * D:2 * I * D].view(I, D) * s[I:2 * I, None]
                dn = q[2 * I * D:].view(D, I) * s[2 * I:, None]
                b.gu["pk"].data[j] = torch.cat([g, u]).to(b.gu["pk"].dtype)
                b.dn["pk"].data[j] = dn.to(b.dn["pk"].dtype)
            b.w["pk"].data.copy_(t[f"model.layers.{li}.mlp.knowledge.gate_w"].to(b.w["pk"].dtype))
            b.b["pk"].data.copy_(t[f"model.layers.{li}.mlp.knowledge.gate_b"])

    def off():
        from packsmith.core.surgery import unwrap_knowledge
        unwrap_knowledge(model)
    return off


def load_polymath(model, pack: Path):
    from polymath_engine import packs as P
    from polymath_engine.layer import attach, detach, layers_of
    attach(model, budget=2)
    banks = P.load_banks(pack, model)
    blocks = layers_of(model)
    for li, b in banks.items():
        blocks[li].set_banks({"pk": b})

    def off():
        detach(model)
    return off


# ---------------------------------------------------------------------------- scoring
THINK_KINDS = ("logic_unseen", "logic_newwording", "repro_trace")   # chat format: asked with thinking on


def prompt_of(t: dict, fmt: str = "plain", think: bool | None = None) -> str:
    """What the model is given. plain: the question in the trained Q&A form. chat (Qwen): the question as a user turn,
    as the Workbench asks it; reasoning questions with thinking on (the assistant turn opens its think block, where
    a Qwen pack was trained to write its steps), the others with thinking off (an empty think block, as a Qwen pack
    was trained to answer a plain question). A statement to reproduce is given as its own first words either way."""
    if t.get("prompt"):
        return t["prompt"]
    if fmt == "chat":
        head = f"<|im_start|>user\n{t['question']}<|im_end|>\n<|im_start|>assistant\n"
        on = (t["kind"] in THINK_KINDS) if think is None else think
        return head + ("<think>\n" if on else "<think>\n\n</think>\n\n")
    return f"Question: {t['question']}\n"


def from_chat(t: dict, reply: str, think: bool | None = None) -> str:
    """A chat reply in the plain 'Reasoning: … / Answer: …' form the scorer reads: what the think block holds is the
    reasoning, what follows it the answer. A thinking-on reply that never closes its think block has no answer."""
    for stop in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        reply = reply.split(stop)[0]
    reply = reply.replace("<pad>", "").strip()
    if t.get("prompt"):
        return reply
    if ((t["kind"] in THINK_KINDS) if think is None else think):
        if "</think>" not in reply:
            return "Reasoning: " + " ".join(reply.split())
        steps, ans = reply.split("</think>", 1)
        steps = " ".join(steps.replace("<think>", "").split())
        return (f"Reasoning: {steps}\n" if steps else "") + "Answer: " + " ".join(ans.split())
    return "Answer: " + " ".join(reply.replace("<think>", "").replace("</think>", "").split())


def answers(model, tok, qs: list[str], dev, max_new: int, bs: int = 16, chat: bool = False) -> list[str]:
    import torch
    hf = tok.hf
    if hf is None:                                        # the byte-level tokenizer of the tiny test models
        pad = tok.eos if tok.eos is not None else 0
        eos = [tok.eos] if tok.eos is not None else []
    else:
        pad = hf.pad_token_id if hf.pad_token_id is not None else (hf.eos_token_id or 0)
        eos = [hf.eos_token_id] if hf.eos_token_id is not None else []
    if chat and hf is not None:                                              # a chat turn ends at <|im_end|> (the base Qwen's eos may differ)
        for t_ in ("<|im_end|>", "<|endoftext|>"):
            i_ = hf.convert_tokens_to_ids(t_)
            if isinstance(i_, int) and i_ >= 0 and i_ != hf.unk_token_id and i_ not in eos:
                eos.append(i_)
    out = []
    for i in range(0, len(qs), bs):
        seqs = [tok.encode(q) for q in qs[i:i + bs]]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s)
            att[j, L - len(s):] = 1
        with torch.no_grad():
            g = model.generate(ids.to(dev), attention_mask=att.to(dev), max_new_tokens=max_new, do_sample=False,
                               pad_token_id=pad, eos_token_id=eos or None)
        out += [hf.decode(r[L:].tolist(), skip_special_tokens=False) if (chat and hf is not None) else tok.decode(r[L:].tolist())
                for r in g]
    return out


def cut(text: str) -> str:
    """The pack's reply up to where it would start a new question."""
    return re.split(r"\n\s*\n|\nQuestion:", text)[0].strip()


def matches(expected: str, answer: str) -> bool:
    """The expected answer (or one of "a|b") as whole words in the answer line: "8" does not match "18"."""
    a = answer.lower()
    return any(re.search(r"(?<![\w])" + re.escape(e.strip().lower()) + r"(?![\w])", a) for e in expected.split("|") if e.strip())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval of a share k/n (Wilson): how far the true rate could be from the measured one."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


# the report's measures: key -> (label, which rows, which field)
FACTS = ("fact_seen", "fact_heldout", "fact_heldout2")
MEASURES = {
    "fact_seen": ("facts, trained wording", lambda x: x["kind"] == "fact_seen", "correct"),
    "fact_heldout": ("facts, new wording 1", lambda x: x["kind"] == "fact_heldout", "correct"),
    "fact_heldout2": ("facts, new wording 2", lambda x: x["kind"] == "fact_heldout2", "correct"),
    "fact_new": ("facts, new wordings (both)", lambda x: x["kind"] in ("fact_heldout", "fact_heldout2"), "correct"),
    "fact_heldout_reasoned": ("facts: reasoned (should be low)", lambda x: x["kind"] in FACTS, "reasoned"),
    "logic_2hop": ("logic 2-hop, unseen chains", lambda x: x["kind"] == "logic_unseen" and x["hops"] == 2, "correct"),
    "logic_3hop": ("logic 3-hop, unseen chains", lambda x: x["kind"] == "logic_unseen" and x["hops"] == 3, "correct"),
    "logic_unseen_new_wording": ("logic, unseen chains, new wording",
                                 lambda x: x["kind"] == "logic_unseen" and x.get("wording") == "new", "correct"),
    "logic_newwording": ("logic, trained chains, new wording", lambda x: x["kind"] == "logic_newwording", "correct"),
    "logic_unseen_reasoned": ("logic: reasoned", lambda x: x["kind"] == "logic_unseen", "reasoned"),
    "logic_right_pier": ("logic: right first step", lambda x: x["kind"] == "logic_unseen", "right_pier_in_steps"),
    "logic_unseen_steps_exact": ("logic, unseen chains: every step right",
                                 lambda x: x["kind"] == "logic_unseen", "reasoning_exact"),
    "general": ("general questions (must stay)", lambda x: x["kind"] == "general", "correct"),
    "dataset_test": ("the dataset's own tests", lambda x: x["kind"] == "dataset_test", "correct"),
    "ds_fact": ("dataset: facts in a wording never trained", lambda x: x.get("sub") == "fact_heldout", "correct"),
    "ds_situation": ("dataset: held-out situations", lambda x: x.get("sub") == "reasoning_test", "correct"),
    "ds_chain": ("dataset: held-out multi-step questions", lambda x: x.get("sub") == "logic_unseen", "correct"),
    "ds_discussion": ("dataset: discussions in a wording never trained", lambda x: x.get("sub") == "discussion_heldout", "correct"),
    "ds_not_covered": ("dataset: says when the material does not cover it", lambda x: x.get("sub") == "not_covered", "correct"),
    "ds_general": ("dataset: ordinary questions far from the subject", lambda x: x.get("sub") == "general", "correct"),
    "ds_near": ("dataset: ordinary questions near the subject", lambda x: x.get("sub") == "general_near", "correct"),
    "reasoning_test": ("reasoning questions, held out (all sources)", lambda x: x["kind"] == "reasoning_test", "correct"),
    "reasoning_text": ("reasoning: worked problems and passages", lambda x: x["kind"] == "reasoning_test" and x.get("src") in ("A", "B", "C"), "correct"),
    "reasoning_chain": ("reasoning: linked facts, 2 and 3 steps", lambda x: x["kind"] == "reasoning_test" and x.get("src") == "D", "correct"),
    "think_open": ("thinking never closed", lambda x: "think_open" in x, "think_open"),
    "think_skipped": ("thinking skipped although on", lambda x: "think_skipped" in x, "think_skipped"),
    "repeating": ("repeating", lambda x: True, "repeating"),
    "repro_qa": ("reproduce: trained Q&A answer", lambda x: x["kind"] == "fact_seen", "correct"),
    "repro_statement": ("reproduce: trained statement completed",
                        lambda x: x["kind"] == "repro_statement", "correct"),
    "repro_trace": ("reproduce: worked question, answer right", lambda x: x["kind"] == "repro_trace", "correct"),
    "repro_trace_exact": ("reproduce: worked question, steps and answer as trained",
                          lambda x: x["kind"] == "repro_trace", "exact"),
}


def norm(x: str) -> str:
    return " ".join(x.split()).strip().rstrip(".").lower()


def sentences(x: str) -> list[str]:
    return [norm(y) for y in re.split(r"(?<=\.)\s+", x.strip()) if y.strip()]


def keyword_match(expected: str, answer: str, need: float = 0.6) -> bool:
    """A long expected answer (a dataset's own tests): right when the answer holds at least `need` of the expected
    answer's content words (words of 4+ letters, numbers kept)."""
    w = {x for x in re.findall(r"[a-z0-9]+", (expected or "").lower()) if len(x) >= 4 or x.isdigit()}
    if not w:
        return matches(expected, answer)
    got = set(re.findall(r"[a-z0-9]+", (answer or "").lower()))
    return len(w & got) / len(w) >= need


def repeating(text: str, n: int = 6, times: int = 3) -> bool:
    """The same run of n words three or more times in one answer (a loop)."""
    w = re.findall(r"\w+", (text or "").lower())
    if len(w) < n * times:
        return False
    seen: dict = {}
    last: dict = {}
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        if i >= last.get(g, -n):
            seen[g] = seen.get(g, 0) + 1
            last[g] = i + n
            if seen[g] >= times:
                return True
    return False


def think_flags(raw: str, think_on: bool) -> dict:
    """With thinking on (the prompt opened the think block): never closed, or closed at once with nothing in it."""
    if not think_on:
        return {}
    r = raw or ""
    for stop in ("<|im_end|>", "<|endoftext|>"):
        r = r.split(stop)[0]
    closed = "</think>" in r
    return {"think_open": not closed, "think_skipped": closed and not r.split("</think>", 1)[0].replace("<think>", "").strip()}


def score(t: dict, reply: str) -> dict:
    if t.get("prompt"):                                   # a statement to complete: its first line is the answer
        r = reply.split("\n")[0].strip()
        full = (t["prompt"] + " " + r).strip()
        return {"correct": matches(t["expected"], r.split(".")[0] if r else ""), "answered": bool(r),
                "reasoned": False, "exact": norm(full).startswith(norm(t.get("trained_text", ""))), "reply": r[:1500]}
    r = cut(reply)
    m = re.search(r"Answer:\s*(.*)", r)
    ans = (m.group(1) if m else r.splitlines()[0] if r else "").strip()
    if t.get("kind") == "dataset_test":                   # a dataset's own test: the whole answer (discussions run long)
        ans = (r[m.start(1):] if m else r).strip()
    reasoned = r.startswith("Reasoning:") or "\nReasoning:" in r or bool(re.match(r"^\s*Reasoning", r))
    ok = keyword_match(t["expected"], ans) if t.get("match") == "keywords" else matches(t["expected"], ans)
    out = {"correct": ok, "answered": m is not None, "reasoned": reasoned, "reply": r[:1500], "repeating": repeating(r)}
    if t.get("steps"):                                    # a reasoning question: its expected steps are known
        pier = re.search(r"(Pier \w+)", t["steps"]).group(1)
        out["right_pier_in_steps"] = pier.lower() in r.lower()
        rm = re.search(r"Reasoning:\s*(.*)", r)
        got = sentences(rm.group(1)) if rm else []
        want = sentences(t["steps"])
        out["steps_right"] = sum(1 for w in want if w in got) / len(want)
        out["reasoning_exact"] = got == want
        out["exact"] = out["reasoning_exact"] and out["correct"]
    return out


def general_nll(model, tok, texts, dev) -> float:
    import torch
    import torch.nn.functional as F
    tot, n = 0.0, 0
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([tok.encode(t)[:256]], device=dev)
            lg = model(input_ids=ids).logits.float()
            tot += float(F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum"))
            n += ids.shape[1] - 1
    return tot / max(1, n)


def evaluate(model, tok, tests, dev, a, gen_texts, think: bool | None = None) -> dict:
    """Every test answered and scored. a: format (plain | chat), max_new, max_new_think. think (chat only): None = by
    kind (reasoning questions on, the rest off), True / False = every question with thinking on / off."""
    chat = a.format == "chat"

    def on(t):                                            # this question is asked with thinking on
        return chat and not t.get("prompt") and ((t["kind"] in THINK_KINDS) if think is None else bool(think))
    replies, flags = [""] * len(tests), [{}] * len(tests)
    groups = {}                                           # reasoning with thinking on gets the longer budget
    for i, t in enumerate(tests):
        groups.setdefault(a.max_new_think if on(t) else a.max_new, []).append(i)
    for budget, idx in groups.items():
        got = answers(model, tok, [prompt_of(tests[i], a.format, think) for i in idx], dev, budget, chat=chat)
        for i, r in zip(idx, got):
            flags[i] = think_flags(r, on(tests[i]))
            replies[i] = from_chat(tests[i], r, think) if chat else r
    sc = [{**score(t, r), **f, "kind": t["kind"], "question": t["question"], "expected": t["expected"],
           "hops": t.get("hops"), "wording": t.get("wording"), "sub": t.get("sub")} for t, r, f in zip(tests, replies, flags)]
    agg, detail = {}, {}
    for key, (_, sel, field) in MEASURES.items():
        rs = [x for x in sc if sel(x)]
        if not rs:
            continue
        k = sum(bool(x.get(field)) for x in rs)
        agg[key] = round(k / len(rs), 3)
        detail[key] = {"k": k, "n": len(rs), "ci95": wilson(k, len(rs))}
    agg["general_nll"] = round(general_nll(model, tok, gen_texts, dev), 4)
    return {"scores": agg, "detail": detail, "rows": sc}


def release() -> None:
    """Give the GPU back: the scorer's model is gone once score() returns; this drops what PyTorch still caches, so
    the next trainer (another process) has the whole card."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_free() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info()
            return f"GPU {f / 2**30:.1f} of {t / 2**30:.1f} GiB free"
    except Exception:                                         # noqa: BLE001
        pass
    return "CPU"


def score_pack(a, pack: Path | None, tests, gen_texts, modes: list | None = None) -> dict:
    """Load the base, put the pack in (None = the base alone), answer the tests. Everything it loads is local and
    freed when it returns (no handle on the model outlives the call). modes: thinking modes to score in this one load
    ([None] by kind, [True, False] the Lab's thinking-on and thinking-off columns): then {mode: result}."""
    from packsmith.core.model import load
    model, tok = load(a.model, device=a.device)
    try:
        dev = next(model.parameters()).device
        off = None
        if pack is not None:
            off = load_polymath(model, pack) if (pack / "pack.json").exists() else load_bslot(model, pack)
        if modes is None:
            ev = evaluate(model, tok, tests, dev, a, gen_texts)
        else:
            ev = {str(m): evaluate(model, tok, tests, dev, a, gen_texts, think=m) for m in modes}
        if off:
            off()
        return ev
    finally:
        del model
        off = None
        release()


def quick_score(model, tok, tests: list[dict], fmt: str = "plain", max_new: int = 48) -> dict:
    """The trainer's score after an epoch (a small fixed sample, never the final tests): {right, n, by_kind}."""
    class A:
        pass
    a = A()
    a.format, a.max_new, a.max_new_think = fmt, max_new, max(max_new, 128)
    was = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        ev = evaluate(model, tok, tests, dev, a, [], think=None)
    finally:
        if was:
            model.train()
    by = {}
    for r in ev["rows"]:
        k = r["kind"]
        by.setdefault(k, [0, 0])
        by[k][0] += bool(r["correct"])
        by[k][1] += 1
    right = sum(v[0] for k, v in by.items() if k != "general")
    n = sum(v[1] for k, v in by.items() if k != "general")
    return {"right": right, "n": n, "by_kind": by}
