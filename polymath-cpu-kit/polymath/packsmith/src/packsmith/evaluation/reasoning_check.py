"""Does a pack stifle reasoning? Same questions with the pack OFF (base) and ON, thinking OFF and ON.

Run from the packsmith folder, in the same venv, with nothing else using the GPU (the script stops Packsmith's engine
and unloads LM Studio first):

    python scripts/reasoning_check.py --packs omc,16_FC                 # on Packsmith's engine model
    python scripts/reasoning_check.py --list                            # packs, containers, the HF cache
    python scripts/reasoning_check.py --probe-only                      # only: does the model think and stop?
    python scripts/reasoning_check.py --convert Qwen/Qwen3-30B-A3B --use --packs omc,16_FC
                                       # convert the cached hybrid for colibri, make it Packsmith's model, run
    python scripts/reasoning_check.py --model ~/moe-knowledge/runs/qwen_base_coli,~/moe-knowledge/runs/qwen3_30b_a3b_coli

Every run starts with a probe (no pack, think on, "What is 2 + 2?") and prints the raw output: a chat model opens a
think block and stops; a pretrained-only model (or chat tokens not reaching the model) does not.

Per pack it runs
  * general control problems with known answers (the pack must not break these),
  * domain tasks chosen from the pack's name: Old Man and the Sea (reasoning + a 1,500-word story) or the 16th Finance
    Commission (grants-in-aid, disaster management grants, local body grants),
and measures for each answer: thinking length, answer length, correctness (controls), loops, story length and voice.
It prints a table and a verdict, and writes everything (with the full outputs) to reasoning_check_<time>.md / .json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from packsmith.packs import store  # noqa: E402
from packsmith.workbench import workbench  # noqa: E402
from packsmith.system.config import SETTINGS  # noqa: E402
from packsmith.system.home import expand

GENERAL = [
    {"id": "g-walls", "kind": "control", "q": "If 3 workers build 3 walls in 3 hours, how many hours do 9 workers need to build 9 walls? "
     "Work it out, then end with a line 'Answer: <number>'.", "expect": "3"},
    {"id": "g-arith", "kind": "control", "q": "Compute 17 × 24 − 150. Show the steps, then end with a line 'Answer: <number>'.", "expect": "258"},
    {"id": "g-order", "kind": "control", "q": "Asha is older than Bimal. Bimal is older than Chitra. Dev is younger than Chitra. "
     "Who is the second youngest? Reason it out, then end with a line 'Answer: <name>'.", "expect": "Chitra"},
]
OMC = [
    {"id": "omc-why", "kind": "reasoning", "q": "In The Old Man and the Sea, why does Santiago keep holding the line through the second "
     "night even though his hands are cut and he could cut the fish loose? Reason step by step from the events of the story "
     "and from what Santiago says and thinks, then conclude."},
    {"id": "omc-victory", "kind": "reasoning", "q": "Santiago returns with only the skeleton of the marlin. Is that a defeat or a victory? "
     "Build the argument for each side from specific episodes in the novella, weigh them, and reach a conclusion."},
    {"id": "omc-story", "kind": "writing", "words": 1500, "q": "Write an original short story of about 1,500 words in the manner of "
     "Ernest Hemingway's The Old Man and the Sea: an old lighthouse keeper alone through a winter storm. Plain words, short "
     "sentences, understatement, nothing explained that can be shown. Write the whole story."},
]
FC = [
    {"id": "fc-gia", "kind": "reasoning", "q": "Under the Indian constitutional scheme, how are grants-in-aid to States recommended and "
     "given (Article 275 and the Finance Commission's role)? Using what the 16th Finance Commission recommended, reason step by "
     "step: what the grants are for, how they are worked out, and what conditions apply. Then summarise."},
    {"id": "fc-disaster", "kind": "reasoning", "q": "A State is hit by a severe flood. Reason step by step how disaster management funding "
     "reaches it under the 16th Finance Commission's recommendations: which funds (response and mitigation, State and "
     "National), how the State's allocation is worked out, cost-sharing between the Union and the State, and what the State "
     "must do to draw on them. Then summarise."},
    {"id": "fc-local", "kind": "reasoning", "q": "Explain step by step how the 16th Finance Commission recommends grants to local bodies: "
     "the split between rural and urban local bodies, how amounts are distributed among States, which parts are tied or "
     "untied, and the conditions a State or local body must meet to receive them. Then summarise."},
]
SYSTEM = "You are a careful expert. Reason before you answer and say so when you are unsure."


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_packs(names: list[str]) -> list[dict]:
    allp = store.list_packs()
    out = []
    for n in names:
        hits = [p for p in allp if norm(n) == norm(p["name"])] or [p for p in allp if norm(n) in norm(p["name"])]
        if not hits:
            sys.exit(f"no pack matching {n!r}. Packs found: {', '.join(p['name'] for p in allp) or 'none'} "
                     f"(in {store.packs_dir()} and registered packs)")
        out.append(hits[0])
    return out


def domain_tasks(pack: dict) -> list[dict]:
    key = norm(pack["name"] + " " + str((pack.get("card") or {}).get("domain", "")) + " " + str((pack.get("card") or {}).get("topic_name", "")))
    if any(w in key for w in ("fc", "finance", "commission", "grant")):
        return FC
    if any(w in key for w in ("omc", "oldman", "sea", "hemingway", "santiago")):
        return OMC
    return OMC if "omc" in key else FC


def prompt(q: str, think: bool) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            + ("" if think else "<think>\n\n</think>\n\n"))


def final_ok(text: str, expect: str) -> bool:
    """The last number (or the last person named) in the answer is the expected one."""
    m = re.findall(r"answer\s*[:：]\s*\**\s*([^\n]{1,60})", text, re.I)   # the first 'Answer: x' line (later text may echo the prompt)
    tail = m[0] if m else text[-300:]
    if expect.isdigit():
        nums = re.findall(r"\d+(?:\.\d+)?", tail.replace(",", ""))
        return bool(nums) and float(nums[-1]) == float(expect)
    names = re.findall(r"\b(Asha|Bimal|Chitra|Dev)\b", tail)
    return bool(names) and names[-1] == expect


def words(s: str) -> int:
    return len(re.findall(r"\w+", s))


def loopiness(s: str) -> float:
    w = re.findall(r"\w+", s.lower())
    if len(w) < 60:
        return 0.0
    grams = [" ".join(w[i:i + 8]) for i in range(len(w) - 7)]
    return round(1 - len(set(grams)) / len(grams), 3)


def voice(s: str) -> dict:
    sents = [x for x in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", s)) if x.strip()]
    lens = [len(x.split()) for x in sents] or [0]
    lw = re.findall(r"[a-z']+", s.lower())
    return {"mean_sentence": round(statistics.mean(lens), 1), "short_share": round(sum(1 for x in lens if x <= 8) / len(lens), 2),
            "and_per_sentence": round(lw.count("and") / max(1, len(sents)), 2)}


def run_one(e, task: dict, think: bool, a) -> dict:
    max_tok = a.max_think if think else (a.max_write if task["kind"] == "writing" else a.max_off)
    temp, top_p = (0.6, 0.95) if think else ((0.7, 0.9) if task["kind"] == "writing" else (0.3, 0.9))
    t0 = time.time()
    raw = workbench.engine_ask(e, prompt(task["q"], think), max_tok, temp, top_p)
    sec = time.time() - t0
    ans, th = workbench.split_answer(raw)
    r = {"task": task["id"], "kind": task["kind"], "think": think, "seconds": round(sec, 1), "think_words": words(th),
         "answer_words": words(ans), "loop": loopiness(th + " " + ans), "hit_token_limit": words(raw) * 1.25 >= max_tok * 0.95,
         "think_block": "</think>" in raw, "answer": ans, "thinking": th, "raw_tail": raw[-300:]}
    if task["kind"] == "control":
        r["correct"] = final_ok(ans or th, task["expect"])
    if task["kind"] == "writing":
        r["target_words"] = task["words"]
        r["voice"] = voice(ans)
    return r


def stop_packsmith() -> None:
    for path in ("/api/workbench/stop",):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{SETTINGS.port}{path}", data=b"{}", method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=20).read()
            print(f"stopped Packsmith's engine (port {SETTINGS.port})")
        except Exception:
            pass


def verdict(rows: list[dict], pack: str) -> list[str]:
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["think"]), {})[r["cond"]] = r
    for (task, think), d in sorted(by.items()):
        b, p = d.get("base"), d.get(pack)
        if not b or not p:
            continue
        mode = "think on" if think else "think off"
        if b["kind"] == "control" and b.get("correct") and not p.get("correct"):
            out.append(f"INTERFERENCE: {task} ({mode}) — base right, pack wrong")
        if think and b["think_words"] > 80 and p["think_words"] < 0.5 * b["think_words"]:
            out.append(f"STIFLED THINKING: {task} — pack thinks {p['think_words']} words vs base {b['think_words']}")
        if b["kind"] == "reasoning" and b["answer_words"] > 80 and p["answer_words"] < 0.5 * b["answer_words"]:
            out.append(f"SHORT ANSWERS: {task} ({mode}) — pack {p['answer_words']} words vs base {b['answer_words']}")
        if b["kind"] == "writing":
            if p["answer_words"] < 0.6 * b["target_words"]:
                out.append(f"STORY TOO SHORT: {task} ({mode}) — pack {p['answer_words']} of ~{b['target_words']} words (base {b['answer_words']})")
            if p["voice"]["mean_sentence"] < b["voice"]["mean_sentence"] - 2:
                out.append(f"VOICE MOVED: {task} ({mode}) — sentences {p['voice']['mean_sentence']} words vs base {b['voice']['mean_sentence']} (shorter = closer to Hemingway)")
        if p["loop"] > 0.3 and p["loop"] > b["loop"] + 0.1:
            out.append(f"LOOPING: {task} ({mode}) — repeated text {p['loop']:.0%} with the pack vs {b['loop']:.0%}")
    if any(r["think"] and not r.get("think_block") for r in rows if r["cond"] == "base"):
        out.insert(0, "NO THINK BLOCK without the pack (think on): this model does not think here (pretrained-only model, or the "
                      "engine drops the <think> tokens) — thinking measures are not meaningful")
    return out or ["No sign of stifling on these tasks: the pack keeps the base model's reasoning and length."]


def resolve_model(spec: str) -> dict:
    """current (Packsmith's setting) or a container folder — colibri loads it as it is."""
    from packsmith.system import models
    if spec == "current":
        return models.current()
    p = expand(spec)
    if not p.is_dir():
        sys.exit(f"{p}: no such folder (give a colibri container, or --convert a Hugging Face model first)")
    return models.info(p)


def provenance(m: dict) -> list[str]:
    """What the container says about where it came from (nothing is validated)."""
    p, out = Path(m["path"]), []
    if m.get("source"):
        out.append(f"converted from {m['source']}")
    for f in ("config.hf.json", "config.json"):
        try:
            c = json.loads((p / f).read_text())
            out.append(f"{f}: model_type {c.get('model_type')}, layers {c.get('num_hidden_layers')}, experts {c.get('num_experts')}"
                       + (f", name {c['_name_or_path']}" if c.get("_name_or_path") else ""))
            break
        except (OSError, ValueError):
            pass
    base_hf = SETTINGS.path("runs_dir") / "qwen3_30b_base_hf"
    if base_hf.exists() and "base" in p.name.lower():
        out.append(f"note: {base_hf} exists — moe-knowledge's g0/run_qwen.sh downloads Qwen/Qwen3-30B-A3B-Base there "
                   "(not into the Hugging Face cache) and converts it into runs/qwen_base_coli")
    return out


def probe(e) -> dict:
    """No pack, think on, a trivial question: does the model open a think block and stop at the end of its turn?"""
    raw = workbench.engine_ask(e, prompt("What is 2 + 2? Answer in one sentence.", True), 600, 0.6, 0.95)
    r = {"raw": raw[:600], "think_block": "</think>" in raw, "stopped": len(re.findall(r"\w+", raw)) < 400,
         "role_debris": bool(re.search(r"(^|\n)\s*(assistant|user|system)\s*[:\n]", raw, re.I))}
    print("  probe (no pack, think on, 'What is 2 + 2?') — raw output as the engine returns it:")
    print("    " + repr(raw[:400]))
    print(f"    think block: {'yes' if r['think_block'] else 'NO'} · stopped by itself: {'yes' if r['stopped'] else 'NO'} · "
          f"role lines written as text: {'YES' if r['role_debris'] else 'no'}")
    if not r["think_block"] or r["role_debris"]:
        print("    → the model does not behave like the hybrid chat model here (pretrained-only model, or the chat tokens are "
              "not reaching it as special tokens). Fix this before judging packs.")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs", default="omc,16_FC")
    ap.add_argument("--model", default="current", help="current (Packsmith's engine model) or a colibri container folder; "
                    "comma-separated to compare")
    ap.add_argument("--convert", default="", help="first convert this for colibri: a Hugging Face model folder, a "
                    "models--Org--Name cache folder, or a repo in the cache (e.g. Qwen/Qwen3-30B-A3B); then run on it")
    ap.add_argument("--use", action="store_true", help="make the converted model Packsmith's engine and training model")
    ap.add_argument("--think", default="both", choices=["both", "on", "off"])
    ap.add_argument("--max-think", type=int, default=6000, help="token limit with thinking (thinking + answer)")
    ap.add_argument("--max-off", type=int, default=1500, help="token limit without thinking")
    ap.add_argument("--max-write", type=int, default=3000, help="token limit for the story without thinking")
    ap.add_argument("--skip-general", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-stop", action="store_true", help="do not stop Packsmith's engine / unload LM Studio first")
    ap.add_argument("--probe-only", action="store_true", help="only run the chat-format probe on each model")
    ap.add_argument("--hf-cache", default="", help="Hugging Face hub cache (default: Packsmith's setting, "
                    "~/.cache/huggingface/hub = /home/<user>/.cache/huggingface/hub)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.hf_cache:
        SETTINGS.hf_cache_dir = a.hf_cache
    from packsmith.system import models as _m
    print(f"Hugging Face cache: {_m.hf_cache()}")
    if a.list:
        from packsmith.system import models
        L = models.listing()
        print("packs:")
        for p in store.list_packs():
            print(f"  {p['name']:38s} {p['path']}  (trained on {Path(str(p['card'].get('base_model', '?'))).name})")
        print("colibri containers:")
        for c in L["containers"]:
            print(f"  {c['path']}{'  [serving]' if c['serving'] else ''}")
            for x in provenance(c):
                print(f"      {x}")
        print(f"Hugging Face cache {L['hf_cache']}:")
        for h in L["hf"]:
            print(f"  {h['repo']:40s} {h['path']}")
        return
    specs = [x.strip() for x in a.model.split(",") if x.strip()]
    if a.convert:
        from packsmith.system import models
        print(f"converting {a.convert} for colibri …")
        last = [0.0]

        def prog(f, m=""):
            if m and (f - last[0] >= 0.02 or "ready" in m or "converting" in m):
                last[0] = f
                print(f"  [{f * 100:5.1f}%] {m}")
        conv = models.convert(a.convert, use_it=a.use, progress=prog)
        specs = [conv["path"]] + [x for x in specs if x != "current"]
    if not a.no_stop:
        stop_packsmith()
        try:
            from packsmith.llm.llm import llm
            un = llm.unload_all()
            if un:
                print("unloaded LM Studio:", ", ".join(un))
        except Exception as ex:                          # noqa: BLE001
            print("could not unload LM Studio:", ex)
    mods = [resolve_model(sp) for sp in specs]
    for m in mods:
        print(f"model: {m['path']}")
        for x in provenance(m):
            print(f"  {x}")
    packs = [] if a.probe_only else find_packs([x.strip() for x in a.packs.split(",") if x.strip()])
    thinks = {"both": [False, True], "on": [True], "off": [False]}[a.think]
    rows = []
    for m in mods:
        mtag = m["name"]
        print(f"\n##### model {mtag}: starting the engine on {'CPU' if a.cpu else 'GPU'} (context {SETTINGS.engine_ctx}) …")
        workbench.stop()
        st = workbench.start(model=m["path"], gpu=not a.cpu)
        if st["status"] != "ready":
            sys.exit(f"engine did not start: {st.get('error')}")
        e = workbench.STATE["engine"]
        m["probe"] = probe(e)
        if a.probe_only:
            workbench.stop()
            continue
        base_cache = {}
        try:
            for pk in packs:
                tasks = ([] if a.skip_general else GENERAL) + domain_tasks(pk)
                print(f"\n=== {mtag} · {pk['name']}: {len(tasks)} tasks × {len(thinks)} thinking modes × no pack/pack ===")
                for cond, path in (("base", None), (pk["name"], pk["path"])):
                    sw = e.swap(path)
                    if not sw["ok"]:
                        sys.exit(f"swap refused for {cond}: {' | '.join(sw['engine_log'][-3:])}")
                    for task in tasks:
                        for think in thinks:
                            key = (task["id"], think)
                            if cond == "base" and key in base_cache:
                                r = dict(base_cache[key])
                            else:
                                r = run_one(e, task, think, a)
                                if cond == "base":
                                    base_cache[key] = r
                            r = {**r, "cond": cond, "pack": pk["name"], "model": mtag}
                            rows.append(r)
                            tag = ("✓" if r.get("correct") else "✗") if task["kind"] == "control" else ""
                            print(f"  {cond:>14s} | {task['id']:12s} | think {'on ' if think else 'off'} | thinking {r['think_words']:5d} w | "
                                  f"answer {r['answer_words']:5d} w | {r['seconds']:6.1f}s {tag}{' | LIMIT' if r['hit_token_limit'] else ''}"
                                  f"{' | loop ' + str(r['loop']) if r['loop'] > 0.3 else ''}"
                                  f"{' | no think block' if think and not r['think_block'] else ''}")
                e.swap(None)
        finally:
            workbench.stop()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / f"reasoning_check_{stamp}"
    groups = [(m["name"], pk["name"]) for m in mods for pk in packs]
    verdicts = {f"{mn} · {pn}": verdict([r for r in rows if r["pack"] == pn and r["model"] == mn], pn) for mn, pn in groups}
    Path(str(out) + ".json").write_text(json.dumps({"rows": rows, "verdicts": verdicts, "models": mods, "settings": vars(a)},
                                                   indent=1, ensure_ascii=False))
    md = [f"# Reasoning check {stamp}\n"]
    for m in mods:
        pr = m.get("probe") or {}
        md.append(f"**Model** {m['path']}  \nprobe: think block {'yes' if pr.get('think_block') else 'NO'}, "
                  f"role lines as text {'YES' if pr.get('role_debris') else 'no'}\n\n```\n{pr.get('raw', '')}\n```\n")
    for mn, pn in groups:
        sel = [x for x in rows if x["pack"] == pn and x["model"] == mn]
        md.append(f"## {mn} · {pn}\n\n**Verdict**\n" + "\n".join(f"- {v}" for v in verdicts[f'{mn} · {pn}']) + "\n")
        md.append("| task | think | condition | thinking words | think block | answer words | correct | loop | seconds |\n|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            md.append(f"| {r['task']} | {'on' if r['think'] else 'off'} | {r['cond']} | {r['think_words']} | {'yes' if r['think_block'] else 'no'} | "
                      f"{r['answer_words']} | {'' if 'correct' not in r else ('yes' if r['correct'] else 'no')} | {r['loop']} | {r['seconds']} |")
        md.append("")
        for r in sel:
            md.append(f"### {r['task']} · think {'on' if r['think'] else 'off'} · {r['cond']}\n")
            if r["thinking"]:
                md.append("<details><summary>thinking</summary>\n\n" + r["thinking"] + "\n\n</details>\n")
            md.append(r["answer"] + "\n")
    Path(str(out) + ".md").write_text("\n".join(md))
    print("\n=== verdict ===")
    for name, vs in verdicts.items():
        print(f"{name}:")
        for v in vs:
            print("  -", v)
    print(f"\nfull outputs: {out}.md  (and .json)")


if __name__ == "__main__":
    main()
