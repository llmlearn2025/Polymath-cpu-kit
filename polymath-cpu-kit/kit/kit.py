"""Polymath CPU test kit: train one B-slot knowledge pack and compare the base with base + pack, both served by
patched colibri on the CPU. Called by run_kit.sh (which builds colibri and the venv first); can be run on its own:

    python kit/kit.py --model <HF folder> --colibri <colibri checkout> --out <run folder> [--tests 40]

Steps
  1. the harbour-world dataset (scripts/regression/make_data.py, seeded: the same files on every machine)
  2. one pack trained with Packsmith's own trainer and its defaults (python -m packsmith.training.train_pack)
  3. the base converted for colibri (tools/convert_olmoe_merged.py, int8 experts)
  4. ONE colibri process answers every question twice: with no pack, then with the pack swapped in live
  5. RESULTS.md: shares in percent with their 95% intervals

Scoring goes through colibri, not PyTorch: what is measured is what colibri serves. Greedy answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
POLY = KIT / "polymath"
SRC = POLY / "packsmith" / "src"
sys.path.insert(0, str(SRC))
from packsmith.lab.scoring import REAL_GENERAL, matches, score, wilson  # noqa: E402

KIT_VERSION = "1.0"
# The kit's columns: (key, label, which test rows). "general" is base-world questions on the small model and
# ordinary questions (capitals, planets, authors) on a real one: what the pack must leave alone.
COLUMNS = [
    ("fact_seen", "Facts, trained wording", lambda t: t["kind"] == "fact_seen"),
    ("fact_new", "Facts, wording never trained", lambda t: t["kind"] in ("fact_heldout", "fact_heldout2")),
    ("logic_2hop", "Reasoning, 2 steps, no worked example", lambda t: t["kind"] == "logic_unseen" and t.get("hops") == 2),
    ("logic_3hop", "Reasoning, 3 steps, no worked example", lambda t: t["kind"] == "logic_unseen" and t.get("hops") == 3),
    ("logic_newwording", "Reasoning, trained chains, new wording", lambda t: t["kind"] == "logic_newwording"),
    ("general", "General questions (must stay)", lambda t: t["kind"] == "general"),
]


def say(msg: str) -> None:
    print(msg, flush=True)


def rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def pick_tests(all_tests: list[dict], per: int, real: bool) -> list[dict]:
    """A fixed sample: the first `per` questions of each kind in the file's (seeded) order, spread over its chains.
    Reasoning: `per` two-step and `per` three-step questions. A real model gets ordinary questions instead of the
    base world."""
    out: list[dict] = []
    def take(sel, n):
        out.extend([t for t in all_tests if sel(t)][:n] if per else [t for t in all_tests if sel(t)])
    take(lambda t: t["kind"] == "fact_seen", per)
    take(lambda t: t["kind"] == "fact_heldout", per // 2)
    take(lambda t: t["kind"] == "fact_heldout2", per - per // 2 if per else 0)
    take(lambda t: t["kind"] == "logic_unseen" and t.get("hops") == 2, per)
    take(lambda t: t["kind"] == "logic_unseen" and t.get("hops") == 3, per)
    take(lambda t: t["kind"] == "logic_newwording", per)
    if real:
        out.extend({"question": q, "expected": e, "kind": "general"} for q, e in (REAL_GENERAL[:per] if per else REAL_GENERAL))
    else:
        take(lambda t: t["kind"] == "general", per)
    return out


def sha12(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def cpu_name() -> str:
    try:
        for l in Path("/proc/cpuinfo").read_text().splitlines():
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(cmd: list[str], log: Path, env: dict | None = None) -> None:
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode:
        tail = "".join(log.read_text(errors="replace").splitlines(True)[-20:])
        raise SystemExit(f"step failed (log: {log.name} in the run folder):\n{tail}")


def train(a, data: Path, out: Path, logs: Path) -> dict:
    pack = out / "pack"
    if (pack / "card.json").exists() and (pack / "overlay.json").exists() and not a.retrain:
        say("  pack already trained in an earlier run: kept (--retrain trains it again)")
        return json.loads((pack / "card.json").read_text())
    anchor = str(data / "anchor_questions.txt") if a.small else "default"
    cmd = [sys.executable, "-m", "packsmith.training.train_pack", "--model", a.model, "--data", str(data / "train.jsonl"),
           "--tests", str(data / "val.jsonl"), "--out", str(pack), "--device", "cpu", "--max-len", "256",
           "--seed", "0", "--anchor-qa", anchor, "--checkpoint", str(out / "ckpt")]
    if a.epochs:
        cmd += ["--epochs", str(a.epochs)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": str(a.threads)}
    t0 = time.time()
    say(f"  Packsmith trainer, its defaults (B-slot pack, auto epochs, copy start, base-answer anchor); log: logs/train.log")
    run(cmd, logs / "train.log", env)
    say(f"  trained in {(time.time() - t0) / 60:.1f} min")
    return json.loads((pack / "card.json").read_text())


def convert(a, out: Path, logs: Path) -> Path:
    coli_model = out / "base_colibri"
    if (coli_model / "config.json").exists() and any(coli_model.glob("model-*.safetensors")):
        return coli_model
    tool = Path(a.colibri) / "c" / "tools" / "convert_olmoe_merged.py"
    run([sys.executable, str(tool), "--model", a.model, "--out", str(coli_model)], logs / "convert.log")
    return coli_model


def ask_all(eng, tests: list[dict], bos: str, max_new: int, label: str) -> list[str]:
    out, t0 = [], time.time()
    for i, t in enumerate(tests):
        prompt = t.get("prompt") or f"Question: {t['question']}\n"
        out.append(eng.ask(bos + prompt, max_new))
        if (i + 1) % 25 == 0 or i + 1 == len(tests):
            el = time.time() - t0
            say(f"  {label}: {i + 1}/{len(tests)} questions, {el / 60:.1f} min")
    return out


def tally(tests: list[dict], replies: list[str]) -> tuple[dict, list[dict]]:
    sc = [{**score(t, r), "kind": t["kind"], "hops": t.get("hops"), "question": t["question"], "expected": t["expected"]}
          for t, r in zip(tests, replies)]
    res = {}
    for key, _, sel in COLUMNS:
        idx = [i for i, t in enumerate(tests) if sel(t)]
        if idx:
            k = sum(1 for i in idx if sc[i]["correct"])
            res[key] = {"k": k, "n": len(idx), "ci95": wilson(k, len(idx))}
    return res, sc


def pct(c: dict) -> str:
    if not c:
        return ""
    p = 100 * c["k"] / c["n"]
    lo, hi = c["ci95"]
    return f"{p:.0f}% ({100 * lo:.0f}–{100 * hi:.0f})"


def verdict(base: dict, pack: dict, key: str) -> str:
    b, p = base.get(key), pack.get(key)
    if not b or not p:
        return ""
    if key == "general":
        if p["ci95"][1] < b["ci95"][0]:
            return "pack lost some"
        return "kept"
    if p["ci95"][0] > b["ci95"][1]:
        return "pack better"
    if p["ci95"][1] < b["ci95"][0]:
        return "pack worse"
    return "no clear difference"


def report(meta: dict, base: dict, pack: dict, samples: list[str]) -> str:
    L = ["# Polymath CPU test kit: results", "",
         "| | |", "|---|---|"]
    for k, v in meta.items():
        L.append(f"| {k} | {v} |")
    L += ["", "Each question was answered by patched colibri on the CPU, greedily, first with no pack and then with the "
          "pack swapped in, in the same engine process. The figure is the share answered right; in brackets its 95% "
          "interval (Wilson). Two figures differ only where their intervals do not overlap.", "",
          "| What was asked | n | Base | Base + pack | |", "|---|---|---|---|---|"]
    for key, label, _ in COLUMNS:
        if key in base:
            L.append(f"| {label} | {base[key]['n']} | {pct(base[key])} | {pct(pack.get(key, {}))} | {verdict(base, pack, key)} |")
    L += ["", "What the rows mean:", "",
          "- **Facts, trained wording**: a fact asked the way it was trained (memorisation).",
          "- **Facts, wording never trained**: the same facts asked in two other wordings (knowledge, not wording).",
          "- **Reasoning, no worked example**: chains whose facts were trained but no worked case of theirs was; the "
          "pack has to chain facts it holds (cargo class → pier → warden → home island).",
          "- **Reasoning, trained chains, new wording**: chains with worked cases, asked in wordings never trained.",
          "- **General questions**: " + (meta.get("General check", "") or "") + " The pack must leave these alone.",
          "", "The world is invented, so the base cannot know it: the base column is the floor.", "",
          "Sample replies:", ""] + samples
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="an OLMoE Hugging Face folder")
    ap.add_argument("--model-label", default="", help="the model's name in the report (no folder paths)")
    ap.add_argument("--colibri", required=True, help="the patched colibri checkout (with c/polymath-bin/cpu/olmoe)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", type=int, default=-1,
                    help="questions per row; 0 = every test question; default: all on the small model (colibri "
                         "answers them in a minute or two), 40 on a real model")
    ap.add_argument("--epochs", type=int, default=0, help="0 = the trainer's auto epochs")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4)))
    ap.add_argument("--max-new", type=int, default=72, help="tokens per answer (three steps need about 60)")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    a.model = str(Path(a.model).expanduser().resolve())
    a.small = (Path(a.model) / "pretrain.json").exists()
    if a.tests < 0:
        a.tests = 0 if a.small else 40
    started = time.strftime("%Y-%m-%d %H:%M")
    T0 = time.time()

    say("== 1/4 dataset (invented harbour world, seeded)")
    data = out / "data"
    if not (data / "tests.jsonl").exists():
        run([sys.executable, str(POLY / "scripts" / "regression" / "make_data.py"), "--out", str(data)], logs / "data.log")
    train_rows = rows(data / "train.jsonl")
    tests = pick_tests(rows(data / "tests.jsonl"), a.tests, real=not a.small)
    say(f"  {len(train_rows)} training items, {len(tests)} test questions")

    say("== 2/4 training the pack (CPU)")
    t1 = time.time()
    card = train(a, data, out, logs)
    train_min = (time.time() - t1) / 60

    say("== 3/4 converting the base for colibri")
    coli_model = convert(a, out, logs)

    say("== 4/4 asking colibri: base, then base + pack")
    from packsmith.engines.colibri.engine import Engine
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    bos = tok.bos_token if tok.bos_token_id is not None else ""     # the small model reads <s> first, as trained
    eng = Engine(a.colibri, str(coli_model), "olmoe", gpu=False, threads=a.threads, build="cpu")
    try:
        t2 = time.time()
        rb = ask_all(eng, tests, bos, a.max_new, "base")
        sw = eng.swap(str(out / "pack"))
        if not sw["ok"]:
            raise SystemExit("colibri did not take the pack: " + " | ".join(sw["engine_log"][-4:]))
        rp = ask_all(eng, tests, bos, a.max_new, "base + pack")
        ask_min = (time.time() - t2) / 60
        engine_line = next((l for l in eng.log if "[POLYMATH]" in l or "[OVERLAY] active" in l), "")
    finally:
        eng.close()
    base, sb = tally(tests, rb)
    pack, sp = tally(tests, rp)

    eng_json = {}
    try:
        eng_json = json.loads((Path(a.colibri) / "c" / "polymath_engine.json").read_text())
    except (OSError, ValueError):
        pass
    try:
        import torch
        torch_v = torch.__version__
    except ImportError:
        torch_v = "?"
    n_gen = sum(1 for t in tests if t["kind"] == "general")
    meta = {
        "Kit": f"polymath-cpu-kit {KIT_VERSION}",
        "Model": a.model_label or Path(a.model).name,
        "Model weights (sha256, first 12)": sha12(Path(a.model) / "model.safetensors") if (Path(a.model) / "model.safetensors").exists() else "several files",
        "Device": f"CPU, {cpu_name()}, {a.threads} threads",
        "Engine": f"patched colibri, CPU build, upstream {str(eng_json.get('commit', '?'))[:7]}, patches {eng_json.get('patch_version', '?')}",
        "Pack": f"B-slot, layers {card.get('layers')}, k {card.get('k')}, {card.get('epochs_run') or len(card.get('epochs') or []) or '?'} epochs "
                f"(best {card.get('best_epoch', '?')}), {len(train_rows)} training items",
        "Python / PyTorch": f"{platform.python_version()} / {torch_v}",
        "General check": (f"{n_gen} questions about the base world the small model was pretrained on." if a.small
                          else f"{n_gen} ordinary questions (capitals, planets, authors)."),
        "Date": started,
        "Time": f"training {train_min:.0f} min, asking {ask_min:.0f} min, total {(time.time() - T0) / 60:.0f} min",
    }
    samples = []
    for kind in ("fact_heldout", "logic_unseen", "general"):
        i = next((j for j, t in enumerate(tests) if t["kind"] == kind), None)
        if i is None:
            continue
        for name, sc in (("base", sb), ("base + pack", sp)):
            x = sc[i]
            samples.append(f"- {name} · Q: {x['question']} · expected {x['expected']} · "
                           f"{'right' if x['correct'] else 'wrong'}: `{x['reply'][:200].replace(chr(10), ' / ')}`")
    md = report(meta, base, pack, samples)
    (out / "RESULTS.md").write_text(md, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"meta": meta, "base": base, "pack": pack,
                                                  "rows": [{"base": x, "pack": y} for x, y in zip(sb, sp)]}, indent=1),
                                      encoding="utf-8")
    say("")
    say(md.split("What the rows mean:")[0])


if __name__ == "__main__":
    main()
