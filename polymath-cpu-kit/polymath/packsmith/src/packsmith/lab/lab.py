"""The Train Lab: ways of training a pack tried on the same data, compared by what the packs answer.

    recipe       a named set of training settings. "baseline" is the Train tab's defaults (read from Settings and
                 the Train tab's own defaults below), cannot be edited, and is always trained and scored first.
    run          one dataset, one base model, one engine kind, several recipes, trained one after another and
                 scored on the same questions (a background job: pause, continue, cancel; it continues at the
                 recipe it was on, and a recipe's training from its last finished epoch).
    comparison   one row per recipe, the base model without a pack as the floor: every share with its count and
                 95% interval; a cell is marked better or worse than the baseline only when the intervals do not
                 overlap.

Nothing here changes how the Train tab, user packs, pipelines or the assistant train: the Lab passes the trainer's
new options (all off by default) only for its own runs, and writes only under <data>/lab/:
    recipes/<name>.json                 saved recipes
    datasets/yardstick/                 the invented harbour set (lab/yardstick.py), made once
    runs/<id>/run.json                  dataset, base model, engine kind, recipes with their resolved settings
    runs/<id>/<recipe>/                 the pack, the per-epoch copies, train.log, scores.json, settings.json
    runs/<id>/base_scores.json          the base model without a pack (the floor)
Paths in run records are relative to the data folder; no user name, home folder or machine name is stored.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from packsmith.packs import store
from packsmith.system.config import SETTINGS

# The Train tab's defaults as it sends them (gui/ui: epochs from Settings, batch / accumulation from Settings,
# training length 1024, learning rate 1e-4, KL anchor 1.0, Q&A items "as written + chat format", learn every token,
# no curriculum, standard size 2 x 4) and the trainer's own defaults for everything the tab does not send.
FIELDS = {   # name: (label, kind, help)
    "lr": ("learning rate", "float", "How far each update moves the weights. The pack's output weights start at zero."),
    "gate_lr_mult": ("gate learning rate ×", "float", "The gates (when the pack speaks) learn this many times faster."),
    "epochs": ("epochs", "int", "0 = auto: about 4,800 texts seen in all, 2 to 16 epochs."),
    "patience": ("patience", "int", "Stop when the held-out loss has not improved for this many epochs (0 = never)."),
    "batch": ("batch", "int", "Texts per step."),
    "accum": ("gradient accumulation", "int", "Steps added up before an update: texts per update = batch × this."),
    "max_len": ("training length", "int", "Tokens per text. Longer items are not cut silently: see the log."),
    "kl": ("anchor weight", "float", "Holds the pack to the base on general text."),
    "anchor_qa": ("anchor questions", "str", "'default' = the base's answers to 40 general questions; 'none' = prose only."),
    "slot_init": ("experts start as", "choice:copy,average", "Copies of the experts the data uses most, or their average."),
    "k": ("experts per layer", "int", "More room in each layer."),
    "n_layers": ("layers", "int", "How many layers get pack experts (a band in the middle)."),
    "loss": ("loss on", "choice:all,answer,final",
             "all: every token, the question too. answer: the reasoning (a filled thinking block) and the answer, "
             "not the question. final: only the final answer, after 'Answer:' or after the thinking block."),
    "chat_format": ("Q&A items", "choice:plain,chat,both", "As written, in the chat format, or both."),
    "think_share": ("trained both ways (thinking)", "float",
                    "0 to 1: this share of the Q&A items is also trained the other way: with an empty thinking block "
                    "if it has reasoning, with a filled one if not (its own reasoning or quoted evidence, else the "
                    "answer as a recall note). Chat form, Qwen models only. 0 = today's behaviour."),
    "curriculum": ("easy to hard", "bool", "Three stages by item kind and length, with replay."),
    "mode": ("Polymath mode", "choice:gate,route", "Polymath packs: added on top (gate) or competing for places (route)."),
    "save_epochs": ("score every epoch", "bool", "Keep a copy of the pack after every epoch and score each. No change to training."),
    "seed": ("seed", "int", "The same seed gives the same pack."),
    "items": ("items trained", "choice:all,source,reasoning",
              "Reasoning datasets made in the Lab only. all: every item; source: only the dataset's own items (the "
              "reasoning left out); reasoning: the reasoning items and only the dataset's items they rest on."),
}
TRAINER_FLAGS = {"lr": "--lr", "gate_lr_mult": "--gate-lr-mult", "epochs": "--epochs", "patience": "--patience",
                 "batch": "--batch", "accum": "--accum", "max_len": "--max-len", "kl": "--kl", "anchor_qa": "--anchor-qa",
                 "slot_init": "--slot-init", "k": "--k", "n_layers": "--n-layers", "loss": "--loss",
                 "chat_format": "--chat-format", "think_share": "--think-share", "mode": "--mode", "seed": "--seed"}


def baseline() -> dict:
    """Today's Train tab defaults, read from where the Train tab reads them."""
    return {"lr": 1e-4, "gate_lr_mult": 10.0, "epochs": int(SETTINGS.train_epochs or 0), "patience": 0,
            "batch": int(SETTINGS.train_batch or 2), "accum": int(SETTINGS.train_accum or 4), "max_len": 1024,
            "kl": 1.0, "anchor_qa": "default", "slot_init": "copy", "k": 2, "n_layers": 4, "loss": "all",
            "chat_format": "both", "think_share": 0.0, "curriculum": False, "mode": "gate", "save_epochs": False, "seed": 0,
            "items": "all"}


# Ready-made experiments: each differs from the baseline in one respect. (Schedules, warm-up, the kept-epoch rule and
# the data-shaping and reasoning recipes come with the next build steps.)
PRESETS = [
    ("score every epoch", {"save_epochs": True},
     "Nothing in training changes: a copy after every epoch, each scored. Did the kept epoch throw away a better one?"),
    ("answer only", {"loss": "answer"}, "Loss on the reasoning and the answer, not the question."),
    ("final answer only", {"loss": "final"}, "Loss on the final answer only: the thinking and the question are context."),
    ("thinking both ways", {"think_share": 0.5},
     "Half of the Q&A items also trained the other way: thinking filled as well as empty. Qwen, chat form."),
    ("longer", {"epochs": 15, "save_epochs": True}, "15 epochs, every epoch scored: does chaining keep improving after recall stops?"),
    ("more layers", {"n_layers": 8}, "8 pack layers instead of 4: too small in depth rather than under-trained?"),
    ("larger effective batch", {"accum": 8}, "2 × 8 texts per update instead of 2 × 4, the same texts seen: smoother, fewer steps."),
    ("route instead of gate", {"mode": "route"}, "Polymath packs only: the other mode."),
    ("easy to hard", {"curriculum": True}, "Three stages by item kind and length (facts, then reasoning), with replay."),
    ("thinking for every item", {"think_share": 1.0},
     "Every Q&A item also trained the other way (thinking filled and empty). Qwen, chat form."),
    # reasoning datasets made in the Lab: the same tests, a different share of the items
    ("without the reasoning", {"items": "source"},
     "Reasoning datasets: only the dataset's own items. Against the baseline (all items): what the reasoning adds."),
    ("reasoning with its facts only", {"items": "reasoning"},
     "Reasoning datasets: the reasoning items and only the facts they rest on."),
]
REASONING_ONLY = {"without the reasoning", "reasoning with its facts only"}


def root() -> Path:
    d = store.data_dir() / "lab"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rj(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _wj(p: Path, v) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(v, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


def _rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(store.data_dir().resolve()))
    except (ValueError, OSError):
        return Path(str(p)).name


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (s or "").strip()).strip("-.")[:60]
    if not s or s.endswith("test_lab") or "(test_lab)" in s:
        raise ValueError("a name is needed (and it may not carry the lab mark)")
    return s


# ------------------------------------------------------------------------------------------------ recipes
def clean_settings(st: dict) -> dict:
    base = baseline()
    out = dict(base)
    for k, v in (st or {}).items():
        if k not in FIELDS:
            continue
        kind = FIELDS[k][1]
        try:
            if kind == "float":
                out[k] = float(v)
            elif kind == "int":
                out[k] = int(v)
            elif kind == "bool":
                out[k] = v in (True, "true", "1", 1, "on")
            elif kind.startswith("choice:"):
                if str(v) not in kind[7:].split(","):
                    raise ValueError(f"{k} must be one of {kind[7:]}")
                out[k] = str(v)
            else:
                out[k] = str(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{FIELDS[k][0]}: {e}") from None
    if not 0 <= out["think_share"] <= 1:
        raise ValueError("trained both ways: a share from 0 to 1")
    if not 1e-6 <= out["lr"] <= 1e-2:
        raise ValueError("learning rate: 1e-6 to 1e-2")
    return out


def differs(st: dict, base: dict | None = None) -> dict:
    """The settings that differ from the baseline ({name: (baseline, recipe)}). Scoring every epoch is not a training
    setting and is left out. base: the run's own baseline (e.g. at another training length)."""
    b = base or baseline()
    return {k: (b[k], st[k]) for k in FIELDS if k != "save_epochs" and st.get(k) != b.get(k)}


def summary_line(st: dict, base: dict | None = None) -> str:
    d = differs(st, base)
    s = ", ".join(f"{FIELDS[k][0]} {v[1]}" for k, v in d.items()) or "the baseline"
    if st.get("save_epochs"):
        s += " · every epoch scored"
    if len(d) > 1:
        s += f" · differs in {len(d)} settings: the cause of any change cannot be told apart"
    return s


def recipes() -> list[dict]:
    """The baseline, the ready-made experiments, and the saved recipes."""
    out = [{"name": "baseline", "kind": "baseline", "settings": baseline(), "about": "Today's Train tab defaults.",
            "editable": False}]
    for n, ch, about in PRESETS:
        st = clean_settings({**baseline(), **ch})
        out.append({"name": n, "kind": "preset", "settings": st, "about": about, "editable": False})
    for f in sorted((root() / "recipes").glob("*.json")):
        r = _rj(f, {}) or {}
        try:
            st = clean_settings(r.get("settings") or {})
        except ValueError:
            continue
        out.append({"name": f.stem, "kind": "saved", "settings": st, "about": r.get("about", ""), "editable": True})
    for r in out:
        r["summary"] = summary_line(r["settings"])
        r["differs"] = {k: list(v) for k, v in differs(r["settings"]).items()}
    return out


def save_recipe(name: str, settings: dict, about: str = "") -> dict:
    n = _slug(name)
    if n in ("baseline",) or n in {p[0] for p in PRESETS}:
        raise ValueError(f"{n} is a built-in recipe: save it under another name")
    st = clean_settings(settings)
    _wj(root() / "recipes" / f"{n}.json", {"name": n, "settings": st, "about": about[:300]})
    return {"name": n, "settings": st, "summary": summary_line(st)}


def delete_recipe(name: str) -> None:
    (root() / "recipes" / f"{_slug(name)}.json").unlink(missing_ok=True)


def sweep(field: str, values: list, start: dict | None = None) -> list[dict]:
    """One recipe per value of one setting (the only way to make several at once)."""
    if field not in FIELDS:
        raise ValueError(f"no setting {field}")
    out = []
    for v in values:
        st = clean_settings({**(start or baseline()), field: v})
        out.append({"name": f"{FIELDS[field][0]} {v}", "settings": st, "summary": summary_line(st)})
    return out


# ------------------------------------------------------------------------------------------------ datasets
def yardstick_dir() -> Path:
    d = root() / "datasets" / "yardstick"
    if not (d / "train.jsonl").exists():
        from packsmith.lab.yardstick import build
        build(d)
    return d


def datasets() -> list[dict]:
    from packsmith.datasets import datasets as ds
    out = [{"id": "yardstick", "name": "yardstick: the invented harbour set (450 items, 1,080 tests)", "train": 450}]
    for m in ds.list_all():
        n = (m.get("counts") or {}).get("train") or 0
        if n:
            out.append({"id": m["id"], "name": f"{m['name']} · {n} items", "train": n})
    from packsmith.lab import reasoning
    for m in reasoning.list_all():
        r = m.get("report") or {}
        out.append({"id": "lab:" + m["id"], "name": f"Lab reasoning: {m['name']} · {r.get('train', 0)} items "
                    f"({r.get('reasoning_items', 0)} reasoning, {r.get('tests', 0)} reasoning tests)",
                    "train": r.get("train", 0), "reasoning": True, "needed_length": (r.get("long_items") or {}).get("needed_length")})
    return out


def _rows(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    except OSError:
        return []


def material(dataset: str, real_model: bool) -> dict:
    """{train, tests, quick, val}: the files and questions a run uses. The quick sample (scored after every epoch)
    is never part of the final tests."""
    from packsmith.lab.scoring import REAL_GENERAL
    if dataset == "yardstick":
        d = yardstick_dir()
        tests = _rows(d / "tests.jsonl")
        if real_model:
            tests = [t for t in tests if t["kind"] != "general"] + [{"question": q, "expected": e, "kind": "general"}
                                                                    for q, e in REAL_GENERAL]
        val = _rows(d / "val.jsonl")
        quick = [{**v, "kind": "fact_seen" if v["kind"] == "val_fact" else "logic_newwording"} for v in val]
        return {"train": d / "train.jsonl", "tests": tests, "quick": quick, "val": d / "val.jsonl",
                "anchor": "default" if real_model else str(d / "anchor_questions.txt")}
    if dataset.startswith("lab:"):
        return _lab_material(dataset[4:], real_model)
    from packsmith.datasets import datasets as ds
    m = ds.manifest(dataset)                               # KeyError when there is none
    tests = [dataset_test(t) for t in _rows(ds.tests_path(dataset)) if t.get("question") and t.get("expected")]
    quick, final = tests[:40], tests[40:]                  # the quick sample is taken out of the final tests
    if len(final) < 20:
        quick, final = tests[: len(tests) // 3], tests[len(tests) // 3:]
    gen = [{"question": q, "expected": e, "kind": "general"} for q, e in REAL_GENERAL]
    return {"train": ds.train_path(dataset), "tests": final + gen, "quick": quick, "val": ds.tests_path(dataset),
            "anchor": "default", "name": m.get("name", dataset)}


def dataset_test(t: dict) -> dict:
    """A dataset's own test, scored as "the dataset's own tests" and also under its kind (sub) when it has one: held-out
    facts, situations, chains, discussions, not covered, ordinary questions near the subject. A long expected answer
    (more than six words) is matched by its keywords, a short one as whole words."""
    long = len(str(t.get("expected", "")).split()) > 6
    return {"question": t["question"], "expected": t.get("expected", ""), "kind": "dataset_test", "sub": t.get("kind", ""),
            **({"match": "keywords"} if long or t.get("match") == "keywords" else {})}


def _lab_material(rid: str, real_model: bool) -> dict:
    """A reasoning dataset made in the Lab: its source dataset's tests (as that dataset is scored), its reasoning
    tests, and the general questions."""
    from packsmith.lab import reasoning
    m = reasoning.get(rid)
    d = reasoning.ddir(rid)
    src = _rows(d / "source_tests.jsonl")
    rt = _rows(d / "tests.jsonl")
    if m.get("source_kind") == "yardstick":
        base = material("yardstick", real_model)
        tests, quick, val, anchor = [t for t in base["tests"]] + rt, base["quick"], base["val"], base["anchor"]
    else:
        from packsmith.lab.scoring import REAL_GENERAL
        quick, final = src[:40], src[40:]
        if len(final) < 20:
            quick, final = src[: len(src) // 3], src[len(src) // 3:]
        tests = final + rt + [{"question": q, "expected": e, "kind": "general"} for q, e in REAL_GENERAL]
        val = d / "val.jsonl"
        if not val.exists():
            val.write_text("".join(json.dumps(x) + "\n" for x in (quick or rt[:20])), encoding="utf-8")
        anchor = "default"
    return {"train": d / "train.jsonl", "tests": tests, "quick": quick, "val": val, "anchor": anchor,
            "name": "Lab reasoning: " + m["name"], "reasoning": True,
            "needed_length": (m["report"].get("long_items") or {}).get("needed_length")}


# ------------------------------------------------------------------------------------------------ runs
def _run_dir(rid: str) -> Path:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", rid or ""):
        raise KeyError(f"no run {rid!r}")
    return root() / "runs" / rid


def _recipe_dir(rid: str, i: int) -> Path:
    return _run_dir(rid) / f"r{i:02d}"


def timing(model: str) -> float | None:
    """Seconds per training step measured on this machine for this base model (the last Lab run), or None."""
    t = _rj(root() / "timing.json", {}) or {}
    return t.get(Path(str(model)).name)


def estimate(dataset: str, recipe_list: list[dict], model: str) -> dict:
    """Steps and time per recipe before Start, from the measured seconds per step of the last run on this base
    model (none measured yet: steps only)."""
    from packsmith.training.train_pack import auto_epochs
    n = next((d["train"] for d in datasets() if d["id"] == dataset), 0)
    sps = timing(model)
    out, total = [], 0.0
    for r in recipe_list:
        st = r["settings"]
        texts = int(n * (1.67 if st["chat_format"] == "both" else 1.0) * (1 + st["think_share"] * 0.67))
        ep = st["epochs"] or auto_epochs(texts)
        steps = -(-texts // max(1, st["batch"])) * ep
        sec = steps * sps if sps else None
        total += sec or 0
        out.append({"name": r["name"], "texts": texts, "epochs": ep, "steps": steps, "seconds": sec})
    return {"recipes": out, "seconds": total if sps else None, "seconds_per_step": sps,
            "note": "" if sps else "no training measured on this base model yet: the time shows after the first run"}


def create_run(body: dict) -> dict:
    """A run: dataset, base model, engine kind and the recipes (baseline first, always)."""
    from packsmith import __version__
    from packsmith.engines import is_polymath
    dataset = str(body.get("dataset") or "yardstick")
    model = str(body.get("model") or SETTINGS.base_train_model or "")
    if not model:
        raise ValueError("no base model (Settings → Paths → base model for training)")
    engine = str(body.get("engine") or "patched-colibri")
    if engine not in ("patched-colibri", "polymath-engine"):
        raise ValueError("the pack kind: patched-colibri (B-slot) or polymath-engine (Polymath packs)")
    real = not (Path(model) / "pretrain.json").exists() and model != "tiny"
    mat = material(dataset, real)                          # checks the dataset exists
    lim = body.get("sample") or {}                         # a trial on a sample: marked, never compared with full runs
    want = list(body.get("recipes") or [])
    have = {r["name"]: r for r in recipes()}
    chosen = [have["baseline"]]
    for r in want:
        if isinstance(r, str):
            if r != "baseline" and r in have:
                chosen.append(have[r])
            elif r != "baseline":
                raise ValueError(f"no recipe {r!r}")
        elif isinstance(r, dict) and r.get("name") and r.get("name") != "baseline":
            st = clean_settings(r.get("settings") or {})
            chosen.append({"name": str(r["name"])[:80], "settings": st})
    raised = None
    for r in chosen:
        if r["settings"]["mode"] == "route" and not is_polymath(engine):
            raise ValueError(f"{r['name']}: route mode is for Polymath packs")
        if r["settings"].get("items", "all") != "all" and not mat.get("reasoning"):
            raise ValueError(f"{r['name']}: 'items trained' needs a reasoning dataset made in the Lab")
    tl = int(body.get("train_length") or 0)              # one training length for every recipe, the baseline too
    if tl:
        if tl not in (1024, 2048, 4096, 8192):
            raise ValueError("training length: 1024, 2048, 4096 or 8192")
        b, acc = (2, 4) if tl <= 1024 else (1, 8)          # longer: the same 8 texts per update, one in memory at a time
        base_acc = max(1, int(chosen[0]["settings"]["accum"]))
        for r in chosen:                                   # a recipe with a larger accumulation keeps its multiple
            mult = max(1, int(r["settings"]["accum"]) // base_acc)
            r["settings"] = {**r["settings"], "max_len": tl, "batch": b, "accum": acc * mult}
    need = mat.get("needed_length") or 0
    if need:                                               # long items kept whole: every recipe gets the length (same data)
        for r in chosen:
            if r["settings"]["max_len"] < need:
                r["settings"] = {**r["settings"], "max_len": int(need)}
                raised = int(need)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run = {"id": rid, "name": str(body.get("name") or "").strip()[:80] or time.strftime("lab %d %b %H:%M"),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "packsmith": __version__, "dataset": dataset,
           "dataset_name": mat.get("name", "yardstick"), "model": Path(model).name, "model_ref": _rel(model) if Path(model).exists() else model,
           "engine": engine, "device": str(body.get("device") or ""), "status": "created",
           "recipes": [{"name": r["name"], "settings": r["settings"], "summary": summary_line(r["settings"], chosen[0]["settings"]),
                        "differs": {k: list(v) for k, v in differs(r["settings"], chosen[0]["settings"]).items()}}
                       for r in chosen],
           "tests": {"n": len(mat["tests"]), "quick": len(mat["quick"])},
           **({"note": f"training length raised to {raised} for every recipe: the dataset keeps long items whole"} if raised else
              {"note": f"every recipe, the baseline too, trained at {tl} tokens (batch {b} x {acc})"} if tl and tl != 1024 else {})}
    d = _run_dir(rid)
    if lim:
        import random
        rng = random.Random(0)
        tr = _rows(mat["train"])
        if lim.get("train"):
            tr = rng.sample(tr, min(len(tr), int(lim["train"])))
        (d / "sample").mkdir(parents=True, exist_ok=True)
        (d / "sample" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in tr) + "\n", encoding="utf-8")
        mat["train"] = d / "sample" / "train.jsonl"
        if lim.get("tests"):
            want_n, by = int(lim["tests"]), {}
            for t_ in mat["tests"]:
                by.setdefault(t_["kind"], []).append(t_)
            share = min(1.0, want_n / max(1, len(mat["tests"])))   # every kind of question keeps at least one
            mat["tests"] = [x for k_ in sorted(by) for x in rng.sample(by[k_], min(len(by[k_]), max(1, round(len(by[k_]) * share))))]
        if lim.get("quick"):
            mat["quick"] = mat["quick"][: int(lim["quick"])]
        run["sample"] = {"train": len(tr), "tests": len(mat["tests"]), "quick": len(mat["quick"])}
        run["tests"] = {"n": len(mat["tests"]), "quick": len(mat["quick"])}
    _wj(d / "run.json", run)
    _wj(d / "tests.json", mat["tests"])
    (d / "quick.jsonl").write_text("\n".join(json.dumps(q) for q in mat["quick"]) + "\n", encoding="utf-8")
    _wj(d / "material.json", {"train": _rel(mat["train"]), "val": _rel(mat["val"]), "reasoning": bool(mat.get("reasoning")),
                              "anchor": mat["anchor"] if mat["anchor"] == "default" else _rel(mat["anchor"])})
    run["_model"] = model
    _wj(d / "model.json", {"model": model})                # the full path stays local to this run (never exported)
    return run


def _abs(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else store.data_dir() / p


def _device(run: dict) -> str:
    if run.get("device"):
        return run["device"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def train_recipe(run: dict, i: int, log) -> dict:
    """Train recipe i of the run with Packsmith's trainer (a subprocess, as the Train tab runs it), resuming from its
    checkpoint after a stop."""
    d = _run_dir(run["id"])
    rd = _recipe_dir(run["id"], i)
    rec = run["recipes"][i]
    st = rec["settings"]
    model = (_rj(d / "model.json") or {}).get("model")
    mat = _rj(d / "material.json")
    rd.mkdir(parents=True, exist_ok=True)
    pack = rd / "pack"
    if (pack / "card.json").exists():
        return _rj(pack / "card.json", {})
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")])}
    data = _abs(mat["train"])
    rows = _rows(data) if mat.get("reasoning") else []
    if rows:                                                # a Lab reasoning dataset: the recipe's share of its items
        from packsmith.training.train_pack import is_qwen
        keep = {"all": None, "source": {"source"}, "reasoning": {"reasoning", "think"}}[st.get("items", "all")]
        sel = [x for x in rows if keep is None or x.get("lab_part") in keep or (st.get("items") == "reasoning" and x.get("needed"))]
        if not is_qwen(model) and model != "tiny":
            n0 = len(sel)
            sel = [x for x in sel if x.get("lab_part") != "think"]
            if n0 != len(sel):
                log(f"{rec['name']}: {n0 - len(sel)} thinking-block items left out (the chat form with a thinking block is Qwen's)")
        if len(sel) != len(rows):
            data = rd / "train.jsonl"
            data.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sel), encoding="utf-8")
    cmd = [sys.executable, "-m", "packsmith.training.train_pack", "--model", model, "--data", str(data),
           "--tests", str(_abs(mat["val"])), "--out", str(pack), "--device", _device(run), "--engine", run["engine"],
           "--checkpoint", str(rd / "ckpt"), "--epoch-score", str(d / "quick.jsonl")]
    for k, flag in TRAINER_FLAGS.items():
        v = st[k]
        if k == "anchor_qa":
            v = mat["anchor"] if v == "default" and mat["anchor"] != "default" else v
            v = str(_abs(v)) if v not in ("default", "none") else v
        if k == "mode" and run["engine"] != "polymath-engine":
            continue
        cmd += [flag, str(v)]
    if st.get("curriculum"):
        cmd.append("--curriculum")
    if st.get("save_epochs"):
        cmd.append("--save-epochs")
    _wj(rd / "settings.json", {"recipe": rec["name"], "resolved": st, "trainer_args": [c if not c.startswith(str(store.data_dir())) else _rel(c) for c in cmd[3:]],
                               "packsmith": run["packsmith"], "defaults_version": __import__("packsmith.system.config", fromlist=["DEFAULTS_VERSION"]).DEFAULTS_VERSION})
    log(f"training {rec['name']}")
    t0 = time.time()
    with open(rd / "train.log", "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        resumed = False
        for line in p.stdout:
            f.write(line)
            f.flush()
            resumed = resumed or line.startswith("[train] resuming")
            if line.startswith(("[epoch]", "[train] resuming", "[done]")):
                log(f"{rec['name']}: {line.strip()[:200]}")
        rc = p.wait()
    if rc != 0 or not (pack / "card.json").exists():
        tail = (rd / "train.log").read_text(errors="replace").splitlines()[-12:]
        raise RuntimeError(f"{rec['name']}: training failed (exit {rc}): " + " | ".join(tail)[-1500:])
    _relative_cards(rd, model)
    card = _rj(pack / "card.json", {}) or {}
    steps = card.get("steps") or (card.get("trained") or {}).get("steps") or 0
    if steps and not resumed:                               # seconds per step on this machine, for the next estimate
        t = _rj(root() / "timing.json", {}) or {}
        t[Path(str(model)).name] = round((time.time() - t0) / max(1, steps), 3)
        _wj(root() / "timing.json", t)
    return card


def _relative_cards(rd: Path, model: str) -> None:
    """The trainer writes full paths into its cards; in the Lab they are made relative to the data folder (and the
    base model reduced to its folder name), so no home folder or user name is kept in a Lab pack."""
    dd = str(store.data_dir().resolve())
    reps = [(dd + os.sep, ""), (dd, "."), (str(model), Path(str(model)).name), (str(Path.home()), "~")]
    for f in rd.rglob("*.json"):
        try:
            t = f.read_text(encoding="utf-8")
        except OSError:
            continue
        t2 = t
        for a, b in reps:
            if a and a not in (".", os.sep):
                t2 = t2.replace(json.dumps(a)[1:-1], json.dumps(b)[1:-1])
        if t2 != t:
            f.write_text(t2, encoding="utf-8")


class _A:
    pass


def _score_args(run: dict, model: str):
    from packsmith.training.train_pack import is_qwen
    a = _A()
    a.model, a.device = model, _device(run)
    a.format = "chat" if (model == "tiny" or is_qwen(model)) and run["recipes"][0]["settings"]["chat_format"] != "plain" else "plain"
    a.max_new, a.max_new_think = 96, 192
    return a


def score(run: dict, pack: Path | None, out: Path, log) -> dict:
    """The full score of one pack (None = the base model), thinking on and off as separate columns in the chat
    format. Saved to out; read back when it is there."""
    if Path(out).exists():
        return _rj(out, {})
    from packsmith.core.general import GENERAL_TEXT
    from packsmith.lab.scoring import score_pack
    d = _run_dir(run["id"])
    model = (_rj(d / "model.json") or {}).get("model")
    a = _score_args(run, model)
    tests = _rj(d / "tests.json", [])
    modes = [True, False] if a.format == "chat" else [None]
    log(f"scoring {'the base model' if pack is None else _rel(pack)} on {len(tests)} questions"
        + (" (thinking on and off)" if len(modes) > 1 else ""))
    ev = score_pack(a, pack, tests, list(GENERAL_TEXT), modes=modes)
    res = {"format": a.format, "modes": {("on" if m == "True" else "off" if m == "False" else "plain"): v for m, v in ev.items()}}
    _wj(out, res)
    return res


def run_job(rid: str, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Train and score every recipe of the run, one after another; the base model is scored once as the floor."""
    from packsmith.workbench.testbench import Hold
    d = _run_dir(rid)
    run = _rj(d / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    from packsmith.system import jobs
    cj = jobs.current()
    if cj is not None:
        run["job"] = cj.id
    run["status"] = "running"
    _wj(d / "run.json", run)
    n = len(run["recipes"])
    state = {"i": 0}

    def log(m):
        progress(min(0.99, state["i"] / max(1, n)), m)
    with Hold(rid, log, "Train Lab run"):                          # the Workbench's engine stopped, put back afterwards
        score(run, None, d / "base_scores.json", log)
        for i, rec in enumerate(run["recipes"]):
            state["i"] = i
            if cancelled():
                break
            rd = _recipe_dir(rid, i)
            if (rd / "scores.json").exists() and (not rec["settings"].get("save_epochs") or (rd / "epoch_scores.json").exists()):
                continue
            train_recipe(run, i, log)
            if cancelled():
                break
            score(run, rd / "pack", rd / "scores.json", log)
            if rec["settings"].get("save_epochs"):
                eps = {}
                for e in sorted((rd / "pack" / "epochs").glob("e*")):
                    if cancelled():
                        break
                    eps[e.name] = score(run, e, rd / "epochs_scored" / f"{e.name}.json", log)
                if not cancelled():
                    _wj(rd / "epoch_scores.json", {k: _brief(v) for k, v in eps.items()})
    run = _rj(d / "run.json")
    run["status"] = "paused" if cancelled() else "done"
    run["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _wj(d / "run.json", run)
    return {"run": rid, "status": run["status"]}


def _brief(res: dict) -> dict:
    return {m: {"scores": v["scores"], "detail": v["detail"]} for m, v in (res.get("modes") or {}).items()}


def start(body: dict) -> dict:
    from packsmith.system import jobs
    run = create_run(body)
    run.pop("_model", None)
    j = jobs.start("lab", run_job, run["name"], rid=run["id"])
    return {"run": run, "job": j.view()}


def resume(rid: str) -> dict:
    from packsmith.system import jobs
    run = _rj(_run_dir(rid) / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    j0 = jobs.get(run.get("job") or "")
    if j0 and j0.get("status") == "running":
        raise ValueError("the run is still running")
    return jobs.start("lab", run_job, run["name"] + " (continued)", rid=rid).view()


def list_runs() -> list[dict]:
    out = []
    for f in sorted((root() / "runs").glob("*/run.json"), reverse=True):
        r = _rj(f, {}) or {}
        done = sum(1 for i in range(len(r.get("recipes") or [])) if (f.parent / f"r{i:02d}" / "scores.json").exists())
        out.append({"id": r.get("id"), "name": r.get("name"), "created": r.get("created"), "status": r.get("status"),
                    "dataset": r.get("dataset_name"), "model": r.get("model"), "engine": r.get("engine"),
                    "recipes": len(r.get("recipes") or []), "done": done, "bytes": _size(f.parent)})
    return out


def _size(p: Path) -> int:
    n = 0
    for root_, _, files in os.walk(p):
        for x in files:
            try:
                n += os.path.getsize(os.path.join(root_, x))
            except OSError:
                pass
    return n


def delete_run(rid: str) -> dict:
    d = _run_dir(rid)
    n = _size(d)
    shutil.rmtree(d, ignore_errors=True)
    return {"freed": n - (_size(d) if d.exists() else 0)}


# ------------------------------------------------------------------------------------------------ the comparison
def _cell(det: dict | None) -> dict | None:
    if not det:
        return None
    return {"k": det["k"], "n": det["n"], "p": round(det["k"] / det["n"], 3) if det["n"] else None, "ci": det["ci95"]}


def _mark(c: dict | None, b: dict | None, higher_better: bool = True) -> str:
    """better / worse than the baseline only when the 95% intervals do not overlap."""
    if not c or not b or not c["n"] or not b["n"]:
        return ""
    if c["ci"][0] > b["ci"][1]:
        return "better" if higher_better else "worse"
    if c["ci"][1] < b["ci"][0]:
        return "worse" if higher_better else "better"
    return ""


LOWER_BETTER = {"think_open", "think_skipped", "repeating", "fact_heldout_reasoned"}


def comparison(rid: str) -> dict:
    """The run with every recipe's scores per thinking mode, cells as {k, n, p, ci, mark}, the per-epoch curves,
    and a plain summary by rule."""
    from packsmith.lab.scoring import MEASURES
    d = _run_dir(rid)
    run = _rj(d / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    base = _rj(d / "base_scores.json")
    rows = []
    for i, rec in enumerate(run["recipes"]):
        rd = _recipe_dir(rid, i)
        sc = _rj(rd / "scores.json")
        card = _rj(rd / "pack" / "card.json", {}) or {}
        eps = _rj(rd / "epoch_scores.json", {}) or {}
        rows.append({"name": rec["name"], "summary": rec["summary"], "differs": rec["differs"],
                     "settings": rec["settings"], "scores": sc, "card": {k: card.get(k) for k in (
                         "best_epoch", "steps", "epochs", "train_seconds", "texts", "layers", "k")},
                     "epoch_scores": eps, "pack_bytes": _size(rd / "pack") if (rd / "pack").exists() else 0,
                     "trained": (rd / "pack" / "card.json").exists()})
    modes = sorted({m for r in rows if r["scores"] for m in r["scores"]["modes"]} | ({m for m in base["modes"]} if base else set()),
                   key=lambda m: ["on", "off", "plain"].index(m))
    keys = [k for k in MEASURES if any(r["scores"] and k in r["scores"]["modes"].get(m, {}).get("detail", {})
                                       for r in rows for m in modes)]
    table = {}
    b0 = rows[0]["scores"]
    for m in modes:
        t = {"floor": {k: _cell((base or {}).get("modes", {}).get(m, {}).get("detail", {}).get(k)) for k in keys}, "rows": []}
        for r in rows:
            cells = {}
            for k in keys:
                c = _cell((r["scores"] or {}).get("modes", {}).get(m, {}).get("detail", {}).get(k))
                bc = _cell((b0 or {}).get("modes", {}).get(m, {}).get("detail", {}).get(k))
                if c is not None and r is not rows[0]:
                    c["mark"] = _mark(c, bc, k not in LOWER_BETTER)
                cells[k] = c
            t["rows"].append(cells)
        table[m] = t
    return {"run": run, "rows": rows, "modes": modes, "measures": [{"key": k, "label": MEASURES[k][0],
            "lower_better": k in LOWER_BETTER} for k in keys], "table": table, "summary": plain_summary(rows, table, keys, modes)}


def plain_summary(rows: list[dict], table: dict, keys: list[str], modes: list[str]) -> list[str]:
    """Three or four sentences from the figures, by rule: which recipes are clearly better or worse than the
    baseline on which measure (the intervals apart), and which cannot be told apart."""
    from packsmith.lab.scoring import MEASURES
    out = []
    if len(rows) < 2 or not rows[0]["scores"]:
        return ["The baseline has not been scored yet."]
    m = "off" if "off" in modes else modes[0]
    t = table[m]["rows"]
    same = []
    for r, cells in list(zip(rows, t))[1:]:
        if not r["scores"]:
            continue
        better = [k for k in keys if (cells.get(k) or {}).get("mark") == "better"]
        worse = [k for k in keys if (cells.get(k) or {}).get("mark") == "worse"]
        if not better and not worse:
            same.append(r["name"])
            continue

        def fmt(k):
            c, b = cells[k], t[0][k]
            return f"{MEASURES[k][0]} ({c['p']:.0%} against {b['p']:.0%})"
        s = r["name"].capitalize()
        if better:
            s += " is better than the baseline on " + "; ".join(fmt(k) for k in better)
        if worse:
            s += (" but" if better else "") + " worse on " + "; ".join(fmt(k) for k in worse)
        out.append(s + ".")
    if same:
        out.append(", ".join(same).capitalize() + (" cannot" if len(same) == 1 else " cannot each") +
                   " be told apart from the baseline on this many questions.")
    out.append(f"(Read with thinking {m}; a mark needs the 95% intervals to be apart.)")
    return out
