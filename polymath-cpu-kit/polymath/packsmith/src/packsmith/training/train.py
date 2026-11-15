"""Pack training job: stops the workbench engine (one GPU), unloads LM Studio models (REST API), runs
scripts/train_pack.py in the same venv, streams its log, restarts the engine and, if asked, loads the LM Studio models
again.

Sources: dataset:<id>, topic:<id> (its current dataset), composition:<id> (its built dataset)."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys

from packsmith.datasets import datasets
from packsmith.system import jobs
from packsmith.packs import store
from packsmith.workbench import workbench
from packsmith.system.config import SETTINGS

OPTS = ("layers", "n_layers", "k", "steps", "epochs", "batch", "accum", "max_len", "max_items", "lr", "kl", "slot_bias",
        "gate_lr_mult", "device", "chat_format", "eval_every_epoch", "seed", "loss", "curriculum",
        "mode", "init", "share", "route", "budget", "patience", "slot_init", "anchor_qa")        # mode / init / share / route: Polymath Engine packs only


def resolve(source: str, kind: str) -> tuple[str, dict]:
    """→ (dataset id, card fields)."""
    if kind == "topic":
        t = store.topic(source)
        did = t.get("current_dataset") or ""
        if not did:
            raise ValueError("the topic has no dataset yet: generate one")
        kind, source = "dataset", did
    if kind in ("composition", "curated"):
        c = store.rj(store.data_dir() / "compositions" / f"{source}.json") or {}
        if not c.get("dataset"):
            raise ValueError("build the composition first")
        kind, source = "dataset", c["dataset"]
    m = datasets.manifest(source)
    card = {"kind": m.get("source", "generate"), "dataset": source, "dataset_name": m["name"], "mode": m.get("mode"),
            "playbook": m.get("playbook")}
    if m.get("topic"):
        try:
            t = store.topic(m["topic"])
            card.update(topic=m["topic"], topic_name=t.get("name"), domain=t.get("domain") or t.get("name"),
                        record=t.get("record"),
                        sources=sorted({store.doc_meta(r["doc"])["name"] for r in t.get("regions", []) if _doc_ok(r["doc"])}))
        except KeyError:
            pass
    if m.get("composition"):
        card.update(composition=m["composition"], components=m.get("components"), domain=card.get("domain") or m["name"])
    return source, card


def _doc_ok(doc: str) -> bool:
    try:
        store.doc_meta(doc)
        return True
    except KeyError:
        return False


def run(source: str, name: str, kind: str = "dataset", opts: dict | None = None, unload_lmstudio: bool = True,
        reload_lmstudio: bool = False, progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    o = {"batch": SETTINGS.train_batch, "accum": SETTINGS.train_accum, "epochs": SETTINGS.train_epochs,
         "max_items": SETTINGS.train_max_items, **{k: v for k, v in (opts or {}).items() if v not in (None, "")}}
    from packsmith.engines import is_polymath
    if is_polymath(SETTINGS.engine):                              # Polymath packs: the reference engine trains them
        o.setdefault("budget", int(SETTINGS.pack_budget))
        # route packs: aim at 40% of the pack's own tokens at the start (was 25%) and push off general text with a
        # third of the old weight; the KL anchor keeps general answers the base's (see userpack.core.train)
        o.setdefault("share", 0.4)
        o.setdefault("route", 0.3)
    did, card = resolve(source, kind)
    data, tests = datasets.train_path(did), datasets.tests_path(did)
    if not data.exists() or not data.read_text().strip():
        raise ValueError("the dataset has no items")
    name = store.slug(name or card.get("dataset_name") or did, 60)
    out = store.packs_dir() / name
    if out.exists():
        shutil.rmtree(out)
    card_file = store.data_dir() / "cache" / f"card-{name}.json"
    store.wj(card_file, card)
    was_running = workbench.STATE["engine"] is not None
    loaded = []
    stopped_early = False
    job = jobs.current()
    with jobs.GPU_LOCK:
        progress(0.01, "stopping the workbench engine")
        workbench.stop()
        if unload_lmstudio:
            from packsmith.llm import llm
            loaded = [(m["key"], i["context_length"]) for m in llm.instances() for i in m["loaded"]]
            from packsmith.system import gpu as _g
            _g._remember(loaded)                  # loaded again on the next LM Studio call (shared GPU)
            un = llm.unload_all()
            progress(0.02, f"LLM server: unloaded {', '.join(un) or 'nothing (none loaded)'}")
        from packsmith.system import gpu as _gpu
        dev_env, device = _gpu.train_env()
        cmd = [sys.executable, "-m", "packsmith.training.train_pack", "--device", device,
               "--model", o.get("model") or str(SETTINGS.path("base_train_model")), "--data", str(data), "--out", str(out),
               "--card", str(card_file), "--engine", SETTINGS.engine]
        if tests.exists() and tests.read_text().strip():
            cmd += ["--tests", str(tests)]
        # a checkpoint after every epoch: a training stopped by a crash or a restart resumes there when the same
        # dataset is trained again under the same name with the same settings (else it starts afresh)
        cmd += ["--checkpoint", str(store.data_dir() / "cache" / f"ckpt-{name}")]
        for k in OPTS:
            if o.get(k) not in (None, "", False):
                cmd += ["--" + k.replace("_", "-")] + ([] if o[k] is True else [str(o[k])])
        progress(0.03, " ".join(cmd[2:6]) + f" … (GPU: {dev_env.get('CUDA_VISIBLE_DEVICES', device)})")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                             env={**os.environ, **dev_env})
        total = int(o.get("steps") or 0)
        signalled, last = False, 0.05
        for line in p.stdout:
            line = line.rstrip()
            if cancelled() and not signalled:
                signalled = True
                if job is not None and job._pause:       # stop and save: the trainer exports the best so far
                    p.send_signal(signal.SIGINT)
                    progress(last, "stopping after the current step and saving the pack…")
                    stopped_early = True
                else:
                    p.kill()
                    break
            m = re.search(r"total (\d+) steps", line)
            if m:
                total = int(m.group(1))
            m = re.search(r"step\s+(\d+)", line)
            if m and total:
                last = min(0.95, 0.05 + 0.9 * int(m.group(1)) / total)
            if line.startswith(("[train]", "[step]", "[epoch]", "[eta]", "[done]", "[stop]")) or "Error" in line or "error" in line:
                progress(last, line[:300])
        rc = p.wait()
    from packsmith.system import gpu
    gpu._set("")
    if was_running:
        progress(0.97, "restarting the workbench engine")
        workbench.start()
        gpu._set("engine")
    if reload_lmstudio and loaded and not gpu.shared():   # shared GPU: LM Studio is loaded again when next needed
        from packsmith.llm import llm
        for key, ctx in loaded:
            try:
                llm.load(key, ctx)
                progress(0.99, f"LLM server: loaded {key} again")
            except llm.LLMError as e:
                progress(0.99, f"LLM server: could not load {key}: {e}")
    if rc != 0 or not store.is_pack(out):
        if cancelled() and not stopped_early:
            raise RuntimeError("training cancelled")
        raise RuntimeError(f"training failed (exit {rc}); see the job log")
    from packsmith.packs import manifest
    if not (out / "overlay.json").exists():       # a Polymath Engine pack: the trainer wrote its pack.json
        man = manifest.read(out)
        datasets.add_pack(did, name)
        return {"pack": name, "path": str(out), "manifest": man, "dataset": did, "stopped_early": stopped_early,
                "card": json.loads((out / "card.json").read_text()), "lmstudio_unloaded": [k for k, _ in loaded]}
    man = manifest.write(out, SETTINGS.engine, base_train_model=o.get("model") or SETTINGS.base_train_model,
                         base_serve_model=SETTINGS.base_serve_model)
    datasets.add_pack(did, name)
    return {"pack": name, "path": str(out), "manifest": man, "dataset": did, "stopped_early": stopped_early,
            "card": json.loads((out / "card.json").read_text()), "lmstudio_unloaded": [k for k, _ in loaded]}
