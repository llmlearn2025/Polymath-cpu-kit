"""Engine models — the simple way: the engine serves the folder in Settings (base_serve_model), colibri loads it.

The list is only a convenience, nothing is validated:
  * colibri containers: the configured model, plus folders in the runs folder and <data_dir>/models that hold
    qwen36_meta.json (Qwen) or a packsmith_model.json marker;
  * the Hugging Face cache (Settings → hf_cache_dir, default /home/<user>/.cache/huggingface/hub) and Polymath's models folder
    ($POLYMATH_HOME/models), listed by name.

How the cache keeps a model: models--Qwen--Qwen3-30B-A3B/
    blobs/<sha256>                 the file contents (the big .safetensors live here)
    snapshots/<commit>/<file>      symlinks into ../../blobs — this folder is "the model folder" for any tool
    refs/main                      the commit that snapshots/<commit> is
So colibri's converter is pointed at snapshots/<commit>; the symlinks resolve to the blobs. No huggingface_hub needed.

`convert(source)` runs colibri's tools/convert_qwen36.py (int4, gs 64) on such a folder → a container in
the runs folder. `use(path)` makes a folder the engine's (and training's) model."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from packsmith.system.config import SETTINGS, save
from packsmith.system.home import expand

MARKER = "packsmith_model.json"


def user_home() -> Path:
    """The real user's home, also under sudo (where $HOME is /root)."""
    from packsmith.system.home import user_home as _uh
    return _uh()


def hf_cache() -> Path:
    """Settings → hf_cache_dir; "~" means the real user's home even under sudo (/home/<user>/.cache/huggingface/hub).
    HF_HUB_CACHE overrides it."""
    if os.environ.get("HF_HUB_CACHE"):
        return expand(os.environ["HF_HUB_CACHE"])
    return expand(str(getattr(SETTINGS, "hf_cache_dir", "") or "~/.cache/huggingface/hub"))


def models_dir() -> Path:
    """Polymath's own models folder: Hugging Face cache layout (models--Org--Name) or plain model folders."""
    return SETTINGS.path("models_dir")


def search_roots() -> list[Path]:
    """Where a downloaded base model is looked for, in order: the user's Hugging Face cache, then Polymath's models."""
    out = []
    for r in (hf_cache(), models_dir()):
        if r not in out:
            out.append(r)
    return out


def snapshot(repo_dir: Path) -> Path | None:
    """models--Org--Name → its snapshots/<commit> folder (the one refs/main names, else the newest)."""
    ref = repo_dir / "refs" / "main"
    if ref.exists():
        s = repo_dir / "snapshots" / ref.read_text().strip()
        if s.is_dir():
            return s
    snaps = sorted((repo_dir / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if (repo_dir / "snapshots").is_dir() else []
    return snaps[0] if snaps else None


def cached() -> list[dict]:
    """Downloaded Hugging Face models in every search root: cache layout (models--Org--Name/snapshots/<rev>) and plain
    folders holding config.json + *.safetensors (repo from their polymath_model.json, else the folder name)."""
    out, seen = [], set()
    for hub in search_roots():
        if not hub.is_dir():
            continue
        for d in sorted(hub.glob("models--*")):
            s = snapshot(d)
            if s and os.path.realpath(s) not in seen:
                seen.add(os.path.realpath(s))
                out.append({"repo": d.name[len("models--"):].replace("--", "/"), "path": str(s), "kind": "hf", "root": str(hub)})
        for d in sorted(hub.iterdir()):
            if d.is_dir() and not d.name.startswith("models--") and (d / "config.json").exists() \
                    and any(d.glob("*.safetensors")) and os.path.realpath(d) not in seen:
                seen.add(os.path.realpath(d))
                repo = ""
                try:
                    repo = json.loads((d / "polymath_model.json").read_text()).get("repo", "")
                except (OSError, ValueError):
                    pass
                out.append({"repo": repo or d.name, "path": str(d), "kind": "hf", "root": str(hub)})
    return out


def _marker(p: Path) -> dict:
    try:
        return json.loads((p / MARKER).read_text())
    except (OSError, ValueError):
        return {}


def info(p) -> dict:
    p = expand(str(p))
    mk = _marker(p)
    engine = "qwen36" if (p / "qwen36_meta.json").exists() else mk.get("engine") or "olmoe"
    return {"path": str(p), "name": p.name, "engine": engine, "repo": mk.get("repo", ""),
            "label": mk.get("label") or p.name, "source": mk.get("source", ""), "kind": "container"}


def containers() -> list[dict]:
    seen, out = set(), []
    cands = [SETTINGS.path("base_serve_model"), SETTINGS.path("base_train_model")]
    for r in (models_dir() / "colibri", SETTINGS.path("runs_dir"), expand(SETTINGS.data_dir) / "models"):
        if r.is_dir():
            cands += [d for d in sorted(r.iterdir()) if (d / "qwen36_meta.json").exists() or (d / MARKER).exists()]
    for c in cands:
        k = os.path.realpath(str(expand(str(c))))
        if k not in seen and os.path.isdir(k):
            seen.add(k)
            out.append(info(c))
    return out


def listing() -> dict:
    serve = os.path.realpath(SETTINGS.path("base_serve_model"))
    train = os.path.realpath(SETTINGS.path("base_train_model"))
    cs = containers()
    for c in cs:
        r = os.path.realpath(c["path"])
        c.update(serving=r == serve, training=r == train)
    return {"containers": cs, "hf": cached(), "hf_cache": str(hf_cache())}


def current() -> dict:
    return info(SETTINGS.path("base_serve_model"))


def _source(src: str) -> tuple[Path, str]:
    """A folder, or a repo name in the HF cache ("Qwen/Qwen3-30B-A3B") → (model folder, repo)."""
    p = expand(src)
    if p.is_dir():
        if p.name.startswith("models--"):
            s = snapshot(p)
            return s, p.name[len("models--"):].replace("--", "/")
        return p, ""
    for c in cached():
        if c["repo"].lower() == src.lower():
            return Path(c["path"]), c["repo"]
    raise FileNotFoundError(f"{src}: not a folder and not in the Hugging Face cache {hf_cache()} "
                            f"(there: {', '.join(c['repo'] for c in cached()) or 'nothing'})")


def convert(src: str, name: str = "", use_it: bool = False, progress=lambda f, m="": None, cancelled=lambda: False,
            purpose: str = "both") -> dict:
    """Convert a Hugging Face model folder (or a downloaded repo) into a colibri container in the shared models
    folder (<models_dir>/colibri). purpose: both | inference | training — which setting(s) the result is used for
    when use_it. Qwen3-MoE: experts int4, one scale per 64 weights (tools/convert_qwen36.py); OLMoE: colibri's merged
    int8 container (tools/convert_olmoe_merged.py)."""
    folder, repo = _source(src)
    try:
        mt = json.loads((folder / "config.json").read_text()).get("model_type", "")
    except (OSError, ValueError):
        mt = ""
    kind = "olmoe" if mt == "olmoe" else "qwen36"
    out = models_dir() / "colibri" / (name or re.sub(r"[^a-z0-9]+", "_", (repo or folder.name).split("/")[-1].lower()) + "_coli")
    done_file = out / ("qwen36_meta.json" if kind == "qwen36" else MARKER)
    if done_file.exists():
        progress(0.95, f"{out} is already converted")
    else:
        if kind == "qwen36":
            tool = SETTINGS.path("colibri_dir") / "c" / "tools" / "convert_qwen36.py"
            cmd_tail = ["--model", str(folder), "--out", str(out), "--ebits", "4", "--gs", "64"]
            what = "int4, scale per 64; ~20 GB, 10–30 min for the 30B"
        else:
            tool = SETTINGS.path("colibri_dir") / "c" / "tools" / "convert_olmoe_merged.py"
            cmd_tail = ["--model", str(folder), "--out", str(out)]
            what = "int8"
        if not tool.exists():
            raise FileNotFoundError(f"{tool} not found (run install_patched_colibri.sh, or Settings → colibri_dir)")
        from packsmith.system.home import ensure_dir
        ensure_dir(out.parent)
        cmd = [sys.executable, str(tool)] + cmd_tail
        progress(0.02, f"converting for colibri ({what}): " + " ".join(cmd[1:]))
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        n = 0
        for line in p.stdout:
            n += 1
            if cancelled():
                p.kill()
                raise RuntimeError("cancelled")
            m = re.search(r"layer\s+(\d+)\s*/\s*(\d+)", line)
            progress(0.05 + 0.9 * int(m.group(1)) / int(m.group(2)) if m else min(0.9, 0.05 + n * 0.002), line.rstrip()[:200])
        if p.wait() != 0 or not out.is_dir() or (kind == "qwen36" and not (out / "qwen36_meta.json").exists()):
            raise RuntimeError(f"conversion failed (exit {p.returncode}); see the log")
        (out / MARKER).write_text(json.dumps({"repo": repo, "label": f"{repo or folder.name} (colibri {'int4' if kind == 'qwen36' else 'int8'})", "engine": kind,
                                              "source": str(folder), "converted": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
        from packsmith.packs import manifest
        manifest.model_identity(out)              # polymath_model.json: the source's weights hash, for pack checks
        from packsmith.system.home import give_to_user
        give_to_user(out)
    if use_it:
        use(str(out), train=purpose in ("both", "training"), serve=purpose in ("both", "inference"))
    progress(1.0, f"ready: {out}")
    return {**info(out), "used": use_it, "purpose": purpose}


def use_for_training_only(path: str) -> dict:
    """Train on a Hugging Face folder as it is (bf16: small models, or cards with ~64 GB for a 30B)."""
    p = expand(path)
    if not (p / "config.json").exists():
        raise ValueError(f"{p} is not a Hugging Face model folder")
    SETTINGS.base_train_model = str(p)
    save(SETTINGS)
    from packsmith.packs import manifest
    manifest.model_identity(p)
    return {"path": str(p), "training": True}


def use(path: str, train: bool = True, serve: bool = True) -> dict:
    """Serve (and by default train on) this folder; restarts the workbench engine if it was running."""
    p = expand(path)
    if not p.is_dir():
        raise ValueError(f"{p} is not a folder")
    from packsmith.workbench import workbench
    was = serve and workbench.STATE.get("engine") is not None
    if was:
        workbench.stop()
    if serve:
        SETTINGS.base_serve_model = str(p)
    if train:                                            # an OLMoE container is for serving: train on its source folder
        from packsmith.core.model import training_source
        try:
            SETTINGS.base_train_model = training_source(str(p))
        except FileNotFoundError:                        # its source is gone: training says so, with what to do
            SETTINGS.base_train_model = str(p)
    save(SETTINGS)
    if was:
        workbench.start()
    return {**info(p), "restarted": was}


def pack_note(card: dict) -> str:
    """When a pack was trained on another model folder than the one served (only a note)."""
    bm = (card or {}).get("base_model")
    if not bm:
        return ""
    same = os.path.realpath(str(expand(bm))) == os.path.realpath(SETTINGS.path("base_serve_model"))
    return "" if same else f"trained on {Path(bm).name}, served on {SETTINGS.path('base_serve_model').name}"


def training_choices() -> dict:
    """The models the Train tab can train on: the one in Settings first, then the colibri Qwen3-MoE containers, the
    Hugging Face folders of the OLMoE containers, and the OLMoE / Qwen3-MoE folders in the Hugging Face cache. Each with
    its family and whether it is the base the engine serves (a pack is used on the model it was trained for)."""
    from packsmith.core.model import training_source
    serve = SETTINGS.path("base_serve_model") if SETTINGS.base_serve_model else None
    serve_src = ""
    if serve:
        try:
            serve_src = os.path.realpath(training_source(str(serve)))
        except FileNotFoundError:
            serve_src = os.path.realpath(str(serve))
    out, seen = [], set()

    def fam_of(path: str) -> str:
        p = expand(path)
        if (p / "qwen36_meta.json").exists():
            return "qwen"
        try:
            mt = json.loads((p / "config.json").read_text()).get("model_type", "")
        except (OSError, ValueError):
            return ""
        return "olmoe" if mt == "olmoe" else "qwen" if mt.startswith("qwen3_moe") else ""

    def add(path: str, label: str, how: str):
        k = os.path.realpath(str(expand(path)))
        if k in seen or not os.path.isdir(k):
            return
        f = fam_of(path)
        if not f:
            return
        seen.add(k)
        out.append({"path": str(path), "label": label, "family": f, "how": how, "serving": k == serve_src,
                    "settings": False})
    st = str(SETTINGS.base_train_model or "")
    if st:
        try:
            src = training_source(str(SETTINGS.path("base_train_model")))
            add(src, f"{Path(src).name} (Settings)", "Settings → base model for training")
        except FileNotFoundError:
            pass
        if out:
            out[0]["settings"] = True
    for c in containers():
        if c["engine"] == "qwen36":
            add(c["path"], f"{c['label']} (colibri container)", "colibri container")
        else:
            try:
                src = training_source(c["path"])
                add(src, f"{c['repo'] or Path(src).name} (Hugging Face folder of {c['name']})", "source of an OLMoE container")
            except FileNotFoundError:
                pass
    for h in cached():
        add(h["path"], f"{h['repo']} (Hugging Face cache)", "Hugging Face cache")
    return {"models": out, "serving": str(serve or ""), "default": out[0]["path"] if out and out[0]["settings"] else ""}
