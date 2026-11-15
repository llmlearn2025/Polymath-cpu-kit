"""The Storage tab: everything Polymath has put on this machine for the logged-in user, by category, and deleting what
you choose. A base model can sit in three places at once (the Hugging Face download, Polymath's models folder, the
colibri container), so the tab looks wherever Polymath writes or downloads, not only in its data folder.

Categories (the tab's drop-down):
    datasets     Packsmith datasets, Lab reasoning datasets
    packs        library packs (also -test_lab ones), packs in the old data/packs folder
    models       colibri containers and model folders in Polymath's models folder, Hugging Face downloads
                 (models--Org--Name in the HF cache), models and outputs in the runs folder
    training     training checkpoints, Train Lab runs with their epoch copies and checkpoints, regression outputs
    temporary    caches (merged packs, LLM-call caches, Lab cache), Test tab runs, logs, Hugging Face's download
                 chunk cache (xet), pip's download cache, leftovers in the temporary folder (control and status files
                 of engines no longer running, test folders)

Deleting is final (no bin). Safety, for every item:
  - only the kinds listed below, each in its own known folder ("root"); the item's path must match the kind's pattern
    relative to that root, resolve inside it, and no part of the path below the root may be a link;
  - links INSIDE an item (the HF cache's snapshots point to its blobs) are removed, never followed;
  - in use is refused: the base model set in Settings (serving or training), a pack loaded in the Workbench, a run or
    dataset whose job is running, the files of an engine still running;
  - more than 1 GB or more than 5 items needs confirm = "DELETE"; with the API token on, the route requires it.
Not listed, so never deleted here: documents, topics, playbooks, library records, settings, jobs, the engines.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

from packsmith.packs import store

GB = 1 << 30
CATEGORIES = [("datasets", "Datasets"), ("packs", "Packs"), ("models", "Models"),
              ("training", "Training checkpoints and runs"), ("temporary", "Caches and temporary files")]
_RID = r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}"
# kind: (category, root, pattern relative to the root, what it is, a note shown beside it)
KINDS = {
    "dataset": ("datasets", "data", r"datasets/[^/]+", "Dataset", ""),
    "lab_dataset": ("datasets", "data", r"lab/datasets/r" + _RID, "Lab reasoning dataset", ""),
    "pack": ("packs", "packs", r"[^/]+", "Pack", ""),
    "legacy_pack": ("packs", "data", r"packs/[^/]+", "Pack (old data/packs folder)", ""),
    "colibri_model": ("models", "models", r"colibri/[^/]+", "colibri model container", "made again by converting the downloaded model"),
    "model_folder": ("models", "models", r"(?!colibri$)[^/]+", "Model in Polymath's models folder", ""),
    "hf_model": ("models", "hf", r"models--[^/]+", "Hugging Face download", "downloaded again when needed"),
    "run_folder": ("models", "runs", r"[^/]+", "Runs folder entry (converted model or research output)", ""),
    "train_ckpt": ("training", "data", r"cache/ckpt-[^/]+", "Training checkpoint", "a finished or abandoned training's resume point"),
    "lab_run": ("training", "data", r"lab/runs/" + _RID, "Train Lab run", "its packs, epoch copies, checkpoints and scores"),
    "lab_epochs": ("training", "data", r"lab/runs/" + _RID + r"/r[0-9]{2}/pack/epochs", "Lab epoch copies", "the scores stay"),
    "lab_ckpt": ("training", "data", r"lab/runs/" + _RID + r"/r[0-9]{2}/ckpt", "Lab checkpoint", ""),
    "regression": ("training", "home", r"regression/[^/]+", "Regression test output", "packs, checkpoints and reports of scripts/regression"),
    "cache": ("temporary", "data", r"cache/(?!ckpt-)[^/]+", "Cached files (merged packs, stacks)", "made again when needed"),
    "unit_cache": ("temporary", "data", r"topics/[^/]+/units", "LLM-call cache of a topic", "the LLM calls are made (and paid) again"),
    "lab_cache": ("temporary", "data", r"lab/cache", "Lab LLM-call cache", "the LLM calls are made again"),
    "test_run": ("temporary", "data", r"tests/runs/[^/]+", "Test tab run", ""),
    "logs": ("temporary", "home", r"logs", "Polymath logs", ""),
    "hf_dataset": ("temporary", "hf", r"datasets--[^/]+", "Hugging Face dataset download", "downloaded again when needed"),
    "hf_xet": ("temporary", "cache", r"huggingface/xet", "Hugging Face download chunk cache", "shared with other programs; only speeds up downloads"),
    "pip_cache": ("temporary", "cache", r"pip", "pip download cache", "shared with other programs; only speeds up installs"),
    "tmp_status": ("temporary", "tmp", r"colibri-packs/[0-9]+\.(json|ctl)", "Status file of an engine no longer running", ""),
    "tmp_ctl": ("temporary", "tmp", r"(coli_packs_ctl_|polymath_try_ctl_)[A-Za-z0-9_]+(\.new)?", "Control file left by a stopped engine", ""),
    "tmp_test": ("temporary", "tmp", r"packsmith_[a-z_]+[A-Za-z0-9_]{6,}", "Test folder left by a test run", ""),
}


def roots() -> dict[str, Path]:
    """The folders Polymath writes to or downloads into, for this user (the real user under sudo)."""
    from packsmith.system.config import SETTINGS
    from packsmith.system.home import polymath_home, user_home
    from packsmith.system.models import hf_cache
    r = {"data": store.data_dir(), "packs": store.packs_dir(), "models": SETTINGS.path("models_dir"),
         "runs": SETTINGS.path("runs_dir"), "hf": hf_cache(), "home": polymath_home(),
         "cache": Path(os.environ.get("XDG_CACHE_HOME") or user_home() / ".cache"), "tmp": Path(tempfile.gettempdir())}
    return {k: Path(v) for k, v in r.items()}


def size(p: Path) -> int:
    if p.is_symlink():
        return 0
    if p.is_file():
        return p.stat().st_size
    n = 0
    for r, _, files in os.walk(p, followlinks=False):
        for x in files:
            f = os.path.join(r, x)
            try:
                if not os.path.islink(f):
                    n += os.path.getsize(f)
            except OSError:
                pass
    return n


def _home_short(p: Path) -> str:
    try:
        from packsmith.system.home import user_home
        return "~/" + str(p.resolve().relative_to(user_home().resolve()))
    except (ValueError, OSError):
        return str(p)


def safe(kind: str, rel: str, root: str | None = None) -> Path:
    """The item's path, or ValueError/KeyError. It matches its kind, resolves inside its root, and no part of it below
    the root is a link."""
    if kind not in KINDS:
        raise ValueError(f"{kind!r} cannot be deleted here")
    _, rk, pat, what, _ = KINDS[kind]
    if root not in (None, rk):
        raise ValueError(f"a {what.lower()} lives in the {rk} folder, not {root}")
    rel = str(rel or "").strip().strip("/")
    if not re.fullmatch(pat, rel) or ".." in rel.split("/") or "\\" in rel:
        raise ValueError(f"{rel!r} is not a {what.lower()}")
    base = roots()[rk].resolve()
    p = base
    for part in rel.split("/"):
        p = p / part
        if p.is_symlink():
            raise ValueError(f"{rel} goes through a link: refused")
    r = p.resolve()
    if r == base or base not in r.parents:
        raise ValueError(f"{rel} is not inside its folder: refused")
    if not r.exists():
        raise KeyError(f"{rel} does not exist")
    return r


# ------------------------------------------------------------------------------------------------ in use
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _in_use_paths() -> list[tuple[Path, str]]:
    from packsmith.system.config import SETTINGS
    out = []
    for key, why in (("base_serve_model", "the base model the engine serves (Settings)"),
                     ("base_train_model", "the base model for training (Settings)")):
        v = getattr(SETTINGS, key, "") or ""
        if v and v != "tiny":
            try:
                out.append((SETTINGS.path(key).resolve(), why))
            except Exception:                              # noqa: BLE001 - an unset or odd path
                pass
    return out


def _running() -> set[str]:
    from packsmith.system import jobs
    ids = set()
    for j in list(jobs.JOBS.values()):
        if j.status == "running":
            ids |= {str(v) for v in (j.inputs or {}).values() if isinstance(v, (str, int))}
    return ids


def in_use(kind: str, p: Path, running: set[str] | None = None, used: list | None = None) -> str:
    """Why this item cannot be deleted now ("" when it can)."""
    running = _running() if running is None else running
    used = _in_use_paths() if used is None else used
    rp = p.resolve()
    for u, why in used:
        if rp == u or rp in u.parents:
            return why
    if kind in ("pack", "legacy_pack"):
        from packsmith.workbench import workbench
        if workbench.STATE.get("engine") is not None and p.name in (workbench.STATE.get("active_names") or []):
            return "loaded in the Workbench"
    if kind in ("lab_run", "lab_epochs", "lab_ckpt", "test_run", "dataset", "lab_dataset"):
        for part in p.parts[-4:]:
            if part in running:
                return "its job is running"
    if kind == "tmp_status":
        pid = int(p.stem)
        if _pid_alive(pid):
            return "its engine is running"
    if kind in ("tmp_ctl", "tmp_test") and time.time() - p.stat().st_mtime < 3600:
        return "less than an hour old: a running engine or test may still use it"
    return ""


# ------------------------------------------------------------------------------------------------ listing
def _children(base: Path, glob: str = "*"):
    try:
        return sorted(base.glob(glob)) if base.is_dir() else []
    except OSError:
        return []


def _candidates(R: dict[str, Path]):
    """(kind, path) for everything that exists, kind by kind."""
    d = R["data"]
    yield from (("dataset", p) for p in _children(d / "datasets") if p.is_dir())
    yield from (("lab_dataset", p) for p in _children(d / "lab" / "datasets", "r*") if p.is_dir())
    for p in _children(R["packs"]):
        if p.is_dir() and store.is_pack(p):
            yield "pack", p
    if (d / "packs").resolve() != R["packs"].resolve():
        yield from (("legacy_pack", p) for p in _children(d / "packs") if p.is_dir() and store.is_pack(p))
    yield from (("colibri_model", p) for p in _children(R["models"] / "colibri") if p.is_dir())
    yield from (("model_folder", p) for p in _children(R["models"]) if p.is_dir() and p.name != "colibri")
    yield from (("hf_model", p) for p in _children(R["hf"], "models--*") if p.is_dir())
    yield from (("hf_dataset", p) for p in _children(R["hf"], "datasets--*") if p.is_dir())
    if R["runs"].resolve() not in (R["models"].resolve(), R["data"].resolve()):
        yield from (("run_folder", p) for p in _children(R["runs"]) if p.is_dir())
    yield from (("train_ckpt", p) for p in _children(d / "cache", "ckpt-*") if p.is_dir())
    for rd in _children(d / "lab" / "runs"):
        if rd.is_dir() and re.fullmatch(_RID, rd.name):
            yield "lab_run", rd
            for r in _children(rd, "r[0-9][0-9]"):
                if (r / "pack" / "epochs").is_dir():
                    yield "lab_epochs", r / "pack" / "epochs"
                if (r / "ckpt").is_dir():
                    yield "lab_ckpt", r / "ckpt"
    yield from (("regression", p) for p in _children(R["home"] / "regression") if p.is_dir())
    yield from (("cache", p) for p in _children(d / "cache") if not p.name.startswith("ckpt-"))
    yield from (("unit_cache", p) for p in _children(d / "topics", "*/units") if p.is_dir())
    if (d / "lab" / "cache").is_dir():
        yield "lab_cache", d / "lab" / "cache"
    yield from (("test_run", p) for p in _children(d / "tests" / "runs") if p.is_dir())
    if (R["home"] / "logs").is_dir():
        yield "logs", R["home"] / "logs"
    for k, rel in (("hf_xet", "huggingface/xet"), ("pip_cache", "pip")):
        if (R["cache"] / rel).is_dir():
            yield k, R["cache"] / rel
    yield from (("tmp_status", p) for p in _children(R["tmp"] / "colibri-packs") if re.fullmatch(r"[0-9]+\.(json|ctl)", p.name))
    for p in _children(R["tmp"]):
        for k in ("tmp_ctl", "tmp_test"):
            if re.fullmatch(KINDS[k][2], p.name):
                yield k, p


def _label(kind: str, p: Path) -> str:
    if kind in ("dataset", "lab_dataset"):
        m = store.rj(p / "manifest.json", {}) or {}
        return m.get("name") or p.name
    if kind in ("hf_model", "hf_dataset"):
        return p.name.split("--", 1)[1].replace("--", "/")
    if kind == "lab_run":
        m = store.rj(p / "run.json", {}) or {}
        return f"{m.get('name', p.name)} · {m.get('dataset_name', '')} · {m.get('model', '')}"
    if kind == "lab_epochs":                    # lab/runs/<run>/rNN/pack/epochs
        return f"run {p.parents[2].name} · recipe {int(p.parents[1].name[1:]) + 1}"
    if kind == "lab_ckpt":                      # lab/runs/<run>/rNN/ckpt
        return f"run {p.parents[1].name} · recipe {int(p.parent.name[1:]) + 1}"
    if kind == "unit_cache":
        return p.parent.name
    return p.name


def _converted_from(R: dict[str, Path]) -> dict[str, str]:
    """HF repo → the colibri container made from it (so the download can be seen as redundant)."""
    out = {}
    for c in _children(R["models"] / "colibri"):
        mk = store.rj(c / "packsmith_model.json", {}) or {}
        if mk.get("repo"):
            out[mk["repo"].lower()] = c.name
    return out


def overview() -> dict:
    R = roots()
    running, used = _running(), _in_use_paths()
    conv = _converted_from(R)
    items = []
    for kind, p in _candidates(R):
        cat, rk, pat, what, note = KINDS[kind]
        try:
            rel = p.resolve().relative_to(R[rk].resolve()).as_posix()
        except (ValueError, OSError):
            continue
        if not re.fullmatch(pat, rel) or p.is_symlink():
            continue
        it = {"kind": kind, "root": rk, "path": rel, "category": cat, "what": what, "label": _label(kind, p),
              "where": _home_short(p), "bytes": size(p), "note": note, "in_use": in_use(kind, p, running, used)}
        try:
            it["modified"] = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
        except OSError:
            pass
        if kind == "hf_model" and it["label"].lower() in conv:
            it["note"] = f"converted for colibri as {conv[it['label'].lower()]}: this download is only needed to convert again or to train on it"
        if kind in ("dataset",):
            m = store.rj(p / "manifest.json", {}) or {}
            if m.get("packs"):
                it["note"] = "packs trained from it: " + ", ".join(m["packs"])
        if kind in ("pack", "legacy_pack") and p.name.endswith("-test_lab"):
            it["what"] = "Pack (Train Lab test pack)"
        items.append(it)
    items.sort(key=lambda x: -x["bytes"])
    by_cat = {c: {"label": lab, "bytes": sum(i["bytes"] for i in items if i["category"] == c),
                  "count": sum(1 for i in items if i["category"] == c)} for c, lab in CATEGORIES}
    return {"categories": [{"key": c, **v} for c, v in by_cat.items()], "total": sum(i["bytes"] for i in items),
            "roots": {k: _home_short(v) for k, v in R.items()}, "items": items,
            "note": "Deleting is final. Items in use cannot be deleted. Documents, topics, playbooks, settings, jobs and the engines are not listed."}


def delete(entries: list[dict], confirm: str = "") -> dict:
    """[{kind, path, root?}] → {deleted, freed}. Every item is checked (kind, folder, links, in use) before any goes."""
    if not entries:
        raise ValueError("nothing chosen")
    running, used = _running(), _in_use_paths()
    targets = []
    for e in entries:
        k, rel = str(e.get("kind") or ""), str(e.get("path") or "")
        p = safe(k, rel, e.get("root"))
        why = in_use(k, p, running, used)
        if why:
            raise ValueError(f"{rel}: in use ({why})")
        targets.append((k, rel, p))
    total = sum(size(p) for _, _, p in targets)
    if (total > GB or len(targets) > 5) and confirm != "DELETE":
        raise ValueError(f"{len(targets)} item(s), {total / GB:.2f} GB: type DELETE to confirm (more than 1 GB or more than 5 items)")
    from packsmith.datasets import datasets as ds
    from packsmith.lab import lab, reasoning
    done = []
    for k, rel, p in targets:
        if k == "lab_run":
            lab.delete_run(p.name)
        elif k == "lab_dataset":
            reasoning.delete(p.name)
        elif k == "dataset":
            ds.delete(p.name)
        elif p.is_dir():
            shutil.rmtree(p)                    # links inside are removed, never followed
        else:
            p.unlink()
        if k in ("pack", "legacy_pack"):
            store.unregister_pack(p.name)
        done.append(rel)
    return {"deleted": done, "freed": total}
