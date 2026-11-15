"""Pack manifest (pack.json, format polymath-pack-v1) and the compatibility check.

A pack folder keeps the files the engine reads (overlay.json + overlay.safetensors, unchanged) and card.json, and gets
pack.json, which records what the pack was made for:

  engine      {name: patched-colibri, kind: qwen36|olmoe, pack_format: colibri-olmoe-overlay-v2}
  base_model  {repo, revision, label, arch_fingerprint, weights_hash, hash_basis, variant}
  slots       {mode, layers, experts_per_layer}
  trained     {packsmith, date, dataset, train_model}
  overlay_sha256

Base-model identity: a model folder carries polymath_model.json (written when Packsmith converts or first looks at it).
Its weights_hash is the SHA-256 over the sorted (file name, file SHA-256) of the source safetensors. In the Hugging Face
cache the file SHA-256 is the blob's name, so nothing is re-read; a plain folder is hashed once and the result kept.
A converted engine container records the hash of the source it was converted from, so the training folder (Hugging
Face) and the serving container (int4) of the same model have the same identity.

Checks when a pack is loaded (check()):
  engine name / kind differ   -> refuse  (the engine cannot read it)
  arch fingerprint differs    -> refuse  (different model shape; the engine refuses too)
  weights hash differs        -> warn    (same shape, other weights: e.g. Base vs hybrid; answers may be off)
  no pack.json (older pack)   -> unverified (loaded; ensure() writes pack.json from overlay.json + card.json)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from packsmith.system.home import expand, give_to_user

FORMAT = "polymath-pack-v1"
MODEL_FILE = "polymath_model.json"
LEGACY_MODEL_FILE = "packsmith_model.json"


# ---------------------------------------------------------------- base-model identity
def _rj(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _file_sha256(f: Path) -> str:
    """SHA-256 of one file. A Hugging Face cache snapshot links each file to blobs/<sha256>: use that name."""
    try:
        if f.is_symlink():
            target = Path(os.path.realpath(f))
            name = target.name
            if len(name) == 64 and all(c in "0123456789abcdef" for c in name):
                return name
    except OSError:
        pass
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_hash(folder: Path) -> tuple[str, str]:
    """(sha256:..., basis) over a Hugging Face model folder's safetensors; ("", "") when it has none."""
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        return "", ""
    h = hashlib.sha256()
    linked = all(f.is_symlink() for f in files)
    for f in files:
        h.update(f.name.encode() + b"\0" + _file_sha256(f).encode() + b"\n")
    return "sha256:" + h.hexdigest(), ("hf-cache-blobs" if linked else "file-contents")


def _hf_repo_rev(folder: Path) -> tuple[str, str]:
    """(repo, revision) for a snapshot folder .../models--Org--Name/snapshots/<rev>."""
    parts = folder.resolve().parts
    for i, p in enumerate(parts):
        if p.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            return p[len("models--"):].replace("--", "/"), parts[i + 2]
    return "", ""


def _arch_from_config(cfg: dict) -> str:
    """Same FNV-1a as the exporter's base_fingerprint (hidden, expert inter, layers, experts, top-k, vocab)."""
    try:
        inter = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size")
        vals = (cfg["hidden_size"], inter, cfg["num_hidden_layers"], cfg.get("num_experts") or cfg.get("n_routed_experts"),
                cfg["num_experts_per_tok"], cfg["vocab_size"])
    except (KeyError, TypeError):
        return ""
    if any(v is None for v in vals):
        return ""
    h = 1469598103934665603
    for v in vals:
        for i in range(8):
            h ^= (int(v) >> (8 * i)) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def model_identity(path: str | os.PathLike, write: bool = True) -> dict:
    """Identity of a base model folder (Hugging Face folder or engine container). Cached in polymath_model.json."""
    folder = expand(path)
    if not folder.is_dir():
        return {}
    cached = _rj(folder / MODEL_FILE)
    if cached.get("weights_hash"):
        return cached
    legacy = _rj(folder / LEGACY_MODEL_FILE)
    ident = {"repo": legacy.get("repo", ""), "revision": "", "label": legacy.get("label", ""),
             "engine_kind": legacy.get("engine") or ("qwen36" if (folder / "qwen36_meta.json").exists() else ""),
             "arch_fingerprint": "", "weights_hash": "", "hash_basis": "", "source": legacy.get("source", "")}
    cfg = _rj(folder / "config.json")
    if cfg:
        ident["arch_fingerprint"] = _arch_from_config(cfg)
        ident["engine_kind"] = ident["engine_kind"] or {"qwen3_moe": "qwen36", "olmoe": "olmoe"}.get(cfg.get("model_type", ""), "")
    repo, rev = _hf_repo_rev(folder)
    if repo:
        ident.update(repo=ident["repo"] or repo, revision=rev)
    wh, basis = weights_hash(folder) if any(folder.glob("*.safetensors")) and not (folder / "qwen36_meta.json").exists() else ("", "")
    if not wh and ident["source"]:                               # a container: the identity of what it was converted from
        src = model_identity(ident["source"], write=False)
        wh, basis = src.get("weights_hash", ""), "source:" + (src.get("hash_basis") or "")
        ident["repo"] = ident["repo"] or src.get("repo", "")
        ident["revision"] = ident["revision"] or src.get("revision", "")
        ident["arch_fingerprint"] = ident["arch_fingerprint"] or src.get("arch_fingerprint", "")
    ident.update(weights_hash=wh, hash_basis=basis)
    if write and wh:
        try:
            (folder / MODEL_FILE).write_text(json.dumps(ident, indent=1), encoding="utf-8")
            give_to_user(folder / MODEL_FILE)
        except OSError:
            pass
    return ident


# ---------------------------------------------------------------- pack.json
def _overlay_sha(pack: Path) -> str:
    st = pack / "overlay.safetensors"
    return _file_sha256(st) if st.exists() else ""


def build(pack: Path, engine: str, base_train_model: str = "", base_serve_model: str = "", extra: dict | None = None) -> dict:
    ov = _rj(pack / "overlay.json")
    card = _rj(pack / "card.json")
    base_path = base_train_model or card.get("base_model") or base_serve_model
    ident = model_identity(base_path) if base_path else {}
    if not ident.get("weights_hash") and base_serve_model:
        ident = model_identity(base_serve_model) or ident
    kn = ov.get("knowledge") or {}
    layers = sorted(int(k) for k in kn) if isinstance(kn, dict) else []
    from packsmith import __version__
    m = {
        "format": FORMAT,
        "name": pack.name,
        "engine": {"name": engine, "kind": ident.get("engine_kind") or card.get("engine") or "",
                   "pack_format": ov.get("format", "")},
        "base_model": {"repo": ident.get("repo", ""), "revision": ident.get("revision", ""), "label": ident.get("label", ""),
                       "arch_fingerprint": ov.get("base_fingerprint", ""), "weights_hash": ident.get("weights_hash", ""),
                       "hash_basis": ident.get("hash_basis", "")},
        "slots": {"mode": card.get("method") or ("Bslot" if kn else "A"), "layers": layers,
                  "experts_per_layer": card.get("k") or card.get("experts_per_layer")},
        "trained": {"packsmith": __version__, "date": card.get("created") or ov.get("created")
                    or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "dataset": card.get("dataset_name") or card.get("dataset") or "", "train_model": str(base_path or "")},
        "overlay_sha256": _overlay_sha(pack),
    }
    if extra:
        m.update(extra)
    return m


def write(pack: str | os.PathLike, engine: str, base_train_model: str = "", base_serve_model: str = "",
          extra: dict | None = None) -> dict:
    p = expand(pack)
    m = build(p, engine, base_train_model, base_serve_model, extra)
    (p / "pack.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    give_to_user(p / "pack.json")
    return m


def read(pack: str | os.PathLike) -> dict:
    return _rj(expand(pack) / "pack.json")


def ensure(pack: str | os.PathLike, engine: str = "patched-colibri", base_serve_model: str = "") -> dict:
    """pack.json for an older pack, from its overlay.json and card.json (marked 'backfilled')."""
    m = read(pack)
    if m:
        return m
    return write(pack, engine, base_serve_model=base_serve_model, extra={"backfilled": True})


# ---------------------------------------------------------------- compatibility
def check(pack: str | os.PathLike, engine: str, serve_model: str) -> dict:
    """{level: ok | warn | refuse | unverified, problems: [...], manifest} for loading `pack` on `engine` serving
    `serve_model`."""
    m = read(pack)
    if not m:
        return {"level": "unverified", "problems": ["no pack.json (made before pack manifests): compatibility not checked"],
                "manifest": {}}
    problems, level = [], "ok"
    me = m.get("engine") or {}
    from packsmith.engines import POLYMATH as polymath   # one pack format, two engines (PyTorch and C)
    if m.get("format") == "polymath-stack-v1":
        return {"level": "ok" if engine in polymath else "refuse", "manifest": m,
                "problems": [] if engine in polymath else [f"a Polymath stack, but {engine} is selected"]}
    if me.get("name") and me["name"] != engine and not (me["name"] in polymath and engine in polymath):
        problems.append(f"made for the engine {me['name']}, but {engine} is selected")
        level = "refuse"
    ident = model_identity(serve_model) if serve_model else {}
    kind = ident.get("engine_kind") or ("qwen36" if (expand(serve_model) / "qwen36_meta.json").exists() else "olmoe") if serve_model else ""
    if me.get("kind") and kind and me["kind"] != kind:
        problems.append(f"made for {me['kind']} models, the engine serves a {kind} model")
        level = "refuse"
    bm = m.get("base_model") or {}
    if bm.get("arch_fingerprint") and ident.get("arch_fingerprint") and bm["arch_fingerprint"] != ident["arch_fingerprint"]:
        problems.append("made for a model of a different shape (architecture fingerprint differs)")
        level = "refuse"
    if level != "refuse" and bm.get("weights_hash") and ident.get("weights_hash") and bm["weights_hash"] != ident["weights_hash"]:
        who = bm.get("label") or bm.get("repo") or "another model"
        problems.append(f"trained on {who} ({bm['weights_hash'][:19]}…), the engine serves "
                        f"{ident.get('label') or ident.get('repo') or 'another model'} ({ident['weights_hash'][:19]}…): "
                        "same shape, different weights — answers may be off")
        level = "warn"
    if level == "ok" and bm.get("weights_hash") and not ident.get("weights_hash"):
        problems.append("the served model's identity is unknown (no polymath_model.json and no source): weights not compared")
    return {"level": level, "problems": problems, "manifest": m}
