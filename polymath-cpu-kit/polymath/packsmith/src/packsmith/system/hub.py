"""Hugging Face models for the engine: search, check, download (MoE models only).

  search(q)      the Hub's models matching q (no config fetched: fast)
  check(repo)    reads the repo's config.json: is it a mixture-of-experts model, which engine converts it, its size,
                 and whether it is already downloaded (the user's Hugging Face cache first, then Polymath's models folder)
  download(repo) a job: refuses a dense model; skips what is already on disk (resumes unfinished files); afterwards the
                 UI asks what the model is for and converts it (models.convert) for the engine.

Only the standard library is used for search/check; the download uses huggingface_hub (installed with Packsmith).
A token for gated models is read from HF_TOKEN (or huggingface-cli login).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from packsmith.system import models

HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# model_type -> the patched-colibri engine that serves it (and converts it)
ENGINE_FOR = {"qwen3_moe": "qwen36", "olmoe": "olmoe"}
MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


class HubError(RuntimeError):
    pass


def _get(url: str, timeout: float = 20) -> bytes:
    headers = {"User-Agent": "packsmith"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise HubError(f"{url}: access denied ({e.code}) — a gated model: accept its licence on huggingface.co and set "
                           "HF_TOKEN") from e
        if e.code == 404:
            raise HubError(f"{url}: not found") from e
        raise HubError(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, OSError) as e:
        raise HubError(f"Hugging Face not reachable ({HUB}): {e}") from e


def search(q: str, limit: int = 25) -> list[dict]:
    qs = urllib.parse.urlencode({"search": q, "limit": int(limit), "sort": "downloads", "direction": -1})
    rows = json.loads(_get(f"{HUB}/api/models?{qs}"))
    out = []
    for m in rows:
        tags = m.get("tags") or []
        out.append({"repo": m.get("id") or m.get("modelId"), "downloads": m.get("downloads"), "likes": m.get("likes"),
                    "pipeline": m.get("pipeline_tag", ""), "moe_hint": any(t.lower() in ("moe", "mixture-of-experts") for t in tags)})
    return out


def moe_info(cfg: dict) -> dict:
    """{moe, experts, active, model_type, engine} from a config.json (looks into text_config for multimodal models)."""
    for c in (cfg, cfg.get("text_config") or {}, cfg.get("language_config") or {}):
        n = next((c.get(k) for k in MOE_KEYS if isinstance(c.get(k), int)), None)
        if n and n > 1:
            mt = c.get("model_type") or cfg.get("model_type") or ""
            return {"moe": True, "experts": n, "active": c.get("num_experts_per_tok") or c.get("top_k") or c.get("moe_k"),
                    "model_type": mt, "engine": ENGINE_FOR.get(mt, "")}
    return {"moe": False, "experts": 0, "active": None, "model_type": cfg.get("model_type", ""), "engine": ""}


def local(repo: str) -> dict | None:
    """Where the repo is already on disk (HF cache first, then Polymath's models), and whether it is complete."""
    for c in models.cached():
        if c["repo"].lower() == repo.lower():
            from pathlib import Path
            p = Path(c["path"])
            idx = p / "model.safetensors.index.json"
            missing = []
            if idx.exists():
                try:
                    need = set(json.loads(idx.read_text())["weight_map"].values())
                    missing = sorted(n for n in need if not (p / n).exists())
                except (OSError, ValueError, KeyError):
                    pass
            elif not any(p.glob("*.safetensors")):
                missing = ["*.safetensors"]
            return {"path": str(p), "root": c.get("root", ""), "complete": not missing, "missing": missing[:10]}
    return None


def check(repo: str) -> dict:
    repo = repo.strip().strip("/")
    cfg = json.loads(_get(f"{HUB}/{repo}/resolve/main/config.json"))
    info = moe_info(cfg)
    size = None
    try:
        meta = json.loads(_get(f"{HUB}/api/models/{repo}?blobs=true"))
        size = sum((s.get("size") or 0) for s in meta.get("siblings") or []
                   if str(s.get("rfilename", "")).endswith((".safetensors", ".json", ".txt", ".jinja", ".model")))
        info["revision"] = meta.get("sha", "")
        info["gated"] = bool(meta.get("gated"))
    except HubError:
        pass
    info.update(repo=repo, size_gb=round(size / 1e9, 1) if size else None, local=local(repo),
                convertible=bool(info["engine"]),
                note=("" if not info["moe"] else
                      "" if info["engine"] else f"MoE, but patched colibri has no converter for {info['model_type']} yet "
                      "(Qwen3-MoE and OLMoE are supported): it can be downloaded, not served"))
    return info


def download(repo: str, where: str = "hf_cache", progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    """Download a MoE model. where: hf_cache (the user's Hugging Face cache, default) | models (Polymath's models
    folder). Nothing is fetched when it is already complete in either place."""
    info = check(repo)
    if not info["moe"]:
        raise HubError(f"{repo} is not a mixture-of-experts model (no experts in its config.json): Packsmith trains packs "
                       "for MoE models only")
    have = info.get("local")
    if have and have["complete"]:
        progress(1.0, f"already downloaded: {have['path']}")
        return {**info, "path": have["path"], "downloaded": False, "ask_convert": info["convertible"]}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise HubError("huggingface_hub is not installed in this venv: re-run install_packsmith.sh") from e
    cache = models.hf_cache() if where != "models" else models.models_dir()
    from packsmith.system.home import ensure_dir, give_to_user
    ensure_dir(cache)
    progress(0.02, f"downloading {repo} ({info.get('size_gb') or '?'} GB) into {cache} — unfinished files resume")
    if cancelled():
        raise RuntimeError("cancelled")
    path = snapshot_download(repo, cache_dir=str(cache), max_workers=8,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "merges.txt", "vocab.json"])
    give_to_user(cache / ("models--" + repo.replace("/", "--")))
    progress(0.98, "checking the download")
    after = local(repo)
    if not after or not after["complete"]:
        raise HubError(f"the download is incomplete ({(after or {}).get('missing')}): run it again to resume")
    from packsmith.packs import manifest
    manifest.model_identity(path)
    progress(1.0, f"downloaded: {path}")
    return {**info, "path": str(path), "downloaded": True, "ask_convert": info["convertible"]}


def main() -> None:
    """python -m packsmith.system.hub search Q | check REPO | get REPO [--purpose both|inference|training-hf] [--where hf_cache|models]"""
    import argparse
    import sys
    from packsmith.system.home import drop_to_real_user
    drop_to_real_user()
    ap = argparse.ArgumentParser(prog="python -m packsmith.system.hub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search").add_argument("q")
    sub.add_parser("check").add_argument("repo")
    g = sub.add_parser("get")
    g.add_argument("repo")
    g.add_argument("--purpose", default="both", choices=["both", "inference", "training-hf", "none"])
    g.add_argument("--where", default="hf_cache", choices=["hf_cache", "models"])
    a = ap.parse_args()
    log = lambda f, m="": print(f"  [{int(f * 100):3d}%] {m}", flush=True)   # noqa: E731
    try:
        if a.cmd == "search":
            for r in search(a.q):
                print(f"{r['repo']:<55} {r['downloads'] or 0:>10} downloads {'MoE' if r['moe_hint'] else ''}")
        elif a.cmd == "check":
            print(json.dumps(check(a.repo), indent=1))
        else:
            r = download(a.repo, a.where, progress=log)
            print(f"model folder: {r['path']}")
            if a.purpose == "none":
                return
            if a.purpose == "training-hf":
                print(models.use_for_training_only(r["path"]))
                return
            if not r["convertible"]:
                sys.exit(r["note"])
            out = models.convert(a.repo, use_it=True, purpose=a.purpose, progress=log)
            print(f"ready: {out['path']} (used for {a.purpose})")
    except HubError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
