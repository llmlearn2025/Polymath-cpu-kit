"""Small command line used by the install scripts (and handy by hand):

  python -m packsmith.system.setup path                 the config file
  python -m packsmith.system.setup get KEY              one setting
  python -m packsmith.system.setup set KEY=VALUE ...    change settings (written to config.toml)
  python -m packsmith.system.setup init [--llm PRESET] [--llm-url URL] [--gpu CHOICE] [--colibri DIR]
                                                         first run: write config.toml (keeps an existing one)
  python -m packsmith.system.setup gpus                 the NVIDIA cards found
  python -m packsmith.system.setup doctor               what is set up and what is missing
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields


def _settings():
    from packsmith.system import config
    return config


def cmd_set(pairs: list[str]) -> None:
    config = _settings()
    fs = {f.name: f for f in fields(config.Settings)}
    for p in pairs:
        k, _, v = p.partition("=")
        k = k.strip()
        if k in config.LEGACY:
            k, conv = config.LEGACY[k]
            v = conv(v)
        if k not in fs:
            sys.exit(f"unknown setting {k!r} (known: {', '.join(fs)})")
        setattr(config.SETTINGS, k, config._coerce(fs[k], v))
        if k == "llm_preset" and not any(q.startswith("llm_url=") for q in pairs):
            url = config.PRESETS.get(config.SETTINGS.llm_preset, ("",))[0]
            if url:
                config.SETTINGS.llm_url = url
    config.save(config.SETTINGS)


def cmd_init(a) -> None:
    config = _settings()
    existed = config.config_file().exists()
    pairs = []
    if a.llm:
        pairs.append(f"llm_preset={a.llm}")
    if a.llm_url:
        pairs.append(f"llm_url={a.llm_url}")
    if a.gpu:
        pairs.append(f"gpu_engine={a.gpu}")
    if a.colibri:
        pairs.append(f"colibri_dir={a.colibri}")
    if a.host:
        pairs.append(f"host={a.host}")
    if existed and not a.force:
        if pairs:
            cmd_set(pairs)
        print(f"kept {config.config_file()}" + (f" (updated: {', '.join(p.split('=')[0] for p in pairs)})" if pairs else ""))
        return
    if pairs:
        cmd_set(pairs)
    else:
        config.save(config.SETTINGS)
    print(f"wrote {config.config_file()}")


def cmd_doctor() -> int:
    from packsmith.system import config, gpu
    from packsmith.llm import llm
    s = config.SETTINGS
    bad = 0
    print(f"config        {config.config_file()} {'(exists)' if config.config_file().exists() else '(not written yet: defaults)'}")
    print(f"python        {sys.executable} ({sys.version.split()[0]})")
    try:
        import torch
        print(f"torch         {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("torch         not installed (training off; install_packsmith.sh installs it)")
    cards = gpu.cards(fresh=True)
    print("GPUs          " + (", ".join(f"{c['index']}: {c['name']} ({c['memory_free_mb'] // 1024}/{c['memory_total_mb'] // 1024} GB free)"
                                       for c in cards) or "none found (nvidia-smi)"))
    print(f"engine GPU    {gpu.engine_device() or 'CPU'}   training GPU {gpu.train_device() or 'CPU'}   shared with LLM: {gpu.shared()}")
    from packsmith.engines.colibri.engine import builds
    have = builds(str(s.path("colibri_dir")))
    print(f"colibri       {s.path('colibri_dir')} "
          + (f"(builds: {', '.join(have)}; Engine build = {s.engine_build})" if have else "(NOT built: run install_patched_colibri.sh)"))
    bad += not have
    from packsmith.engines import polymath, polymath_c
    pc = polymath_c.builds(str(s.path("colibri_dir")))
    print(f"polymath-c    " + (f"in the colibri builds: {', '.join(pc)}" if pc else "not built (run.sh rebuilds colibri with it)"))
    pg = polymath_c.gpu_builds(str(s.path("colibri_dir")))
    print("polymath-c-gpu " + ("the GPU build has the CUDA chain port" if pg else
                               "not built (needs the CUDA toolkit; errors, if any: ~/.polymath/logs/polymath-c-gpu-build.txt)"))
    print(f"polymath-eng  " + ("installed (PyTorch reference, trainer of Polymath packs)" if polymath.installed() else
                               "not installed (needs PyTorch: install_packsmith.sh without --no-train)"))
    print(f"engine        Settings → Engine = {s.engine}")
    try:
        ms = llm.models()
        print(f"LLM           {llm.server_name()} at {s.llm_url}: {len(ms)} model(s)")
    except llm.LLMError as e:
        print(f"LLM           {s.llm_url}: not reachable ({e})")
        bad += 1
    for k in ("base_serve_model", "base_train_model"):
        v = getattr(s, k)
        print(f"{k:<17}{s.path(k) if v else '(not set: Settings → Engine model, or run.sh --get-model)'}")
    for k in ("data_dir", "packs_dir", "models_dir", "hf_cache_dir"):
        print(f"{k:<17}{s.path(k)}")
    return 1 if bad else 0


def main() -> None:
    from packsmith.system.home import drop_to_real_user
    drop_to_real_user()
    ap = argparse.ArgumentParser(prog="python -m packsmith.system.setup")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("path")
    g = sub.add_parser("get")
    g.add_argument("key")
    st = sub.add_parser("set")
    st.add_argument("pairs", nargs="+")
    i = sub.add_parser("init")
    i.add_argument("--llm")
    i.add_argument("--llm-url")
    i.add_argument("--gpu")
    i.add_argument("--colibri")
    i.add_argument("--host")
    i.add_argument("--force", action="store_true")
    sub.add_parser("gpus")
    sub.add_parser("doctor")
    sub.add_parser("reset-token", help="remove the API token (on this machine only: it edits the local config)")
    a = ap.parse_args()
    config = _settings()
    if a.cmd == "path":
        print(config.config_file())
    elif a.cmd == "get":
        print(getattr(config.SETTINGS, a.key))
    elif a.cmd == "set":
        cmd_set(a.pairs)
    elif a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "gpus":
        from packsmith.system import gpu
        print(json.dumps(gpu.cards(fresh=True), indent=1))
    elif a.cmd == "doctor":
        sys.exit(cmd_doctor())
    elif a.cmd == "reset-token":
        config.SETTINGS.api_token = ""
        config.save(config.SETTINGS)
        print(f"API token removed from {config.config_file()}: restart Packsmith, then set a new one in Settings → Server")


if __name__ == "__main__":
    main()
