"""Router tuning for a Polymath stack, from Packsmith: finds each pack's training data and runs the engine's tuner.

    python -m packsmith.training.tune_router --stack <stack folder or pack name> [--method calibrate|train|both]
                                             [--share 0.25] [--general-cap 0.02] [--steps 200] [--dry-run] [--reset]

  * the stack: a folder with a polymath-stack-v1 pack.json (Packsmith writes one when packs are merged on a Polymath
    engine), or the name of one under the packs folder; or --packs a,b to make the stack first (--out names it)
  * each pack's data: its card.json "dataset" (a Packsmith dataset id → that dataset's train.jsonl), or --data
  * the model: Settings → Paths → Base model for training (the model the packs were trained on)
  * the GPU: the Workbench engine is stopped first (one card), as for training

Writes stack/router.safetensors (+ "router" in stack/pack.json, stack/router_tune.json). Both Polymath engines
(polymath-engine and polymath-c) apply the tuned rows whenever the stack is loaded; the packs are not modified.
See engines/polymath-c/ROUTER_TUNING.md for when to use which method and how to read the report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from packsmith.system.config import SETTINGS


def _packs_dir() -> Path:
    from packsmith.packs import store
    return store.packs_dir()


def resolve_stack(s: str) -> Path:
    p = Path(s).expanduser()
    if not p.exists():
        p = _packs_dir() / s
    if not (p / "pack.json").exists():
        raise SystemExit(f"{s}: not a pack folder (no pack.json)")
    return p.resolve()


def pack_data(pack: Path) -> str:
    """The train.jsonl a Packsmith pack was trained on ('' when unknown)."""
    try:
        card = json.loads((pack / "card.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    for k in ("data_file", "train_file"):
        if card.get(k) and Path(card[k]).expanduser().is_file():
            return str(Path(card[k]).expanduser())
    did = card.get("dataset")
    if isinstance(did, str) and did:
        if Path(did).expanduser().is_file():
            return str(Path(did).expanduser())
        try:
            from packsmith.datasets import datasets
            f = datasets.train_path(did)
            if f.is_file():
                return str(f)
        except Exception:                        # noqa: BLE001
            pass
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", default="", help="stack folder or name under the packs folder")
    ap.add_argument("--packs", default="", help="make a stack of these packs first (names or folders, comma-separated)")
    ap.add_argument("--out", default="", help="with --packs: the stack's name (default: the packs' names joined)")
    ap.add_argument("--method", default="calibrate", choices=["calibrate", "train", "both"])
    ap.add_argument("--data", default="", help="override: one data file per pack, comma-separated, in stack order")
    ap.add_argument("--general", default="")
    ap.add_argument("--share", type=float, default=0.25)
    ap.add_argument("--general-cap", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=int(SETTINGS.pack_budget))
    ap.add_argument("--model", default="", help="default: Settings → Base model for training")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="start from the packs' own rows")
    ap.add_argument("--remove", action="store_true", help="remove the stack's tuned rows and stop")
    a = ap.parse_args(argv)
    if a.packs:
        from packsmith.engines import polymath
        ps = [resolve_stack(x) for x in a.packs.split(",") if x]
        out = _packs_dir() / (a.out or "+".join(p.name for p in ps))
        stack = polymath.merge(ps, out, source="tune_router")
        print(f"[tune] stack written: {stack}")
    elif a.stack:
        stack = resolve_stack(a.stack)
    else:
        raise SystemExit("--stack or --packs")
    m = json.loads((stack / "pack.json").read_text(encoding="utf-8"))
    if m.get("format") != "polymath-stack-v1":
        raise SystemExit(f"{stack} is a single pack; router tuning fits several packs together (use --packs a,b)")
    if a.remove:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "engines" / "polymath-engine"))
        from polymath_engine.tune import reset
        reset(stack)
        print(f"[tune] removed the tuned router rows of {stack}")
        return 0
    from packsmith.engines import polymath
    packs = polymath.expand_packs(stack)                  # the expanded list: the engine numbers the rows (and data) this way
    data = a.data.split(",") if a.data else [pack_data(p) for p in packs]
    if len(data) != len(packs):
        raise SystemExit(f"--data has {len(data)} file(s), the stack expands to {len(packs)} pack(s): "
                         + ", ".join(p.name for p in packs))
    for p, d in zip(packs, data):
        print(f"[tune] {p.name}: data {d or '(none found: its rows are left as they are)'}")
    model = a.model or str(SETTINGS.path("base_train_model"))
    try:                                          # the GPU: stop the Workbench engine (best effort)
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{SETTINGS.port}/api/workbench/stop", data=b"{}",
                                                      method="POST", headers={"Content-Type": "application/json"}), timeout=20)
    except Exception:                             # noqa: BLE001
        pass
    cmd = [sys.executable, "-m", "polymath_engine", "tune-router", "--model", model, "--device", a.device,
           "--stack", str(stack), "--method", a.method, "--data", ",".join(data), "--share", str(a.share),
           "--general-cap", str(a.general_cap), "--steps", str(a.steps), "--lr", str(a.lr), "--budget", str(a.budget)]
    if a.general:
        cmd += ["--general", a.general]
    if a.dry_run:
        cmd.append("--dry-run")
    if a.reset:
        cmd.append("--reset")
    print("[tune] " + " ".join(cmd[2:]), flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
