"""Lab packs in the library: marked (test_lab), promoted to a normal pack, or deleted (final).

    add      copies a Lab recipe's pack into the packs folder as "<name>-test_lab", its card marked test_lab. It can then
             be ticked by hand in the Workbench and the Test tab. The assistant, agents and integrations never see it
             (userpack/agent.packs leaves it out, and load refuses it).
    promote  renames it to a normal pack (the mark and the card flag removed; the card keeps where it came from).
    delete   removes the pack folder. Final: there is no bin.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from packsmith.packs import store

MARK = "-test_lab"


def _name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (s or "").strip()).strip("-.")[:60]
    if not s:
        raise ValueError("a pack name is needed")
    return s


def _folder(name: str) -> Path:
    """A pack folder directly inside the packs folder: no other path is accepted."""
    root = store.packs_dir().resolve()
    p = (root / name)
    if p.is_symlink() or p.resolve().parent != root or name in ("", ".", ".."):
        raise ValueError(f"{name!r} is not a pack in the packs folder")
    return p


def _rename_inside(p: Path, name: str) -> None:
    """The pack's own name in its manifest (B-slot overlay.json shard, Polymath pack.json name) follows the folder,
    so two Lab packs loaded together are not both called 'pack'. The weights and their checksums are unchanged."""
    for f, key in (("overlay.json", "shard"), ("pack.json", "name")):
        q = p / f
        if q.exists():
            m = json.loads(q.read_text(encoding="utf-8"))
            if key in m:
                m[key] = name
                q.write_text(json.dumps(m, indent=1), encoding="utf-8")


def is_test(card: dict | None) -> bool:
    return bool((card or {}).get("test_lab"))


def add(rid: str, i: int, name: str = "") -> dict:
    from packsmith.lab import lab
    run = lab._rj(lab._run_dir(rid) / "run.json")
    if run is None:
        raise KeyError(f"no run {rid}")
    if not 0 <= int(i) < len(run["recipes"]):
        raise KeyError(f"no recipe {i} in run {rid}")
    src = lab._recipe_dir(rid, int(i)) / "pack"
    if not (src / "card.json").exists():
        raise ValueError("that recipe has not been trained yet")
    rec = run["recipes"][int(i)]
    n = _name(name or f"lab-{rec['name']}-{rid[-6:]}")
    n = n if n.endswith(MARK) else n + MARK
    dst = _folder(n)
    if dst.exists():
        raise ValueError(f"the packs folder already has {n}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("epochs", "ckpt"))
    _rename_inside(dst, n)
    card = json.loads((dst / "card.json").read_text(encoding="utf-8"))
    card.update(test_lab=True, lab={"run": rid, "run_name": run["name"], "recipe": rec["name"], "summary": rec["summary"],
                                    "dataset": run.get("dataset_name"), "base_model": run.get("model")})
    (dst / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    return {"name": n}


def list_test() -> list[dict]:
    return [{"name": p["name"], "lab": (p.get("card") or {}).get("lab") or {}} for p in store.list_packs()
            if is_test(p.get("card")) and Path(p["path"]).resolve().parent == store.packs_dir().resolve()]


def promote(name: str, new_name: str = "") -> dict:
    p = _folder(name)
    card = store.rj(p / "card.json", {}) or {}
    if not is_test(card):
        raise ValueError(f"{name} is not a Lab test pack")
    n = _name(new_name or (name[: -len(MARK)] if name.endswith(MARK) else name))
    if n.endswith(MARK):
        raise ValueError("a promoted pack does not carry the lab mark")
    dst = _folder(n)
    if dst.exists():
        raise ValueError(f"the packs folder already has {n}")
    p.rename(dst)
    _rename_inside(dst, n)
    card.pop("test_lab", None)
    card["promoted_from_lab"] = {**card.pop("lab", {}), "was": name}
    (dst / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    return {"name": n}


def delete(name: str) -> dict:
    p = _folder(name)
    if not is_test(store.rj(p / "card.json", {})):
        raise ValueError(f"{name} is not a Lab test pack (delete library packs from the Library)")
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
    shutil.rmtree(p)
    return {"freed": n}
