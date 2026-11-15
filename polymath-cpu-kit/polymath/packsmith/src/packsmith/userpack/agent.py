"""The small surface agents use (REST /api/agent/*, the MCP server, Hermes' slash commands): list packs, load and
unload them in the running engine, the engine's status. Packs are named, never given as paths. Agents may name
folders or files as user-pack sources only inside Settings → Server → Folders agents may read (agent_folders)."""
from __future__ import annotations

from packsmith.packs import store
from packsmith.system.config import SETTINGS


def packs() -> list[dict]:
    from packsmith.workbench import dials, workbench
    out = []
    active = set(workbench.STATE.get("active_names") or []) if workbench.STATE.get("engine") else set()
    for p in store.list_packs():
        card = p.get("card") or {}
        if card.get("test_lab"):                         # Train Lab test packs: chosen by hand only, never by an agent
            continue
        c = workbench.compat(p["name"], p["path"])
        out.append({"name": p["name"], "kind": card.get("kind") or "knowledge", "personal": bool(card.get("personal")),
                    "domain": card.get("domain") or card.get("topic_name") or "", "compat": c["level"],
                    "problems": c["problems"], "active": p["name"] in active, "dials": dials.get(p["name"])})
    return out


def status() -> dict:
    from packsmith.workbench import workbench
    st = workbench.status()
    return {"engine": SETTINGS.engine, "status": st["status"], "model": st.get("model"), "active": st.get("active_names"),
            "device": st.get("device"), "budget": SETTINGS.pack_budget, "dials": st.get("active_dials", {})}


def load(names: list[str], start: bool = True) -> dict:
    """Make exactly these packs active (several: stacked or merged by the engine's pack module). Each pack gets its
    Workbench dials (strength, shift): those are set in the page only, never by an agent."""
    from packsmith.workbench import workbench
    if not names:
        return unload()
    lab = {p["name"] for p in store.list_packs() if (p.get("card") or {}).get("test_lab")}
    if lab & set(names):
        raise ValueError("Train Lab test packs are loaded by hand in the Workbench only: " + ", ".join(sorted(lab & set(names))))
    e = workbench.STATE.get("engine")
    if (e is None or not e.alive()) and start:
        st = workbench.start()
        if st["status"] != "ready":
            raise RuntimeError(f"the engine did not start: {st.get('error')}")
        e = workbench.STATE["engine"]
    if e is None:
        raise RuntimeError("the engine is not running")
    paths = [workbench.pack_path(n) for n in names]
    if len(paths) == 1:
        target = paths[0]
    elif __import__("packsmith.engines", fromlist=["is_polymath"]).is_polymath(SETTINGS.engine):
        from packsmith.engines import polymath
        import hashlib
        key = hashlib.sha1("|".join(names).encode()).hexdigest()[:12]
        out = store.data_dir() / "cache" / f"agent-stack-{key}"
        polymath.merge(paths, out, source="agent: " + " + ".join(names))
        target = str(out)
        workbench.COMBOS[str(out.resolve())] = list(names)
    else:
        target = workbench.merged(names)
    r = e.swap(target)
    if not r["ok"]:
        raise RuntimeError("the engine refused: " + " | ".join(r.get("engine_log", [])[-3:]))
    return {"ok": True, "active": names, "dials": workbench.STATE.get("active_dials", {}), "seconds": r.get("seconds"),
            "warnings": [workbench.WARNINGS[n] for n in names
                                                                                   if n in workbench.WARNINGS]}


def unload() -> dict:
    from packsmith.workbench import workbench
    e = workbench.STATE.get("engine")
    if e is None or not e.alive():
        return {"ok": True, "active": [], "note": "the engine is not running"}
    r = e.swap(None)
    return {"ok": bool(r["ok"]), "active": []}
