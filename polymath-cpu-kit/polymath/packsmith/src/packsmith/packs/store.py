"""On-disk storage. Everything is a plain file under SETTINGS.data_dir:

  docs/<doc>/meta.json, regions.json, pages/0001.txt, img/0001.png
  topics/<topic>/topic.json, prompt.html, plan.html, skeleton.html, fragments/*.html,
                 playbook.html, playbook.<doc>.html, backtest.json, dataset/*.jsonl, notes.jsonl
  packs/<pack>/overlay.json, overlay.safetensors, card.json
  compositions/<id>.json
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from pathlib import Path

from packsmith.system.config import SETTINGS
from packsmith.system.home import expand

REGION_TYPES = ["chapter", "section", "other", "contents", "index", "table_of_cases", "table_of_statutes"]
ROLES = ["train", "context", "heldout", "exclude"]
# Region types that describe the book (contents, index, tables): read for the skeleton and by discovery, never mined
# for moves, facts or examples, whatever their role.
CONTEXT_TYPES = ("contents", "index", "table_of_cases", "table_of_statutes")
# What people see: the stored values stay as they are (older data and topics keep working).
TYPE_LABELS = {"chapter": "chapter", "section": "section", "other": "other", "contents": "table of contents",
               "index": "book index", "table_of_cases": "table of cases", "table_of_statutes": "table of statutes"}


def type_label(t: str) -> str:
    return TYPE_LABELS.get(t, t)


def data_dir() -> Path:
    d = SETTINGS.path("data_dir")
    for sub in ("docs", "topics", "packs", "compositions", "library"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def slug(s: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()[:n] or "item"
    return s


def new_id(name: str) -> str:
    return f"{slug(name, 32)}-{secrets.token_hex(3)}"


def rj(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def wj(p: Path, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def rt(p: Path, default: str = "") -> str:
    try:
        return Path(p).read_text(encoding="utf-8")
    except FileNotFoundError:
        return default


def wt(p: Path, s: str) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


# ---------------------------------------------------------------- documents
def doc_dir(doc: str) -> Path:
    return data_dir() / "docs" / doc


def list_docs() -> list[dict]:
    out = []
    for d in sorted((data_dir() / "docs").iterdir()):
        m = rj(d / "meta.json")
        if m:
            m["regions"] = rj(d / "regions.json", [])
            out.append(m)
    return out


def doc_meta(doc: str) -> dict:
    m = rj(doc_dir(doc) / "meta.json")
    if not m:
        raise KeyError(f"no document {doc}")
    return m


def save_meta(doc: str, m: dict) -> None:
    wj(doc_dir(doc) / "meta.json", m)


def page_text(doc: str, page: int) -> str:
    return rt(doc_dir(doc) / "pages" / f"{page:04d}.txt")


def set_page_text(doc: str, page: int, text: str) -> None:
    wt(doc_dir(doc) / "pages" / f"{page:04d}.txt", text)


def page_image(doc: str, page: int) -> Path | None:
    p = doc_dir(doc) / "img" / f"{page:04d}.png"
    return p if p.exists() else None


def regions(doc: str) -> list[dict]:
    return rj(doc_dir(doc) / "regions.json", [])


def save_region(doc: str, r: dict) -> dict:
    rs = regions(doc)
    n = doc_meta(doc)["pages"]
    r = {k: r[k] for k in ("id", "type", "role", "name", "start", "end") if k in r}   # nothing else is stored
    r["start"], r["end"] = int(r["start"]), int(r["end"])
    if r["start"] > r["end"]:
        r["start"], r["end"] = r["end"], r["start"]
    if not (1 <= r["start"] <= n and 1 <= r["end"] <= n):
        raise ValueError(f"pages must be within 1..{n}")
    if r.get("type") not in REGION_TYPES:
        raise ValueError(f"type must be one of {REGION_TYPES}")
    r["role"] = r.get("role") or ("context" if r["type"] in CONTEXT_TYPES else "train")
    if r["role"] not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if r["type"] in CONTEXT_TYPES and r["role"] in ("train", "heldout"):
        # a contents/index page is never mined: "train" on it only hid the mistake (the tag said train, the miners
        # skipped it). It is context.
        r["role"] = "context"
    r["name"] = r.get("name") or f"{type_label(r['type'])} {r['start']}-{r['end']}"
    if not r.get("id"):
        r["id"] = "r" + secrets.token_hex(3)
        rs.append(r)
    else:
        rs = [r if x["id"] == r["id"] else x for x in rs]
    wj(doc_dir(doc) / "regions.json", sorted(rs, key=lambda x: (x["start"], x["end"])))
    return r


def delete_region(doc: str, rid: str) -> None:
    wj(doc_dir(doc) / "regions.json", [x for x in regions(doc) if x["id"] != rid])


def region(doc: str, rid: str) -> dict:
    for r in regions(doc):
        if r["id"] == rid:
            return r
    raise KeyError(f"no region {rid} in {doc}")


def region_text(doc: str, r: dict) -> list[tuple[int, str]]:
    return [(p, page_text(doc, p)) for p in range(r["start"], r["end"] + 1)]


def delete_doc(doc: str) -> None:
    shutil.rmtree(doc_dir(doc), ignore_errors=True)


# ---------------------------------------------------------------- topics
def topic_dir(t: str) -> Path:
    return data_dir() / "topics" / t


def list_topics() -> list[dict]:
    out = []
    for d in sorted((data_dir() / "topics").iterdir()):
        m = rj(d / "topic.json")
        if m:
            m["status"] = topic_status(m["id"])
            out.append(m)
    return out


def topic(t: str) -> dict:
    m = rj(topic_dir(t) / "topic.json")
    if not m:
        raise KeyError(f"no topic {t}")
    return m


def save_topic(m: dict) -> dict:
    if not m.get("id"):
        m["id"] = new_id(m.get("name", "topic"))
        m["created"] = time.strftime("%Y-%m-%d %H:%M")
    m.setdefault("regions", [])
    m.setdefault("strategy", "unified")
    wj(topic_dir(m["id"]) / "topic.json", m)
    return m


def topic_status(t: str) -> dict:
    d = topic_dir(t)
    return {k: (d / f).exists() for k, f in (("prompt", "prompt.html"), ("plan", "plan.html"), ("skeleton", "skeleton.html"),
                                              ("playbook", "playbook.html"), ("backtest", "backtest.json"),
                                              ("dataset", "dataset/train.jsonl"))}


def topic_regions(m: dict, roles: tuple[str, ...] | None = None) -> list[tuple[str, dict]]:
    """The topic's regions: the ones ticked in it, plus every region of its whole documents (whole_docs: a topic
    that follows its books, so regions tagged or tagged again later belong to it)."""
    out, seen = [], set()
    refs = list(m.get("regions", []))
    for doc in m.get("whole_docs", []) or []:
        refs += [{"doc": doc, "region": r["id"]} for r in regions(doc)]
    for ref in refs:
        key = (ref.get("doc"), ref.get("region"))
        if key in seen:
            continue
        seen.add(key)
        try:
            r = region(ref["doc"], ref["region"])
        except (KeyError, OSError):
            continue
        if roles is None or r["role"] in roles:
            out.append((ref["doc"], r))
    return out


def mined_regions(m: dict, roles: tuple[str, ...] = ("train",)) -> list[tuple[str, dict]]:
    """The topic's regions that the miners read (map, facts, examples): these roles, and not a contents/index type."""
    return [(d, r) for d, r in topic_regions(m, roles) if r["type"] not in CONTEXT_TYPES]


def regions_signature(m: dict, roles: tuple[str, ...] = ("train",)) -> str:
    """Changes whenever the mined regions of these roles change (added, removed, re-typed, moved): cached mining
    results keyed on it start afresh instead of mixing in regions that are no longer part of the topic."""
    import hashlib
    sig = sorted((d, r["id"], r["start"], r["end"], r["type"], r["role"]) for d, r in mined_regions(m, roles))
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:12]


def why_no_regions(m: dict, roles: tuple[str, ...] = ("train",)) -> str:
    """Why the topic has no regions of these roles to mine, in words a person can act on."""
    want = "/".join(roles)
    refs = m.get("regions", [])
    if not refs and not m.get("whole_docs"):
        return f"the topic has no regions: in Topics, tick the {want} regions of its documents and save"
    found, missing = [], 0
    for ref in refs:
        try:
            found.append((ref["doc"], region(ref["doc"], ref["region"])))
        except (KeyError, OSError):
            missing += 1
    parts = []
    if missing:
        parts.append(f"{missing} of its {len(refs)} region(s) no longer exist (deleted or tagged again in the Library, "
                     "which gives a region a new id): tick them again in Topics")
    typed = [(d, r) for d, r in found if r["role"] in roles and r["type"] in CONTEXT_TYPES]
    if typed:
        names = ", ".join(f"'{r['name']}' (p.{r['start']}–{r['end']}, type {type_label(r['type'])})" for _, r in typed[:4])
        parts.append(f"{len(typed)} region(s) have the role {want} but a table-of-contents / book-index type, and those are never "
                     f"mined: {names}{' …' if len(typed) > 4 else ''}. In the Library, change their type to chapter "
                     "(or section / other)")
    other = sorted({r["role"] for _, r in found if r["role"] not in roles and r["type"] not in CONTEXT_TYPES})
    if other and not typed:
        parts.append(f"its chapter/section regions have the role(s) {', '.join(other)}, not {want}: change the role in "
                     "the Library")
    ticked = {(ref["doc"], ref["region"]) for ref in refs}
    for doc in sorted({ref["doc"] for ref in refs}):
        try:
            loose = [r for r in regions(doc) if (doc, r["id"]) not in ticked and r["role"] in roles
                     and r["type"] not in CONTEXT_TYPES]
            name = doc_meta(doc)["name"]
        except (KeyError, OSError):
            continue
        if loose:
            parts.append(f"'{name}' has {len(loose)} {want} region(s) that are not ticked in this topic "
                         f"({', '.join(r['name'] for r in loose[:4])}{' …' if len(loose) > 4 else ''}): tick them in "
                         "Topics and save, or tick 'every region of these documents'")
    if not parts:
        parts.append(f"tag {want} regions (type chapter, section or other) in the Library and tick them in Topics")
    return f"the topic has no {want} regions to mine: " + "; ".join(parts)


def delete_topic(t: str) -> None:
    shutil.rmtree(topic_dir(t), ignore_errors=True)


# ---------------------------------------------------------------- packs, compositions
def packs_dir() -> Path:
    """Settings → packs_dir (shared across installs/containers; POLYMATH_PACKS)."""
    from packsmith.system.home import ensure_dir
    return ensure_dir(SETTINGS.packs_dir)


def _pack_folders() -> list[Path]:
    """The packs folder, plus <data_dir>/packs where packs lived before the shared packs folder."""
    roots = [packs_dir()]
    legacy = data_dir() / "packs"
    if legacy.resolve() != roots[0].resolve():
        roots.append(legacy)
    seen, out = set(), []
    for r in roots:
        for d in sorted(r.iterdir()) if r.is_dir() else []:
            if d.name not in seen:
                seen.add(d.name)
                out.append(d)
    return out


def is_pack(d: Path) -> bool:
    """A patched-colibri pack (overlay.json) or a Polymath Engine pack / stack (pack.json, polymath-experts-v1)."""
    if (d / "overlay.json").exists():
        return True
    from packsmith.engines import polymath
    return polymath.is_pack(d)


def list_packs() -> list[dict]:
    out = []
    for d in _pack_folders():
        if is_pack(d):
            card = rj(d / "card.json", {})
            out.append({"name": d.name, "path": str(d), "card": card, "manifest": rj(d / "pack.json", {})})
    extra = data_dir() / "external_packs.json"           # packs registered from elsewhere (e.g. the runs folder)
    for name, path in (rj(extra, {}) or {}).items():
        p = expand(path)
        if is_pack(p):
            out.append({"name": name, "path": str(p), "card": rj(p / "card.json", {"domain": name, "external": True}),
                        "manifest": rj(p / "pack.json", {})})
    return out


def register_pack(name: str, path: str) -> dict:
    p = expand(path).resolve()
    if not is_pack(p):
        raise ValueError(f"{p} holds no pack (no overlay.json, and no Polymath Engine pack.json)")
    for q in list_packs():
        if Path(q["path"]).resolve() == p:
            return {"already": True, "name": q["name"]}
    extra = data_dir() / "external_packs.json"
    d = rj(extra, {}) or {}
    if name in d or (packs_dir() / name).exists():
        raise ValueError(f"the name {name!r} is taken")
    d[name] = str(p)
    wj(extra, d)
    return {"already": False, "name": name}


def unregister_pack(name: str) -> None:
    extra = data_dir() / "external_packs.json"
    d = rj(extra, {}) or {}
    d.pop(name, None)
    wj(extra, d)


def scan_packs(folder: str = "") -> list[dict]:
    """Packs found on disk (the runs folder, the packs folder, a folder the user names, two levels deep),
    with whether and under which name each is registered."""
    from packsmith.system.config import SETTINGS
    roots = [packs_dir(), SETTINGS.path("runs_dir")] + ([expand(folder)] if folder else [])
    known = {str(Path(q["path"]).resolve()): q["name"] for q in list_packs()}
    mine = {(q["card"] or {}).get("base_fingerprint") for q in list_packs() if (q["card"] or {}).get("base_fingerprint")}
    seen, out = set(), []
    for root in roots:
        if not root.exists():
            continue
        cands = [root] + [x for x in root.glob("*") if x.is_dir()] + [x for x in root.glob("*/*") if x.is_dir()]
        for d in cands:
            ov = d / "overlay.json"
            r = str(d.resolve())
            if r in seen or not ov.exists():
                continue
            seen.add(r)
            o = rj(ov, {}) or {}
            card = rj(d / "card.json", {}) or {}
            fp = o.get("base_fingerprint", "")
            out.append({"path": r, "name": d.name, "registered_as": known.get(r, ""), "format": o.get("format", ""),
                        "created": o.get("created") or card.get("created", ""), "fingerprint": fp,
                        "same_base": (fp in mine) if mine else None, "domain": card.get("domain") or card.get("topic_name") or "",
                        "texts": card.get("texts")})
    return sorted(out, key=lambda x: (x["registered_as"] != "", x["name"]))


def list_compositions() -> list[dict]:
    return [rj(p) for p in sorted((data_dir() / "compositions").glob("*.json"))]


def save_composition(c: dict) -> dict:
    if not c.get("id"):
        c["id"] = new_id(c.get("name", "curated"))
    wj(data_dir() / "compositions" / f"{c['id']}.json", c)
    return c
