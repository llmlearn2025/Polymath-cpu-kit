"""FastAPI app: REST API + the single-page UI."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from packsmith.workbench import assistant, pipelines, workbench
from packsmith.evaluation import backtest
from packsmith.datasets import compose, curated, datasets, generate, prompts, quick
from packsmith.system import config, jobs
from packsmith.mining import discover, facts, factbook, ingest, mine, ocr, structure
from packsmith.packs import library, store
from packsmith.llm import llm
from packsmith.training import train

UI = Path(__file__).parent / "ui"


@asynccontextmanager
async def lifespan(_app):
    store.data_dir()
    _auto_register()
    try:
        datasets.adopt_legacy()                   # datasets made by earlier versions appear as datasets
    except Exception as e:                        # noqa: BLE001 - never block startup
        print(f"[packsmith] could not adopt earlier datasets: {e}")
    if os.environ.get("PACKSMITH_NO_ENGINE") != "1":
        threading.Thread(target=workbench.start, daemon=True).start()
    if exposed_without_token():
        print("[packsmith] WARNING: listening beyond this machine with no API token: anyone on the network can use "
              "Packsmith. Set one in Settings → Server → API token.", flush=True)
    yield
    workbench.stop()


app = FastAPI(title="Packsmith", lifespan=lifespan)

# ---------------------------------------------------------------- the optional API token
import hashlib as _hashlib  # noqa: E402
import hmac as _hmac  # noqa: E402
from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

SESSION_COOKIE = "packsmith_session"
OPEN = ("/api/auth", "/api/login", "/api/logout")
LOOPBACK = ("127.0.0.1", "localhost", "::1", "")


def _same(a: str, b: str) -> bool:
    return _hmac.compare_digest(a.encode(), b.encode())


def _session(token: str) -> str:
    """The browser's cookie: derived from the token (it changes when the token does), never the token itself."""
    return _hmac.new(token.encode(), b"packsmith-session-v1", _hashlib.sha256).hexdigest()


def _authed(request: Request, token: str) -> bool:
    h = request.headers.get("authorization", "")
    if h.lower().startswith("bearer ") and _same(h[7:].strip(), token):
        return True
    if _same(request.headers.get("x-packsmith-token", ""), token):
        return True
    return _same(request.cookies.get(SESSION_COOKIE, ""), _session(token))


def exposed_without_token() -> bool:
    """Listening beyond this machine with no token. In Docker the container listens on 0.0.0.0; compose tells it
    where the port is published (PACKSMITH_PUBLISHED, 127.0.0.1 by default)."""
    if config.SETTINGS.api_token:
        return False
    host = os.environ.get("PACKSMITH_HOST") or config.SETTINGS.host
    published = os.environ.get("PACKSMITH_PUBLISHED")
    return (published if published is not None else host) not in LOOPBACK


@app.middleware("http")
async def _auth(request: Request, call_next):
    tok = config.SETTINGS.api_token
    if tok and request.url.path.startswith("/api/") and request.url.path not in OPEN and not _authed(request, tok):
        return JSONResponse({"detail": "this Packsmith needs its API token", "token_required": True}, status_code=401)
    return await call_next(request)


@app.get("/api/auth")
def auth_status(request: Request):
    tok = config.SETTINGS.api_token
    return {"token_required": bool(tok), "authed": (not tok) or _authed(request, tok),
            "exposed_without_token": exposed_without_token()}


@app.post("/api/login")
def login(d: dict = Body(...)):
    tok = config.SETTINGS.api_token
    if not tok:
        return {"ok": True, "token_required": False}
    if not _same(str(d.get("token") or ""), tok):
        raise HTTPException(401, "wrong token")
    r = JSONResponse({"ok": True})
    r.set_cookie(SESSION_COOKIE, _session(tok), httponly=True, samesite="strict", max_age=30 * 86400, path="/")
    return r


@app.post("/api/logout")
def logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie(SESSION_COOKIE, path="/")
    return r


def _auto_register():
    """First run: register packs already trained in the runs folder that match the base model's family."""
    if (store.data_dir() / "external_packs.json").exists():
        return
    runs = config.SETTINGS.path("runs_dir")
    qwen = (config.SETTINGS.path("base_serve_model") / "qwen36_meta.json").exists()
    found = {}
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if (d / "overlay.json").exists() and (("qwen" in d.name) == qwen):
                found[d.name] = str(d)
    store.wj(store.data_dir() / "external_packs.json", found)
app.mount("/ui", StaticFiles(directory=UI), name="ui")


def _err(e: Exception):
    raise HTTPException(400, f"{type(e).__name__}: {e}")


def _par(d: dict):
    try:
        return max(1, min(32, int(d.get("parallel")))) if d.get("parallel") not in (None, "") else None
    except (TypeError, ValueError):
        return None


def job(kind, fn, label="", /, **kw):        # positional-only: job args may be called kind/fn/label
    return jobs.start(kind, fn, label, **kw).view()


@app.get("/", response_class=HTMLResponse)
def index():
    return (UI / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- settings, models, jobs
def _public() -> dict:
    """The settings without their secrets (the API token, the LLM API key): each is replaced by <name>_set."""
    d = asdict(config.SETTINGS)
    for k in config.SECRETS:
        d[k + "_set"] = bool(d.pop(k, ""))
    return d


@app.get("/api/settings")
def get_settings():
    return _public()


@app.post("/api/settings")
def set_settings(request: Request, d: dict = Body(...)):
    fs = {f.name: f for f in fields(config.Settings)}
    bad = []
    d = dict(d)
    for k in config.SECRETS:                               # an empty secret field means "unchanged" unless cleared
        if k in d and d[k] in ("", None) and not d.get("clear_" + k):
            d.pop(k)
    if "api_token" in d and d["api_token"] != config.SETTINGS.api_token and config.SETTINGS.api_token:
        if not _same(str(d.get("current_api_token") or ""), config.SETTINGS.api_token):
            raise HTTPException(403, "changing or clearing the API token needs the current token")
    for k in [k for k in d if k.startswith("clear_") or k == "current_api_token"]:
        d.pop(k)
    if "agent_folders" in d and d["agent_folders"] != config.SETTINGS.agent_folders and not _from_page(request):
        raise HTTPException(403, "the folders agents may read are changed in Packsmith's page (Settings → Server), "
                                 "not through the API")
    for k, v in d.items():
        if k in fs:
            try:
                setattr(config.SETTINGS, k, config._coerce(fs[k], v))
            except (TypeError, ValueError):
                bad.append(k)
    if bad:
        raise HTTPException(400, f"not a valid value for: {', '.join(bad)}")
    if "llm_preset" in d and "llm_url" not in d:          # a new provider without a URL: its usual address
        url = config.PRESETS.get(config.SETTINGS.llm_preset, ("",))[0]
        if url:
            config.SETTINGS.llm_url = url
    config.save(config.SETTINGS)
    return _public()


@app.get("/api/settings/meta")
def settings_meta():
    from packsmith import engines
    from packsmith.system import gpu
    help_ = {}
    for k, v in config.HELP.items():
        g, lab, h, kind, rs = v[:5]
        help_[k] = {"group": g, "label": lab, "help": h, "kind": kind, "restart": rs, "options": v[5] if len(v) > 5 else None}
    return {"values": _public(), "help": help_, "config_file": str(config.config_file()),
            "presets": {k: {"url": u, "label": lab} for k, (u, lab) in config.PRESETS.items()},
            "cards": gpu.cards(fresh=True), "engines": engines.available()}


@app.get("/api/gpus")
def gpus():
    from packsmith.system import gpu
    return {"cards": gpu.cards(fresh=True), "engine_device": gpu.engine_device(), "train_device": gpu.train_device(),
            "shared": gpu.shared()}


@app.get("/api/engines")
def engines_list():
    from packsmith import engines
    return engines.available()


@app.get("/api/llm/test")
def llm_test():
    """Settings → Test connection: the models the configured server lists."""
    try:
        ms = llm.models()
        return {"ok": True, "server": llm.server_name(), "url": llm.server_url(), "models": [m["id"] for m in ms],
                "can_manage_models": llm.can_manage_models()}
    except llm.LLMError as e:
        return {"ok": False, "server": llm.server_name(), "url": llm.server_url(), "error": str(e)}


@app.get("/api/packs/{name}/compat")
def pack_compat(name: str):
    from packsmith.workbench import workbench as wb
    for p in store.list_packs():
        if p["name"] == name:
            return wb.compat(name, p["path"])
    raise HTTPException(404, f"no pack {name}")


@app.post("/api/packs/{name}/manifest")
def pack_manifest(name: str):
    """Write pack.json for an older pack (from overlay.json + card.json)."""
    from packsmith.packs import manifest
    for p in store.list_packs():
        if p["name"] == name:
            return manifest.ensure(p["path"], config.SETTINGS.engine, config.SETTINGS.base_serve_model)
    raise HTTPException(404, f"no pack {name}")


@app.get("/api/llm/instances")
def llm_instances():
    return llm.instances()


@app.post("/api/llm/unload")
def llm_unload():
    return {"unloaded": llm.unload_all()}


@app.get("/api/llm/models")
def llm_models():
    try:
        return {"models": llm.models(), "default": config.SETTINGS.default_model}
    except llm.LLMError as e:
        return {"models": [], "error": str(e)}


@app.get("/api/jobs")
def list_jobs():
    return jobs.listing()


@app.get("/api/jobs/{jid}")
def get_job(jid: str, full: bool = False):
    if jid in jobs.JOBS:
        return jobs.JOBS[jid].view(0 if full else 60)
    r = jobs.get(jid)
    if not r:
        raise HTTPException(404, "no such job")
    if not full:
        r["log"] = (r.get("log") or [])[-60:]
    return r


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    jobs.cancel(jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/pause")
def pause_job(jid: str):
    jobs.pause(jid)
    return {"ok": True}


@app.post("/api/jobs/{jid}/resume")
def resume_job(jid: str):
    try:
        return jobs.resume(jid).view()
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    try:
        jobs.delete(jid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.post("/api/jobs/clear")
def clear_jobs():
    return {"deleted": jobs.clear_finished()}


# ---------------------------------------------------------------- documents
@app.get("/api/docs")
def docs():
    return store.list_docs()


@app.post("/api/docs/upload")
async def upload(file: UploadFile = File(...), name: str = Form("")):
    tmpd = Path(tempfile.mkdtemp(prefix="packsmith_up_"))
    dst = tmpd / Path(file.filename or "upload").name
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def work(progress):
        try:
            return ingest.ingest(dst, name or None, progress)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    return job("ingest", work, dst.name)


@app.get("/api/docs/{doc}")
def doc(doc: str):
    try:
        m = store.doc_meta(doc)
    except KeyError as e:
        raise HTTPException(404, str(e))
    m["regions"] = store.regions(doc)
    m["has_images"] = (store.doc_dir(doc) / "img").exists()
    return m


@app.delete("/api/docs/{doc}")
def del_doc(doc: str):
    store.delete_doc(doc)
    return {"ok": True}


@app.get("/api/docs/{doc}/page/{n}")
def page(doc: str, n: int):
    m = store.doc_meta(doc)
    if not 1 <= n <= m["pages"]:
        raise HTTPException(404, "page out of range")
    return {"page": n, "pages": m["pages"], "text": store.page_text(doc, n), "image": store.page_image(doc, n) is not None,
            "needs_ocr": n in m.get("needs_ocr", []), "ocr_done": n in m.get("ocr_done", [])}


@app.put("/api/docs/{doc}/page/{n}")
def put_page(doc: str, n: int, d: dict = Body(...)):
    store.set_page_text(doc, n, d.get("text", ""))
    return {"ok": True}


@app.get("/api/docs/{doc}/img/{n}")
def page_img(doc: str, n: int):
    p = store.page_image(doc, n)
    if not p:
        raise HTTPException(404, "no image")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.post("/api/docs/{doc}/ocr")
def run_ocr(doc: str, d: dict = Body({})):
    pages = d.get("pages") or None
    return job("ocr", ocr.ocr_pages, doc, doc=doc, pages=pages, model=d.get("model") or None, parallel=_par(d))


@app.post("/api/docs/{doc}/regions")
def add_region(doc: str, r: dict = Body(...)):
    try:
        return store.save_region(doc, r)
    except (ValueError, KeyError) as e:
        _err(e)


@app.delete("/api/docs/{doc}/regions/{rid}")
def del_region(doc: str, rid: str):
    store.delete_region(doc, rid)
    return {"ok": True}


@app.post("/api/docs/{doc}/calibration")
def calibrate(doc: str, d: dict = Body(...)):
    m = store.doc_meta(doc)
    cal = {int(a): int(b) for a, b in m.get("calibration", [])}
    cal[int(d["scan"])] = int(d["printed"])
    m["calibration"] = sorted([a, b] for a, b in cal.items())
    store.save_meta(doc, m)
    return m["calibration"]


def printed_to_scan(m: dict, printed: int) -> int | None:
    cal = sorted(m.get("calibration", []), key=lambda x: x[1])
    if not cal:
        return None
    best = min(cal, key=lambda x: abs(x[1] - printed))
    return best[0] + (printed - best[1])


@app.get("/api/docs/{doc}/search")
def search(doc: str, q: str = "", scope: str = "all"):
    """One row per page that contains the term: its number of matches, up to 3 matching lines, and (index scope) the
    page numbers those lines cite. An empty or one-letter term is refused: it matched everywhere, and an empty one
    matched at every character, so the list filled with the first page that has text."""
    q = (q or "").strip()
    if len(q) < 2:
        raise HTTPException(400, "type a search term of at least 2 characters")
    m = store.doc_meta(doc)
    hits, total = [], 0
    rx = re.compile(re.escape(q), re.I)
    pages = range(1, m["pages"] + 1)
    if scope == "index":
        pages = [p for r in store.regions(doc) if r["type"] == "index" for p in range(r["start"], r["end"] + 1)]
    for p in pages:
        t = store.page_text(doc, p)
        lines = []
        n = 0
        for mm in rx.finditer(t):
            n += 1
            a, b = t.rfind("\n", 0, mm.start()) + 1, t.find("\n", mm.end())
            line = t[a: b if b >= 0 else len(t)].strip()[:240]
            if line not in lines:
                lines.append(line)
        if not n:
            continue
        total += n
        refs = [int(x) for ln in lines for x in re.findall(r"\b(\d{1,4})\b", ln)] if scope == "index" else []
        hits.append({"page": p, "count": n, "line": lines[0], "lines": lines[:3],
                     "refs": [{"printed": r, "scan": printed_to_scan(m, r)} for r in refs[:12]]})
        if len(hits) >= 300:
            break
    return hits


# ---------------------------------------------------------------- topics
@app.get("/api/topics")
def topics():
    return store.list_topics()


@app.post("/api/topics")
def save_topic(t: dict = Body(...)):
    t["whole_docs"] = [d for d in (t.get("whole_docs") or []) if isinstance(d, str)]
    return store.save_topic(t)


@app.get("/api/topics/{tid}")
def get_topic(tid: str):
    t = store.topic(tid)
    t["status"] = store.topic_status(tid)
    t["region_details"] = [{"doc": d, "doc_name": store.doc_meta(d)["name"], **r} for d, r in store.topic_regions(t)]
    mined = store.mined_regions(t, ("train",))
    t["mining"] = {"train_regions": len(mined), "train_pages": sum(r["end"] - r["start"] + 1 for _, r in mined),
                   "heldout_regions": len(store.mined_regions(t, ("heldout",))),
                   "problem": "" if mined else store.why_no_regions(t)}
    t["playbooks"] = mine.playbooks(tid)
    return t


@app.delete("/api/topics/{tid}")
def del_topic(tid: str):
    store.delete_topic(tid)
    return {"ok": True}


@app.get("/api/topics/{tid}/file", response_class=PlainTextResponse)
def topic_file(tid: str, name: str):
    if "/" in name.replace("dataset/", "", 1) or ".." in name:
        raise HTTPException(400, "bad name")
    return store.rt(store.topic_dir(tid) / name)


# ---------------------------------------------------------------- approach library
@app.get("/api/library")
def lib():
    return {"patterns": library.patterns(), "records": library.records()}


@app.get("/api/library/nearest")
def lib_nearest(domain: str, intent: str = ""):
    return library.nearest(domain, intent, 3)


@app.post("/api/library/derive")
def lib_derive(d: dict = Body(...)):
    return job("derive", library.derive, d.get("domain", ""), domain=d.get("domain", ""), intent=d.get("intent", "reasoning"),
               model=d.get("model") or None)


@app.post("/api/library/save")
def lib_save(r: dict = Body(...)):
    return library.save(r)


# ---------------------------------------------------------------- discovery, structure, mapping, playbook
@app.post("/api/topics/{tid}/prompt")
def build_prompt(tid: str, d: dict = Body({})):
    t = store.topic(tid)
    if d.get("record") is not None:
        t["record"] = d["record"]
        store.save_topic(t)
    return {"prompt": discover.build_prompt(tid, t.get("record"))}


@app.post("/api/topics/{tid}/discover")
def run_discover(tid: str, d: dict = Body(...)):
    return job("discover", discover.run, tid, topic_id=tid, system_prompt=d["prompt"], model=d.get("model") or None,
               pages_per_round=int(d.get("pages_per_round", 6)), stable_rounds=int(d.get("stable_rounds", 2)),
               min_coverage=float(d.get("min_coverage", 0.6)), max_pages=int(d.get("max_pages", 120)))


@app.get("/api/topics/{tid}/rounds")
def rounds(tid: str):
    return {"rounds": discover.rounds(tid), "draft": store.rt(store.topic_dir(tid) / "plan.draft.html"),
            "plan": store.rt(store.topic_dir(tid) / "plan.html")}


@app.post("/api/topics/{tid}/plan")
def approve(tid: str, d: dict = Body(...)):
    try:
        return discover.approve(tid, d["html"])
    except ValueError as e:
        _err(e)


@app.post("/api/topics/{tid}/structure")
def run_structure(tid: str, d: dict = Body({})):
    return job("structure", structure.run, tid, topic_id=tid, model=d.get("model") or None)


@app.post("/api/topics/{tid}/map")
def run_map(tid: str, d: dict = Body({})):
    return job("map", mine.map_topic, tid, topic_id=tid, model=d.get("model") or None, parallel=_par(d))


@app.post("/api/topics/{tid}/reduce")
def run_reduce(tid: str, d: dict = Body({})):
    return job("reduce", mine.reduce_topic, tid, topic_id=tid, model=d.get("model") or None, parallel=_par(d))


@app.get("/api/topics/{tid}/alignment")
def alignment(tid: str):
    return mine.alignment(tid)


@app.get("/api/topics/{tid}/coverage")
def coverage(tid: str):
    return mine.coverage(tid)


@app.get("/api/topics/{tid}/playbook", response_class=PlainTextResponse)
def get_playbook(tid: str, doc: str = ""):
    d = store.topic_dir(tid)
    return store.rt(d / (f"playbook.{doc}.html" if doc and doc != "unified" else "playbook.html")) or store.rt(d / "skeleton.html")


@app.put("/api/topics/{tid}/playbook")
def put_playbook(tid: str, d: dict = Body(...)):
    try:
        return mine.save_playbook(tid, d["html"], "" if d.get("doc") in (None, "", "unified") else d["doc"])
    except ValueError as e:
        _err(e)


@app.post("/api/topics/{tid}/backtest")
def run_backtest(tid: str, d: dict = Body({})):
    return job("backtest", backtest.run, tid, topic_id=tid, playbook=d.get("playbook", "unified"), role=d.get("role", "train"),
               model=d.get("model") or None, limit=int(d.get("limit", 60)), parallel=_par(d))


@app.get("/api/topics/{tid}/backtest")
def get_backtest(tid: str):
    return store.rj(store.topic_dir(tid) / "backtest.json", {}) or {}


@app.post("/api/topics/{tid}/generate")
def run_generate(tid: str, d: dict = Body({})):
    return job("generate", generate.run, tid, topic_id=tid, playbook=d.get("playbook", "unified"), opts=d.get("opts") or {},
               model=d.get("model") or None, parallel=_par(d), dataset=d.get("dataset") or "", name=d.get("name") or "")


# ---------------------------------------------------------------- datasets
@app.get("/api/datasets")
def ds_list(topic: str | None = None):
    return datasets.list_all(topic)


@app.get("/api/datasets/{did}")
def ds_get(did: str):
    try:
        return datasets.manifest(did)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/datasets/{did}")
def ds_delete(did: str):
    try:
        return datasets.delete(did)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/datasets/{did}/rename")
def ds_rename(did: str, d: dict = Body(...)):
    return datasets.rename(did, d["name"])


@app.post("/api/datasets/{did}/current")
def ds_current(did: str):
    m = datasets.manifest(did)
    if not m.get("topic"):
        raise HTTPException(400, "this dataset belongs to no topic")
    datasets.set_current(m["topic"], did)
    return {"ok": True}


@app.post("/api/datasets/{did}/continue")
def ds_continue(did: str, d: dict = Body({})):
    m = datasets.manifest(did)
    if m.get("source") == "compose" and m.get("composition"):
        return job("compose", compose.build, m["composition"], cid=m["composition"], model=d.get("model") or None, parallel=_par(d))
    if not m.get("topic"):
        raise HTTPException(400, "nothing to continue")
    return job("generate", generate.run, m["topic"], topic_id=m["topic"], dataset=did, model=d.get("model") or None,
               parallel=_par(d))


@app.post("/api/datasets/{did}/query")
def ds_query(did: str, d: dict = Body({})):
    try:
        return datasets.query(did, d.get("part", "train"), d.get("q", ""), bool(d.get("regex")), bool(d.get("case")),
                              d.get("kind", ""), d.get("check", ""), d.get("source", ""), int(d.get("min_len") or 0),
                              int(d.get("max_len") or 0), bool(d.get("has_numbers")), d.get("edited", ""),
                              int(d.get("offset") or 0), int(d.get("limit") or 100))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/datasets/{did}/replace-preview")
def ds_replace_preview(did: str, d: dict = Body(...)):
    try:
        return datasets.preview_replace(did, d.get("hashes") or [], d["find"], d.get("repl", ""), bool(d.get("regex")),
                                        bool(d.get("case")), d.get("part", "train"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/datasets/{did}/ops")
def ds_ops(did: str, d: dict = Body(...)):
    try:
        return datasets.apply_ops(did, d["ops"], d.get("note", ""))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/datasets/{did}/revert")
def ds_revert(did: str, d: dict = Body(...)):
    return datasets.revert(did, int(d.get("version", 0)))


@app.get("/api/datasets/{did}/edits")
def ds_edits(did: str):
    return {"versions": datasets.manifest(did).get("versions", []), "edits": datasets.edits(did)}


@app.post("/api/datasets/preview")
def ds_preview(d: dict = Body(...)):
    return compose.preview(d.get("datasets") or [])


@app.post("/api/curated/{cid}/dataset")
def curated_dataset(cid: str, d: dict = Body({})):
    return job("import", compose.from_curated, cid, cid=cid, name=d.get("name", ""))


@app.get("/api/playbooks")
def all_playbooks():
    return mine.all_playbooks()


@app.get("/api/mining/options")
def mining_options():
    from packsmith.mining import miners
    return miners.options()


@app.get("/api/topics/{tid}/mining")
def mining_report(tid: str):
    d = store.topic_dir(tid) / "mining"
    return {"report": store.rj(d / "report.json", {}), "voice": store.rj(d / "voice.json", {})}


@app.post("/api/topics/{tid}/pilot")
def run_pilot(tid: str, d: dict = Body({})):
    from packsmith.workbench import pilot
    return job("pilot", pilot.run, tid, topic_id=tid, opts=d.get("opts") or {}, model=d.get("model") or None, parallel=_par(d))


@app.get("/api/pilot")
def pilot_list():
    from packsmith.workbench import pilot
    return [{k: v for k, v in r.items() if k != "discipline"} for r in pilot.listing()]


@app.get("/api/pilot/{pid}")
def pilot_one(pid: str):
    from packsmith.workbench import pilot
    r = pilot.get(pid)
    if not r:
        raise HTTPException(404, "no such pilot")
    return r


@app.get("/api/data/options")
def data_options():
    return {"lenses": {k: {"name": v[0], "desc": v[1]} for k, v in prompts.LENSES.items()}, "presets": prompts.PRESETS,
            "modes": generate.MODES, "defaults": {k: v for k, v in facts.DEFAULTS.items()}}


@app.post("/api/topics/{tid}/capture")
def run_capture(tid: str, d: dict = Body({})):
    return job("capture", facts.extract, tid, topic_id=tid, opts=d.get("data") or {}, model=d.get("model") or None,
               parallel=_par(d))


@app.get("/api/topics/{tid}/data")
def get_data(tid: str, n: int = 40):
    rs = facts.records(tid, grounded_only=False)
    return {"report": store.rj(store.topic_dir(tid) / "data" / "report.json", {}),
            "sample": [r for r in rs if r.get("grounded")][:n], "ungrounded": [r for r in rs if not r.get("grounded")][:n]}


@app.get("/api/curated")
def curated_sets():
    return curated.list_sets()


@app.post("/api/curated/upload")
async def curated_upload(file: UploadFile = File(...), name: str = Form("")):
    tmpd = Path(tempfile.mkdtemp(prefix="packsmith_cur_"))
    dst = tmpd / Path(file.filename or "upload").name
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def work(progress):
        try:
            return curated.import_file(dst, name or None)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    return job("curated", work, dst.name)


@app.get("/api/curated/{cid}")
def curated_one(cid: str, n: int = 30):
    try:
        return {**curated.meta(cid), "sample": curated.units(cid)[:n]}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/curated/{cid}")
def curated_del(cid: str):
    curated.delete(cid)
    return {"ok": True}


@app.post("/api/topics/{tid}/factbook")
def run_factbook(tid: str, d: dict = Body({})):
    return job("factbook", factbook.build, tid, topic_id=tid, opts=d.get("factbook") or {}, model=d.get("model") or None,
               parallel=_par(d))


@app.get("/api/topics/{tid}/factbook", response_class=HTMLResponse)
def view_factbook(tid: str):
    return store.rt(store.topic_dir(tid) / "factbook" / "factbook.html") or "<p>No fact book yet.</p>"


@app.get("/api/topics/{tid}/factbook/state")
def factbook_state(tid: str):
    d = store.topic_dir(tid) / "factbook"
    return {"report": store.rj(d / "report.json", {}), "conflicts": store.rj(d / "conflicts.json", []),
            "flagged": store.rj(d / "flagged.json", [])}


@app.post("/api/topics/{tid}/factbook/resolve")
def factbook_resolve(tid: str, d: dict = Body(...)):
    factbook.resolve(tid, d["group"], d.get("choice", ""))
    return {"ok": True}


@app.post("/api/quick")
def run_quick(d: dict = Body(...)):
    return job("quick", quick.run, d.get("doc") or d.get("curated", ""), doc=d.get("doc") or "", curated_set=d.get("curated") or "",
               factbook_opts=d.get("factbook") or {}, start=d.get("start") or None, end=d.get("end") or None,
               mode=d.get("mode", "facts"), playbook=d.get("playbook", ""), data=d.get("data") or {}, name=d.get("name", ""),
               max_len=int(d.get("max_len", 1024)), immersion=bool(d.get("immersion", True)), model=d.get("model") or None,
               parallel=_par(d))


@app.get("/api/topics/{tid}/dataset")
def get_dataset(tid: str, n: int = 30):
    tr, te = generate.load(tid)
    return {"report": store.rj(store.topic_dir(tid) / "dataset" / "report.json", {}), "sample": tr[:n], "tests": te[:n],
            "counts": {"train": len(tr), "tests": len(te)}}


# ---------------------------------------------------------------- training, packs
@app.post("/api/train")
def run_train(d: dict = Body(...)):
    src = d["source"]
    kind = d.get("kind", "topic")
    if ":" in src:
        kind, src = src.split(":", 1)
    return job("train", train.run, d.get("name", ""), source=src, name=d.get("name") or src,
               kind=kind, opts=d.get("opts") or {}, unload_lmstudio=bool(d.get("unload_lmstudio", config.SETTINGS.llm_unload_for_gpu)),
               reload_lmstudio=bool(d.get("reload_lmstudio", False)))


@app.get("/api/train/models")
def train_models():
    """The base models the Train tab offers (Settings first)."""
    from packsmith.system import models
    return models.training_choices()


@app.get("/api/train/sizing")
def train_sizing(dataset: str = "", model: str = "", mode: str = "gate"):
    """Base model shape, what the size presets cost on it, and a suggested size for the dataset (a rule of thumb)."""
    from packsmith.training import sizing
    base = model or config.SETTINGS.base_train_model or config.SETTINGS.base_serve_model
    geom = sizing.base_geometry(base)
    eng = config.SETTINGS.engine
    out = {"engine": eng, "geometry": geom, "expert_weights": sizing.expert_weights(geom),
           "bytes_per_weight": 2 if __import__("packsmith.engines", fromlist=["is_polymath"]).is_polymath(eng) else 1, "budget": int(config.SETTINGS.pack_budget),
           "presets": {name: {"k": k, "n_layers": n, **(sizing.pack_size(geom, k, n, eng) if geom else {})}
                       for name, (k, n) in sizing.PRESETS.items()}}
    if dataset:
        try:
            m = datasets.manifest(dataset)
            mining = ((m.get("report") or {}).get("mining") or {})
            items, tokens = sizing.dataset_tokens(datasets.train_path(dataset))
            out["suggestion"] = sizing.suggest(items, tokens, bool(mining.get("perform")),
                                               bool(mining.get("traces")) or (m.get("mode") == "pilot" and m.get("arm") == "B"),
                                               geom, eng, "route" if mode == "route" else "gate")
        except Exception as e:                           # noqa: BLE001
            out["suggestion_error"] = str(e)
    return out


@app.get("/api/packs")
def packs():
    from packsmith.system import models
    from packsmith.workbench import dials
    d = dials.all_dials()
    return [{**p, "note": models.pack_note(p.get("card")), "dials": d.get(p["name"], dict(dials.DEFAULT))}
            for p in store.list_packs()]


@app.post("/api/packs/{name}/dials")
def pack_dials(name: str, request: Request, d: dict = Body(...)):
    """{strength?, shift?}: the pack's Workbench dials (workbench/dials.py). Set from the page only: with an API token
    set, a call with the token in a header (an agent) is refused. The active set is applied again at once."""
    from packsmith.workbench import dials, workbench
    if not _from_page(request):
        raise HTTPException(403, "a pack's strength and routing are set in the Packsmith page (Workbench), not by agents")
    if not any(p["name"] == name for p in store.list_packs()):
        raise HTTPException(404, f"no pack {name}")
    try:
        cur = dials.set_dials(name, d.get("strength"), d.get("shift"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    out = {"name": name, "dials": cur, "applied": False}
    if name in (workbench.STATE.get("active_names") or []):
        r = workbench.reapply()
        out["applied"] = bool(r and r.get("ok"))
        if r and not r.get("ok"):
            out["error"] = "the engine refused the new setting: " + " | ".join((r.get("engine_log") or [])[-2:])
    return out


@app.get("/api/engine-models")
def engine_models():
    from packsmith.system import models
    return models.listing()


@app.post("/api/engine-models/convert")
def engine_models_convert(d: dict = Body(...)):
    from packsmith.system import models
    purpose = d.get("purpose") or "both"
    if purpose == "training-hf":                     # train on the Hugging Face folder as it is: nothing to convert
        folder, _ = models._source(d["source"])
        return models.use_for_training_only(str(folder))
    return job("model", models.convert, d["source"], src=d["source"], name=d.get("name", ""), use_it=bool(d.get("use", True)),
               purpose=purpose)


@app.get("/api/hub/search")
def hub_search(q: str, limit: int = 25):
    from packsmith.system import hub
    try:
        return hub.search(q, limit)
    except hub.HubError as e:
        raise HTTPException(502, str(e))


@app.get("/api/hub/check")
def hub_check(repo: str):
    from packsmith.system import hub
    try:
        return hub.check(repo)
    except hub.HubError as e:
        raise HTTPException(502, str(e))


@app.post("/api/hub/download")
def hub_download(d: dict = Body(...)):
    """Download a MoE model (refused for dense ones) into the shared Hugging Face cache (or Polymath's models)."""
    from packsmith.system import hub
    return job("download", hub.download, d["repo"], repo=d["repo"], where=d.get("where", "hf_cache"))


@app.post("/api/engine-models/use")
def engine_models_use(d: dict = Body(...)):
    from packsmith.system import models
    try:
        return models.use(d["path"], train=bool(d.get("train", True)))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/packs/scan")
def scan_packs(folder: str = ""):
    return store.scan_packs(folder)


@app.post("/api/packs/register")
def register(d: dict = Body(...)):
    try:
        r = store.register_pack(d.get("name") or Path(d["path"]).name, d["path"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {**r, "packs": store.list_packs()}


@app.delete("/api/packs/{name}")
def del_pack(name: str):
    p = store.packs_dir() / name
    if name in ("", ".", "..") or p.is_symlink() or p.resolve().parent != store.packs_dir().resolve():
        raise HTTPException(400, "not a pack in the packs folder")
    if p.exists():
        shutil.rmtree(p)
    else:
        store.unregister_pack(name)          # a registered external pack: forget it, never delete its files
    return {"ok": True}


# ---------------------------------------------------------------- workbench
@app.get("/api/workbench")
def wb_status():
    return workbench.status()


@app.post("/api/workbench/start")
def wb_start(d: dict = Body({})):
    from packsmith.system import gpu

    def go():
        try:
            gpu.need_engine()                      # shared GPU: unloads LM Studio first
        except Exception as e:                     # noqa: BLE001 - shown by the status endpoint
            workbench.STATE.update(status="error", error=str(e))
    threading.Thread(target=go, daemon=True).start()
    return {"status": "starting"}


@app.get("/api/gpu")
def gpu_status():
    from packsmith.system import gpu
    return gpu.status()


@app.post("/api/workbench/stop")
def wb_stop():
    return workbench.stop()


def _ask_kw(d: dict) -> dict:
    return dict(question=d["question"], packs=d.get("packs", []), mode=d.get("mode", "single"), template=d.get("template", "chat"),
                domain=d.get("domain", "this field"), max_tokens=int(d.get("max_tokens", 1024)),
                temperature=None if d.get("temperature") in (None, "") else float(d["temperature"]),
                top_p=None if d.get("top_p") in (None, "") else float(d["top_p"]), history=d.get("history") or [],
                system=d.get("system", ""), thinking=bool(d.get("thinking")),
                # an empty dict is "every configuration starts its own conversation", not "no threads sent": treating it as
                # None made the first compare turn fall back to the shared history, so the base saw the pack's answers
                histories=({str(k): v for k, v in (d.get("histories") or {}).items() if isinstance(v, list)}
                           if isinstance(d.get("histories"), dict) else None))


@app.post("/api/workbench/ask")
def wb_ask(d: dict = Body(...)):
    try:
        return workbench.ask(**_ask_kw(d))
    except (RuntimeError, KeyError, ValueError) as e:
        _err(e)


@app.post("/api/workbench/ask/stream")
def wb_ask_stream(d: dict = Body(...)):
    """Single mode, the answer streamed as it is written (text chunks, then a final JSON line after \x1e)."""
    import queue
    q: queue.Queue = queue.Queue()

    def work():
        try:
            r = workbench.ask(**_ask_kw(d), on_chunk=lambda s: q.put(("c", s)))
            q.put(("r", r))
        except Exception as e:                         # noqa: BLE001 - sent to the client
            q.put(("e", f"{type(e).__name__}: {e}"))
    threading.Thread(target=work, daemon=True).start()

    def gen():
        yield "\u200b"                                  # an immediate first byte: the browser starts reading at once
        while True:
            k, v = q.get()
            if k == "c":
                yield v
            else:
                yield "\x1e" + json.dumps({"result": v} if k == "r" else {"error": v})
                return
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/workbench/evaluations/{pack}")
def wb_evals(pack: str):
    return workbench.evaluations(pack)


@app.get("/api/workbench/evaluations/{pack}/{eid}")
def wb_eval_one(pack: str, eid: str):
    r = workbench.evaluation(pack, eid)
    if not r:
        raise HTTPException(404, "no such evaluation")
    return r


@app.post("/api/workbench/analyse")
def wb_analyse(d: dict = Body(...)):
    try:
        return workbench.analyse(d["question"], d["answer"], d.get("topic", ""), d.get("model") or None,
                                 packs=[p for p in (d.get("packs") or []) if isinstance(p, str)])
    except llm.LLMError as e:
        _err(e)


@app.post("/api/workbench/judge")
def wb_judge(d: dict = Body(...)):
    try:
        return {"html": workbench.judge(d["question"], d["answers"], d.get("model") or None)}
    except llm.LLMError as e:
        _err(e)


@app.post("/api/workbench/test")
def wb_test(d: dict = Body(...)):
    workbench.add_test(d["topic"], d["question"], d["expected"])
    return {"ok": True}


@app.post("/api/workbench/flag")
def wb_flag(d: dict = Body(...)):
    workbench.flag_miss(d["topic"], d["question"], d.get("answer", ""), d.get("note", ""))
    return {"ok": True}


# ---------------------------------------------------------------- the Test tab (workbench/testbench.py)
def _tb(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/test/options")
def test_options():
    from packsmith.workbench import testbench
    return testbench.options()


@app.get("/api/test/draft")
def test_draft():
    from packsmith.workbench import testbench
    return testbench.draft()


@app.post("/api/test/draft")
def test_draft_save(d: dict = Body(...)):
    from packsmith.workbench import testbench
    return _tb(testbench.save_draft, d)


@app.post("/api/test/check")
def test_check(d: dict = Body(...)):
    from packsmith.workbench import testbench
    return testbench.check_candidate(d)


@app.post("/api/test/send")
def test_send(d: dict = Body(...)):
    """The Workbench's Send to Test: the question and the packs it was asked with."""
    from packsmith.workbench import testbench
    return _tb(testbench.send, str(d.get("question") or ""), [p for p in (d.get("packs") or []) if isinstance(p, str)],
               d.get("engine") or None)


@app.get("/api/test/sets")
def test_sets():
    from packsmith.workbench import testbench
    return testbench.list_sets()


@app.get("/api/test/sets/{name}")
def test_set(name: str):
    from packsmith.workbench import testbench
    return _tb(testbench.get_set, name)


@app.post("/api/test/sets/{name}")
def test_set_save(name: str, d: dict = Body(...)):
    from packsmith.workbench import testbench
    return _tb(testbench.save_set, name, d.get("sets") or [])


@app.delete("/api/test/sets/{name}")
def test_set_delete(name: str):
    from packsmith.workbench import testbench
    _tb(testbench.delete_set, name)
    return {"ok": True}


@app.get("/api/test/runs")
def test_runs():
    from packsmith.workbench import testbench
    return testbench.list_runs()


@app.post("/api/test/runs")
def test_run_start(d: dict = Body(...)):
    """Start a run (a background job). Agents may start runs; candidates name library packs only."""
    from packsmith.workbench import testbench
    return _tb(testbench.start, d)


@app.get("/api/test/runs/{rid}")
def test_run(rid: str):
    from packsmith.workbench import testbench
    return _tb(testbench.get_run, rid)


@app.post("/api/test/runs/{rid}/continue")
def test_run_continue(rid: str):
    """Run again what a paused, cancelled or failed run has not answered yet."""
    from packsmith.workbench import testbench
    j = _tb(testbench.resume, rid)
    return {"job": j}


@app.post("/api/test/runs/{rid}/evaluate")
def test_run_evaluate(rid: str, d: dict = Body({})):
    from packsmith.workbench import testbench
    _tb(testbench.get_run, rid)
    kw = {"rid": rid, "model": str(d.get("model") or ""), "topic": str(d.get("topic") or "")}
    for k, k2 in (("set", "set_index"), ("q", "q_index")):
        if d.get(k) not in (None, ""):
            kw[k2] = int(d[k])
    if d.get("cand"):
        kw["cand"] = str(d["cand"])
    return job("test-evaluate", testbench.evaluate_job, f"evaluate {rid}", **kw)


@app.delete("/api/test/runs/{rid}")
def test_run_delete(rid: str):
    from packsmith.workbench import testbench
    _tb(testbench.delete_run, rid)
    return {"ok": True}


# ---------------------------------------------------------------- the Train Lab (lab/lab.py)
@app.get("/api/lab/options")
def lab_options():
    from packsmith.lab import lab
    return {"recipes": lab.recipes(), "fields": {k: {"label": v[0], "kind": v[1], "help": v[2]} for k, v in lab.FIELDS.items()},
            "datasets": lab.datasets(), "model": Path(str(config.SETTINGS.base_train_model or "")).name,
            "engines": ["patched-colibri", "polymath-engine"]}


@app.post("/api/lab/recipes/{name}")
def lab_recipe_save(name: str, d: dict = Body(...)):
    from packsmith.lab import lab
    return _tb(lab.save_recipe, name, d.get("settings") or {}, str(d.get("about") or ""))


@app.delete("/api/lab/recipes/{name}")
def lab_recipe_delete(name: str):
    from packsmith.lab import lab
    _tb(lab.delete_recipe, name)
    return {"ok": True}


@app.post("/api/lab/sweep")
def lab_sweep(d: dict = Body(...)):
    from packsmith.lab import lab
    return _tb(lab.sweep, str(d.get("field") or ""), list(d.get("values") or [])[:8], d.get("start"))


@app.post("/api/lab/estimate")
def lab_estimate(d: dict = Body(...)):
    from packsmith.lab import lab
    have = {r["name"]: r for r in lab.recipes()}
    rs = [have["baseline"]] + [have[n] if isinstance(n, str) else {"name": str(n.get("name")), "settings": lab.clean_settings(n.get("settings") or {})}
                               for n in (d.get("recipes") or []) if (isinstance(n, str) and n in have and n != "baseline") or isinstance(n, dict)]
    return _tb(lab.estimate, str(d.get("dataset") or "yardstick"), rs, str(config.SETTINGS.base_train_model or ""))


@app.get("/api/lab/runs")
def lab_runs():
    from packsmith.lab import lab
    return lab.list_runs()


@app.post("/api/lab/runs")
def lab_run_start(d: dict = Body(...)):
    """Start a Lab run (a background job). Agents may start runs. The base model is the one in Settings: a run
    cannot name another folder to read."""
    from packsmith.lab import lab
    d = dict(d)
    d["model"] = str(config.SETTINGS.base_train_model or "")
    d.pop("device", None)
    return _tb(lab.start, d)


@app.get("/api/lab/runs/{rid}")
def lab_run(rid: str):
    from packsmith.lab import lab
    return _tb(lab.comparison, rid)


@app.post("/api/lab/runs/{rid}/continue")
def lab_run_continue(rid: str):
    from packsmith.lab import lab
    return {"job": _tb(lab.resume, rid)}


@app.delete("/api/lab/runs/{rid}")
def lab_run_delete(rid: str):
    from packsmith.lab import lab
    return _tb(lab.delete_run, rid)


@app.get("/api/lab/reasoning")
def lab_reasoning_list():
    from packsmith.lab import reasoning
    return {"datasets": reasoning.list_all(), "defaults": reasoning.DEFAULTS}


@app.post("/api/lab/reasoning")
def lab_reasoning_build(d: dict = Body(...)):
    """Make a reasoning dataset in the Lab's folder (a background job; the LLM chosen at the top of the page)."""
    from packsmith.lab import reasoning
    o = {k: d[k] for k in reasoning.DEFAULTS if k in d}
    if not o.get("dataset"):
        raise HTTPException(400, "choose the dataset whose items are kept")
    return job("lab-reasoning", reasoning.job, "reasoning dataset", opts=o, model=str(d.get("model") or ""), parallel=_par(d))


@app.get("/api/lab/reasoning/{rid}")
def lab_reasoning_get(rid: str):
    from packsmith.lab import reasoning
    return _tb(reasoning.get, rid)


@app.delete("/api/lab/reasoning/{rid}")
def lab_reasoning_delete(rid: str):
    from packsmith.lab import reasoning
    _tb(reasoning.get, rid)
    reasoning.delete(rid)
    return {"ok": True}


@app.post("/api/lab/runs/{rid}/recipes/{i}/library")
def lab_to_library(rid: str, i: int, d: dict = Body({})):
    """A Lab pack into the library, marked test_lab: ticked by hand only, never chosen by an agent."""
    from packsmith.lab import testpacks
    return _tb(testpacks.add, rid, i, str(d.get("name") or ""))


@app.get("/api/lab/testpacks")
def lab_testpacks():
    from packsmith.lab import testpacks
    return testpacks.list_test()


@app.post("/api/lab/testpacks/{name}/promote")
def lab_testpack_promote(name: str, d: dict = Body({})):
    from packsmith.lab import testpacks
    return _tb(testpacks.promote, name, str(d.get("name") or ""))


@app.delete("/api/lab/testpacks/{name}")
def lab_testpack_delete(name: str):
    from packsmith.lab import testpacks
    return _tb(testpacks.delete, name)


# ---------------------------------------------------------------- the Storage tab (lab/storage.py)
@app.get("/api/storage")
def storage_overview():
    from packsmith.lab import storage
    return storage.overview()


@app.post("/api/storage/delete")
def storage_delete(request: Request, d: dict = Body(...)):
    """Final. Only items inside the data folder; with the API token on it is required here without exception."""
    from packsmith.lab import storage
    tok = config.SETTINGS.api_token
    if tok and not _authed(request, tok):
        raise HTTPException(401, "this Packsmith needs its API token")
    return _tb(storage.delete, [e for e in (d.get("items") or []) if isinstance(e, dict)], str(d.get("confirm") or ""))


@app.post("/api/workbench/evaluate")
def wb_eval(d: dict = Body(...)):
    if d.get("dataset"):
        tests = str(datasets.tests_path(d["dataset"]))
    elif d.get("topic"):
        tests = str(store.topic_dir(d["topic"]) / "dataset" / "tests.jsonl")
    else:
        card = next((p["card"] for p in store.list_packs() if p["name"] == d["pack"]), {}) or {}
        if not card.get("dataset"):
            raise HTTPException(400, "choose the dataset whose tests to use")
        tests = str(datasets.tests_path(card["dataset"]))
    return job("evaluate", workbench.evaluate, d["pack"], pack=d["pack"], tests_file=tests, limit=int(d.get("limit", 40)),
               template=d.get("template", "qa"), judge_model=d.get("model") or None, parallel=_par(d))


# ---------------------------------------------------------------- assistant and pipelines
@app.get("/api/assistant")
def as_list():
    return assistant.listing()


@app.post("/api/assistant")
def as_new(d: dict = Body({})):
    return assistant.new(d.get("title", ""))


@app.get("/api/assistant/{cid}")
def as_get(cid: str):
    try:
        c = assistant.conv(cid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {**c, "pipeline_view": assistant._pipeline_view(c)}


@app.delete("/api/assistant/{cid}")
def as_del(cid: str):
    assistant.delete(cid)
    return {"ok": True}


@app.post("/api/assistant/{cid}/message")
def as_msg(cid: str, d: dict = Body(...)):
    try:
        return assistant.message(cid, d.get("text", ""), d.get("docs") or [], d.get("model") or None, d.get("action", ""))
    except llm.LLMError as e:
        _err(e)


@app.get("/api/assistant/{cid}/status")
def as_status(cid: str):
    return assistant._pipeline_view(assistant.conv(cid))


@app.get("/api/pipelines")
def pl_list():
    return pipelines.listing()


@app.post("/api/pipelines")
def pl_new(d: dict = Body(...)):
    try:
        p = pipelines.create(d["plan"])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    j = jobs.start("pipeline", pipelines.run, p["name"], pid=p["id"])
    return {"pipeline": p, "job": j.view()}


@app.post("/api/pipelines/{pid}/resume")
def pl_resume(pid: str):
    j = jobs.start("pipeline", pipelines.run, f"{pid} (resumed)", pid=pid)
    return j.view()


# ---------------------------------------------------------------- composer
@app.get("/api/compositions")
def comps():
    out = store.list_compositions()
    for c in out:
        c["stale"] = compose.stale(c) if c.get("built") else False
    return out


@app.post("/api/compositions")
def save_comp(c: dict = Body(...)):
    return store.save_composition(c)


@app.post("/api/compositions/{cid}/build")
def build_comp(cid: str, d: dict = Body({})):
    return job("compose", compose.build, cid, cid=cid, model=d.get("model") or None, parallel=_par(d))


# ---------------------------------------------------------------- user packs and the agent API
from fastapi import Depends, Header  # noqa: E402
from packsmith.userpack import agent as _agent, core as _up, recipes as _recipes  # noqa: E402


def _token():
    """Kept for the routes that declare it: the token is checked for every /api/ route by the _auth middleware."""
    return None


def _from_page(request: Request) -> bool:
    """A call made by Packsmith's own page (the person at the keyboard), not by an agent: with an API token set, the
    page is the caller signed in with the session cookie (agents send the token in a header). With no token there is
    no way to tell them apart, and every caller counts as the page: the folder limit holds only with a token."""
    tok = config.SETTINGS.api_token
    if not tok:
        return True
    return _same(request.cookies.get(SESSION_COOKIE, ""), _session(tok)) and not request.headers.get("authorization") \
        and not request.headers.get("x-packsmith-token")


def _check_paths(request: Request, sections=None, from_folder: str = "", paths=None) -> None:
    """Agents may name folders or files only inside Settings → Server → Folders agents may read (agent_folders)."""
    if _from_page(request):
        return
    names = [from_folder] if from_folder else []
    names += list(paths or [])
    for sec in sections if isinstance(sections, list) else []:
        names += list((sec or {}).get("paths") or [])
    names = [n for n in names if n]
    if not names:
        return
    roots = [Path(r.strip()).expanduser().resolve() for r in config.SETTINGS.agent_folders.split(",") if r.strip()]
    for n in names:
        q = Path(n).expanduser().resolve()
        if not any(q == r or r in q.parents for r in roots):
            raise HTTPException(403, f"{n} is not inside a folder agents may read (Settings → Server → Folders agents "
                                     f"may read: {config.SETTINGS.agent_folders or 'none set'}); send the text instead")


def _up_call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except (ValueError, RuntimeError) as e:
        _err(e)


@app.get("/api/userpack/types", dependencies=[Depends(_token)])
def up_types():
    """Section types, with the prompts that mine them (what 'How this is mined' shows)."""
    return [{**t, "prompts": _recipes.prompts(t["type"])} for t in _recipes.describe()]


@app.get("/api/userpacks", dependencies=[Depends(_token)])
def up_list():
    return _up.list_all()


@app.post("/api/userpacks", dependencies=[Depends(_token)])
def up_new(request: Request, d: dict = Body(...)):
    """{name, sections?: [{type, title?, about?, text?, paths?}] | from_folder?: "<folder>", mode?: stack|merged, options?}"""
    _check_paths(request, d.get("sections"), d.get("from_folder", ""))
    secs = _up_call(_up.sections_from_folder, d["from_folder"]) if d.get("from_folder") else d.get("sections")
    return _up_call(_up.new_spec, d.get("name", ""), secs, d.get("mode", "stack"), d.get("options"))


@app.get("/api/userpacks/{name}", dependencies=[Depends(_token)])
def up_get(name: str):
    return _up_call(_up.status, name)


@app.delete("/api/userpacks/{name}", dependencies=[Depends(_token)])
def up_delete(name: str):
    _up.delete(name)
    return {"ok": True}


@app.post("/api/userpacks/{name}/sections", dependencies=[Depends(_token)])
def up_add_section(name: str, request: Request, d: dict = Body(...)):
    _check_paths(request, paths=d.get("paths"))
    return _up_call(_up.add_section, name, d.get("type", "custom"), d.get("title", ""), d.get("about", ""),
                    d.get("text", ""), d.get("paths"))


@app.post("/api/userpacks/{name}/sections/{sid}", dependencies=[Depends(_token)])
def up_edit_section(name: str, sid: str, request: Request, d: dict = Body(...)):
    _check_paths(request, paths=d.get("paths"))
    return _up_call(_up.update_section, name, sid, **{k: d.get(k) for k in ("title", "about", "text", "paths", "enabled", "type")})


@app.delete("/api/userpacks/{name}/sections/{sid}", dependencies=[Depends(_token)])
def up_del_section(name: str, sid: str):
    return _up_call(_up.remove_section, name, sid)


@app.post("/api/userpacks/{name}/sections/{sid}/files", dependencies=[Depends(_token)])
async def up_file(name: str, sid: str, file: UploadFile = File(...)):
    return _up_call(_up.add_file, name, sid, file.filename or "file.txt", await file.read())


@app.post("/api/userpacks/{name}/extract", dependencies=[Depends(_token)])
def up_extract(name: str, d: dict = Body({})):
    _up_call(_up.spec, name)
    return job("userpack-extract", _up.extract, name, name=name, sections=d.get("sections"))


@app.get("/api/userpacks/{name}/statements", dependencies=[Depends(_token)])
def up_statements(name: str):
    return _up_call(_up.statements, name)


@app.post("/api/userpacks/{name}/statements", dependencies=[Depends(_token)])
def up_edit_statements(name: str, d: dict = Body(...)):
    """{edits: [{section, id, confirmed?, text?, kind?, stable?}]} or {auto: true} (accept the grounded ones)."""
    if d.get("auto"):
        return _up_call(_up.auto_confirm, name)
    return _up_call(_up.edit_statements, name, d.get("edits") or [])


@app.get("/api/userpacks/{name}/plan", dependencies=[Depends(_token)])
def up_plan(name: str):
    return _up_call(_up.plan, name)


@app.post("/api/userpacks/{name}/build", dependencies=[Depends(_token)])
def up_build(name: str, d: dict = Body({})):
    """Generate the items and train (sections: retrain only those parts of a stack)."""
    _up_call(_up.spec, name)
    return job("userpack-build", _up.build, name, name=name, sections=d.get("sections"))


@app.post("/api/userpacks/{name}/create", dependencies=[Depends(_token)])
def up_create(name: str, request: Request, d: dict = Body({})):
    """One shot: {sections?, mode?, options?, confirm: review|auto}. review stops for the statements' ticks."""
    _check_paths(request, d.get("sections"), d.get("from_folder", ""))
    if not d.get("sections") and not d.get("from_folder") and not _from_page(request):   # a saved spec's own paths
        try:
            _check_paths(request, _up.spec(name)["sections"])
        except KeyError:
            pass
    return job("userpack-create", _up.create, name, name=name, sections=d.get("from_folder") or d.get("sections"),
               mode=d.get("mode", "stack"),
               options=d.get("options"), confirm=d.get("confirm", "review"))


@app.post("/api/userpacks/{name}/subject", dependencies=[Depends(_token)])
def up_subject(name: str, d: dict = Body(...)):
    """{name, pronoun}: who the user pack is about (a name adds items that ask about the person by name)."""
    return _up_call(_up.set_subject, name, str(d.get("name") or ""), str(d.get("pronoun") or ""))


@app.post("/api/userpacks/{name}/validate", dependencies=[Depends(_token)])
def up_validate(name: str, d: dict = Body({})):
    _up_call(_up.spec, name)
    curve = d.get("curve")
    if curve is True:
        curve = _up.CURVE
    if curve is not None and (not isinstance(curve, list) or len(curve) > 8):
        raise HTTPException(400, "curve: true, or a list of up to 8 strengths")
    return job("userpack-validate", _up.validate, name, name=name, curve=curve or None)


@app.get("/api/userpacks/{name}/profile", dependencies=[Depends(_token)], response_class=PlainTextResponse)
def up_profile(name: str):
    return _up_call(_up.profile, name)


@app.get("/api/agent/packs", dependencies=[Depends(_token)])
def ag_packs():
    return _agent.packs()


@app.get("/api/agent/status", dependencies=[Depends(_token)])
def ag_status():
    return _agent.status()


@app.post("/api/agent/load", dependencies=[Depends(_token)])
def ag_load(d: dict = Body(...)):
    """{packs: [name, ...]}: exactly these packs active (starts the engine when needed), each with the strength and
    routing set in the Workbench page. Agents cannot set those: a "strengths" field is ignored, and the answer says so."""
    from packsmith.workbench.workbench import PackIncompatible
    try:
        r = _agent.load(d.get("packs") or [], bool(d.get("start", True)))
        if d.get("strengths") is not None:
            r["note"] = "strengths are set in the Packsmith page (Workbench), not by agents: the page's settings were used"
        return r
    except PackIncompatible as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except RuntimeError as e:
        _err(e)


@app.post("/api/agent/unload", dependencies=[Depends(_token)])
def ag_unload():
    return _agent.unload()


@app.get("/api/agent/jobs/{jid}", dependencies=[Depends(_token)])
def ag_job(jid: str):
    """A job's progress for agents: status, progress, the last log lines, the result."""
    v = get_job(jid)
    return {k: v.get(k) for k in ("id", "kind", "status", "progress", "result", "error")} | {"log": (v.get("log") or [])[-8:]}
