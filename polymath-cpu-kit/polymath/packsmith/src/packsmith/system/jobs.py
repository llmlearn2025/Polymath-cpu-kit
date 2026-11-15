"""Background jobs (threads) with progress, log, results, cancel / pause / resume, kept on disk.

Every job is saved to <data_dir>/jobs/<id>.json (inputs, log, result, error, timings), so results — an evaluation's
rows, a dataset report, a training card — stay readable after the page or the server restarts. Cancelling closes the
job's open LM Studio connections at once (llm registers them); pausing does the same but marks the job resumable.
Resume runs the same function with the same inputs; the steps cache their work, so it continues where it stopped."""
from __future__ import annotations

import importlib
import inspect
import json
import secrets
import threading
import time
import traceback

JOBS: dict[str, "Job"] = {}
GPU_LOCK = threading.Lock()          # training and engine restarts take the GPU exclusively
_local = threading.local()


def current() -> "Job | None":
    """The job the calling thread works for (worker threads inherit it through par.pmap)."""
    return getattr(_local, "job", None)


def set_current(job: "Job | None") -> None:
    _local.job = job


def _dir():
    from packsmith.packs import store
    d = store.data_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(v):
    from pathlib import PurePath
    if isinstance(v, PurePath):                     # kept as a path string, so Resume passes a usable value back
        return str(v)
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)[:200]


def _trace(e: BaseException) -> str:
    """The traceback from the job's own function down: this module's runner frame (`job.result = fn(**kw)`) is only
    the caller, and showing it first made jobs.py look like the place of the error."""
    tb = e.__traceback__
    if tb is not None and tb.tb_frame.f_code.co_filename == __file__:
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(e), e, tb))


class Job:
    def __init__(self, kind: str, label: str = "", fn_ref: str = "", inputs: dict | None = None, jid: str | None = None):
        self.id = jid or secrets.token_hex(4)
        self.kind, self.label, self.fn_ref = kind, label, fn_ref
        self.inputs = {k: _safe(v) for k, v in (inputs or {}).items()}
        self.status, self.progress, self.log = "running", 0.0, []
        self.result, self.error = None, ""
        self.started, self.ended = time.time(), None
        self._cancel = self._pause = False
        self.conns: set = set()
        self._saved = 0.0
        self.resumable = bool(fn_ref)

    def report(self, frac: float, msg: str = ""):
        self.progress = max(0.0, min(1.0, float(frac)))
        if msg:
            self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
            del self.log[:-2000]
        if time.time() - self._saved > 3:
            self.save()

    def cancelled(self) -> bool:
        return self._cancel or self._pause

    def stop(self, pause: bool = False):
        if pause:
            self._pause = True
        else:
            self._cancel = True
        for c in list(self.conns):                    # abort calls in flight
            try:
                if c.sock is not None:
                    c.sock.shutdown(2)
                c.close()
            except OSError:
                pass

    def view(self, tail: int = 60) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "status": self.status, "progress": round(self.progress, 3),
                "log": self.log[-tail:] if tail else self.log, "result": self.result, "error": self.error,
                "started": self.started, "ended": self.ended, "inputs": self.inputs, "resumable": self.resumable,
                "seconds": round((self.ended or time.time()) - self.started, 1)}

    def save(self):
        self._saved = time.time()
        try:
            from packsmith.packs import store
            store.wj(_dir() / f"{self.id}.json", {**self.view(0), "fn_ref": self.fn_ref})
        except Exception:                              # noqa: BLE001 - saving a job record must never break the job
            pass


def _ref(fn) -> str:
    mod, name = getattr(fn, "__module__", ""), getattr(fn, "__qualname__", "")
    return f"{mod}:{name}" if mod.startswith("packsmith") and "<" not in name else ""


def start(kind: str, fn, label: str = "", /, **kwargs) -> Job:
    job = Job(kind, label, _ref(fn), kwargs)
    JOBS[job.id] = job
    _launch(job, fn, kwargs)
    return job


def _launch(job: Job, fn, kwargs: dict):
    params = inspect.signature(fn).parameters
    kw = dict(kwargs)
    if "progress" in params:
        kw["progress"] = job.report
    if "cancelled" in params:
        kw["cancelled"] = job.cancelled

    def run():
        set_current(job)
        try:
            job.result = fn(**kw)
            job.status = "paused" if job._pause else "cancelled" if job._cancel else "done"
            job.progress = 1.0 if job.status == "done" else job.progress
        except Exception as e:                           # noqa: BLE001 - reported to the UI
            if job._pause or job._cancel:
                job.status = "paused" if job._pause else "cancelled"
                job.log.append(f"{time.strftime('%H:%M:%S')} stopped: {type(e).__name__}: {e}")
            else:
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                job.log.append(_trace(e)[-3000:])
        job.ended = time.time()
        job.save()
        set_current(None)

    threading.Thread(target=run, daemon=True).start()


def cancel(jid: str) -> None:
    if jid in JOBS:
        JOBS[jid].stop(pause=False)


def pause(jid: str) -> None:
    if jid in JOBS:
        JOBS[jid].stop(pause=True)


def resume(jid: str) -> Job:
    """Run a stopped/failed job's function again with its inputs (its steps skip finished work)."""
    rec = get(jid)
    if not rec:
        raise KeyError("no such job")
    if rec["status"] == "running":
        raise ValueError("the job is still running")
    ref = rec.get("fn_ref") or ""
    if not ref:
        raise ValueError("this job cannot be resumed")
    mod, name = ref.split(":", 1)
    fn = importlib.import_module(mod)
    for part in name.split("."):
        fn = getattr(fn, part)
    job = Job(rec["kind"], rec["label"] + " (resumed)", ref, rec.get("inputs") or {})
    job.log.append(f"{time.strftime('%H:%M:%S')} resuming job {jid}")
    JOBS[job.id] = job
    _launch(job, fn, rec.get("inputs") or {})
    return job


def get(jid: str) -> dict | None:
    if jid in JOBS:
        return {**JOBS[jid].view(0), "fn_ref": JOBS[jid].fn_ref}
    from packsmith.packs import store
    return store.rj(_dir() / f"{jid}.json")


def delete(jid: str) -> None:
    j = JOBS.get(jid)
    if j is not None and j.status == "running":
        raise ValueError("cancel the job first")
    JOBS.pop(jid, None)
    (_dir() / f"{jid}.json").unlink(missing_ok=True)


def clear_finished() -> int:
    n = 0
    for rec in listing(500):
        if rec["status"] != "running":
            delete(rec["id"])
            n += 1
    return n


def listing(limit: int = 100) -> list[dict]:
    """Running jobs from memory, finished ones from disk, newest first."""
    from packsmith.packs import store
    seen, out = set(), []
    for j in JOBS.values():
        out.append(j.view(3))
        seen.add(j.id)
    for p in _dir().glob("*.json"):
        if p.stem in seen:
            continue
        r = store.rj(p)
        if r:
            if r.get("status") == "running":           # the server stopped while it ran
                r["status"] = "interrupted"
            r["log"] = (r.get("log") or [])[-3:]
            out.append(r)
    out.sort(key=lambda r: -(r.get("started") or 0))
    return out[:limit]
