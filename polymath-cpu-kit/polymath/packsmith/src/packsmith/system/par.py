"""Parallel LLM calls: an ordered thread map with progress, cancel and per-item error capture.
LM Studio serves parallel requests (set "Max concurrent predictions" / parallel slots in LM Studio to match)."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from packsmith.system.config import SETTINGS


def workers(n: int | None = None) -> int:
    return max(1, int(n or SETTINGS.parallel or 1))


def pmap(fn, items: list, n: int | None = None, on_done=None, cancelled=lambda: False) -> list:
    """fn(item) for every item with up to n calls in flight; results in input order. A failed item yields
    {"_error": "..."} (reported through on_done) instead of stopping the run; if every item fails the first
    error is raised."""
    items = list(items)
    out: list = [None] * len(items)
    if not items:
        return out
    n = workers(n)
    lock, done, errors = threading.Lock(), [0], []

    from packsmith.system import jobs
    parent = jobs.current()                          # worker threads work for the same job (cancel reaches them)
    if parent is not None:
        outer = cancelled
        cancelled = lambda: outer() or parent.cancelled()  # noqa: E731

    def run(i):
        if parent is not None:
            jobs.set_current(parent)
        if cancelled():
            return i, {"_error": "cancelled"}
        try:
            return i, fn(items[i])
        except Exception as e:                       # noqa: BLE001 - reported per item
            errors.append(e)
            return i, {"_error": f"{type(e).__name__}: {e}"}

    if n == 1:
        for i in range(len(items)):
            _, out[i] = run(i)
            done[0] += 1
            on_done and on_done(done[0], len(items), out[i])
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            for f in as_completed([ex.submit(run, i) for i in range(len(items))]):
                i, r = f.result()
                out[i] = r
                with lock:
                    done[0] += 1
                    on_done and on_done(done[0], len(items), r)
    if parent is not None and parent.cancelled():     # stop the whole step: finished units stay cached for resume
        from packsmith.llm.llm import Cancelled
        raise Cancelled("paused" if parent._pause else "cancelled")
    if errors and len(errors) == len(items):
        raise errors[0]
    return out


def failed(r) -> bool:
    return isinstance(r, dict) and "_error" in r
