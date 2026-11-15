"""The assistant: a chat (LM Studio model) that turns "train me on this" into a plan and runs it.

It asks a few questions to pin down what the user wants, proposes a plan (shown as a card), and on the user's go-ahead
starts a pipeline (pipelines.py) — dataset only, dataset + train, or dataset + train + test. It reports progress and
can pause, resume or cancel the run. The model works through tools (OpenAI tool calls); models without tool calling
fall back to a JSON block in the reply that the server reads the same way.

  assistant/<id>.json  {id, title, messages: [...], plan: {...}|null, pipeline: id|null}
"""
from __future__ import annotations

import json
import re
import time

from packsmith.datasets import curated
from packsmith.system import jobs
from packsmith.llm import llm
from packsmith.mining import mine
from packsmith.workbench import pipelines
from packsmith.packs import store

SYSTEM = """You are Packsmith's assistant. You help a user turn documents (books, reports, manuals) or curated data
(training data or notes from other apps) into a specialization pack for their local model, end to end.

How you work:
1. Find out what they want, briefly: which document(s) and which pages or chapters (use document_info to look at the
   contents pages), what the pack should know (facts and figures, the method or reasoning of the field, procedures),
   who will use it and for what, and whether to only build the training data, or also train, and also test.
   Ask at most two short questions at a time. If the user says "just do it", choose sensible defaults.
2. Propose a plan with propose_plan. Choose:
   - mode: "facts" (facts & data from the pages; works on any text), "reasoning" (worked cases from a playbook),
     or "both". When the user wants method or reasoning and no playbook exists for the field, use the built-in
     playbook "builtin:facts" (fact reasoning, works on any text).
   - profile (extra mining, so the pack can PERFORM like the expert, not only know and reason): "writer" (write in
     an author's voice), "drafter" (produce documents like the book's forms and clauses), "reader" (from findings or
     facts to the expert's report), "reviewer" (check and correct work); empty when the user wants knowledge only.
     mode "mining" uses only the profile's miners.
   - preset: generic (any text), report (commission/policy reports), manual (guides, manuals), field-guide
     (catalogues: items with attributes), everything.
   - train and evaluate: true unless the user wants only the dataset.
   - epochs: leave them out (auto: about 4,800 texts seen in all, 2 to 16 epochs) unless the user asks; max_len 1024.
   Then tell the user the plan in two or three lines and ask them to press Run (or say "go").
3. When they agree, call run_plan. While it runs, use pipeline_status when asked; pause_pipeline, resume_pipeline and
   cancel_pipeline on request. Training unloads the LM Studio models (you included) and loads them again afterwards;
   say so before training starts. A failed step's error says what to fix; tell the user in plain words.

Other things you can help with:
- Regions and topics. A document's regions have a type (chapter, section, other = the teaching pages, the only ones
  mined; table of contents, book index, table of cases, table of statutes = read for structure, never mined) and a
  role (train = learned from; context = structure only; heldout = kept for the final test; exclude = ignored). When a
  topic's mining fails, call topic_status: it says exactly why (chapters not ticked in the topic, a region typed
  table of contents with the role train, regions tagged again with a new id) and the user fixes it in the Library or
  Topics tab. A text without a table of contents or book index is fine: the skeleton step is optional.
- Packs: list_packs shows the library, which engine each pack fits, what is loaded, and each pack's strength and
  routing. Strength (how strongly a pack speaks, 100% = as trained) and routing (how often it steps in, Polymath
  engines) are set by the user in Workbench → Packs only; you cannot change them: say where they are.
- User packs (a person's preferences, conventions and projects baked into a pack): userpack_start takes squares of
  text the user gives you (type about | answer_style | coding | projects | custom with a title) and extracts short
  statements; the user ticks them in the User pack tab, then userpack_build builds it (stack of small packs, or one
  combined pack). userpack_status shows where it is; userpack_check compares base and pack on its probe questions
  (curve: true also at several strengths, to choose the strength). Uploads and folders are added in the User pack
  tab, not by you. Do not put secrets or other people's private details into a user pack.
Be brief and concrete. Never invent documents, page numbers or pack names: look them up with the tools."""

TOOLS = [
    {"type": "function", "function": {"name": "list_documents", "description": "Documents in the library (id, name, pages, tagged regions).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "document_info", "description": "A document's details and the text of some pages (default: the first 4, where the contents usually are).",
                                      "parameters": {"type": "object", "properties": {"doc": {"type": "string"}, "pages": {"type": "array", "items": {"type": "integer"}}},
                                                     "required": ["doc"]}}},
    {"type": "function", "function": {"name": "search_document", "description": "Pages of a document that mention a term (to find a chapter).",
                                      "parameters": {"type": "object", "properties": {"doc": {"type": "string"}, "term": {"type": "string"}}, "required": ["doc", "term"]}}},
    {"type": "function", "function": {"name": "list_curated", "description": "Curated data sets (training data / notes imported from other apps).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_playbooks", "description": "Playbooks available (built-in and built from texts).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "propose_plan", "description": "Propose the plan to the user (shown as a card with a Run button). Does not start anything.",
                                      "parameters": {"type": "object", "properties": {
                                          "name": {"type": "string", "description": "short pack name"},
                                          "goal": {"type": "string", "description": "what the pack should know or do, in the user's words"},
                                          "sources": {"type": "array", "items": {"type": "object", "properties": {"doc": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}, "required": ["doc"]}},
                                          "curated": {"type": "array", "items": {"type": "string"}},
                                          "mode": {"type": "string", "enum": ["facts", "reasoning", "both", "mining"]},
                                          "profile": {"type": "string", "enum": ["", "writer", "drafter", "reader", "reviewer"],
                                                      "description": "extra mining so the pack can perform like the expert"},
                                          "miners": {"type": "array", "items": {"type": "string"}},
                                          "playbook": {"type": "string"},
                                          "preset": {"type": "string", "enum": ["generic", "report", "manual", "field-guide", "everything"]},
                                          "fields": {"type": "array", "items": {"type": "string"}},
                                          "train": {"type": "boolean"}, "evaluate": {"type": "boolean"},
                                          "epochs": {"type": "integer"}, "max_len": {"type": "integer"}},
                                          "required": ["name", "mode"]}}},
    {"type": "function", "function": {"name": "topic_status", "description": "A topic's regions and what will be mined, and why nothing will be when that is the case. Without topic: all topics, briefly.",
                                      "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "list_packs", "description": "Packs in the library: kind, personal, fit with the served model, active, strength and routing (read only), last evaluation; and the engine in use.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "userpack_status", "description": "User packs (without name: all of them), or one: squares, statements found / ticked, build and check results.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "userpack_start", "description": "Start a user pack from squares of text the user gave: extracts statements for the user to tick in the User pack tab. Runs as a job.",
                                      "parameters": {"type": "object", "properties": {
                                          "name": {"type": "string"},
                                          "squares": {"type": "array", "items": {"type": "object", "properties": {
                                              "type": {"type": "string", "enum": ["about", "answer_style", "coding", "projects", "custom"]},
                                              "title": {"type": "string"}, "text": {"type": "string"}}, "required": ["type", "text"]}},
                                          "mode": {"type": "string", "enum": ["stack", "merged"]}}, "required": ["name", "squares"]}}},
    {"type": "function", "function": {"name": "userpack_build", "description": "Build (generate and train) a user pack once its statements are ticked. Runs as a job.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "userpack_check", "description": "Compare base and pack on the user pack's probe questions (curve: also at strengths 25-125%). Runs as a job.",
                                      "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "curve": {"type": "boolean"}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "run_plan", "description": "Start the proposed plan (only after the user agreed).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "pipeline_status", "description": "Progress of the running or last pipeline.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "pause_pipeline", "description": "Pause the running pipeline (resumable).",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "resume_pipeline", "description": "Resume a paused or failed pipeline.",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "cancel_pipeline", "description": "Cancel the running pipeline.",
                                      "parameters": {"type": "object", "properties": {}}}},
]


def adir():
    d = store.data_dir() / "assistant"
    d.mkdir(parents=True, exist_ok=True)
    return d


def conv(cid: str) -> dict:
    c = store.rj(adir() / f"{cid}.json")
    if not c:
        raise KeyError(f"no conversation {cid}")
    return c


def save(c: dict) -> dict:
    c["updated"] = time.strftime("%Y-%m-%d %H:%M")
    store.wj(adir() / f"{c['id']}.json", c)
    return c


def new(title: str = "") -> dict:
    return save({"id": store.new_id(title or "chat"), "title": title or "New chat", "messages": [], "plan": None, "pipeline": None,
                 "created": time.strftime("%Y-%m-%d %H:%M"), "tools": True})


def listing() -> list[dict]:
    return sorted(({k: v for k, v in (store.rj(p) or {}).items() if k != "messages"} for p in adir().glob("*.json")),
                  key=lambda c: c.get("updated", ""), reverse=True)


def delete(cid: str) -> None:
    (adir() / f"{cid}.json").unlink(missing_ok=True)


# ---------------------------------------------------------------- tools
def _pipeline_view(c: dict) -> dict:
    if not c.get("pipeline"):
        return {"pipeline": None}
    p = pipelines.get(c["pipeline"])
    jid = p.get("job")
    j = jobs.get(jid) if jid else None
    return {"pipeline": p["id"], "status": p["status"], "steps": [{k: s.get(k) for k in ("name", "status", "error", "result")} for s in p["steps"]],
            "job": jid, "progress": (j or {}).get("progress"), "last": ((j or {}).get("log") or [""])[-1]}


def call_tool(c: dict, name: str, args: dict) -> dict:
    if name == "list_documents":
        return {"documents": [{"id": d["id"], "name": d["name"], "pages": d["pages"],
                               "regions": [{"name": r["name"], "start": r["start"], "end": r["end"], "type": store.type_label(r["type"]),
                                            "role": r["role"], "mined": r["role"] == "train" and r["type"] not in store.CONTEXT_TYPES}
                                           for r in d.get("regions", [])],
                               "scanned_pages": len(d.get("needs_ocr", []))} for d in store.list_docs()]}
    if name == "document_info":
        m = store.doc_meta(args["doc"])
        pages = [p for p in (args.get("pages") or [1, 2, 3, 4]) if 1 <= int(p) <= m["pages"]][:8]
        return {"id": m["id"], "name": m["name"], "pages": m["pages"], "regions": store.regions(m["id"]),
                "text": {str(p): store.page_text(m["id"], int(p))[:2500] for p in pages}}
    if name == "search_document":
        term = args["term"].lower()
        m = store.doc_meta(args["doc"])
        hits = [p for p in range(1, m["pages"] + 1) if term in store.page_text(m["id"], p).lower()]
        return {"pages": hits[:60], "count": len(hits)}
    if name == "list_curated":
        return {"curated": [{"id": s["id"], "name": s["name"], "records": s["units"], "kinds": s.get("kinds")} for s in curated.list_sets()]}
    if name == "list_playbooks":
        return {"playbooks": [{"ref": p["ref"], "name": f"{p['topic_name']} · {p['which']}", "checks": p["checks"]} for p in mine.all_playbooks()]}
    if name == "topic_status":
        if not args.get("topic"):
            return {"topics": [{"id": t["id"], "name": t["name"], "regions": len(t.get("regions", [])),
                                "train_regions": len(store.mined_regions(t)), "done": [k for k, v in t["status"].items() if v]}
                               for t in store.list_topics()]}
        t = store.topic(args["topic"])
        mined = store.mined_regions(t)
        return {"id": t["id"], "name": t["name"], "follows_whole_documents": bool(t.get("whole_docs")),
                "regions": [{"doc": d, "name": r["name"], "type": store.type_label(r["type"]), "role": r["role"],
                             "pages": f"{r['start']}-{r['end']}", "mined": r["role"] == "train" and r["type"] not in store.CONTEXT_TYPES}
                            for d, r in store.topic_regions(t)],
                "train_regions": len(mined), "heldout_regions": len(store.mined_regions(t, ("heldout",))),
                "problem": "" if mined else store.why_no_regions(t), "done": [k for k, v in store.topic_status(t["id"]).items() if v]}
    if name == "list_packs":
        from packsmith.userpack import agent
        from packsmith.system.config import SETTINGS
        cards = {p["name"]: p.get("card") or {} for p in store.list_packs()}
        out = []
        for p in agent.packs():
            ev = (cards.get(p["name"], {}).get("evaluations") or [])[-1:]
            out.append({**{k: p[k] for k in ("name", "kind", "personal", "compat", "active", "dials")},
                        "last_evaluation": ev[0] if ev else None})
        return {"engine": SETTINGS.engine, "packs": out,
                "note": "strength and routing are set by the user in Workbench → Packs"}
    if name in ("userpack_status", "userpack_start", "userpack_build", "userpack_check"):
        from packsmith.userpack import core as up
        if name == "userpack_status":
            if not args.get("name"):
                return {"user_packs": up.list_all()}
            st = up.status(args["name"])
            b = st["build"]
            return {"squares": [{"id": s["id"], "type": s["type"], "title": s.get("title")} for s in st["spec"]["sections"]],
                    "mode": st["spec"].get("mode"), "statements": st["statements"],
                    "build": {k: b.get(k) for k in ("status", "pack", "path", "updated", "validation", "tuned") if k in b}}
        if name == "userpack_start":
            secs = [{"type": q.get("type") or "custom", "title": q.get("title") or "", "text": q.get("text") or ""}
                    for q in args.get("squares") or [] if (q.get("text") or "").strip()]
            if not secs:
                return {"error": "give at least one square with text"}
            j = jobs.start("userpack-create", up.create, args["name"], name=args["name"], sections=secs,
                           mode=args.get("mode") if args.get("mode") in ("stack", "merged") else "stack", confirm="review")
            return {"job": j.id, "next": "when the job is done, the user ticks the statements in the User pack tab; then userpack_build"}
        if name == "userpack_build":
            st = up.status(args["name"])
            pending = sum(v["pending"] for v in st["statements"].values())
            if not st["statements"]:
                return {"error": "no statements yet: run userpack_start (or Create pack in the User pack tab) first"}
            j = jobs.start("userpack-build", up.build, args["name"], name=args["name"])
            return {"job": j.id, **({"note": f"{pending} statement(s) were not ticked either way and are left out"} if pending else {})}
        j = jobs.start("userpack-validate", up.validate, args["name"], name=args["name"], curve=up.CURVE if args.get("curve") else None)
        return {"job": j.id}
    if name == "propose_plan":
        c["plan"] = pipelines.normalise(args)
        return {"ok": True, "plan": c["plan"], "note": "shown to the user with a Run button"}
    if name == "run_plan":
        if not c.get("plan"):
            return {"error": "propose a plan first"}
        p = pipelines.create(c["plan"])
        c["pipeline"] = p["id"]
        j = jobs.start("pipeline", pipelines.run, p["name"], pid=p["id"])
        return {"started": p["id"], "job": j.id, "steps": [s["name"] for s in p["steps"] if s["status"] != "skipped"]}
    if name == "pipeline_status":
        return _pipeline_view(c)
    if name in ("pause_pipeline", "cancel_pipeline", "resume_pipeline"):
        v = _pipeline_view(c)
        if not v.get("pipeline"):
            return {"error": "no pipeline"}
        if name == "resume_pipeline":
            if v["status"] == "running":
                return {"error": "it is running"}
            j = jobs.start("pipeline", pipelines.run, f"{v['pipeline']} (resumed)", pid=v["pipeline"])
            return {"resumed": v["pipeline"], "job": j.id}
        if not v.get("job"):
            return {"error": "not running"}
        (jobs.pause if name == "pause_pipeline" else jobs.cancel)(v["job"])
        return {"ok": True, "action": name.split("_")[0]}
    return {"error": f"unknown tool {name}"}


# ---------------------------------------------------------------- a turn
_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def message(cid: str, text: str, docs: list[str] | None = None, model: str | None = None, action: str = "") -> dict:
    """Add the user's message, let the model answer (calling tools as needed), return the new messages and state."""
    c = conv(cid)
    n0 = len(c["messages"])
    if action in ("pause", "resume", "cancel"):           # the run buttons: no model call needed
        r = call_tool(c, f"{action}_pipeline", {})
        c["messages"].append({"role": "tool_note", "content": json.dumps({"tool": f"{action}_pipeline", "result": r})})
        save(c)
        return {"messages": c["messages"][n0:], "plan": c.get("plan"), "pipeline": _pipeline_view(c), "title": c["title"]}
    if action == "run":                                    # the Run button on the plan card: start, no model call
        r = call_tool(c, "run_plan", {})
        c["messages"].append({"role": "user", "content": "Run the plan."})
        c["messages"].append({"role": "tool_note", "content": json.dumps({"tool": "run_plan", "result": r})})
        steps = ", ".join(r.get("steps") or [])
        c["messages"].append({"role": "assistant", "content": (f"Started: {steps}. Progress shows on the left; ask me for the status, "
                                                               "or pause, resume or cancel at any time."
                                                               + (" Training will unload the LM Studio models (me included) and load "
                                                                  "them again afterwards." if "train" in steps else ""))
                              if not r.get("error") else f"Could not start: {r['error']}"})
        save(c)
        return {"messages": c["messages"][n0:], "plan": c.get("plan"), "pipeline": _pipeline_view(c), "title": c["title"]}
    else:
        note = ""
        if docs:
            names = []
            for d in docs:
                try:
                    names.append(f"{store.doc_meta(d)['name']} (id {d}, {store.doc_meta(d)['pages']} pages)")
                except KeyError:
                    pass
            note = f"\n[attached documents: {'; '.join(names)}]"
        c["messages"].append({"role": "user", "content": text + note})
        if c["title"] == "New chat" and text:
            c["title"] = text[:60]
    msgs = [{"role": "system", "content": SYSTEM + ("" if c.get("tools", True) else NO_TOOLS)}]
    for m in c["messages"]:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "tool_note":
            msgs.append({"role": "user", "content": f"[result of an action: {m['content'][:3000]}]"})
    for _ in range(8):
        try:
            out = llm.chat_raw(msgs, TOOLS if c.get("tools", True) else None, model=model)
        except llm.LLMError as e:
            if c.get("tools", True) and "tool" in str(e).lower():
                c["tools"] = False                         # this model cannot call tools: use the JSON fallback
                msgs[0]["content"] = SYSTEM + NO_TOOLS
                continue
            raise
        calls = out.get("tool_calls") or []
        if not calls and not c.get("tools", True):
            m = _JSON.search(out.get("content") or "")
            if m:
                try:
                    j = json.loads(m.group(1))
                    calls = [{"id": "x", "function": {"name": j.get("tool", ""), "arguments": json.dumps(j.get("args") or {})}}]
                    out["content"] = _JSON.sub("", out["content"]).strip()
                except json.JSONDecodeError:
                    calls = []
        if not calls:
            c["messages"].append({"role": "assistant", "content": out.get("content") or "(no answer)"})
            break
        if out.get("content"):
            c["messages"].append({"role": "assistant", "content": out["content"]})
        msgs.append({"role": "assistant", "content": out.get("content") or "", **({"tool_calls": calls} if c.get("tools", True) else {})})
        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                res = call_tool(c, fn, args)
            except Exception as e:                         # noqa: BLE001 - the model sees the error and can recover
                res = {"error": f"{type(e).__name__}: {e}"}
            c["messages"].append({"role": "tool_note", "content": json.dumps({"tool": fn, "result": res}, default=str)[:6000]})
            if c.get("tools", True):
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", fn), "content": json.dumps(res, default=str)[:6000]})
            else:
                msgs.append({"role": "user", "content": f"[result of {fn}: {json.dumps(res, default=str)[:3000]}]"})
    save(c)
    return {"messages": c["messages"][n0:], "plan": c.get("plan"), "pipeline": _pipeline_view(c), "title": c["title"]}


NO_TOOLS = """

You cannot call tools directly. To use one, end your reply with a JSON block like
```json
{"tool": "document_info", "args": {"doc": "<id>"}}
```
(one tool per reply; the result comes back as the next message). Tools: list_documents, document_info(doc, pages),
search_document(doc, term), list_curated, list_playbooks, topic_status(topic), list_packs, userpack_status(name),
userpack_start(name, squares[{type,title,text}], mode), userpack_build(name), userpack_check(name, curve),
propose_plan(name, goal, sources[{doc,start,end}], curated, mode, profile, playbook, preset, fields, train, evaluate,
epochs, max_len), run_plan, pipeline_status, pause_pipeline, resume_pipeline, cancel_pipeline."""
