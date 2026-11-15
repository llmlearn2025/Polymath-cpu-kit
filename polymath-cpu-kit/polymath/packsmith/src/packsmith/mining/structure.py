"""Contents + index (user-tagged regions) → skeleton playbook."""
from __future__ import annotations

from packsmith.mining import htmlconv
from packsmith.llm import llm
from packsmith.datasets import prompts
from packsmith.packs import store
from packsmith.system.config import SETTINGS


def structure_text(t: dict, budget: int | None = None) -> tuple[str, str]:
    """The contents and index pages of the topic's own regions (not every region of its documents)."""
    contents, index = [], []
    for doc, r in sorted(store.topic_regions(t), key=lambda x: (x[0], x[1]["start"])):
        if r["role"] == "exclude" or r["type"] not in store.CONTEXT_TYPES:
            continue
        name = store.doc_meta(doc)["name"]
        txt = "\n".join(store.page_text(doc, p) for p in range(r["start"], r["end"] + 1))
        if r["type"] == "contents":
            contents.append(f"[{doc}: {name}]\n{txt}")
        else:
            index.append(f"[{doc}: {name} — {r['type']}]\n{txt}")
    half = (budget or llm.data_chars()) // 2
    return "\n\n".join(contents)[:half], "\n\n".join(index)[:half]


def run(topic_id: str, model: str | None = None) -> dict:
    t = store.topic(topic_id)
    plan = store.rt(store.topic_dir(topic_id) / "plan.html")
    contents, index = structure_text(t, llm.budget_chars(model, 8000, len(plan) + 3000))
    if not contents and not index:
        # optional: many texts have neither. The playbook is then built from the chapters alone (Map → Build).
        return {"checks": 0, "skipped": "this topic has no table of contents or book index, so there is no skeleton; "
                "that is fine: Map regions and Build playbook work from the chapters alone (to use one, tag those "
                "pages in the Library and tick them in Topics)"}
    out = llm.chat(prompts.structure_prompt(plan, contents, index), model=model, max_tokens=8000)
    ok, why = htmlconv.validate(out, "playbook")
    if not ok:
        raise ValueError(f"structure mining returned no checks: {why}")
    pb = htmlconv.parse_playbook(out)
    for c in pb["checks"]:
        c["origin"] = "structure"
    html = htmlconv.render_playbook(pb)
    store.wt(store.topic_dir(topic_id) / "skeleton.html", html)
    return {"checks": len(pb["checks"])}
