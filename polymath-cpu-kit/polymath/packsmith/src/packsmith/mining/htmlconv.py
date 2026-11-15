"""The HTML conventions (DESIGN.md §5): a tiny DOM, extractors for plans, fragments, playbooks,
cases and examples, and validation."""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

VOID = {"br", "hr", "img", "meta", "link", "input"}


@dataclass
class Node:
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)   # Node or str
    parent: "Node | None" = None

    @property
    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def text(self) -> str:
        out = []
        for c in self.children:
            out.append(c if isinstance(c, str) else c.text())
            if not isinstance(c, str) and c.tag in ("p", "li", "br", "div", "section", "article", "blockquote"):
                out.append(" ")
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def own_text(self) -> str:
        """Text of this node without nested lists (the label of an outline item)."""
        out = []
        for c in self.children:
            if isinstance(c, str):
                out.append(c)
            elif c.tag not in ("ul", "ol"):
                out.append(c.text())
        return re.sub(r"\s+", " ", " ".join(out)).strip()

    def find_all(self, cls: str | None = None, tag: str | None = None) -> list["Node"]:
        out = []
        for c in self.children:
            if isinstance(c, Node):
                if (cls is None or cls in c.classes) and (tag is None or c.tag == tag):
                    out.append(c)
                out.extend(c.find_all(cls, tag))
        return out

    def find(self, cls: str | None = None, tag: str | None = None) -> "Node | None":
        r = self.find_all(cls, tag)
        return r[0] if r else None

    def direct(self, cls: str | None = None, tag: str | None = None) -> list["Node"]:
        return [c for c in self.children if isinstance(c, Node) and (cls is None or cls in c.classes)
                and (tag is None or c.tag == tag)]


class _P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.cur = self.root

    # elements that an unclosed sibling ends (as browsers do): <p>a<p>b, <li>a<li>b, <td>a<td>b
    IMPLICIT = {"p": {"p"}, "li": {"li", "p"}, "td": {"td", "th", "p"}, "th": {"td", "th", "p"}, "tr": {"tr", "td", "th", "p"},
                "article": {"p", "li"}}

    def handle_starttag(self, tag, attrs):
        while self.cur.parent is not None and self.cur.tag in self.IMPLICIT.get(tag, ()):
            self.cur = self.cur.parent
        n = Node(tag, dict(attrs), [], self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, dict(attrs), [], self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def parse(s: str) -> Node:
    p = _P()
    p.feed(s or "")
    p.close()
    return p.root


def esc(s) -> str:
    return _html.escape(str(s or ""), quote=True)


def cites(n: Node) -> list[dict]:
    return [{"doc": c.attrs.get("data-doc", ""), "page": c.attrs.get("data-page", "")} for c in n.find_all(tag="cite")]


def _t(n: Node, cls: str) -> str:
    x = n.find(cls)
    return x.text() if x else ""


# ---------------------------------------------------------------- plans
def parse_plan(s: str) -> dict:
    root = parse(s)
    moves = []
    for m in root.find_all("move"):
        moves.append({"name": m.attrs.get("data-name") or _t(m, "desc")[:40], "pattern": m.attrs.get("data-pattern", ""),
                      "desc": _t(m, "desc"), "extract": _t(m, "extract"),
                      "examples": [{"text": e.text(), **(cites(e)[0] if cites(e) else {})} for e in m.find_all("example")]})
    conf = _t(root, "confidence")
    try:
        conf = float(re.findall(r"[0-9.]+", conf)[0])
    except (IndexError, ValueError):
        conf = None
    return {"about": _t(root, "about"), "moves": moves, "confidence": conf}


def render_plan(p: dict) -> str:
    out = ['<section class="plan">', f'<p class="about">{esc(p.get("about"))}</p>']
    for m in p["moves"]:
        out.append(f'<section class="move" data-name="{esc(m["name"])}" data-pattern="{esc(m.get("pattern"))}">')
        out.append(f'<p class="desc">{esc(m.get("desc"))}</p><p class="extract">{esc(m.get("extract"))}</p>')
        for e in m.get("examples", [])[:5]:
            out.append(f'<blockquote class="example"><cite data-doc="{esc(e.get("doc"))}" data-page="{esc(e.get("page"))}"></cite>{esc(e.get("text"))}</blockquote>')
        out.append("</section>")
    if p.get("confidence") is not None:
        out.append(f'<p class="confidence">{p["confidence"]}</p>')
    out.append("</section>")
    return "\n".join(out)


# ---------------------------------------------------------------- fragments
FRAG_PARTS = ("condition", "outcome", "cue", "exception", "step", "distinction", "fact")


def parse_fragments(s: str, doc: str = "", page: int | str = "") -> list[dict]:
    out = []
    for f in parse(s).find_all("fragment"):
        d = {"move": f.attrs.get("data-move", ""), "doc": f.attrs.get("data-doc") or doc,
             "page": f.attrs.get("data-page") or page, "inferred": bool(f.find("inferred")), "text": f.text()}
        for k in FRAG_PARTS:
            d[k] = [x.text() for x in f.find_all(k)]
        out.append(d)
    return out


def render_fragment(d: dict) -> str:
    parts = "".join(f'<p class="{k}">{esc(v)}</p>' for k in FRAG_PARTS for v in d.get(k, []))
    inf = '<span class="inferred"></span>' if d.get("inferred") else ""
    return (f'<article class="fragment" data-move="{esc(d["move"])}" data-doc="{esc(d["doc"])}" '
            f'data-page="{esc(d["page"])}">{parts}{inf}</article>')


# ---------------------------------------------------------------- playbook
ITEM_KINDS = ("branch", "cue", "exception", "distinction", "fact", "step", "source", "note")


def parse_playbook(s: str) -> dict:
    root = parse(s)
    pb = root.find("playbook") or root
    checks = []
    for i, c in enumerate(pb.find_all("check")):
        items = []
        for li in c.find_all(tag="li"):
            kind = next((k for k in ITEM_KINDS if k in li.classes), None)
            if kind:
                items.append({"kind": kind, "text": li.text(), "attrs": {k: v for k, v in li.attrs.items() if k.startswith("data-")},
                              "cites": cites(li)})
        checks.append({"id": c.attrs.get("data-id") or f"c{i + 1}", "move": c.attrs.get("data-move", ""),
                       "origin": c.attrs.get("data-origin", ""),
                       "label": (c.find("label").text() if c.find("label") else c.own_text())[:300], "items": items,
                       "cites": [x for it in items for x in it["cites"]]})
    return {"topic": pb.attrs.get("data-topic", ""), "title": _t(root, "title"), "checks": checks}


def render_playbook(pb: dict) -> str:
    out = [f'<section class="playbook" data-topic="{esc(pb.get("topic"))}">']
    if pb.get("title"):
        out.append(f'<h2 class="title">{esc(pb["title"])}</h2>')
    out.append('<ol class="checks">')
    for c in pb["checks"]:
        org = f' data-origin="{esc(c["origin"])}"' if c.get("origin") else ""
        out.append(f'<li class="check" data-id="{esc(c["id"])}" data-move="{esc(c.get("move"))}"{org}>'
                   f'<span class="label">{esc(c["label"])}</span><ul>')
        for it in c.get("items", []):
            attrs = "".join(f' {k}="{esc(v)}"' for k, v in it.get("attrs", {}).items())
            cs = "".join(f'<cite data-doc="{esc(x["doc"])}" data-page="{esc(x["page"])}"></cite>' for x in it.get("cites", []))
            out.append(f'<li class="{it["kind"]}"{attrs}>{esc(it["text"])}{cs}</li>')
        out.append("</ul></li>")
    out.append("</ol></section>")
    return "\n".join(out)


# ---------------------------------------------------------------- cases and examples
def parse_cases(s: str) -> list[dict]:
    out = []
    for a in parse(s).find_all("case"):
        steps = a.find("steps")
        out.append({"kind": a.attrs.get("data-kind", "walkthrough"), "check": a.attrs.get("data-check", ""),
                    "pair": a.attrs.get("data-pair", ""),
                    "situation": _t(a, "situation"), "steps": [li.text() for li in (steps.find_all(tag="li") if steps else [])],
                    "conclusion": _t(a, "conclusion")})
    return [c for c in out if c["situation"] and (c["conclusion"] or c["steps"])]


def parse_examples(s: str) -> list[dict]:
    out = []
    for a in parse(s).find_all("example"):
        c = cites(a)
        out.append({"facts": _t(a, "facts"), "conclusion": _t(a, "conclusion"),
                    "doc": c[0]["doc"] if c else "", "page": c[0]["page"] if c else ""})
    return [e for e in out if e["facts"] and e["conclusion"]]


def parse_items(s: str, cls: str) -> list[str]:
    return [n.text() for n in parse(s).find_all(cls) if n.text()]


# ---------------------------------------------------------------- validation
REQUIRED = {"plan": ("plan", "move"), "fragments": ("fragment",), "playbook": ("check",), "cases": ("case",),
            "examples": ("example",), "record": ("record",)}


def validate(s: str, kind: str) -> tuple[bool, str]:
    root = parse(s)
    for cls in REQUIRED.get(kind, ()):
        if not root.find(cls):
            return False, f'missing element with class="{cls}"'
    return True, ""
