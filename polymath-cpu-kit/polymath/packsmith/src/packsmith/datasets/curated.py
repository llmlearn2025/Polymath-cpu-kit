"""Curated data: training data or notes made in other apps (Easy Dataset exports, Obsidian notes, dataset
generators, spreadsheets …) imported losslessly as units. Nothing is summarised or dropped here: every record, row,
bullet, paragraph and front-matter field becomes a unit with an id and its source location.

Formats (auto-detected, also inside a .zip):
  JSON / JSONL   Alpaca {instruction, input, output}, ShareGPT {conversations|messages}, Q&A {question, answer,
                 cot?, chunk?, label?/tags?}, or any other records (every scalar field kept)
  CSV / TSV      one unit per row, the header as field names
  HTML           headings → subject path; list items → facts; paragraphs → passages; tables → rows; dt/dd → terms
  YAML / XML / XLSX   records (lists of mappings, repeated elements, sheet rows)
  PDF / DOCX / EPUB   their text, read as notes (headings, lists, paragraphs)
  anything else that is text: read as Markdown
  Markdown       Obsidian notes (and the dictionaries made in Obsidian): front matter → fields; headings → subject path; bullets → facts; paragraphs →
                 passages; tables → rows; [[links]] and #tags recorded
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

from packsmith.packs import store

try:
    import yaml
except ImportError:                                   # front matter falls back to "key: value" lines
    yaml = None

DATA_EXTS = (".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".xlsx", ".md", ".markdown", ".zip")
DOC_EXTS = (".html", ".htm", ".xhtml", ".txt", ".pdf", ".docx", ".epub", ".rtf", ".org", ".rst")
EXTS = DATA_EXTS + DOC_EXTS
SUBJECT_KEYS = ("subject", "title", "name", "topic", "label", "heading", "term", "section", "id")
Q_KEYS = ("question", "instruction", "prompt", "query", "q")
A_KEYS = ("answer", "output", "response", "completion", "a")
CTX_KEYS = ("chunk", "context", "input", "source_text", "passage", "evidence")
COT_KEYS = ("cot", "reasoning", "rationale", "chain_of_thought", "thinking")


def cdir(cid: str) -> Path:
    return store.data_dir() / "curated" / cid


def list_sets() -> list[dict]:
    d = store.data_dir() / "curated"
    d.mkdir(exist_ok=True)
    return [m for m in (store.rj(p / "meta.json") for p in sorted(d.iterdir()) if p.is_dir()) if m]


def meta(cid: str) -> dict:
    m = store.rj(cdir(cid) / "meta.json")
    if not m:
        raise KeyError(f"no curated set {cid}")
    return m


def units(cid: str) -> list[dict]:
    return [json.loads(x) for x in store.rt(cdir(cid) / "units.jsonl").splitlines() if x.strip()]


def delete(cid: str) -> None:
    shutil.rmtree(cdir(cid), ignore_errors=True)


# ---------------------------------------------------------------- helpers
def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return re.sub(r"[ \t]+", " ", str(v)).strip()


def _first(d: dict, keys) -> tuple[str, str]:
    low = {k.lower(): k for k in d}
    for k in keys:
        if k in low and _s(d[low[k]]):
            return low[k], _s(d[low[k]])
    return "", ""


def _unit(src: str, kind: str, subject: str = "", text: str = "", **kw) -> dict:
    return {"src": src, "kind": kind, "subject": subject.strip(), "text": text.strip(), "question": kw.get("question", ""),
            "answer": kw.get("answer", ""), "reasoning": kw.get("reasoning", ""), "context": kw.get("context", ""),
            "fields": kw.get("fields", {}), "tags": kw.get("tags", []), "links": kw.get("links", [])}


# ---------------------------------------------------------------- records (JSON / JSONL / CSV)
def from_record(rec, src: str) -> dict:
    if not isinstance(rec, dict):
        return _unit(src, "fact", "", _s(rec))
    conv = rec.get("conversations") or rec.get("messages")
    if isinstance(conv, list) and conv:                                   # ShareGPT / chat
        turns = [(str(m.get("from") or m.get("role") or ""), _s(m.get("value") or m.get("content"))) for m in conv if isinstance(m, dict)]
        sys_ = " ".join(v for r, v in turns if r in ("system",))
        qs = [v for r, v in turns if r in ("human", "user")]
        an = [v for r, v in turns if r in ("gpt", "assistant", "model")]
        if qs and an:
            return _unit(src, "qa", _first(rec, SUBJECT_KEYS)[1], question="\n".join(qs), answer="\n".join(an), context=sys_,
                         tags=_tags(rec))
    qk, q = _first(rec, Q_KEYS)
    ak, a = _first(rec, A_KEYS)
    if q and a:                                                           # Alpaca / Q&A / Easy Dataset
        ck, ctx = _first(rec, CTX_KEYS)
        _, cot = _first(rec, COT_KEYS)
        if ck.lower() == "input" and ctx and len(ctx) < 400:              # Alpaca input is part of the question
            q, ctx = f"{q}\n{ctx}", ""
        used = {qk, ak, ck} | {k for k in rec if k.lower() in COT_KEYS}
        extra = {k: _s(v) for k, v in rec.items() if k not in used and _s(v) and k.lower() not in ("system",)}
        return _unit(src, "qa", _first(rec, SUBJECT_KEYS)[1] or extra.get("label", ""), question=q, answer=a, reasoning=cot,
                     context=ctx, fields=extra, tags=_tags(rec))
    fields = {k: _s(v) for k, v in rec.items() if _s(v)}
    sk, subj = _first(rec, SUBJECT_KEYS)
    body = {k: v for k, v in fields.items() if k != sk}
    if len(body) == 1:                                                    # {subject, fact}
        return _unit(src, "fact", subj, next(iter(body.values())), fields=fields, tags=_tags(rec))
    return _unit(src, "entity", subj, "; ".join(f"{k}: {v}" for k, v in body.items()), fields=fields, tags=_tags(rec))


def _tags(rec: dict) -> list[str]:
    t = rec.get("tags") or rec.get("label") or rec.get("labels") or []
    return [str(x) for x in (t if isinstance(t, list) else re.split(r"[,;]\s*", str(t)) if t else [])]


def parse_json(text: str, name: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        rows = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [data]) \
            if isinstance(data, dict) else [data]
        return [from_record(r, f"{name}#{i + 1}") for i, r in enumerate(rows)]
    except json.JSONDecodeError:                                          # JSON Lines
        out = []
        for i, line in enumerate(text.splitlines()):
            if line.strip():
                try:
                    out.append(from_record(json.loads(line), f"{name}:{i + 1}"))
                except json.JSONDecodeError:
                    out.append(_unit(f"{name}:{i + 1}", "fact", "", line.strip(), fields={"_unparsed": "1"}))
        return out


def parse_csv(text: str, name: str, delim: str) -> list[dict]:
    rd = csv.DictReader(io.StringIO(text), delimiter=delim)
    return [from_record({k: v for k, v in r.items() if k}, f"{name}:{i + 2}") for i, r in enumerate(rd)]


# ---------------------------------------------------------------- Markdown (Obsidian)
_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")
_TAG = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]*)")


def _clean_md(s: str, links: list, tags: list) -> str:
    links += [m.group(1).strip() for m in _LINK.finditer(s)]
    tags += _TAG.findall(s)
    s = _LINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), s)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)                           # embedded images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)                        # [text](url)
    s = re.sub(r"==([^=]+)==|\*\*([^*]+)\*\*|__([^_]+)__", lambda m: next(g for g in m.groups() if g), s)
    s = re.sub(r"^>\s*\[![^\]]+\][+-]?\s*", "", s)                        # callout header
    return re.sub(r"\s+", " ", s).strip()


def _term(raw: str) -> str:
    """A dictionary line names its term first: **Term**: …, **Term** — …, [[Term]]: …"""
    m = re.match(r"^\s*(?:\*\*|__)(.{1,80}?)(?:\*\*|__)\s*(?:[:—–-]|\()", raw) or \
        re.match(r"^\s*\[\[([^\]|]{1,80})(?:\|[^\]]*)?\]\]\s*[:—–-]", raw)
    return m.group(1).strip().rstrip(":") if m else ""


def _front(text: str) -> tuple[dict, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end():]
    if yaml:
        try:
            d = yaml.safe_load(raw) or {}
            return (d if isinstance(d, dict) else {}), body
        except yaml.YAMLError:
            pass
    d = {}
    for line in raw.splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d, body


def parse_md(text: str, name: str) -> list[dict]:
    title = Path(name).stem
    fm, body = _front(text)
    out = []
    if fm:
        fields = {k: _s(v) for k, v in fm.items() if _s(v)}
        out.append(_unit(f"{name}#front-matter", "entity", _s(fm.get("title")) or title,
                         "; ".join(f"{k}: {v}" for k, v in fields.items()), fields=fields, tags=_tags(fm)))
    path = [title]
    para, table, line_no = [], [], 0

    def subject():
        return " › ".join(dict.fromkeys(p for p in path if p))

    def flush_para():
        if para:
            links, tags = [], []
            t = _clean_md(" ".join(para), links, tags)
            term = _term(para[0])
            if t:
                out.append(_unit(f"{name}:{line_no - len(para) + 1}", "fact" if term else "passage",
                                 f"{subject()} › {term}" if term else subject(), t, links=links, tags=tags))
            para.clear()

    def flush_table():
        rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", r)]
        if rows:
            head = rows[0]
            for i, r in enumerate(rows[1:]):
                links, tags = [], []
                cells = [_clean_md(c, links, tags) for c in r]
                f = {h or f"col{j + 1}": c for j, (h, c) in enumerate(zip(head, cells)) if c}
                out.append(_unit(f"{name}:table{len(out)}:{i + 1}", "entity", f"{subject()} › {cells[0]}" if cells else subject(),
                                 "; ".join(f"{k}: {v}" for k, v in f.items()), fields=f, links=links, tags=tags))
        table.clear()

    in_code = False
    stack: list[tuple[int, str]] = []                                     # nested bullets keep their parent
    for line_no, line in enumerate(body.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            flush_para()
            continue
        if in_code:
            para.append(line)
            continue
        if line.lstrip().startswith("|"):
            flush_para()
            table.append(line)
            continue
        if table:
            flush_table()
        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            flush_para()
            lvl = len(h.group(1))
            path[:] = path[:lvl] if lvl > 1 else path[:1]
            while len(path) < lvl:
                path.append("")
            path.append(_clean_md(h.group(2), [], []))
            stack.clear()
            continue
        f = re.match(r"^\s*(?:[-*+]\s+)?([^:\[\]#|]{1,60}?)::\s*(.+)$", line)        # Dataview inline field key:: value
        if f:
            flush_para()
            links, tags = [], []
            k, v = f.group(1).strip(), _clean_md(f.group(2), links, tags)
            out.append(_unit(f"{name}:{line_no}", "entity", subject(), f"{k}: {v}", fields={k: v}, links=links, tags=tags))
            continue
        b = re.match(r"^(\s*)(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s*)?(.*)", line)
        if b:
            flush_para()
            ind = len(b.group(1).replace("\t", "    "))
            links, tags = [], []
            t = _clean_md(b.group(2), links, tags)
            while stack and stack[-1][0] >= ind:
                stack.pop()
            parent = stack[-1][1] if stack else ""
            stack.append((ind, t))
            if t:
                term = _term(b.group(2))
                out.append(_unit(f"{name}:{line_no}", "fact", f"{subject()} › {term}" if term else subject(),
                                 f"{parent} — {t}" if parent else t, links=links, tags=tags))
            continue
        if not line.strip():
            flush_para()
            stack.clear()
            continue
        if para and _term(line):                     # a dictionary: each term line starts its own entry
            flush_para()
        para.append(line.strip())
    flush_para()
    if table:
        flush_table()
    return out


# ---------------------------------------------------------------- HTML
def parse_html(text: str, name: str) -> list[dict]:
    from packsmith.mining import htmlconv
    root = htmlconv.parse(re.sub(r"(?is)<(script|style|nav|head)\b.*?</\1>", " ", text))
    title_m = re.search(r"(?is)<title>(.*?)</title>", text)
    path = [_s(title_m.group(1)) if title_m else Path(name).stem]
    out, n = [], [0]

    def subject():
        return " › ".join(dict.fromkeys(p for p in path if p))

    def add(kind, txt, **kw):
        txt = re.sub(r"\s+", " ", txt or "").strip()
        if txt:
            n[0] += 1
            out.append(_unit(f"{name}#{n[0]}", kind, kw.pop("subject", subject()), txt, **kw))

    def walk(node):
        for c in node.children:
            if isinstance(c, str):
                if c.strip() and node.tag in ("root", "body", "div", "section", "article", "main"):
                    add("passage", c)
                continue
            tag = c.tag
            if re.fullmatch(r"h[1-6]", tag):
                lvl = int(tag[1])
                del path[lvl:]
                while len(path) < lvl:
                    path.append("")
                path.append(c.text())
            elif tag in ("p", "blockquote", "pre"):
                add("passage", c.text())
            elif tag == "li":
                own = c.own_text()
                add("fact", own)
                for sub in c.children:
                    if not isinstance(sub, str) and sub.tag in ("ul", "ol"):
                        for li in sub.find_all(tag="li"):
                            add("fact", f"{own} — {li.own_text()}")
            elif tag == "table":
                rows = c.find_all(tag="tr")
                head = [x.text() for x in rows[0].children if not isinstance(x, str) and x.tag in ("th", "td")] if rows else []
                for r in rows[1:]:
                    cells = [x.text() for x in r.children if not isinstance(x, str) and x.tag in ("th", "td")]
                    f = {h or f"col{j + 1}": v for j, (h, v) in enumerate(zip(head, cells)) if v}
                    if f:
                        add("entity", "; ".join(f"{k}: {v}" for k, v in f.items()),
                            subject=f"{subject()} › {cells[0]}" if cells else subject(), fields=f)
            elif tag == "dl":
                term = ""
                for x in c.children:
                    if isinstance(x, str):
                        continue
                    if x.tag == "dt":
                        term = x.text()
                    elif x.tag == "dd":
                        add("fact", f"{term}: {x.text()}" if term else x.text(), subject=term or subject())
            else:
                walk(c)
    walk(root)
    return out


# ---------------------------------------------------------------- import
def parse_yaml(text: str, name: str) -> list[dict]:
    if not yaml:
        return parse_md(text, name)
    docs = [x for x in yaml.safe_load_all(text) if x is not None]
    rows = []
    for x in docs:
        if isinstance(x, list):
            rows += x
        elif isinstance(x, dict) and any(isinstance(v, list) and v and isinstance(v[0], dict) for v in x.values()):
            rows += next(v for v in x.values() if isinstance(v, list) and v and isinstance(v[0], dict))
        elif isinstance(x, dict) and all(not isinstance(v, (dict, list)) for v in x.values()):
            rows.append(x)
        elif isinstance(x, dict):                       # a dictionary: term → definition / fields
            rows += [{"term": k, **(v if isinstance(v, dict) else {"definition": v})} for k, v in x.items()]
    return [from_record(r, f"{name}#{i + 1}") for i, r in enumerate(rows)]


def parse_xml(text: str, name: str) -> list[dict]:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return parse_md(re.sub(r"<[^>]+>", "\n", text), name)
    kids = list(root)
    tags = Counter(k.tag for k in kids)
    rec_tag = tags.most_common(1)[0][0] if tags else None
    out = []
    for i, k in enumerate(x for x in kids if x.tag == rec_tag):
        rec = {**k.attrib, **{c.tag: " ".join(c.itertext()).strip() for c in k}}
        if not list(k) and (k.text or "").strip():
            rec[k.tag] = k.text.strip()
        out.append(from_record(rec, f"{name}#{i + 1}"))
    return out


def parse_xlsx(data: bytes, name: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True) if any(c not in (None, "") for c in r)]
        if not rows:
            continue
        head = [str(c) if c not in (None, "") else f"col{j + 1}" for j, c in enumerate(rows[0])]
        for i, r in enumerate(rows[1:]):
            out.append(from_record({h: c for h, c in zip(head, r) if c not in (None, "")}, f"{name}!{ws.title}:{i + 2}"))
    return out


def parse_document(name: str, data: bytes) -> list[dict]:
    """PDF / Word / EPUB / RTF: their text through the document reader, then read as notes."""
    import tempfile
    from packsmith.mining import ingest
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / Path(name).name
        p.write_bytes(data)
        try:
            pages = ingest.read_text_pages(p)
        except Exception:                               # noqa: BLE001 - an unreadable file is reported as one unit
            return [_unit(name, "passage", Path(name).stem, "", fields={"_unreadable": "1"})]
    out = []
    for i, txt in enumerate(pages):
        for u in parse_md(txt, f"{name} p.{i + 1}"):
            u["subject"] = u["subject"].replace(f"{Path(name).stem} p.{i + 1}", Path(name).stem)
            out.append(u)
    return out


def parse_file(name: str, data: bytes) -> list[dict]:
    low = name.lower()
    if low.endswith(".zip"):
        out = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for n in sorted(z.namelist()):
                if n.endswith("/") or "/." in "/" + n or "__MACOSX" in n or n.lower().endswith((".png", ".jpg", ".jpeg", ".gif",
                                                                                                  ".webp", ".mp3", ".mp4", ".zip")):
                    continue
                out += parse_file(n, z.read(n))
        return out
    if low.endswith(".xlsx"):
        return parse_xlsx(data, name)
    if low.endswith((".pdf", ".docx", ".epub", ".rtf")):
        return parse_document(name, data)
    text = data.decode("utf-8-sig", errors="replace")
    if low.endswith((".yaml", ".yml")):
        return parse_yaml(text, name)
    if low.endswith(".xml"):
        return parse_xml(text, name)
    if low.endswith((".json", ".jsonl", ".ndjson")):
        return parse_json(text, name)
    if low.endswith(".csv"):
        return parse_csv(text, name, ",")
    if low.endswith(".tsv"):
        return parse_csv(text, name, "\t")
    if low.endswith((".html", ".htm", ".xhtml")):
        return parse_html(text, name)
    return parse_md(text, name)                                           # .md / .markdown / .txt


def import_file(path: Path, name: str = "") -> dict:
    path = Path(path)
    us = parse_file(path.name, path.read_bytes())
    cid = store.new_id(name or path.stem)
    d = cdir(cid)
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, d / ("source" + path.suffix.lower()))
    for i, u in enumerate(us):
        u["id"] = f"{cid}:u{i + 1}"
    store.wt(d / "units.jsonl", "\n".join(json.dumps(u, ensure_ascii=False) for u in us) + ("\n" if us else ""))
    kinds = {}
    for u in us:
        kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    m = {"id": cid, "name": name or path.stem, "file": path.name, "created": time.strftime("%Y-%m-%d %H:%M"),
         "units": len(us), "kinds": kinds}
    store.wj(d / "meta.json", m)
    return m


def is_curated(filename: str) -> bool:
    """Default reading of a file in the quick path (the user can switch): data/notes formats are curated; PDF, Word,
    EPUB, HTML and plain text are documents to read page by page."""
    return filename.lower().endswith(DATA_EXTS)
