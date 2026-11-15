"""Documents → numbered pages. PDF pages keep their order in the file (scan position); other formats
are split into pseudo-pages at paragraph boundaries."""
from __future__ import annotations

import html
import re
import shutil
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path

from packsmith.packs import store

PSEUDO_PAGE_CHARS = 3000
KINDS = {".pdf": "pdf", ".docx": "docx", ".html": "html", ".htm": "html", ".xhtml": "html", ".epub": "epub",
         ".txt": "text", ".md": "text", ".markdown": "text"}


class _Text(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote", "pre"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3", "h4"):
            self.out.append("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(s: str) -> str:
    p = _Text()
    p.feed(s)
    t = html.unescape("".join(p.out))
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def paginate(text: str, size: int = PSEUDO_PAGE_CHARS) -> list[str]:
    paras, pages, cur = re.split(r"\n\s*\n", text), [], ""
    for para in paras:
        if cur and len(cur) + len(para) > size:
            pages.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
        while len(cur) > size * 1.6:                     # a single huge paragraph
            pages.append(cur[:size])
            cur = cur[size:]
    if cur.strip():
        pages.append(cur.strip())
    return pages or [""]


def _epub_text(path: Path) -> str:
    z = zipfile.ZipFile(path)
    names = z.namelist()
    opf = next((n for n in names if n.endswith(".opf")), None)
    order = []
    if opf:
        o = z.read(opf).decode(errors="replace")
        base = opf.rsplit("/", 1)[0] + "/" if "/" in opf else ""
        items = dict(re.findall(r'<item[^>]*id="([^"]+)"[^>]*href="([^"]+)"', o))
        items.update({i: h for h, i in re.findall(r'<item[^>]*href="([^"]+)"[^>]*id="([^"]+)"', o)})
        for idref in re.findall(r'<itemref[^>]*idref="([^"]+)"', o):
            if idref in items:
                order.append(base + items[idref])
    if not order:
        order = sorted(n for n in names if n.endswith((".xhtml", ".html", ".htm")))
    return "\n\n".join(html_to_text(z.read(n).decode(errors="replace")) for n in order if n in names)


def ingest(src: Path, name: str | None = None, progress=lambda f, msg="": None, scale: float = 1.6) -> dict:
    src = Path(src)
    kind = KINDS.get(src.suffix.lower())
    if not kind:
        raise ValueError(f"unsupported file type {src.suffix}; use {sorted(KINDS)}")
    doc = store.new_id(name or src.stem)
    d = store.doc_dir(doc)
    (d / "pages").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, d / ("source" + src.suffix.lower()))
    meta = {"id": doc, "name": name or src.stem, "file": src.name, "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M"),
            "needs_ocr": [], "ocr_done": [], "calibration": []}
    if kind == "pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(d / "source.pdf"))
        n = len(pdf)
        (d / "img").mkdir(exist_ok=True)
        for i in range(n):
            page = pdf[i]
            text = page.get_textpage().get_text_range() or ""
            store.set_page_text(doc, i + 1, text.replace("\r", ""))
            page.render(scale=scale).to_pil().save(d / "img" / f"{i + 1:04d}.png", optimize=True)
            if len(text.strip()) < 40:
                meta["needs_ocr"].append(i + 1)
            progress((i + 1) / n, f"page {i + 1}/{n}")
        meta["pages"] = n
    else:
        if kind == "docx":
            import docx
            dd = docx.Document(str(src))
            text = "\n\n".join(("# " if p.style.name.lower().startswith("heading") else "") + p.text
                               for p in dd.paragraphs if p.text.strip())
        elif kind == "html":
            text = html_to_text(src.read_text(encoding="utf-8", errors="replace"))
        elif kind == "epub":
            text = _epub_text(src)
        else:
            text = src.read_text(encoding="utf-8", errors="replace")
        pages = paginate(text)
        for i, t in enumerate(pages):
            store.set_page_text(doc, i + 1, t)
        meta["pages"] = len(pages)
        progress(1.0, f"{len(pages)} pseudo-pages")
    store.save_meta(doc, meta)
    store.wj(d / "regions.json", [])
    return meta


def read_text_pages(src: Path) -> list[str]:
    """Text of a PDF (per page), Word, EPUB or RTF file, without storing a document (used by curated import).
    Scanned PDF pages come back empty: import those as documents instead (they need OCR)."""
    src = Path(src)
    ext = src.suffix.lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(src))
        return [(pdf[i].get_textpage().get_text_range() or "").replace("\r", "") for i in range(len(pdf))]
    if ext == ".docx":
        import docx
        dd = docx.Document(str(src))
        lv = lambda p: int(re.sub(r"\D", "", p.style.name) or 1) if p.style.name.lower().startswith("heading") else 0  # noqa: E731
        return ["\n\n".join(("#" * lv(p) + " " if lv(p) else "") + p.text for p in dd.paragraphs if p.text.strip())]
    if ext == ".epub":
        return [_epub_text(src)]
    if ext == ".rtf":
        t = src.read_text(encoding="utf-8", errors="replace")
        return [re.sub(r"\\[a-z]+-?\d* ?|[{}]", "", t)]
    return [src.read_text(encoding="utf-8", errors="replace")]
