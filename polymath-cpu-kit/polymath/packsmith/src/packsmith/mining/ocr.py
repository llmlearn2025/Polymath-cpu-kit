"""OCR through an LM Studio vision model, one page at a time (keeps every text tied to its scan page)."""
from __future__ import annotations

from packsmith.llm import llm
from packsmith.system import par
from packsmith.packs import store

OCR_PROMPT = ("Transcribe this book page exactly as plain text. Keep headings, paragraph breaks, numbered "
              "lists, footnote markers and the printed page number if visible. Do not summarise, do not "
              "translate, do not add commentary. Output only the page text.")


def ocr_pages(doc: str, pages: list[int] | None = None, model: str | None = None, parallel: int | None = None,
              progress=lambda f, m="": None, cancelled=lambda: False) -> dict:
    meta = store.doc_meta(doc)
    pages = pages or meta.get("needs_ocr") or list(range(1, meta["pages"] + 1))
    def one(p):
        img = store.page_image(doc, p)
        if not img:
            raise ValueError(f"page {p} has no image")
        store.set_page_text(doc, p, llm.vision(img.read_bytes(), OCR_PROMPT, model=model))
        return p
    res = par.pmap(one, pages, parallel, lambda k, n, r: progress(k / n, f"{k}/{n} pages" + (f" — {r['_error'][:120]}" if par.failed(r) else "")),
                   cancelled)
    done = [r for r in res if not par.failed(r)]
    failed = [p for p, r in zip(pages, res) if par.failed(r)]
    meta = store.doc_meta(doc)
    meta["ocr_done"] = sorted(set(meta.get("ocr_done", [])) | set(done))
    meta["needs_ocr"] = sorted(set(meta.get("needs_ocr", [])) - set(done))
    store.save_meta(doc, meta)
    return {"done": done, "failed": failed}
