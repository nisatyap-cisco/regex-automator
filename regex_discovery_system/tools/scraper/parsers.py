"""Generic HTML extractors — works on any page without site-specific adapters.

Strategies (tried in order):
  1. HTML tables  → extract cell text from <table> rows
  2. List items   → extract <li> / <dd> text
  3. Plain text   → regex-scan the full visible text for the target pattern
"""
from __future__ import annotations

import re
import logging
from typing import Optional

from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_tables(html: str) -> list[list[str]]:
    """Return every non-empty cell value across all <table> elements."""
    tree = HTMLParser(html)
    values: list[list[str]] = []
    for table in tree.css("table"):
        rows: list[str] = []
        for tr in table.css("tr"):
            cells = [_clean(td.text()) for td in tr.css("td, th") if _clean(td.text())]
            rows.extend(cells)
        if rows:
            values.append(rows)
    return values


def extract_lists(html: str) -> list[str]:
    """Extract text from <li>, <dd>, and <dt> elements."""
    tree = HTMLParser(html)
    items: list[str] = []
    for tag in tree.css("li, dd, dt"):
        txt = _clean(tag.text())
        if txt and len(txt) < 200:
            items.append(txt)
    return items


def extract_visible_text(html: str) -> str:
    """Return the full visible text of the page."""
    tree = HTMLParser(html)
    for tag in tree.css("script, style, noscript, svg, head"):
        tag.decompose()
    return _clean(tree.text(separator="\n"))


def extract_by_regex(html: str, pattern: re.Pattern) -> list[str]:
    """Scan the visible page text for all matches of a compiled regex."""
    text = extract_visible_text(html)
    return list({m.group(0) for m in pattern.finditer(text)})


def _looks_like_pdf(content: str) -> bool:
    return content.lstrip()[:20].startswith("%PDF")


def _extract_pdf_text(content: bytes | str) -> str:
    """Try to extract readable text from raw PDF bytes."""
    raw = content.encode("latin-1", errors="replace") if isinstance(content, str) else content
    try:
        import io
        import importlib
        pdfplumber = importlib.import_module("pdfplumber")
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages_text)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        pass
    text_chunks = re.findall(rb"\(([^)]{2,80})\)", raw)
    return " ".join(chunk.decode("latin-1", errors="replace") for chunk in text_chunks)


def extract_all_values(html: str, value_regex: Optional[re.Pattern] = None) -> list[str]:
    """Best-effort extraction: tables first, then lists, then regex scan.

    If value_regex is given, it's used to filter/extract the actual identifier
    values from whatever raw text we pull out.
    Handles PDF content transparently if the input looks like a PDF.
    """
    if _looks_like_pdf(html):
        pdf_text = _extract_pdf_text(html)
        if not pdf_text.strip():
            logger.debug("[parser] PDF text extraction returned empty")
            return []
        logger.info("[parser] extracted %d chars of text from PDF", len(pdf_text))
        if value_regex:
            return list({m.group(0) for m in value_regex.finditer(pdf_text)})
        return [line.strip() for line in pdf_text.splitlines() if line.strip()]

    candidates: list[str] = []

    for table_cells in extract_tables(html):
        candidates.extend(table_cells)

    if not candidates:
        candidates.extend(extract_lists(html))

    if value_regex:
        matched = []
        for c in candidates:
            matched.extend(m.group(0) for m in value_regex.finditer(c))
        if not matched:
            matched = extract_by_regex(html, value_regex)
        return matched

    return candidates
