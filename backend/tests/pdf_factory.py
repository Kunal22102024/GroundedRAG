"""Build small PDFs for parser and ingestion tests (no binary fixtures in the repo)."""

from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF
from PIL import Image

BODY = 11
HEADING = 18


class _Doc(FPDF):
    """An FPDF with optional running header and footer (to test their removal)."""

    running_header: str | None = None
    running_footer: bool = False

    def header(self) -> None:
        if self.running_header:
            self.set_font("Helvetica", size=8)
            self.cell(0, 6, self.running_header, new_x="LMARGIN", new_y="NEXT")
            self.ln(4)

    def footer(self) -> None:
        if self.running_footer:
            self.set_y(-12)
            self.set_font("Helvetica", size=8)
            self.cell(0, 6, f"Page {self.page_no()}", align="C")


# A page is a list of blocks. A block is ("h", text) heading, ("p", text) paragraph, ("b", text)
# bold line, ("cols", "left||right") two columns, or ("row", "a||b||c") a table row. "image" makes an image-only page (what a scan looks like); None is a blank page.
Block = tuple[str, str]
Page = list[Block] | str | None


def make_pdf(
    pages: list[Page],
    *,
    outline: list[tuple[int, str, int]] | None = None,
    title: str | None = None,
    created: datetime | None = None,
    header: str | None = None,
    footer: bool = False,
) -> bytes:
    """Return PDF bytes. ``outline`` entries are (level, title, 1-based page)."""
    pdf = _Doc()
    pdf.set_auto_page_break(False)
    pdf.running_header, pdf.running_footer = header, footer
    if title:
        pdf.set_title(title)
    if created:
        pdf.set_creation_date(created)

    entries: dict[int, list[tuple[int, str]]] = {}
    for level, name, page in outline or []:
        entries.setdefault(page, []).append((level, name))

    for number, page in enumerate(pages, start=1):
        pdf.add_page()
        for level, name in entries.get(number, []):
            pdf.start_section(name, level=level - 1)
        if page is None:
            continue
        if page == "image":
            buffer = io.BytesIO()
            Image.new("RGB", (240, 240), (200, 200, 200)).save(buffer, "PNG")
            buffer.seek(0)
            pdf.image(buffer, x=20, y=30, w=120)
            continue
        assert isinstance(page, list)
        for kind, text in page:
            if kind == "h":
                pdf.set_font("Helvetica", "B", HEADING)
                pdf.multi_cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
            elif kind == "cols":  # two side-by-side paragraphs: "left text||right text"
                left, right = text.split("||")
                pdf.set_font("Helvetica", "", BODY)
                top = pdf.get_y()
                pdf.set_xy(15, top)
                pdf.multi_cell(85, 6, left, new_x="LMARGIN", new_y="NEXT")
                bottom_left = pdf.get_y()
                pdf.set_xy(112, top)
                pdf.multi_cell(85, 6, right, new_x="LMARGIN", new_y="NEXT")
                pdf.set_xy(10, max(bottom_left, pdf.get_y()) + 5)
            elif kind == "row":  # a table row of short cells: "a||b||c"
                pdf.set_font("Helvetica", "", BODY)
                top = pdf.get_y()
                for x, cell in zip((15, 60, 110), text.split("||"), strict=False):
                    pdf.set_xy(x, top)
                    pdf.cell(45, 6, cell)
                pdf.set_xy(10, top + 8)
            elif kind == "b":
                pdf.set_font("Helvetica", "B", BODY)
                pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", BODY)
                pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(5)  # visible gap = paragraph break
    return bytes(pdf.output())


ZERO_PAGE_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
    b"trailer<</Root 1 0 R/Size 3>>\nstartxref\n0\n%%EOF\n"
)
