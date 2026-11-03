"""Minimal Markdown -> docx converter for this project's reports.

Only the syntax these reports actually use is supported: headings (#/##/###),
paragraphs, bullet and numbered lists, pipe tables, fenced code blocks, blockquotes,
horizontal rules and inline `code` / **bold**. Anything else is written as plain text.

The ASCII diagrams rely on SimSun's exact 2:1 CJK/ASCII advance ratio, so code blocks
are set in SimSun 9 pt (ascii + eastAsia) or the frames drift.

Usage:
    python tools/md2docx.py docs/a.md docs/b.md --out-dir materials
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_FONT = "等线"
MONO_FONT = "SimSun"


def _set_font(run, name: str, size: float, bold: bool = False) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(fonts)
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:eastAsia"), name)


INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`)")


def _clean(text: str) -> str:
    """Strip inline markup so it never reaches the document as literal characters."""
    if text.count("**") % 2:
        text = text.replace("**", "")
    return text.replace("**", "").replace("`", "")


def _add_runs(par, text: str) -> None:
    if text.count("**") % 2:
        text = text.replace("**", "")
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            _set_font(par.add_run(piece[2:-2]), BODY_FONT, 10.5, bold=True)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            _set_font(par.add_run(piece[1:-1]), MONO_FONT, 9.5)
        else:
            _set_font(par.add_run(piece), BODY_FONT, 10.5)


def _table(doc: Document, rows: list[str]) -> None:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
    if not cells:
        return
    ncol = max(len(r) for r in cells)
    table = doc.add_table(rows=len(cells), cols=ncol)
    table.style = "Table Grid"
    for i, row in enumerate(cells):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            text = row[j] if j < len(row) else ""
            for piece in INLINE.split(text):
                if not piece:
                    continue
                bold = i == 0
                txt = _clean(piece)
                _set_font(par.add_run(txt), BODY_FONT, 9, bold=bold)
            if i == 0:
                shading = cell._tc.get_or_add_tcPr().makeelement(
                    qn("w:shd"), {qn("w:fill"): "EDEDED"}
                )
                cell._tc.get_or_add_tcPr().append(shading)
    doc.add_paragraph()


def convert(src: Path, dst: Path) -> Path:
    lines = src.read_text(encoding="utf-8").splitlines()
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(2.3)
    section.top_margin = section.bottom_margin = Cm(2.2)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            for text in block:
                par = doc.add_paragraph()
                par.paragraph_format.space_after = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                _set_font(par.add_run(text if text else " "), MONO_FONT, 9)
            doc.add_paragraph()
        elif stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(10)
            par.paragraph_format.space_after = Pt(4)
            _set_font(par.add_run(_clean(stripped[level:].strip())), BODY_FONT,
                      {1: 16, 2: 13, 3: 11.5}.get(level, 11), bold=True)
        elif stripped.startswith(">"):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.6)
            _add_runs(par, stripped.lstrip("> ").strip())
        elif stripped in ("---", "***", "___"):
            doc.add_paragraph()
        elif stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            _table(doc, block)
            continue
        elif re.match(r"^\s*[-*]\s+", line):
            par = doc.add_paragraph(style="List Bullet")
            _add_runs(par, re.sub(r"^\s*[-*]\s+", "", line))
        elif re.match(r"^\s*\d+[.)]\s+", line):
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.75)
            par.paragraph_format.first_line_indent = Cm(-0.75)
            _add_runs(par, re.sub(r"^\s*(\d+)[.)]\s+", r"\1. ", line))
        elif stripped:
            par = doc.add_paragraph()
            par.paragraph_format.space_after = Pt(6)
            _add_runs(par, stripped)
        i += 1

    for par in doc.paragraphs:
        if not par.alignment:
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dst))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out-dir", default="materials")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    for item in args.sources:
        src = Path(item)
        if not src.is_absolute():
            src = root / src
        dst = convert(src, out_dir / f"{src.stem}.docx")
        print(f"[docx] {dst}  ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
