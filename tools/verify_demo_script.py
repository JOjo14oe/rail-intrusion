"""Verify materials/demo-script.docx structure and leftover Markdown markers."""
from pathlib import Path

from docx import Document

DOC = Path(__file__).resolve().parents[1] / "materials" / "demo-script.docx"


def main() -> int:
    d = Document(str(DOC))
    print(f"[file] {DOC}  {DOC.stat().st_size} bytes")
    print(f"[count] paragraphs={len(d.paragraphs)} tables={len(d.tables)}")
    print("[head] first 6 paragraphs:")
    for i, p in enumerate(d.paragraphs[:6]):
        run = p.runs[0] if p.runs else None
        size = run.font.size.pt if run is not None and run.font.size else None
        bold = run.font.bold if run is not None else None
        print(f"  {i}: style={p.style.name!r} size={size} bold={bold} text={p.text[:50]!r}")

    text = "\n".join(p.text for p in d.paragraphs)
    for t in d.tables:
        for row in t.rows:
            for cell in row.cells:
                text += "\n" + cell.text
    stray = {
        "bold_marker": text.count("**"),
        "code_tick": text.count("`"),
        "table_pipe_line": sum(1 for ln in text.splitlines() if ln.strip().startswith("|")),
        "md_heading_hash": sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#")),
    }
    print(f"[stray] {stray}")
    for probe in ("口播稿", "电梯稿", "已达成", "未验证"):
        print(f"[probe] {probe!r} -> {probe in text}")
    return 0 if all(v == 0 for v in stray.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
