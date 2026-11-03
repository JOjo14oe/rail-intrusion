from pathlib import Path

from docx import Document

for p in sorted(Path("materials").glob("*.docx")):
    d = Document(str(p))
    heads = [
        x.text
        for x in d.paragraphs
        if x.runs and x.runs[0].font.bold and x.runs[0].font.size and x.runs[0].font.size.pt >= 11
    ]
    stray = sum(1 for x in d.paragraphs if "**" in x.text or chr(96) in x.text)
    rows = sum(len(t.rows) for t in d.tables)
    print(f"{p.name} | paras {len(d.paragraphs)} | tables {len(d.tables)} ({rows} rows) | heads {len(heads)} | stray-md {stray}")
    print(f"    first heads: {heads[:3]}")
