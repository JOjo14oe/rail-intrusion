from pathlib import Path

from docx import Document

for p in sorted(Path("materials").glob("*.docx")):
    d = Document(str(p))
    hits = [x.text for x in d.paragraphs if "**" in x.text or chr(96) in x.text]
    for t in d.tables:
        for row in t.rows:
            for c in row.cells:
                if "**" in c.text or chr(96) in c.text:
                    hits.append("TABLE: " + c.text[:80])
    if hits:
        print(p.name)
        for h in hits:
            print("   ", h[:120])
