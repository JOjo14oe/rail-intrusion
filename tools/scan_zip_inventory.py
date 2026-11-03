"""Scan every OSDaR23 zip without extracting images: frame count, per-camera class
counts, and whether the 'track' class exists (needed to decide "on track" for persons).

Reads only the *_labels.json entry inside each zip, so a 8 GB archive costs a few seconds.

Usage:
    python out/probe/scan_zip_inventory.py "C:\\Users\\jojo as\\Desktop\\数据集" --out out/osdar23/zip_inventory.json
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path


def scan_one(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("_labels.json")]
        if not names:
            return {"zip": zip_path.name, "error": "no labels json"}
        data = json.loads(zf.read(min(names, key=len)))
        image_streams = sorted(
            {
                Path(n).parts[0]
                for n in zf.namelist()
                if "/" in n and Path(n).suffix.lower() in (".png", ".jpg")
            }
        )
    frames = data.get("openlabel", {}).get("frames", {})
    per_cam: dict[str, Counter] = {}
    for frame in frames.values():
        for obj in (frame.get("objects") or {}).values():
            for bb in (obj.get("object_data", {}).get("bbox") or []):
                name = bb.get("name", "")
                if "__bbox__" not in name:
                    continue
                cam, cls = name.split("__bbox__", 1)
                per_cam.setdefault(cam, Counter())[cls] += 1
    return {
        "zip": zip_path.name,
        "frames": len(frames),
        "streams": image_streams,
        "per_cam": {cam: dict(c.most_common()) for cam, c in sorted(per_cam.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="out/osdar23/zip_inventory.json")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for zip_path in sorted(root.glob("*.zip")):
        try:
            row = scan_one(zip_path)
        except Exception as exc:  # noqa: BLE001 - a broken archive must not stop the sweep
            row = {"zip": zip_path.name, "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        rgb = row.get("per_cam", {}).get("rgb_center", {})
        has_track = any("track" in c for c in rgb)
        print(
            f"{row['zip'][:44]:<46} frames {row.get('frames', '?'):>4} | "
            f"rgb_center boxes {sum(rgb.values()):>6} | track-class {has_track}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
