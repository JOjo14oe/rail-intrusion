"""Build contact sheets for manual "person inside the running rails" labelling.

The labelling unit is one frame: `intrusion = 1` if any person's foot point is between the
running rails of the track nearest the ego train, else 0. Sheets are drawn with the current
model's person boxes so a wrong proposal is visible immediately.

Writes:
    <out>/sheets/sheet_<n>.jpg   grid of frames, index printed per cell
    <out>/candidates.csv          frame_id,sequence,path,model_boxes

Usage:
    python tools/make_label_sheets.py --per-seq 20 --grid 4 5 --conf 0.35
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEQUENCES = [
    "3_fire_site_3.1",
    "7_approach_underground_station_7.2",
    "7_approach_underground_station_7.3",
    "12_vegetation_steady_12.1",
    "18_vegetation_switch_18.1",
    "19_vegetation_curve_19.1",
]


def frames_of(seq_dir: Path, per_seq: int) -> list[Path]:
    images = sorted((seq_dir / "rgb_center").glob("*.png"))
    if not images:
        return []
    if len(images) <= per_seq:
        return images
    step = len(images) / per_seq
    return [images[int(i * step)] for i in range(per_seq)]


def detect(model, path: Path, conf: float) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    res = model.predict(str(path), conf=conf, imgsz=1280, device=0, verbose=False, plots=False)[0]
    names = res.names
    out = []
    for box, cls, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        out.append((names[int(cls)], float(score), tuple(box)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/osdar23")
    ap.add_argument("--out", default="data/label_curation")
    ap.add_argument("--weights", default="runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt")
    ap.add_argument("--per-seq", type=int, default=20)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cell", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    from ultralytics import YOLO  # imported late so --help stays fast

    data_dir = ROOT / args.data
    out_dir = ROOT / args.out
    sheets = out_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ROOT / args.weights))
    rows = []
    for seq in SEQUENCES:
        seq_dir = data_dir / seq
        picked = frames_of(seq_dir, args.per_seq)
        if not picked:
            print(f"[skip] {seq}: no rgb_center images")
            continue
        for path in picked:
            dets = detect(model, path, args.conf)
            persons = [d for d in dets if d[0] == "person"]
            rows.append(
                {
                    "sequence": seq,
                    "frame": path.stem,
                    "path": str(path.relative_to(ROOT)),
                    "model_persons": len(persons),
                    "boxes": "|".join(f"{c}:{s:.2f}:" + ",".join(f"{v:.0f}" for v in b) for c, s, b in persons),
                }
            )
        print(f"[ok] {seq}: {len(picked)} frames")

    with (out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "frame", "path", "model_persons", "boxes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_dir / 'candidates.csv'} ({len(rows)} frames)")

    per_sheet = args.cols * args.rows
    cell = args.cell
    sheet_h = int(cell * 0.5625)
    for index in range(0, len(rows), per_sheet):
        chunk = rows[index:index + per_sheet]
        canvas = np.full((args.rows * sheet_h, args.cols * cell, 3), 32, np.uint8)
        for slot, row in enumerate(chunk):
            img = cv2.imread(str(ROOT / row["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            img = cv2.resize(img, (cell, sheet_h))
            sx, sy = cell / w, sheet_h / h
            for item in filter(None, row["boxes"].split("|")):
                _, _, coords = item.split(":", 2)
                x0, y0, x1, y1 = (float(v) for v in coords.split(","))
                p0 = (int(x0 * sx), int(y0 * sy))
                p1 = (int(x1 * sx), int(y1 * sy))
                cv2.rectangle(img, p0, p1, (0, 220, 0), 2)
            col, line = slot % args.cols, slot // args.cols
            top = line * sheet_h
            canvas[top:top + sheet_h, col * cell:(col + 1) * cell] = img
            label = f"#{index + slot} p={row['model_persons']}"
            cv2.putText(canvas, label, (col * cell + 4, top + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.rectangle(canvas, (col * cell, top), (col * cell + cell - 1, top + sheet_h - 1), (90, 90, 90), 1)
        path = sheets / f"sheet_{index // per_sheet:02d}.jpg"
        cv2.imwrite(str(path), canvas)
        print(f"[sheet] {path}  frames #{index}-#{index + len(chunk) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
