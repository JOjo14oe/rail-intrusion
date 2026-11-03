"""Report the two measurable results: YOLO11n vs YOLOv8n, and VLM presence vs baseline."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["buffer_stop", "catenary_pole", "person", "road_vehicle", "signal_pole"]


def last_results_csv(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row]
    if not rows:
        return None
    return {k.strip(): v.strip() for k, v in rows[-1].items()}


def best_map50(run_dir: Path) -> tuple[float, int]:
    path = run_dir / "results.csv"
    best, best_epoch = 0.0, -1
    if not path.is_file():
        return (0.0, -1)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = [k for k in row if "mAP50(B)" in k]
            if not key:
                continue
            try:
                value = float(row[key[0]])
                epoch = int(float(row.get("epoch", -1)))
            except (TypeError, ValueError):
                continue
            if value > best:
                best, best_epoch = value, epoch
    return (best, best_epoch)


def main() -> int:
    for name in ("osdar23-yolov8n-516img-1280", "osdar23-yolo11n-1280"):
        run_dir = ROOT / "runs" / "detect" / name
        last = last_results_csv(run_dir)
        best, best_epoch = best_map50(run_dir)
        if last is None:
            print(f"[{name}] no results.csv")
            continue
        m50 = next((v for k, v in last.items() if "mAP50(B)" in k), "?")
        m5095 = next((v for k, v in last.items() if "mAP50-95(B)" in k), "?")
        p = next((v for k, v in last.items() if "precision(B)" in k), "?")
        r = next((v for k, v in last.items() if "recall(B)" in k), "?")
        print(f"[{name}] last-epoch mAP50={m50} mAP50-95={m5095} P={p} R={r} | "
              f"best mAP50={best:.4f} @epoch={best_epoch}")

    vlm_path = ROOT / "out" / "vlm" / "vlm_presence.json"
    if not vlm_path.is_file():
        print("[vlm] no vlm_presence.json")
        return 0
    data = json.loads(vlm_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    print(f"\n[vlm] frames={len(rows)} classes={list(rows[0]['answers'])}")
    print(f"{'class':15s} {'truth+':>6s} {'tp':>4s} {'fp':>3s} {'fn':>3s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s} | {'always-yes F1':>13s} {'verdict':>9s}")
    for cls in rows[0]["answers"]:
        cls_id = CLASS_NAMES.index(cls)
        tp = fp = fn = pos = 0
        for row in rows:
            truth = cls_id in row["truth"]
            pred = bool(row["answers"][cls]["yes"])
            pos += truth
            tp += truth and pred
            fp += pred and not truth
            fn += truth and not pred
        n = len(rows)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        base_p = pos / n
        base_f1 = 2 * base_p / (base_p + 1.0) if base_p else 0.0
        verdict = "beats" if f1 > base_f1 + 0.02 else ("ties" if f1 > base_f1 - 0.02 else "WORSE")
        print(f"{cls:15s} {pos:6d} {tp:4d} {fp:3d} {fn:3d} {precision:6.3f} {recall:6.3f} "
              f"{f1:6.3f} | {base_f1:13.3f} {verdict:>9s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
