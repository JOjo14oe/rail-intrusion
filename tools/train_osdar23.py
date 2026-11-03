"""OSDaR23 YOLOv8n training + evaluation, producing the numbers used in the report.

Two rows are compared:
  zero-shot  -- COCO pretrained weights evaluated directly on the railway val split
  fine-tuned -- best.pt after fine-tuning on the OSDaR23 train split

Usage:
  python tools/train_osdar23.py --epochs 60 --batch 16 --device 0

NOTE: this file is ASCII-only on purpose. Its previous version was destroyed by a
UTF-8 <-> CP936 round trip (every non-ASCII string became mojibake and the `?`
replacement ate closing quotes, so the file stopped parsing). Do not reintroduce
non-ASCII text here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def metrics_of(model, data: Path, imgsz: int = 640) -> dict:
    """Run val on the val split and return json-safe metrics.

    imgsz must match training: training at 1280 but evaluating at 640 shifts
    mAP50 by as much as 0.17 on this dataset, so the numbers stop being comparable.
    """
    r = model.val(data=str(data), split="val", imgsz=imgsz, device=0, verbose=False, plots=True)
    names = r.names if isinstance(r.names, dict) else {i: n for i, n in enumerate(r.names)}
    per_class = {}
    try:
        for idx, ap in zip(r.box.ap_class_index.tolist(), r.box.maps.tolist()):
            per_class[names[int(idx)]] = round(float(ap), 4)
    except Exception as exc:  # noqa: BLE001
        per_class = {"error": str(exc)}
    d = r.results_dict
    return {
        "mAP50": round(float(d.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(d.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(d.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(d.get("metrics/recall(B)", 0.0)), 4),
        "per_class_mAP50-95": per_class,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/osdar23_yolo/dataset.yaml")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    # 8 dataloader workers intermittently crash at the train->val handover on this
    # 15.6 GB box (child dies with `EOFError: Ran out of input` in spawn_main).
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="osdar23-yolov8n")
    ap.add_argument("--out", default="out/osdar23_yolo")
    ap.add_argument("--seed", type=int, default=0, help="training seed; repeat runs with different seeds to estimate variance")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--after", default=None, help="only evaluate this weights file")
    args = ap.parse_args()

    from ultralytics import YOLO

    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"data": str(data), "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "seed": args.seed}
    t0 = time.time()

    if args.skip_train and args.after:
        model = YOLO(str(ROOT / args.after if not Path(args.after).is_absolute() else args.after))
        report["fine_tuned"] = metrics_of(model, data, args.imgsz)
        report["weights"] = str(args.after)
    else:
        base = YOLO(str(ROOT / args.weights if not Path(args.weights).is_absolute() else args.weights))
        print("[1/3] zero-shot baseline (COCO weights on the railway val split)")
        report["zero_shot"] = metrics_of(base, data, args.imgsz)

        print(f"[2/3] fine-tuning for {args.epochs} epochs")
        res = base.train(
            data=str(data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            name=args.name,
            seed=args.seed,
            plots=True,
            val=True,
        )
        best = Path(str(res.save_dir)) / "weights" / "best.pt"
        report["weights"] = str(best)

        print("[3/3] evaluating the fine-tuned weights")
        report["fine_tuned"] = metrics_of(YOLO(str(best)), data, args.imgsz)

    report["elapsed_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
