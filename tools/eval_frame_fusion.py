"""C1 第二版：帧级（presence）可见光/红外融合对照。

第一版在「跨模态几何配准」上失败了 —— 同一批 GT 配对拟合出的相似变换尺度
在 1.28 和 0.63 之间跳，说明 2D bbox 底边中点不足以唯一确定 IR->RGB 关联。
这一版彻底绕开配准：只用「同一帧里有没有某类目标」这个信息，
RGB 与 IR 各自回答，再按融合规则合并，评测也是帧级的。

用法：
    python tools/eval_frame_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from railintrude.evaluate import evaluate_presence  # noqa: E402
from railintrude.fusion import frame_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["buffer_stop", "catenary_pole", "person", "road_vehicle", "signal_pole"]
W_IR, H_IR = 640, 480


def read_labels(path: Path) -> list[int]:
    """Return class ids present in a YOLO label file."""
    out: list[int] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append(int(parts[0]))
    return out


def dark_fraction(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float((img < 40).mean())


def detect_classes(weights: Path, image: Path, imgsz: int, conf: float, device: str) -> set[int]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    res = model.predict(str(image), conf=conf, imgsz=imgsz, device=device, verbose=False, plots=False)
    if not res:
        return set()
    boxes = res[0].boxes
    if boxes is None or boxes.cls is None:
        return set()
    return {int(c) for c in boxes.cls.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-weights", default="runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt")
    ap.add_argument("--ir-weights", default="runs/detect/osdar23-yolov8n-ir/weights/best.pt")
    ap.add_argument("--rgb-data", default="data/osdar23_yolo")
    ap.add_argument("--ir-data", default="data/osdar23_yolo_ir")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz-rgb", type=int, default=1280)
    ap.add_argument("--imgsz-ir", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dark", type=float, default=0.30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="out/osdar23_yolo/fusion_frame.json")
    args = ap.parse_args()

    rgb_root, ir_root = ROOT / args.rgb_data, ROOT / args.ir_data
    rgb_imgs, ir_imgs = rgb_root / "images" / args.split, ir_root / "images" / args.split
    rgb_lbls = rgb_root / "labels" / args.split
    ir_by_key = {frame_key(p.name): p for p in ir_imgs.glob("*.jpg")}
    ir_lbl_by_key = {frame_key(p.name): p for p in (ir_root / "labels" / args.split).glob("*.txt")}

    from ultralytics import YOLO

    rgb_model = YOLO(str(ROOT / args.rgb_weights))
    ir_model = YOLO(str(ROOT / args.ir_weights))

    gt_sets: list[set[int]] = []
    rgb_sets: list[set[int]] = []
    ir_sets: list[set[int]] = []
    dark_flags: list[bool] = []

    for rgb_img in sorted(rgb_imgs.glob("*.jpg")):
        key = frame_key(rgb_img.name)
        ir_img = ir_by_key.get(key)
        gt_sets.append(set(read_labels(rgb_lbls / (rgb_img.stem + ".txt"))))
        dark_flags.append(dark_fraction(rgb_img) >= args.dark)

        def predict(model, path: Path, imgsz: int) -> set[int]:
            res = model.predict(str(path), conf=args.conf, imgsz=imgsz, device=args.device,
                                verbose=False, plots=False)
            if not res or res[0].boxes is None or res[0].boxes.cls is None:
                return set()
            return {int(c) for c in res[0].boxes.cls.tolist()}

        rgb_sets.append(predict(rgb_model, rgb_img, args.imgsz_rgb))
        ir_sets.append(predict(ir_model, ir_img, args.imgsz_ir) if ir_img is not None else set())

    union = [a | b for a, b in zip(rgb_sets, ir_sets)]
    inter = [a & b for a, b in zip(rgb_sets, ir_sets)]

    def summarise(gts, preds, mask=None):
        if mask is None:
            return evaluate_presence(preds, gts)
        return evaluate_presence([p for p, m in zip(preds, mask) if m],
                                 [g for g, m in zip(gts, mask) if m])

    arms = {
        "rgb_only": rgb_sets,
        "ir_only": ir_sets,
        "fusion_or": union,
        "fusion_and": inter,
    }
    out = {
        "frames": len(gt_sets),
        "dark_frames": int(sum(dark_flags)),
        "conf": args.conf,
        "dark_threshold": args.dark,
        "all": {k: summarise(gt_sets, v) for k, v in arms.items()},
        "dark": {k: summarise(gt_sets, v, dark_flags) for k, v in arms.items()},
        "bright": {k: summarise(gt_sets, v, [not d for d in dark_flags]) for k, v in arms.items()},
    }
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[frames] {out['frames']}  dark={out['dark_frames']}  conf={args.conf}")
    for group in ("all", "dark", "bright"):
        for arm, m in out[group].items():
            print(f"{group:6s} {arm:11s} P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} tp={m['tp']} fp={m['fp']} fn={m['fn']}")
    print(f"[out] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
