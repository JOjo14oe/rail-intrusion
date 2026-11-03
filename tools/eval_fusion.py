"""C1 三路对照：RGB-only / IR-only（单应映射到 RGB 像平面）/ 晚融合。

同一份 OSDaR23 验证集、同一套类别、同一个自写 COCO 口径评测器（railintrude.evaluate），
唯一变化的是"喂什么给检测器、怎么合并"。

- 单应 H 只在 **train** 划分上用两路 GT 框拟合，验证集完全不参与拟合。
- IR 框经 H 投到 RGB 像平面后与 RGB 框按类做"以 RGB 为主、IR 补盲"的并集合并。
- 额外报"大暗区子集"（RGB 中 <40 灰度像素占比 >= 30%）的指标，用来验证夜间/弱光主张。

用法：
  python tools/eval_fusion.py --imgsz-rgb 1280 --imgsz-ir 640
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from railintrude.evaluate import evaluate_predictions  # noqa: E402
from railintrude.fusion import apply_affine, fit_similarity, frame_key, merge_by_affine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DARK_FRACTION = 0.30
DARK_PIXEL = 40


def list_images(data_dir: Path, split: str) -> list[Path]:
    return sorted((data_dir / "images" / split).glob("*.jpg"))


def load_gt(label_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        rows.append((cls, ((cx - bw / 2) * width, (cy - bh / 2) * height,
                           (cx + bw / 2) * width, (cy + bh / 2) * height)))
    return rows


def centre(box):
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def fit_pairs_from_split(rgb_images, rgb_data: Path, ir_index: dict[str, Path], ir_data: Path):
    """同类内按中心 x 排序后一一配对，只保留两侧数量相等的帧/类。"""
    src, dst = [], []
    for rgb_path in rgb_images:
        key = frame_key(rgb_path.name)
        ir_path = ir_index.get(key)
        if ir_path is None:
            continue
        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue
        h_rgb, w_rgb = rgb_img.shape[:2]
        h_ir, w_ir = ir_img.shape[:2]
        rgb_gt = load_gt(rgb_data / "labels" / rgb_path.parent.name / (rgb_path.stem + ".txt"), w_rgb, h_rgb)
        ir_gt = load_gt(ir_data / "labels" / ir_path.parent.name / (ir_path.stem + ".txt"), w_ir, h_ir)
        for cls in sorted({c for c, _ in rgb_gt} & {c for c, _ in ir_gt}):
            a = sorted((b for c, b in rgb_gt if c == cls), key=lambda b: centre(b)[0])
            b = sorted((b for c, b in ir_gt if c == cls), key=lambda b: centre(b)[0])
            if len(a) != len(b) or not a:
                continue
            for rb, ib in zip(a, b):
                src.append(centre(ib))
                dst.append(centre(rb))
    return np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64)


def predict_boxes(model, image_path: Path, imgsz: int, conf: float, device: str) -> list[tuple[int, float, tuple]]:
    res = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
    if not res:
        return []
    boxes = res[0].boxes
    out = []
    for cls, cf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
        out.append((int(cls), float(cf), (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))))
    return out


def summarise(arm_name: str, dets, gts, n_classes: int) -> dict:
    m = evaluate_predictions(dets, gts, n_classes)
    return {"arm": arm_name, "mAP50": round(m["mAP50"], 4), "mAP50-95": round(m["mAP50-95"], 4),
            "per_class": {str(k): {"ap50": round(v["ap50"], 4), "ap": round(v["ap"], 4)}
                          for k, v in m["per_class"].items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-data", default="data/osdar23_yolo")
    ap.add_argument("--ir-data", default="data/osdar23_yolo_ir")
    ap.add_argument("--rgb-weights", required=True)
    ap.add_argument("--ir-weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--fit-split", default="train")
    ap.add_argument("--imgsz-rgb", type=int, default=1280)
    ap.add_argument("--imgsz-ir", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--max-centre-px", type=float, default=40.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/osdar23_yolo/fusion")
    args = ap.parse_args()

    from ultralytics import YOLO

    rgb_data = ROOT / args.rgb_data
    ir_data = ROOT / args.ir_data
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    yaml_text = (rgb_data / "dataset.yaml").read_text(encoding="utf-8")
    n_classes = sum(1 for line in yaml_text.splitlines() if line.strip()[:1].isdigit() and ":" in line)

    ir_index = {frame_key(p.name): p for p in list_images(ir_data, args.split)}
    rgb_val = list_images(rgb_data, args.split)
    if args.limit:
        rgb_val = rgb_val[: args.limit]
    paired = [p for p in rgb_val if frame_key(p.name) in ir_index]
    print(f"[pair] rgb {args.split}={len(rgb_val)} ir={len(ir_index)} paired={len(paired)}")

    rgb_train = list_images(rgb_data, args.fit_split)
    ir_train_index = {frame_key(p.name): p for p in list_images(ir_data, args.fit_split)}
    src, dst = fit_pairs_from_split(rgb_train, rgb_data, ir_train_index, ir_data)
    print(f"[fit] correspondence pairs = {len(src)}")
    A = fit_similarity(src, dst)
    print(f"[fit] A = {np.round(A, 4).tolist()}")

    proj = np.asarray(src, dtype=np.float64) @ A[:, :2].T + A[:, 2]
    resid = np.linalg.norm(proj - np.asarray(dst, dtype=np.float64), axis=1)
    print(f"[fit] residual px: median={np.median(resid):.1f} mean={resid.mean():.1f} "
          f"p90={np.percentile(resid, 90):.1f} max={resid.max():.1f}")
    (out / "similarity.json").write_text(json.dumps({"A": A.tolist(), "pairs": len(src),
                                                     "residual_median_px": float(np.median(resid))},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")

    rgb_model = YOLO(str(ROOT / args.rgb_weights))
    ir_model = YOLO(str(ROOT / args.ir_weights))

    arms = {"rgb_only": [], "ir_only": [], "fusion": []}
    gts, dark_flags = [], []
    for i, rgb_path in enumerate(paired, 1):
        ir_path = ir_index[frame_key(rgb_path.name)]
        img = cv2.imread(str(rgb_path))
        h_rgb, w_rgb = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_flags.append(float((gray < DARK_PIXEL).mean()) >= DARK_FRACTION)

        gts.append(load_gt(rgb_data / "labels" / args.split / (rgb_path.stem + ".txt"), w_rgb, h_rgb))
        rgb_dets = predict_boxes(rgb_model, rgb_path, args.imgsz_rgb, args.conf, args.device)
        ir_dets_raw = predict_boxes(ir_model, ir_path, args.imgsz_ir, args.conf, args.device)
        ir_mapped = apply_affine([b for _, _, b in ir_dets_raw], A)
        ir_dets = [(c, s, b) for (c, s, _), b in zip(ir_dets_raw, ir_mapped)]

        arms["rgb_only"].append(rgb_dets)
        arms["ir_only"].append(ir_dets)
        arms["fusion"].append(merge_by_affine(rgb_dets, ir_dets, A=np.eye(2, 3),
                                             max_centre_px=args.max_centre_px))
        if i % 20 == 0:
            print(f"[{i}/{len(paired)}] rgb_dets={len(rgb_dets)} ir_dets={len(ir_dets)}")

    dark_idx = [i for i, flag in enumerate(dark_flags) if flag]
    results = {"paired_frames": len(paired), "dark_frames": len(dark_idx),
               "dark_rule": f"rgb gray<{DARK_PIXEL} fraction >= {DARK_FRACTION}",
               "arms": {}, "dark_subset": {}}
    for arm, dets in arms.items():
        results["arms"][arm] = summarise(arm, dets, gts, n_classes)
        if dark_idx:
            results["dark_subset"][arm] = summarise(arm, [dets[i] for i in dark_idx],
                                                   [gts[i] for i in dark_idx], n_classes)

    (out / "fusion_c1.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["| 臂 | mAP50 | mAP50-95 | 大暗区 mAP50 | 大暗区帧数 |", "|---|---|---|---|---|"]
    for arm in arms:
        d = results["dark_subset"].get(arm, {})
        lines.append(f"| {arm} | {results['arms'][arm]['mAP50']:.4f} | {results['arms'][arm]['mAP50-95']:.4f} "
                     f"| {d.get('mAP50', float('nan')):.4f} | {len(dark_idx)} |")
    table = "\n".join(lines)
    (out / "fusion_c1.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
