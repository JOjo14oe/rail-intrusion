"""整图推理 vs 切片推理 对照（都用自写评测器评）。

结论要回答的问题：catenary_pole / signal_pole 在 2464x1600 上一根只有十几像素宽，
整图缩到 640 基本被抹平；切成 640 小块按原尺度喂给模型，能不能救回来？

正确性闸门：整图自评必须复现官方 mAP50 0.5196，否则评测器或流程有错，先别信切片数字。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from railintrude.evaluate import evaluate_predictions  # noqa: E402
from railintrude.slicing import remap_boxes  # noqa: E402


def load_gt(label_path: Path, w: int, h: int) -> list:
    if not label_path.is_file():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(v) for v in p[1:])
        out.append((c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def tile_origins(w: int, h: int, size: int, overlap: float) -> list[tuple[int, int]]:
    step = max(int(round(size * (1.0 - overlap))), 1)
    xs = list(range(0, max(w - size, 0) + 1, step))
    ys = list(range(0, max(h - size, 0) + 1, step))
    if not xs:
        xs = [0]
    elif xs[-1] + size < w:
        xs.append(max(w - size, 0))
    if not ys:
        ys = [0]
    elif ys[-1] + size < h:
        ys.append(max(h - size, 0))
    return [(x, y) for y in ys for x in xs]


def _boxes(res, offset_x: int, offset_y: int, keep: set, canvas_w: int, canvas_h: int) -> list:
    """canvas_w/h 是"贴回去的原图画布"尺寸，不是块尺寸——传块尺寸会把框全裁在块内。"""
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    raw = [(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])) for i in range(len(cls))]
    remapped = remap_boxes(raw, offset_x, offset_y, canvas_w, canvas_h)
    for i, (c, s, box) in enumerate(zip(cls, conf, remapped)):
        if c in keep:
            out.append((int(c), float(s), box))
    return out


def _nms_per_class(dets: list, iou: float = 0.7) -> list:
    if not dets:
        return []
    boxes = torch.tensor([d[2] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d[1] for d in dets], dtype=torch.float32)
    classes = torch.tensor([d[0] for d in dets], dtype=torch.int64)
    # 类别偏移技巧：整框平移 (max_coord+1)*cls，跨类必然不相交 → 一次 NMS 即可
    shift = classes.to(boxes.dtype).unsqueeze(1) * (float(boxes.abs().max()) + 1.0)
    keep = torch.ops.torchvision.nms(boxes + shift, scores, iou)
    return [dets[i] for i in keep.tolist()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "osdar23_yolo"))
    ap.add_argument("--weights", default=str(ROOT / "runs" / "detect" / "osdar23-yolov8n-v2" / "weights" / "best.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "out" / "osdar23_yolo" / "slicing.json"))
    args = ap.parse_args()

    data = Path(args.data)
    img_dir = data / "images" / args.split
    lbl_dir = data / "labels" / args.split
    names = {}
    for line in (data / "dataset.yaml").read_text().splitlines():
        if ":" in line and line.strip().split(":")[0].strip().isdigit():
            k, v = line.split(":", 1)
            names[int(k)] = v.strip()
    n_classes = max(names) + 1 if names else 5
    keep = set(names)

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    print(f"验证图 {len(images)} 张，类别 {names}")

    model = YOLO(str(Path(args.weights)))
    gts, whole, sliced = [], [], []
    t0 = time.time()

    for i, ip in enumerate(images, 1):
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        gts.append(load_gt(lbl_dir / (ip.stem + ".txt"), w, h))

        r = model.predict(img, conf=args.conf, iou=0.7, imgsz=640, device=args.device, verbose=False)[0]
        whole.append(_boxes(r, 0, 0, keep, w, h))

        pool = []
        for x, y in tile_origins(w, h, args.tile, args.overlap):
            crop = img[y : y + args.tile, x : x + args.tile]
            if crop.size == 0:
                continue
            rt = model.predict(crop, conf=args.conf, iou=0.7, imgsz=640, device=args.device, verbose=False)[0]
            pool.extend(_boxes(rt, x, y, keep, w, h))
        sliced.append(_nms_per_class(pool))

        if i % 6 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  {time.time() - t0:.0f}s", flush=True)

    res_whole = evaluate_predictions(whole, gts, n_classes)
    res_sliced = evaluate_predictions(sliced, gts, n_classes)

    def row(tag, r):
        print(f"{tag:>10}  mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f}")
        for c, v in sorted(r["per_class"].items()):
            print(f"   {names.get(c, c):<14} AP50 {v['ap50']:.4f}  P {v['precision']:.4f}  R {v['recall']:.4f}")

    print("\n=== 整图推理（应接近官方 0.5196）===")
    row("whole", res_whole)
    print(f"\n=== 切片推理 tile={args.tile} overlap={args.overlap} ===")
    row("sliced", res_sliced)

    out = {
        "images": len(images),
        "tile": args.tile,
        "overlap": args.overlap,
        "names": names,
        "whole": res_whole,
        "sliced": res_sliced,
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
