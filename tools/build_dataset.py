"""抽帧 + COCO 预训练 YOLO 自动打伪标签，生成可训练的 YOLO 数据集。

诚实边界：伪标签来自 COCO 预训练模型本身，微调是在"学自己的预测"。
它验证的是训练管线与域适配趋势，不是新监督信号。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import yaml

from railintrude.dataset import CLASS_MAP, select_frames, temporal_split, to_yolo_lines
from railintrude.vision import YoloDetector


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/platform-arrival-1080p.webm")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--stride", type=int, default=5, help="抽帧间隔")
    ap.add_argument("--conf", type=float, default=0.35, help="伪标签置信度阈值")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--neg-stride", type=int, default=5,
                    help="无目标帧每 N 个取 1 个当负样本，0=不要负样本")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"找不到权重：{weights}")
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"找不到视频：{video}")

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    detector = YoloDetector(weights=str(weights), conf=args.conf, device=args.device)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频：{video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = list(range(0, total, args.stride))

    # 第一遍：只算伪标签，记录"有目标的帧"。目标帧在时间轴上并不均匀，
    # 所以必须按目标帧切分，否则验证集会只剩个位数（实测 145 帧里只有 3 张落在后 30%）。
    cls_counter: Counter[str] = Counter()
    kept: dict[int, list[str]] = {}
    empty: list[int] = []
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        dets = detector(frame, idx)
        lines = to_yolo_lines(dets, w, h)
        if not lines:
            empty.append(idx)
            continue
        kept[idx] = lines
        for det in dets:
            if det.cls_name in CLASS_MAP:
                cls_counter[det.cls_name] += 1

    kept = select_frames(kept, empty, args.neg_stride)
    kept_idx = sorted(kept)
    train_frames, val_frames = temporal_split(kept_idx, args.train_ratio)
    split_of = {f: "train" for f in train_frames} | {f: "val" for f in val_frames}

    # 第二遍：按切分结果落盘
    for idx in kept_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        split = split_of[idx]
        name = f"{idx:06d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame)
        lines = kept[idx]
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
        )
    cap.release()

    names = {v: k for k, v in CLASS_MAP.items()}
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
             "names": names},
            allow_unicode=True, sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats = {
        "video": str(video), "stride": args.stride, "conf": args.conf,
        "frames_total": total, "frames_sampled": len(frames), "frames_kept": len(kept),
        "frames_with_targets": len(kept) - sum(1 for v in kept.values() if not v),
        "negatives": sum(1 for v in kept.values() if not v), "neg_stride": args.neg_stride,
        "train_images": len(list((out / "images" / "train").glob("*.jpg"))),
        "val_images": len(list((out / "images" / "val").glob("*.jpg"))),
        "instances_by_class": dict(cls_counter),
        "instances_total": sum(cls_counter.values()),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n数据集：{yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
