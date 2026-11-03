"""微调 yolov8n 并在留出时间段的验证集上对照原权重。

注意：验证集标签是同一 COCO 模型自动生成的伪标签，因此"原权重"在它上面接近满分
（它是出题人）。这个对照检验的是**微调有没有把教师模型的行为学下来 / 有没有训坏**，
不是真实精度。真实品质要靠人工抽检 val 帧。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def metrics_of(model: YOLO, data: str, device: str) -> dict:
    r = model.val(data=data, split="val", imgsz=640, device=device, verbose=False, plots=False)
    names = {int(k): v for k, v in model.names.items()}
    per_class = {}
    for i, ci in enumerate(r.box.ap_class_index):
        per_class[names.get(int(ci), str(ci))] = {
            "ap50": round(float(r.box.ap50[i]), 4),
            "recall": round(float(r.box.r[i]), 4),
        }
    return {
        "mAP50": round(float(r.box.map50), 4),
        "mAP50-95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class": per_class,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset/dataset.yaml")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freeze", type=int, default=10, help="冻结前 N 层，保住预训练特征")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=15, help="验证集不再提升就早停")
    ap.add_argument("--mosaic", type=float, default=1.0,
                    help="马赛克增强概率；要教'整帧空场景'时必须设 0")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--after", default="", help="已有微调权重：给了就跳过训练，只做对照评估")
    ap.add_argument("--out", default="out/experiment")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"找不到数据集配置：{data}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 评估原权重 {args.weights}")
    base = YOLO(args.weights)
    before = metrics_of(base, str(data), args.device)

    if args.after:
        best = Path(args.after)
        print(f"[2/3] 跳过训练，直接用已有权重 {best}")
    else:
        print(f"[2/3] 微调 {args.epochs} epoch (imgsz={args.imgsz}, batch={args.batch})")
        res = base.train(
            data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project="runs", name=args.name, exist_ok=True,
            seed=0, workers=0, verbose=True, plots=False,
            freeze=args.freeze, lr0=args.lr0, patience=args.patience,
        )
        # ultralytics 会把 project 再拼一层，所以只用它回报的 save_dir，不自己猜路径
        best = Path(str(res.save_dir)) / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"找不到微调权重：{best}")
    tuned = YOLO(str(best))

    print(f"[3/3] 评估微调权重 {best}")
    after = metrics_of(tuned, str(data), args.device)

    rows = [("mAP50", before["mAP50"], after["mAP50"]),
            ("mAP50-95", before["mAP50-95"], after["mAP50-95"]),
            ("precision", before["precision"], after["precision"]),
            ("recall", before["recall"], after["recall"])]
    table = "| 指标 | 原 yolov8n | 微调后 | 差值 |\n|---|---|---|---|\n"
    for name, b, a in rows:
        table += f"| {name} | {b} | {a} | {a - b:+.4f} |\n"

    result = {
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "weights_before": args.weights, "weights_after": str(best),
        "caveat": "验证集为同模型伪标签，原权重接近满分；本表只检验微调是否学住教师行为。",
        "before": before, "after": after,
    }
    (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "comparison.md").write_text("# 微调前后对照\n\n" + table + "\n" + result["caveat"] + "\n",
                                       encoding="utf-8")
    print("\n" + table)
    print(result["caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
