"""PointPillars 基线训练（D3 Rail-PillarNet 的 Model A 复现）。

刻意偏差（写进 docs/readings/D3.md）：
- OpenPCDet -> 自写纯 PyTorch，8 GB 卡、batch=1；
- lr 1e-3（原文 Adam 0.03 / batch 2，batch=1 时按比例下调）；
- 每格 8 anchor（4 类 × 2 朝向），原文按类平均尺寸设 anchor。

数据按帧惰性读取（load_ids + read_frame）：全量读入 1069 帧约 3.6 GB，会被系统杀掉。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from railintrude.det import pillarize                      # noqa: E402
from railintrude.det.anchors import make_anchors          # noqa: E402
from railintrude.det.data import CLASS_NAMES, load_ids, read_frame  # noqa: E402
from railintrude.det.loss import pp_loss                  # noqa: E402
from railintrude.det.metric import evaluate_frames          # noqa: E402
from railintrude.det.pipeline import (decode, sample_targets,           # noqa: E402
                                      sample_to_batch)
from railintrude.det.pointpillars import PointPillars     # noqa: E402

X_MIN, X_MAX = -30.0, 354.0
Y_MIN, Y_MAX = -25.0, 39.0
Z_MIN, Z_MAX = -4.0, 8.0
# pillarize 的 point_range 顺序是 (xmin, ymin, zmin, xmax, ymax, zmax)，别按 (x0,x1,y0,y1,...) 写
POINT_RANGE = (X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX)
VOXEL = (0.16, 0.16, 12.0)
BEV_STRIDE = 4
MAX_PILLARS, MAX_POINTS = 16000, 32


def load_sample(pts, names, boxes):
    pillars, coords, npts = pillarize(pts, VOXEL, POINT_RANGE, MAX_PILLARS, MAX_POINTS)
    return {"pillars": torch.as_tensor(pillars, dtype=torch.float32),
            "coords": torch.as_tensor(coords, dtype=torch.long),
            "num_points": torch.as_tensor(npts, dtype=torch.long),
            "boxes": boxes, "names": names}


def to_dev(d, dev):
    return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in d.items()}


def grid_hw(model, device):
    b = {"pillars": torch.zeros(1, MAX_POINTS, 4, device=device),
         "coords": torch.zeros(1, 3, dtype=torch.long, device=device),
         "num_points": torch.ones(1, dtype=torch.long, device=device), "batch_size": 1}
    with torch.no_grad():
        return model(b)["grid_hw"]


def evaluate_split(model, anchors, root, ids, device, conf=0.01, frames=None):
    """conf 必须低：AP 是对整条 PR 曲线积分，用 0.1 会把低分真目标全剪掉。"""
    pred_by_class = {n: [] for n in CLASS_NAMES}      # 必须按类别名做 key（metric.evaluate 的约定）
    gt_by_class = {n: [] for n in CLASS_NAMES}
    for fid in (ids[:frames] if frames else ids):
        pts, names, boxes = read_frame(root, fid)
        s = load_sample(pts, names, boxes)
        del pts
        with torch.no_grad():
            out = model(to_dev(sample_to_batch(s), device))
        dets = decode(out["cls_preds"], out["box_preds"], out["dir_preds"], anchors,
                      conf_thr=conf)
        for i, cn in enumerate(CLASS_NAMES):
            pred_by_class[cn].append([(bx, sc) for bx, sc, cc in dets if cc == i])
            gt_by_class[cn].append([np.array(bb, float) for bb, nm in
                                    zip(s["boxes"], s["names"]) if nm == cn])
    return evaluate_frames(pred_by_class, gt_by_class, CLASS_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/pp_kitti")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="每轮只用前 N 帧（冒烟）")
    ap.add_argument("--overfit", action="store_true",
                    help="小批过拟合自检：验证集改用训练集前 6 帧，用来区分“欠训练”和“实现有 bug”")
    ap.add_argument("--val-frames", type=int, default=25,
                    help="训练中每轮只评前 N 帧，末轮评全量")
    ap.add_argument("--val-every", type=int, default=5,
                    help="每隔多少轮评一次 val（decode 是 Python 循环，很慢）")
    ap.add_argument("--dense", action="store_true",
                    help="密集正样本：pos_thr=0.5 / 每真值 top3 / focal alpha=0.5")
    ap.add_argument("--conf-eval", type=float, default=0.01, help="验证时解框的置信度下限")
    ap.add_argument("--out", type=Path, default=ROOT / "out/pp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    kw = {"pos_thr": 0.5, "max_pos": 3} if args.dense else {}
    alpha = 0.5 if args.dense else 0.25

    model = PointPillars(num_classes=len(CLASS_NAMES), garrid_hw=(
        int(round((Y_MAX - Y_MIN) / VOXEL[1])),
        int(round((X_MAX - X_MIN) / VOXEL[0]))),
        bev_stride=BEV_STRIDE, anchors_per_cell=len(CLASS_NAMES) * 2).to(dev)
    g = grid_hw(model, dev)
    anchors, anchor_cls = make_anchors(g, BEV_STRIDE * 2, class_names=CLASS_NAMES)
    print(f"[pp] grid={g} anchors={len(anchors)} params={sum(p.numel() for p in model.parameters()):,}")

    train_ids = load_ids(args.data, "train")
    val_ids = load_ids(args.data, "val")
    if args.limit:
        train_ids = train_ids[:args.limit]
        val_ids = val_ids[:max(1, args.limit // 4)]
    if args.overfit:                      # 自检：在训练帧自身上评，考“能不能背下来”
        val_ids = train_ids[:6]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for i, fid in enumerate(train_ids, 1):
            pts, names, boxes = read_frame(args.data, fid)
            s = load_sample(pts, names, boxes)
            del pts
            out = model(to_dev(sample_to_batch(s), dev))
            tgt = to_dev(sample_targets(s, anchors, anchor_cls, CLASS_NAMES, **kw), dev)
            loss, _parts = pp_loss(out["cls_preds"], out["box_preds"], out["dir_preds"],
                                   tgt, len(CLASS_NAMES), alpha=alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach())
            if i % 50 == 0 or i == len(train_ids):
                print(f"[pp] ep{ep} {i}/{len(train_ids)} loss={tot / i:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        if ep % max(1, args.val_every) == 0 or ep == args.epochs:
            m = evaluate_split(model, anchors, args.data, val_ids, dev, conf=args.conf_eval,
                                  frames=None if ep == args.epochs else args.val_frames)
            print(f"[pp] ep{ep} val mAP50={m['mAP50']:.4f} mAP50-95={m.get('mAP50_95', 0):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        else:
            m = None
            print(f"[pp] ep{ep} train loss={tot / max(len(train_ids), 1):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        hist.append({"epoch": ep, "loss": tot / max(len(train_ids), 1), "val": m})
        torch.save(model.state_dict(), args.out / "last.pt")
        (args.out / "metrics.json").write_text(
            json.dumps({"anchors": len(anchors), "grid": list(g), "history": hist,
                        "config": vars(args) | {"data": str(args.data), "out": str(args.out)}},
                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("[pp] done ->", args.out / "metrics.json")


if __name__ == "__main__":
    main()
