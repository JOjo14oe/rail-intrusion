"""自写 AP 评测器。

为什么要有它：ultralytics 的 val 只吃整图，切片推理的结果它评不了。
口径刻意对齐 COCO（101 点插值、IoU 0.5:0.95、无 GT 的类别跳过），
这样"整图自评"必须复现官方 mAP50 0.5196 —— 这是评测器正确性的硬证据。
"""

from __future__ import annotations

# 0.50, 0.55, … 0.95
IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def box_iou(a, b) -> float:
    """两个 xyxy 框的 IoU；不相交返回 0。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _class_ap(dets, gts, cls, iou_thr):
    """单类单 IoU 阈值的 AP；该类无 GT 时返回 (None, None, None)。"""
    gt_pool = {i: [b for c, b in g if c == cls] for i, g in enumerate(gts)}
    n_gt = sum(len(v) for v in gt_pool.values())
    if n_gt == 0:
        return None, None, None

    cand = [(conf, i, box) for i, d in enumerate(dets) for c, conf, box in d if c == cls]
    cand.sort(key=lambda t: -t[0])

    matched = {i: [False] * len(v) for i, v in gt_pool.items()}
    tp, fp = [], []
    for _conf, i, box in cand:
        best, best_j = 0.0, -1
        for j, gbox in enumerate(gt_pool.get(i, [])):
            if matched[i][j]:
                continue
            v = box_iou(box, gbox)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= iou_thr:
            matched[i][best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    ctp = cfp = 0
    prec, rec = [], []
    for t, f in zip(tp, fp):
        ctp += t
        cfp += f
        prec.append(ctp / (ctp + cfp))
        rec.append(ctp / n_gt)

    # COCO 101 点插值
    points = []
    for k in range(101):
        r = k / 100.0
        points.append(max([p for p, rr in zip(prec, rec) if rr >= r], default=0.0))
    return sum(points) / 101.0, (prec[-1] if prec else 0.0), (rec[-1] if rec else 0.0)


def evaluate_predictions(dets, gts, n_classes, iou_thrs=IOU_THRS) -> dict:
    """dets: 每图 [(cls, conf, (x0,y0,x1,y1))]；gts: 每图 [(cls, (x0,y0,x1,y1))]。"""
    per_class = {}
    for cls in range(n_classes):
        aps, p50, r50 = [], None, None
        for t in iou_thrs:
            ap, p, r = _class_ap(dets, gts, cls, t)
            if ap is None:
                break
            aps.append(ap)
            if abs(t - 0.5) < 1e-9:
                p50, r50 = p, r
        if not aps:
            continue
        per_class[cls] = {
            "ap50": aps[0],
            "ap": sum(aps) / len(aps),
            "precision": p50,
            "recall": r50,
        }

    n = len(per_class)
    return {
        "mAP50": sum(v["ap50"] for v in per_class.values()) / n if n else 0.0,
        "mAP50-95": sum(v["ap"] for v in per_class.values()) / n if n else 0.0,
        "per_class": per_class,
    }

def evaluate_presence(preds, gts):
    """Frame-level presence metrics: does this frame contain class X at all?

    ``preds`` and ``gts`` are equal-length lists of class-id sets, one per frame.
    This is the metric for late fusion: RGB and IR each answer "present?" and the
    fusion rule combines the two answers. No pixel registration is involved, which
    is exactly why it survives the fact that IR and RGB boxes cannot be matched
    reliably (see docs/c1-fusion-report.md).
    """
    if len(preds) != len(gts):
        raise ValueError("preds and gts must be the same length")
    tp = fp = fn = 0
    for pred, gt in zip(preds, gts):
        tp += len(pred & gt)
        fp += len(pred - gt)
        fn += len(gt - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def segments(flags):
    """Split a per-frame boolean list into inclusive (start, end) alarm events."""
    out = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out
