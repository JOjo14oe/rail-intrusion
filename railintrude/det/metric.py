"""3D/BEV 评测：旋转框 IoU + 40 点插值 AP@0.5（对齐 Rail-PillarNet 原文口径）。

原文口径：IoU 0.5 下取 40 个 recall 阈值的精度平均（与 KITTI 的 40 点插值一致）。
"""
from __future__ import annotations

import cv2
import numpy as np


def _poly(box):
    """box=[x,y,z,l,w,h,yaw] -> 4 角点 (4,2)。"""
    x, y, l, w, yaw = float(box[0]), float(box[1]), float(box[3]), float(box[4]), float(box[6])
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def bev_iou(a, b):
    """单个 BEV 旋转框 IoU（用 cv2 多边形求交，避免自写裁剪）。"""
    pa = _poly(a).astype(np.float32)
    pb = _poly(b).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb, handleNested=True)
    if inter <= 0:
        return 0.0
    ua = a[3] * a[4]
    ub = b[3] * b[4]
    union = ua + ub - float(inter)
    return float(inter) / float(union) if union > 0 else 0.0


def _match(preds, gts, iou_thr):
    """在单类、单阈值下做按分数排序的贪心匹配。

    返回的 tp 与 preds 同序（原始下标），调用方不要再按分数排序一次——
    早期版本在这里返回“按名次排”的数组，调用方又排一次，导致 PR 曲线错位。
    """
    order = np.argsort([-p[1] for p in preds])
    used = np.zeros(len(gts), dtype=bool)
    tp = np.zeros(len(preds), dtype=bool)
    for idx in order:
        iou, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if used[j]:
                continue
            cur = bev_iou(preds[idx][0], g)
            if cur >= iou:
                iou, best_j = cur, j
        if best_j >= 0:
            used[best_j] = True
            tp[idx] = True
    return tp


def ap_40(preds, gts, iou_thr=0.5, n_recall=40):
    """preds=[(box, score)], gts=[box]；返回 40 点插值 AP（百分制）。"""
    if not gts:
        return None
    if not preds:
        return 0.0
    tp = _match(preds, gts, iou_thr)
    order = np.argsort([-p[1] for p in preds])
    tp = tp[order]
    ctp = np.cumsum(tp)
    prec = ctp / np.arange(1, len(tp) + 1)
    rec = ctp / len(gts)
    grid = np.linspace(0, 1, n_recall)
    out = []
    for r in grid:
        mask = rec >= r
        out.append(prec[mask].max() if mask.any() else 0.0)
    return float(np.clip(np.mean(out) * 100.0, 0.0, 100.0))


def evaluate(pred_by_class, gt_by_class, class_names, iou_thr=0.5):
    """返回 {"mAP50": float, "per_class": {name: ap}}；无真值的类跳过（记为 None）。"""
    per, vals = {}, []
    for name in class_names:
        ap = ap_40(pred_by_class.get(name, []), gt_by_class.get(name, []), iou_thr)
        per[name] = ap
        if ap is not None:
            vals.append(ap)
    return {"mAP50": float(np.mean(vals)) if vals else 0.0, "per_class": per}


def evaluate_frames(pred_by_class, gt_by_class, class_names, iou_thr=0.5):
    """逐帧版本的 evaluate：入参是每类一个「帧列表」，每帧是 [(box, score)] 或 [box]。

    额外产出 mAP50_95（IoU 0.50:0.05:0.95 的均值）；早期版本没有这个键，
    调用方读 m.get("mAP50_95", 0) 所以永远是 0。
    """
    per, vals, per95, vals95 = {}, [], {}, []
    thrs = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    for name in class_names:
        g_frames = gt_by_class.get(name, [])
        p_frames = pred_by_class.get(name, [])
        if not any(len(g) for g in g_frames):
            per[name] = None
            continue
        p = [kv for fr in p_frames for kv in fr]
        g = [b for fr in g_frames for b in fr]
        ap = ap_40(p, g, iou_thr)
        per[name] = ap
        if ap is not None:
            vals.append(ap)
        aps = [a for a in (ap_40(p, g, t) for t in thrs) if a is not None]
        per95[name] = float(np.mean(aps)) if aps else None
        if aps:
            vals95.append(per95[name])
    return {"mAP50": float(np.mean(vals)) if vals else 0.0,
            "mAP50_95": float(np.mean(vals95)) if vals95 else 0.0,
            "per_class": per, "per_class_50_95": per95}
