"""anchor 生成与标签分配（SSD 风格，阈值沿用 OpenPCDet 的 0.6 / 0.45）。"""
from __future__ import annotations

import numpy as np

from .data import ANCHOR_DIMS, CLASS_NAMES
from .metric import bev_iou

POS_THR = 0.6
NEG_THR = 0.45


def make_anchors(grid_hw, bev_stride, voxel_xy=(0.16, 0.16),
                 x_range=(-30.0, 354.0), y_range=(-25.0, 39.0),
                 class_names=CLASS_NAMES):
    """返回 (anchors[A,7], anchor_cls[A])，每格按 (类, 0°/90°) 展开。"""
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    sx = bev_stride * voxel_xy[0]
    sy = bev_stride * voxel_xy[1]
    cx = x_range[0] + (np.arange(gw) + 0.5) * sx
    cy = y_range[0] + (np.arange(gh) + 0.5) * sy
    rows, classes = [], []
    # 顺序：格 -> 类 -> 朝向，且 x 在外层（与 BEV 特征图的 (x, y) 展平顺序一致）
    for x in cx:
        for y in cy:
            for ci, name in enumerate(class_names):
                l, w, h = ANCHOR_DIMS[name]
                for yaw in (0.0, np.pi / 2):
                    rows.append([x, y, h / 2.0, l, w, h, yaw])
                    classes.append(ci)
    return np.asarray(rows, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def assign(anchors, anchor_cls, boxes, names, pos_thr=POS_THR, neg_thr=NEG_THR,
           class_names=CLASS_NAMES, max_pos=1):
    """返回 dict(pos, neg, cls, reg)；ignore 区既不参与分类也不参与回归。"""
    n = len(anchors)
    pos = np.zeros(n, dtype=bool)
    best_iou = np.zeros(n, dtype=np.float32)
    best_gt = np.full(n, -1, dtype=np.int64)
    best_cls = np.zeros(n, dtype=np.int64)

    cls_index = {nm: i for i, nm in enumerate(class_names)}
    for gi, (box, name) in enumerate(zip(boxes, names)):
        ci = cls_index[name]
        cand = np.nonzero(anchor_cls == ci)[0]
        if len(cand) == 0:
            continue
        a = anchors[cand]
        half = 0.5 * float(np.hypot(box[3], box[4])) + 2.0
        near = (np.abs(a[:, 0] - box[0]) < half) & (np.abs(a[:, 1] - box[1]) < half)
        for j in cand[near]:
            iou = bev_iou(anchors[j], box)
            if iou > best_iou[j]:
                best_iou[j], best_gt[j], best_cls[j] = iou, gi, ci

    # 每个真值保留 IoU 最大的 max_pos 个 anchor 为正样本（密集正样本，缓解极度不平衡）
    for gi in range(len(boxes)):
        own = np.nonzero(best_gt == gi)[0]
        if len(own) == 0:
            continue
        order = own[np.argsort(-best_iou[own])][:max(1, int(max_pos))]
        pos[order] = True
    pos |= best_iou >= pos_thr
    neg = (best_iou < neg_thr) & ~pos

    reg = np.zeros((n, 7), dtype=np.float32)
    for j in np.nonzero(pos)[0]:
        a = anchors[j]
        b = boxes[best_gt[j]]
        diag = float(np.hypot(a[3], a[4]))
        reg[j, 0] = (b[0] - a[0]) / diag
        reg[j, 1] = (b[1] - a[1]) / diag
        reg[j, 2] = (b[2] - a[2]) / a[5]
        reg[j, 3] = np.log(max(b[3], 1e-6) / a[3])
        reg[j, 4] = np.log(max(b[4], 1e-6) / a[4])
        reg[j, 5] = np.log(max(b[5], 1e-6) / a[5])
        dyaw = (b[6] - a[6] + np.pi) % (2 * np.pi) - np.pi
        reg[j, 6] = dyaw
    return {"pos": pos, "neg": neg, "cls": best_cls, "reg": reg}
