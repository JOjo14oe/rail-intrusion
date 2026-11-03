"""单帧样本 -> 模型输入 / 标签目标的最小胶水层。

batch 固定为 1（8 GB 卡上 190k 点的 pillar 化后仍以单帧训练最稳），
因此这里不做多帧 collate，只把 coords 的 batch 维显式写成 0。

解框/NMS 全部向量化：conf 阈值放到 0.01 时单帧候选可达上万，逐个 Python 循环
会把一次验证拖到分钟级（实测 6 帧 430 s）。
"""
from __future__ import annotations

import numpy as np
import torch

from .anchors import assign
from .data import CLASS_NAMES
from .metric import bev_iou


def sample_to_batch(sample):
    coords = sample["coords"].clone()
    coords[:, 0] = 0
    return {
        "pillars": sample["pillars"],
        "coords": coords,
        "num_points": sample["num_points"],
        "batch_size": 1,
    }


def sample_targets(sample, anchors, anchor_cls, class_names=CLASS_NAMES, **assign_kw):
    """类别顺序必须与 make_anchors 生成 anchor_cls 时用的一致，否则正样本会分配不到。"""
    boxes = sample["boxes"]
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    out = assign(anchors, anchor_cls, np.asarray(boxes, dtype=np.float32), sample["names"],
                 class_names=class_names, **assign_kw)
    return {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)) for k, v in out.items()}


# 回归目标编码（与 anchors.assign 里写 reg 的方式必须一致）
_DIAG = 1.0


def _decode_boxes(anchors, reg):
    """(N,7) anchors + (N,7) reg -> (N,7) boxes [x, y, z, l, w, h, yaw]，与 assign 互逆。"""
    a = np.asarray(anchors, dtype=np.float32)
    r = np.asarray(reg, dtype=np.float32)
    ax, ay, az, al, aw, ah, ayaw = (a[:, i] for i in range(7))
    dx, dy, dz, dl, dw, dh, dyaw = (r[:, i] for i in range(7))
    diag = np.maximum(np.hypot(np.maximum(al, 1e-3), np.maximum(aw, 1e-3)), _DIAG)
    out = np.empty_like(a)
    out[:, 0] = ax + dx * diag
    out[:, 1] = ay + dy * diag
    out[:, 2] = az + dz * ah
    out[:, 3] = al * np.exp(np.clip(dl, -4.0, 4.0))
    out[:, 4] = aw * np.exp(np.clip(dw, -4.0, 4.0))
    out[:, 5] = ah * np.exp(np.clip(dh, -4.0, 4.0))
    out[:, 6] = ayaw + dyaw
    return out


def _decode_box(anchor, reg):
    """单框版本，保留给调试/测试。"""
    return _decode_boxes(np.asarray(anchor, dtype=np.float32)[None, :],
                         np.asarray(reg, dtype=np.float32)[None, :])[0]


def _aabb(boxes):
    """旋转框 -> 轴对齐包围盒 (N,4)：[x1, y1, x2, y2]，只用于 NMS 预筛。"""
    b = np.asarray(boxes, dtype=np.float32)
    x, y, l, w, yaw = b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]
    c, s = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
    ex = c * l / 2 + s * w / 2
    ey = s * l / 2 + c * w / 2
    return np.stack([x - ex, y - ey, x + ex, y + ey], 1)


def nms_bev(boxes, scores, iou_thr=0.05, keep_top=200):
    """按分数降序做 BEV 抑制，返回保留的下标。

    加速关键：旋转框内含于其 AABB，故 IoU_rotated <= IoU_AABB。
    先用向量化的 AABB IoU 一次性筛出“有可能被抑制”的候选（上界 >= 阈值），
    再只对这几对算精确的旋转矩形 IoU。两条循环都只做 O(候选) 次多边形运算。
    """
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    n = len(boxes)
    ab = _aabb(boxes)
    x1, y1, x2, y2 = ab[:, 0], ab[:, 1], ab[:, 2], ab[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    alive = np.ones(n, dtype=bool)
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(int(i))
        if len(keep) >= keep_top:
            break
        ix = np.clip(np.minimum(x2[i], x2) - np.maximum(x1[i], x1), 0, None)
        iy = np.clip(np.minimum(y2[i], y2) - np.maximum(y1[i], y1), 0, None)
        inter = ix * iy
        ub = inter / np.maximum(area[i] + area - inter, 1e-9)      # 上界
        cand = np.nonzero(alive & (ub >= iou_thr))[0]
        for k in cand:
            if k == i:
                continue
            if bev_iou(boxes[i], boxes[k]) >= iou_thr:
                alive[k] = False
    return keep


def _bev_iou_many(box, boxes):
    """单个框与一批框的 BEV IoU；AABB 不相交的直接记 0，避免成千上万次精确多边形求交。"""
    ious = np.zeros(len(boxes), dtype=np.float32)
    if len(boxes) == 0:
        return ious
    a = _aabb(np.asarray(box, dtype=np.float32)[None, :])[0]
    bb = _aabb(boxes)
    ix = np.minimum(a[2], bb[:, 2]) - np.maximum(a[0], bb[:, 0])
    iy = np.minimum(a[3], bb[:, 3]) - np.maximum(a[1], bb[:, 1])
    cand = np.nonzero((ix > 0) & (iy > 0))[0]
    for k in cand:
        ious[k] = bev_iou(box, boxes[k])
    return ious


def decode(cls_preds, box_preds, dir_preds, anchors, conf_thr=0.5, nms_thr=0.05,
           keep_top=200, pre_top=300):
    """(1,N,C)/(1,N,7)/(1,N,2) + anchors -> [(box(7,), score, class_id)]，已做 BEV NMS。

    pre_top 是进 NMS 前的候选上限（按分数截断），否则 conf 低时会把所有 anchor 送进 NMS。
    """
    cls = cls_preds[0].detach().cpu().numpy()
    box = box_preds[0].detach().cpu().numpy()
    dire = dir_preds[0].detach().cpu().numpy()
    prob = 1.0 / (1.0 + np.exp(-np.clip(cls, -30.0, 30.0)))   # 稳定 sigmoid，避免 exp 溢出
    out = []
    for c in range(prob.shape[1]):
        idx = np.nonzero(prob[:, c] >= conf_thr)[0]
        if idx.size == 0:
            continue
        if idx.size > pre_top:                       # 只留分数最高的 pre_top 个候选
            idx = idx[np.argsort(-prob[idx, c])[:pre_top]]
        boxes = _decode_boxes(anchors[idx], box[idx])
        # 朝向二分类：index 1 表示 yaw 在 +pi/2 一侧
        flip = dire[idx].argmax(1) == 1
        boxes[flip, 6] += np.pi
        scores = prob[idx, c]
        for k in nms_bev(boxes, scores, iou_thr=nms_thr, keep_top=keep_top):
            out.append((boxes[k], float(scores[k]), int(c)))
    out.sort(key=lambda r: -r[1])
    return out[:keep_top]
