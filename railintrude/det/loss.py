"""PointPillars 的三支损失（分类 focal / 回归 smooth-L1 / 朝向二分类）。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BETA = 1 / 9


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t).pow(gamma) * ce).sum()


def smooth_l1_loss(pred, target, beta=BETA):
    return F.smooth_l1_loss(pred, target, beta=beta, reduction="sum")


def pp_loss(cls_preds, box_preds, dir_preds, targets, num_classes, alpha=0.25):
    """cls_preds (1,N,C) / box_preds (1,N,7) / dir_preds (1,N,2) / targets 见 labels.assign。

    方向支只 targets 带 "yaw"（真值朝向）时才监督；否则该项为 0，避免用回归残差反推朝向。
    """
    cls = cls_preds[0]
    box = box_preds[0]
    dire = dir_preds[0]
    n = cls.shape[0]
    pos = targets["pos"]
    hot = torch.zeros(n, num_classes, device=cls.device)
    if pos.any():
        hot[pos, targets["cls"][pos].long()] = 1.0
    npos = max(int(pos.sum()), 1)
    l_cls = focal_loss(cls, hot, alpha=alpha) / npos
    l_reg = smooth_l1_loss(box[pos], targets["reg"][pos]) / npos if pos.any() else cls.new_zeros(())
    yaw_t = targets.get("yaw")
    if pos.any() and yaw_t is not None:
        l_dir = F.binary_cross_entropy_with_logits(
            dire[pos], (yaw_t[pos] > 0).float().unsqueeze(1).expand(-1, 2).to(cls.device)
        ) / max(int(pos.sum()), 1)
    else:
        l_dir = cls.new_zeros(())
    total = l_cls + l_reg + l_dir
    return total, {"cls": float(l_cls.detach()), "reg": float(l_reg.detach()), "dir": float(l_dir.detach())}
