"""分类/回归/方向三支损失：形状、数值、可反传。"""
import math

import torch

from railintrude.det.loss import focal_loss, pp_loss, smooth_l1_loss


def test_focal_loss_matches_analytic_for_single_positive():
    # 单个 anchor、单个类别、gt=1、预测概率 p：loss = -alpha*(1-p)^gamma*log(p)
    logits = torch.tensor([[0.0]], requires_grad=True)
    target = torch.tensor([[1.0]])
    p = torch.sigmoid(logits)
    alpha, gamma = 0.25, 2.0
    want = -alpha * (1 - p) ** gamma * torch.log(p)
    got = focal_loss(logits, target, alpha=alpha, gamma=gamma)
    assert torch.allclose(got, want.sum(), atol=1e-6)


def test_focal_loss_zero_when_perfect():
    z = 20.0
    logits = torch.tensor([[z, -z]])
    target = torch.tensor([[1.0, 0.0]])
    assert focal_loss(logits, target).item() < 1e-6


def test_smooth_l1_zero_and_quadratic_small():
    a = torch.zeros(3, 7)
    assert smooth_l1_loss(a, a).item() == 0.0
    x = torch.full((1, 7), 0.1)
    # beta=1/9 时 |x|<beta 段为 0.5*x^2/beta
    want = 7 * 0.5 * 0.1 ** 2 / (1 / 9)
    assert abs(smooth_l1_loss(x, torch.zeros(1, 7)).item() - want) < 1e-5


def test_pp_loss_finite_and_backward():
    n, ncls = 40, 4
    cls_preds = torch.randn(1, n, ncls, requires_grad=True)
    box_preds = torch.randn(1, n, 7, requires_grad=True)
    dir_preds = torch.randn(1, n, 2, requires_grad=True)
    targets = {
        "pos": torch.zeros(n, dtype=torch.bool),
        "neg": torch.ones(n, dtype=torch.bool),
        "cls": torch.zeros(n, dtype=torch.long),
        "reg": torch.zeros(n, 7),
    }
    targets["pos"][:3] = True
    targets["neg"][:3] = False
    targets["cls"][:3] = 2
    targets["reg"][:3] = torch.randn(3, 7) * 0.1
    loss, parts = pp_loss(cls_preds, box_preds, dir_preds, targets, ncls)
    assert torch.isfinite(loss)
    assert set(parts) == {"cls", "reg", "dir"}
    loss.backward()
    assert cls_preds.grad is not None and box_preds.grad is not None


def test_pp_loss_no_positive_is_finite():
    n, ncls = 8, 4
    t = {"pos": torch.zeros(n, dtype=torch.bool), "neg": torch.ones(n, dtype=torch.bool),
         "cls": torch.zeros(n, dtype=torch.long), "reg": torch.zeros(n, 7)}
    loss, _ = pp_loss(torch.zeros(1, n, ncls), torch.zeros(1, n, 7), torch.zeros(1, n, 2), t, ncls)
    assert math.isfinite(float(loss)) and float(loss) >= 0.0
