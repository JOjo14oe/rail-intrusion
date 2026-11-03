import numpy as np
import torch

from railintrude.det.pointpillars import PointPillars


def _batch(P=64, M=32, device="cpu"):
    return {
        "pillars": torch.randn(P, M, 4, device=device),
        "coords": torch.stack([torch.zeros(P, dtype=torch.long, device=device),
                               torch.randint(0, 400, (P,), device=device),
                               torch.randint(0, 2400, (P,), device=device)], 1),
        "num_points": torch.full((P,), M, dtype=torch.long, device=device),
        "batch_size": 1,
    }


def test_forward_shapes():
    m = PointPillars(num_classes=4)
    out = m(_batch())
    b, n, c = out["cls_preds"].shape
    assert b == 1 and c == 4
    assert out["box_preds"].shape == (1, n, 7)
    assert out["dir_preds"].shape == (1, n, 2)
    hb, wb = out["grid_hw"]
    assert n == hb * wb * m.A


def test_grid_hw_follows_garrid_hw_and_stride():
    """backbone 只降 2 倍，输出网格 = (gh/4, gw/4)；每格 = 0.16*4*2 = 1.28 m。"""
    m = PointPillars(num_classes=4, garrid_hw=(40, 240), bev_stride=4)
    out = m(_batch(P=8))
    assert out["grid_hw"] == (5, 30)


def test_anchor_flatten_order_matches_forward():
    """make_anchors 的生成顺序必须与 forward 的展平顺序一致：x 外层、y 中层、anchor 内层。"""
    from railintrude.det.anchors import make_anchors
    from railintrude.det.data import CLASS_NAMES
    m = PointPillars(num_classes=4, garrid_hw=(40, 240), bev_stride=4)
    hb, wb = m(_batch(P=8))["grid_hw"]
    a, _ = make_anchors((hb, wb), 8, class_names=CLASS_NAMES)   # 8*0.16 = 1.28 m/格
    assert len(a) == hb * wb * m.A
    assert a.shape[1] == 7
    assert np.allclose(a[0][:2], a[m.A - 1][:2])          # 同一格内各 anchor 的 x/y 相同（z 按类高不同）
    assert abs((a[m.A][1] - a[0][1]) - 1.28) < 1e-5      # x 外层 -> 下一个是 y 方向一格
    assert a[m.A][0] == a[0][0]


def test_gradients_reach_pfn_and_head():
    m = PointPillars(num_classes=4)
    out = m(_batch())
    loss = out["cls_preds"].sum() + out["box_preds"].sum() + out["dir_preds"].sum()
    loss.backward()
    assert next(m.pfn.parameters()).grad is not None
    assert next(m.head.parameters()).grad is not None
    assert m.pfn.net[0].weight.grad is not None


def test_empty_pillars_still_returns_grid():
    m = PointPillars(num_classes=4, bev_stride=4)
    b = {"pillars": torch.zeros(0, 32, 4), "coords": torch.zeros(0, 3, dtype=torch.long),
         "num_points": torch.zeros(0, dtype=torch.long), "batch_size": 1}
    out = m(b)
    hb, wb = out["grid_hw"]
    assert out["cls_preds"].shape == (1, hb * wb * m.A, 4)
    assert torch.isfinite(out["cls_preds"]).all()


def test_param_count_is_small():
    m = PointPillars(num_classes=4)
    n = sum(p.numel() for p in m.parameters())
    assert 1_000_000 < n < 12_000_000, n
