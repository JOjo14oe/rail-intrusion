"""网络头输出的 anchor 数目必须等于 make_anchors 在**最终特征图**上生成的 anchor 数。"""
import torch

from railintrude.det.anchors import make_anchors
from railintrude.det.pointpillars import PointPillars

DOWN = 2   # 本实现 backbone 只降 2 倍（原文 PointPillars 为 4 倍，属刻意偏差，见 docs/readings/D3.md）


def _batch(coords, batch_size=1):
    n = len(coords)
    return {"pillars": torch.zeros(n, 8, 4), "coords": torch.as_tensor(coords, dtype=torch.int32),
            "num_points": torch.full((n,), 8, dtype=torch.int32), "batch_size": batch_size}


def test_anchor_count_matches_head_output():
    grid, stride, A, ncls = (64, 96), 2, 8, 4
    feats_hw = (grid[0] // stride // DOWN, grid[1] // stride // DOWN)     # (8, 12)
    anchors, anchor_cls = make_anchors(feats_hw, stride * DOWN)
    model = PointPillars(num_classes=ncls, garrid_hw=grid, bev_stride=stride,
                         anchors_per_cell=A, feat_ch=16, bev_out=32).eval()
    out = model(_batch([[0, 5, 7], [0, 62, 95]]))
    assert out["grid_hw"] == feats_hw
    assert out["cls_preds"].shape == (1, len(anchors), ncls)
    assert out["box_preds"].shape == (1, len(anchors), 7)
    assert out["dir_preds"].shape == (1, len(anchors), 2)
    assert len(anchor_cls) == len(anchors)


def test_anchor_order_is_x_outer_then_y():
    """同一格内先换朝向，下一格先换 y（x 在外层）。"""
    anchors, _ = make_anchors((4, 6), 1)
    a, b = anchors[0], anchors[1]
    assert a[0] == b[0] and a[1] == b[1]          # 同格
    assert abs(a[6] - b[6]) > 1.0                 # 朝向不同（0 vs 90°）
    c = anchors[8]
    assert c[1] != a[1]                           # y 先变
