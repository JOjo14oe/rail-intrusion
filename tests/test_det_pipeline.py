"""det 端到端接口：一帧样本 -> (模型输入 batch, 标签 targets)。"""
import numpy as np
import torch

from railintrude.det.anchors import make_anchors
from railintrude.det.pipeline import decode, sample_targets, sample_to_batch


def _sample(P=3, M=4):
    return {
        "pillars": torch.zeros(P, M, 4),
        "coords": torch.tensor([[0, 5, 7], [0, 6, 7], [0, 7, 7]]),
        "num_points": torch.tensor([M, M, 2]),
        "boxes": torch.tensor([[10.0, 0.0, 1.0, 3.0, 2.0, 1.6, 0.0]]),   # x,y,z,l,w,h,yaw
        "names": ["road_vehicle"],
    }


def test_sample_to_batch_marks_batch_index():
    batch = sample_to_batch(_sample())
    assert batch["batch_size"] == 1
    assert batch["pillars"].shape == (3, 4, 4)
    assert (batch["coords"][:, 0] == 0).all()
    assert batch["num_points"].tolist() == [4, 4, 2]


def test_sample_targets_shapes_match_anchor_count():
    # 网格必须覆盖 x=10 m：40×60 格 × 1.28 m = 51 m × 77 m，从 x=-30/-25 起算
    anchors, anchor_cls = make_anchors((40, 60), 8, class_names=("road_vehicle", "person"))
    targets = sample_targets(_sample(), anchors, anchor_cls, ("road_vehicle", "person"))
    n = len(anchors)
    assert targets["pos"].shape == (n,) and targets["pos"].dtype == torch.bool
    assert targets["neg"].shape == (n,)
    assert targets["cls"].shape == (n,)
    assert targets["reg"].shape == (n, 7)
    assert targets["pos"].any()          # 10 m 处的车必须分配到正样本


def test_decode_recovers_gt_box_from_zero_residual():
    """cls 饱和 + reg 全 0 时，decode 出来的框应等于 anchor 本身。"""
    anchors, _ = make_anchors((40, 60), 8, class_names=("road_vehicle", "person"))
    n = len(anchors)
    ncls = 2
    cls = torch.full((1, n, ncls), -10.0)
    m = ((anchors[:, 0] > 10) & (anchors[:, 0] < 11)
         & (np.abs(anchors[:, 1]) < 1.0) & (anchors[:, 6] == 0.0))
    j = int(np.nonzero(m)[0][0])
    cls[0, j, 0] = 10.0
    box = torch.zeros(1, n, 7)
    dire = torch.zeros(1, n, 2)
    out = decode(cls, box, dire, anchors, conf_thr=0.5)
    assert len(out) >= 1
    got = max(out, key=lambda r: r[1])
    assert got[2] == 0
    assert np.allclose(got[0][:2], anchors[j][:2], atol=0.2)
