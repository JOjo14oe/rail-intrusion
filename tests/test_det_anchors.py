"""anchor 生成 / 标签分配 / BEV IoU 与 AP 的单测。"""
import numpy as np

from railintrude.det.anchors import assign, make_anchors
from railintrude.det.metric import ap_40, bev_iou


def test_make_anchors_layout():
    a, cls = make_anchors((2, 3), bev_stride=4)
    assert a.shape == (6 * 4 * 2, 7)          # 格 × 类 × 朝向
    assert cls.shape == (48,)
    # 同格相邻的两个 anchor 是同一类的 0° 与 90°
    assert cls[0] == 0 and cls[1] == 0
    assert a[0, 6] == 0.0 and abs(a[1, 6] - np.pi / 2) < 1e-5
    # 首格中心与尺寸（buffer_stop 1.78 × 2.85 × 2.50）
    assert a[0, 0] == np.float32(-30.0 + 0.5 * 0.64)
    assert a[0, 1] == np.float32(-25.0 + 0.5 * 0.64)
    assert abs(a[0, 2] - 1.25) < 1e-6
    assert (a[0, 3], a[0, 4], a[0, 5]) == (np.float32(1.78), np.float32(2.85), np.float32(2.50))
    # 第 3 个 anchor 进入下一类（person）
    assert cls[2] == 1


def test_assign_exact_hit_is_positive_with_zero_regression():
    a, cls = make_anchors((2, 2), bev_stride=4, class_names=("train",))
    gt = np.array([[a[0, 0], a[0, 1], a[0, 2], 62.95, 3.99, 4.28, 0.0]], dtype=np.float32)
    out = assign(a, cls, gt, ["train"], class_names=("train",))
    idx = np.nonzero(out["pos"])[0]
    # 列车长 63 m，整个小网格里朝向 0° 的 anchor 都与真值高度重叠，故不止一个正样本
    assert len(idx) >= 1
    assert (a[idx, 6] == 0.0).all()            # 只有 0° 朝向命中
    assert (out["cls"][idx] == 0).all()
    # 与真值同心的那个 anchor 回归目标为 0；邻居格只有亚格偏移（≤ 1 格 / 对角线）
    assert np.abs(out["reg"][idx, 3:]).max() < 1e-6          # 尺寸比 = 1 -> log = 0
    assert np.abs(out["reg"][idx, :2]).max() <= 0.64 / 63.0 + 1e-4   # 一格 / 长对角线
    assert np.abs(out["reg"][0]).max() == 0.0
    assert out["neg"].sum() == len(a) - len(idx)


def test_assign_far_object_leaves_everything_negative():
    a, cls = make_anchors((2, 2), bev_stride=4, class_names=("person",))
    gt = np.array([[500.0, 500.0, 1.0, 0.88, 0.85, 1.89, 0.0]], dtype=np.float32)
    out = assign(a, cls, gt, ["person"])
    assert out["pos"].sum() == 0
    assert out["neg"].all()


def test_bev_iou_identical_disjoint_and_half_overlap():
    box = [10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0]
    assert abs(bev_iou(box, box) - 1.0) < 1e-5
    assert bev_iou(box, [100.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0]) == 0.0
    half = [10.0, 0.0, 1.0, 2.0, 2.0, 1.5, 0.0]   # 面积 8 与 4，重叠 4
    assert abs(bev_iou(box, half) - 0.5) < 1e-5


def test_ap40_perfect_and_empty():
    gts = [[0.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0], [20.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0]]
    preds = [(gts[0], 0.9), (gts[1], 0.8)]
    assert abs(ap_40(preds, gts) - 100.0) < 1e-6
    assert ap_40([], gts) == 0.0
    assert ap_40(preds, []) is None
