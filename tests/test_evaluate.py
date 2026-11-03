"""自写评测器的单元测试：先用合成数据把 AP 的行为钉死，再去测切片。

为什么要自己写：ultralytics 的 val 只吃整图，切片推理的结果它评不了。
自写评测器必须先被证明是对的（否则 0.52 → 0.7 这种数字毫无意义）。
"""

from railintrude.evaluate import box_iou, evaluate_predictions

IOU_THRS = [0.5, 0.75]


def test_box_iou_basic():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # 一半重叠：交 50，并 150
    assert abs(box_iou((0, 0, 10, 10), (0, 0, 10, 5)) - 0.5) < 1e-9


def test_perfect_detections_give_ap_one():
    gts = [[(0, (0, 0, 10, 10)), (0, (20, 20, 40, 40))]]
    dets = [[(0, 0.9, (0, 0, 10, 10)), (0, 0.8, (20, 20, 40, 40))]]

    r = evaluate_predictions(dets, gts, n_classes=1, iou_thrs=IOU_THRS)

    assert r["mAP50"] == 1.0
    assert r["per_class"][0]["ap50"] == 1.0


def test_missed_box_halves_ap():
    # 2 个 GT，只检出 1 个（排在最前）→ 该类的 AP50 应为 0.5 左右
    gts = [[(0, (0, 0, 10, 10)), (0, (100, 100, 120, 120))]]
    dets = [[(0, 0.9, (0, 0, 10, 10))]]

    r = evaluate_predictions(dets, gts, n_classes=1, iou_thrs=IOU_THRS)

    assert abs(r["per_class"][0]["ap50"] - 0.5) < 0.02


def test_wrong_location_gives_zero_ap():
    gts = [[(0, (0, 0, 10, 10))]]
    dets = [[(0, 0.9, (500, 500, 510, 510))]]

    r = evaluate_predictions(dets, gts, n_classes=1, iou_thrs=IOU_THRS)

    assert r["per_class"][0]["ap50"] == 0.0


def test_class_without_ground_truth_is_skipped():
    gts = [[(0, (0, 0, 10, 10))]]
    dets = [[(0, 0.9, (0, 0, 10, 10)), (1, 0.9, (0, 0, 10, 10))]]

    r = evaluate_predictions(dets, gts, n_classes=2, iou_thrs=IOU_THRS)

    # 只有 class 0 参与均值，class 1 无 GT 不计入
    assert r["mAP50"] == 1.0
    assert 1 not in r["per_class"]
