from railintrude.dataset import (
    CLASS_MAP, select_frames, temporal_split, to_yolo_lines,
)
from railintrude.vision import Detection


def test_class_map_covers_rail_relevant_coco_classes():
    assert CLASS_MAP == {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
                         "bus": 4, "truck": 5, "train": 6}


def test_to_yolo_lines_normalizes_center_and_size():
    det = Detection((100.0, 200.0, 300.0, 400.0), "person", 0.9)
    lines = to_yolo_lines([det], 400, 400)
    assert lines == ["0 0.500000 0.750000 0.500000 0.500000"]


def test_to_yolo_lines_drops_other_coco_classes():
    dets = [Detection((0, 0, 10, 10), "chair", 0.9), Detection((0, 0, 10, 10), "train", 0.8)]
    lines = to_yolo_lines(dets, 100, 100)
    assert len(lines) == 1 and lines[0].startswith("6 ")


def test_to_yolo_lines_accepts_custom_class_map():
    """OSDaR23 的类别不在 COCO 表里，必须能传入自己的映射，否则标签会被静默丢弃。"""
    det = Detection((10, 10, 20, 20), "buffer_stop", 0.9)
    assert to_yolo_lines([det], 100, 100, {"buffer_stop": 3}) == [
        "3 0.150000 0.150000 0.100000 0.100000"
    ]


def test_to_yolo_lines_clamps_box_to_image():
    det = Detection((-20.0, -20.0, 120.0, 50.0), "person", 0.9)
    line = to_yolo_lines([det], 100, 100)[0]
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0
    assert float(w) == 1.0 and 0.0 < float(h) <= 1.0


def test_temporal_split_keeps_order_and_no_overlap():
    frames = list(range(100, 200))
    train, val = temporal_split(frames, train_ratio=0.7)
    assert len(train) == 70 and len(val) == 30
    assert train[-1] < val[0]                      # 按时间切开，验证集在训练集之后
    assert set(train).isdisjoint(val)


def test_select_frames_keeps_all_labelled_and_samples_empty():
    labelled = {10: ["0 0.5 0.5 0.1 0.1"], 30: ["6 0.5 0.5 0.9 0.9"]}
    empty = [0, 6, 12, 18, 24]
    kept = select_frames(labelled, empty, neg_stride=2)
    assert set(kept) == {0, 10, 12, 24, 30}   # 空帧每 2 个取 1 个
    assert kept[10] and kept[30]                # 有目标的帧标签不被动
    assert kept[0] == [] and kept[12] == []     # 空帧作为负样本，标签为空


def test_select_frames_neg_stride_zero_drops_all_empty():
    kept = select_frames({10: ["0 0.5 0.5 0.1 0.1"]}, [0, 6], neg_stride=0)
    assert set(kept) == {10}


def test_temporal_split_needs_frames():
    try:
        temporal_split([], 0.7)
    except ValueError as exc:
        assert "frames" in str(exc)
    else:
        raise AssertionError("空列表应报错")
