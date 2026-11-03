"""OSDaR23 openlabel → YOLO 数据集转换的单元测试（TDD：先写测试）。"""

from __future__ import annotations

from railintrude.osdar23 import parse_openlabel


def _ol():
    """最小可用的 openlabel 片段：2 帧，含 rgb_center 与 rgb_left 两种 bbox。"""
    return {
        "openlabel": {
            "frames": {
                "10": {
                    "frame_properties": {
                        "streams": {
                            "rgb_center": {"uri": "/rgb_center/10_1631705117.900000000.png"},
                            "rgb_left": {"uri": "/rgb_left/10_1631705117.900000000.png"},
                        }
                    },
                    "objects": {
                        "u1": {
                            "object_data": {
                                "bbox": [
                                    {
                                        "name": "rgb_center__bbox__person",
                                        "val": [100.0, 200.0, 50.0, 80.0],
                                    },
                                    {
                                        "name": "rgb_left__bbox__person",
                                        "val": [1.0, 2.0, 3.0, 4.0],
                                    },
                                ]
                            }
                        },
                        "u2": {
                            "object_data": {
                                "bbox": [
                                    {
                                        "name": "rgb_center__bbox__buffer_stop",
                                        "val": [0.0, 0.0, 10.0, 10.0],
                                    }
                                ]
                            }
                        },
                    },
                },
                "2": {
                    "frame_properties": {
                        "streams": {"rgb_center": {"uri": "/rgb_center/2_1631705116.800000000.png"}}
                    },
                    "objects": {},
                },
                "5": {
                    "frame_properties": {"streams": {"ir_center": {"uri": "/ir_center/5_x.png"}}},
                    "objects": {},
                },
            }
        }
    }


def test_parse_openlabel_returns_frames_in_numeric_order():
    frames = parse_openlabel(_ol()["openlabel"], "rgb_center")
    assert [f.frame_id for f in frames] == [2, 10]


def test_parse_openlabel_skips_frames_without_the_camera():
    frames = parse_openlabel(_ol()["openlabel"], "rgb_center")
    assert 5 not in [f.frame_id for f in frames]


def test_parse_openlabel_maps_xywh_to_xyxy_and_keeps_only_that_camera():
    frames = parse_openlabel(_ol()["openlabel"], "rgb_center")
    f10 = [f for f in frames if f.frame_id == 10][0]
    assert f10.image == "10_1631705117.900000000.png"
    assert ("person", (100.0, 200.0, 150.0, 280.0)) in f10.boxes
    assert ("buffer_stop", (0.0, 0.0, 10.0, 10.0)) in f10.boxes
    # rgb_left 的框不能混进来
    assert all(b[1] != (1.0, 2.0, 4.0, 6.0) for b in f10.boxes)
    assert len(f10.boxes) == 2


def test_parse_openlabel_yields_empty_boxes_for_unannotated_frame():
    frames = parse_openlabel(_ol()["openlabel"], "rgb_center")
    f2 = [f for f in frames if f.frame_id == 2][0]
    assert f2.boxes == []
