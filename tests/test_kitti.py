"""cuboid → KITTI 3D 标签的几何与格式（对齐 Rail-PillarNet 复现口径）"""
import math

from railintrude.kitti import (
    CATEGORIES,
    frame_labels,
    to_kitti_line,
    within_range,
    yaw_from_quaternion,
)


def test_identity_quaternion_gives_zero_yaw():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == 0.0


def test_quarter_turn_gives_half_pi():
    s = math.sin(math.pi / 4)
    assert abs(yaw_from_quaternion(0.0, 0.0, s, s) - math.pi / 2) < 1e-6


def test_frame_labels_filters_classes_and_reads_val():
    frame = {
        "objects": {
            "a": {"object_data": {"cuboid": [{"name": "lidar__cuboid__person",
                                             "val": [30.0, 1.0, 1.0, 0, 0, 0, 1, 1.0, 0.7, 1.9]}]}},
            "b": {"object_data": {"cuboid": [{"name": "lidar__cuboid__signal",
                                             "val": [40.0, 1.0, 1.0, 0, 0, 0, 1, 1, 1, 1]}]}},
            "c": {"object_data": {"bbox": []}},
        }
    }
    got = frame_labels(frame)
    assert len(got) == 1
    cls, x, y, z, l, w, h, yaw = got[0]
    assert cls == "person" and (x, y, z, l, w, h) == (30.0, 1.0, 1.0, 1.0, 0.7, 1.9) and yaw == 0.0


def test_frame_labels_honours_custom_classes():
    frame = {"objects": {"a": {"object_data": {"cuboid": [
        {"name": "lidar__cuboid__train", "val": [100.0, 0.0, 2.0, 0, 0, 0, 1, 62.0, 4.0, 4.0]}]}}}}
    assert frame_labels(frame, classes={"train"})[0][0] == "train"
    assert frame_labels(frame, classes={"person"}) == []


def test_within_range_boundaries():
    assert within_range(-30.0, -25.0, -4.0) is True
    assert within_range(354.1, 0.0, 0.0) is False
    assert within_range(0.0, 39.1, 0.0) is False
    assert within_range(0.0, 0.0, 8.1) is False


def test_to_kitti_line_follows_kitti_field_order():
    line = to_kitti_line("buffer_stop", 10.0, -1.0, 0.5, 1.78, 2.85, 2.50, 0.25)
    parts = line.split()
    assert len(parts) == 15
    assert parts[0] == "buffer_stop" and parts[1] == "0.00" and parts[2] == "0"
    # KITTI: cls trunc occ alpha bbox4 h w l x y z ry
    assert parts[8:11] == ["2.50", "2.85", "1.78"]      # h, w, l
    assert parts[11:14] == ["10.00", "-1.00", "0.50"]   # x, y, z
    assert parts[14] == "0.25"                          # ry


def test_categories_are_the_four_paper_classes():
    assert set(CATEGORIES) == {"person", "road_vehicle", "train", "buffer_stop"}
