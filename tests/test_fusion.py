"""融合层的最小单测：单应拟合、框映射、晚融合并集规则。"""

import numpy as np

from railintrude.fusion import (apply_affine, fit_homography, fit_similarity, frame_key,
                                ground_homography, map_boxes_h, merge_by_affine,
                                parse_calibration, project_ground)

# 相机坐标系与车体坐标系同轴（x 前进、y 左、z 上），Z=0 为轨面。
_CAM_A = """
CAMERA
data_folder: cam_a
position: [0.0, 0.0, 1.0]
camera_matrix: [100.0, 0, 320.0;
 0, 100.0, 240.0;
 0, 0, 1]
homogeneous transform:
[1.0, 0.0, 0.0, 0.0;
 0.0, 1.0, 0.0, 0.0;
 0.0, 0.0, 1.0, 1.0;
 0, 0, 0, 1]
"""

_CAM_B = """
CAMERA
data_folder: cam_b
position: [0.0, 0.5, 1.0]
camera_matrix: [100.0, 0, 320.0;
 0, 100.0, 240.0;
 0, 0, 1]
homogeneous transform:
[1.0, 0.0, 0.0, 0.0;
 0.0, 1.0, 0.0, 0.5;
 0.0, 0.0, 1.0, 1.0;
 0, 0, 0, 1]
"""

CALIB = _CAM_A + _CAM_B


def test_parse_calibration_reads_intrinsics_and_transform():
    cams = parse_calibration(CALIB)
    assert set(cams) == {"cam_a", "cam_b"}
    assert cams["cam_a"]["K"][0, 0] == 100.0
    assert cams["cam_a"]["K"][0, 2] == 320.0
    assert cams["cam_b"]["T"][1, 3] == 0.5
    assert cams["cam_a"]["T"][2, 3] == 1.0
    assert cams["cam_a"]["T"].shape == (4, 4)


def test_ground_homography_is_identity_for_same_camera():
    cams = parse_calibration(CALIB)
    H = ground_homography(cams["cam_a"], cams["cam_a"])
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-9)


def test_ground_homography_shifts_by_baseline_over_depth():
    cams = parse_calibration(CALIB)
    H = ground_homography(cams["cam_a"], cams["cam_b"])
    # 同一地面点（车体坐标 X=10, Y=0）在两台相机上的像点应满足 v = H u
    p_a = project_ground(cams["cam_a"], 10.0, 0.0)
    p_b = project_ground(cams["cam_b"], 10.0, 0.0)
    proj = H @ np.array([p_a[0], p_a[1], 1.0])
    proj = proj[:2] / proj[2]
    assert np.allclose(proj, p_b, atol=1e-6)
    # 基线 0.5 m、深度 10 m -> 像面上正好右移 0.05 m * fx = 5 px
    assert abs(proj[0] - p_a[0] - 5.0) < 1e-6


def test_frame_key_ignores_per_stream_timestamp():
    rgb = "14_signals_station_14.1__098_1631449460.900000056.jpg"
    ir = "14_signals_station_14.1__098_1631449460.900464720.jpg"
    assert frame_key(rgb) == frame_key(ir) == "14_signals_station_14.1__098"


def test_frame_key_keeps_sequences_apart():
    a = "7_approach_underground_station_7.1__148_1631705117.900464580.jpg"
    b = "7_approach_underground_station_7.2__148_1631705117.900464580.jpg"
    assert frame_key(a) != frame_key(b)


def test_fit_homography_recovers_identity():
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0],
                    [50.0, 40.0], [20.0, 60.0]])
    dst = src.copy()
    H = fit_homography(src, dst)
    assert H.shape == (3, 3)
    got = map_boxes_h([(0.0, 0.0, 100.0, 80.0)], H)[0]
    assert np.allclose(got, (0.0, 0.0, 100.0, 80.0), atol=1e-6)


def test_fit_homography_recovers_scaled_translation():
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0],
                    [50.0, 40.0], [20.0, 60.0]])
    dst = src * 2.0 + np.array([10.0, 20.0])
    H = fit_homography(src, dst)
    got = map_boxes_h([(0.0, 0.0, 100.0, 80.0)], H)[0]
    assert np.allclose(got, (10.0, 20.0, 210.0, 180.0), atol=1e-6)


def test_map_boxes_h_transforms_corners_not_just_origin():
    H = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    assert map_boxes_h([(10.0, 10.0, 20.0, 20.0)], H) == [(20.0, 20.0, 40.0, 40.0)]


def test_fit_similarity_recovers_scale_and_translation():
    src = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (200.0, 150.0), (50.0, 80.0)]
    dst = [(10.0 + 1.28 * x, 700.0 + 1.28 * y) for x, y in src]
    A = fit_similarity(src, dst)
    assert abs(A[0, 0] - 1.28) < 1e-4
    assert abs(A[1, 1] - 1.28) < 1e-4
    assert abs(A[0, 1]) < 1e-4
    assert abs(A[0, 2] - 10.0) < 1e-4
    assert abs(A[1, 2] - 700.0) < 1e-4


def test_fit_similarity_rejects_too_few_points():
    try:
        fit_similarity([(1.0, 1.0)], [(2.0, 2.0)])
    except ValueError as exc:
        assert "2" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_apply_affine_maps_all_four_corners():
    A = np.array([[2.0, 0.0, 5.0], [0.0, 3.0, 7.0]])
    out = apply_affine([(10.0, 10.0, 20.0, 30.0)], A)
    assert out == [(25.0, 37.0, 45.0, 97.0)]


def test_merge_by_affine_adds_ir_box_that_maps_far_away():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rgb = [(2, 0.9, (100.0, 100.0, 200.0, 300.0))]
    ir = [(2, 0.8, (600.0, 100.0, 700.0, 300.0))]
    out = merge_by_affine(rgb, ir, A, max_centre_px=40.0)
    assert len(out) == 2


def test_merge_by_affine_drops_ir_duplicate_of_rgb_same_class():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rgb = [(2, 0.6, (100.0, 100.0, 200.0, 300.0))]
    ir = [(2, 0.9, (104.0, 101.0, 201.0, 299.0))]
    out = merge_by_affine(rgb, ir, A, max_centre_px=40.0)
    assert len(out) == 1
    assert out[0][1] == 0.9  # 重复时保留更高置信度，几何仍用 RGB 的框
    assert out[0][2] == (100.0, 100.0, 200.0, 300.0)


def test_merge_by_affine_keeps_ir_box_when_class_differs():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rgb = [(2, 0.6, (100.0, 100.0, 200.0, 300.0))]
    ir = [(3, 0.9, (104.0, 101.0, 201.0, 299.0))]
    out = merge_by_affine(rgb, ir, A, max_centre_px=40.0)
    assert len(out) == 2
