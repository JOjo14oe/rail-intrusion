import pytest

from railintrude.geometry import DistanceCalibration, RoiPolygon, ttc

ROI = RoiPolygon.from_pairs([(0.1, 0.5), (0.9, 0.5), (0.98, 1.0), (0.02, 1.0)])


def test_calibration_reproduces_anchor_points():
    cal = DistanceCalibration.from_two_points(y1=0.95, d1=6.0, y2=0.62, d2=50.0)
    assert cal.distance(0.95) == pytest.approx(6.0, rel=1e-9)
    assert cal.distance(0.62) == pytest.approx(50.0, rel=1e-9)


def test_distance_increases_as_point_moves_up():
    cal = DistanceCalibration.from_two_points(0.95, 6.0, 0.62, 50.0)
    assert cal.distance(0.90) < cal.distance(0.70) < cal.distance(0.63)


def test_distance_above_horizon_is_infinite():
    cal = DistanceCalibration.from_two_points(0.95, 6.0, 0.62, 50.0)
    assert cal.distance(cal.y_h) == float("inf")


def test_bad_calibration_is_rejected():
    with pytest.raises(ValueError):
        # 近处的点反而画在更高的位置 -> a<=0，物理上不可能
        DistanceCalibration.from_two_points(0.62, 6.0, 0.95, 50.0)


def test_roi_contains_bottom_center_and_rejects_sky():
    assert ROI.contains(0.5, 0.9)
    assert not ROI.contains(0.5, 0.2)
    assert not ROI.contains(0.005, 0.95)   # 梯形左边界在 y=0.95 处约 x=0.028


def test_ttc_is_distance_over_speed():
    assert ttc(11.111, 40.0) == pytest.approx(1.0, rel=1e-3)
    assert ttc(50.0, 0.0) == float("inf")
    assert ttc(float("inf"), 40.0) == float("inf")
