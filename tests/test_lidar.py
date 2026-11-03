import numpy as np

from railintrude.lidar import (
    depth_at_foot,
    project_points,
    read_pcd,
    to_optical,
    unproject,
)

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
I4 = np.eye(4)


def test_to_optical_maps_forward_to_z():
    # non-optical (x forward, y left, z up) -> optical (x right, y down, z forward)
    p = np.array([[10.0, 2.0, 3.0]])
    assert np.allclose(to_optical(p), [[-2.0, -3.0, 10.0]])


def test_project_points_puts_centre_on_principal_point():
    uv, depth, ok = project_points(np.array([[10.0, 0.0, 0.0]]), I4, K)
    assert np.allclose(uv, [[50.0, 50.0]])
    assert np.allclose(depth, [10.0])
    assert ok.tolist() == [True]


def test_project_points_puts_left_point_left_of_centre():
    # a point 1 m to the left (y = +1) must land left of the principal point
    uv, _, _ = project_points(np.array([[10.0, 1.0, 0.0]]), I4, K)
    assert uv[0, 0] < 50.0


def test_project_points_drops_points_behind_camera():
    uv, depth, ok = project_points(np.array([[-5.0, 0.0, 0.0]]), I4, K)
    assert ok.tolist() == [False]
    assert len(uv) == 0


def test_unproject_round_trips_through_projection():
    xyz = np.array([[12.0, -1.5, 0.4], [30.0, 0.7, -0.2]])
    uv, depth, _ = project_points(xyz, I4, K)
    back = unproject(uv, depth, I4, K)
    assert np.allclose(back, xyz, atol=1e-6)


def test_depth_at_foot_uses_points_inside_box():
    uv = np.array([[10.0, 10.0], [50.0, 90.0], [52.0, 92.0], [200.0, 200.0]])
    depth = np.array([5.0, 20.0, 22.0, 60.0])
    box = (40.0, 80.0, 60.0, 100.0)
    assert abs(depth_at_foot(uv, depth, box) - 21.0) < 1e-9


def test_depth_at_foot_returns_none_without_points():
    uv = np.zeros((0, 2))
    depth = np.zeros((0,))
    assert depth_at_foot(uv, depth, (0.0, 0.0, 100.0, 100.0)) is None


def test_read_pcd_reads_ascii_points(tmp_path):
    path = tmp_path / "tiny.pcd"
    path.write_text(
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 2\n"
        "DATA ascii\n"
        "1.0 2.0 3.0 0.5\n"
        "4.0 5.0 6.0 0.1\n",
        encoding="ascii",
    )
    xyz = read_pcd(path)
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz[1], [4.0, 5.0, 6.0])

def test_pick_depth_returns_near_quantile_and_count():
    from railintrude.lidar import pick_depth

    uv = np.array([[50.0, 50.0]] * 5)
    depth = np.array([9.0, 7.0, 5.0, 8.0, 6.0])
    d, n = pick_depth(uv, depth, (40.0, 40.0, 60.0, 60.0), min_points=3, quantile=0.2)
    assert n == 5
    assert d == 6.0


def test_pick_depth_returns_none_when_box_is_empty():
    from railintrude.lidar import pick_depth

    uv = np.array([[5.0, 5.0]])
    depth = np.array([10.0])
    assert pick_depth(uv, depth, (40.0, 40.0, 60.0, 60.0)) == (None, 0)


def test_pick_depth_needs_min_points():
    from railintrude.lidar import pick_depth

    uv = np.array([[50.0, 50.0], [51.0, 51.0]])
    depth = np.array([10.0, 11.0])
    assert pick_depth(uv, depth, (40.0, 40.0, 60.0, 60.0), min_points=3) == (None, 2)
