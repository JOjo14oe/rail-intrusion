import numpy as np
from railintrude.det import pillarize

VOX = (0.16, 0.16, 12.0)
RNG = (-30.0, -25.0, -4.0, 354.0, 39.0, 8.0)


def _pts(coords):
    if len(coords) == 0:
        return np.zeros((0, 4), np.float32)
    a = np.array(coords, np.float32)
    return np.concatenate([a, np.ones((len(a), 1), np.float32)], 1)


def test_empty_input_returns_zero_pillars():
    p, c, n = pillarize(_pts([]), VOX, RNG, 100, 32)
    assert p.shape == (0, 32, 4) and c.shape == (0, 3) and n.shape == (0,)


def test_out_of_range_points_dropped():
    p, c, n = pillarize(_pts([(1000.0, 0.0, 1.0)]), VOX, RNG, 100, 32)
    assert len(c) == 0


def test_same_pillar_points_grouped():
    p, c, n = pillarize(_pts([(0.0, 0.0, 1.0), (0.01, 0.01, 1.0), (10.0, 10.0, 1.0)]), VOX, RNG, 100, 32)
    assert len(c) == 2
    assert sorted(n.tolist()) == [1, 2]


def test_max_points_truncates():
    pts = [(0.001 * k, 0.0, 1.0) for k in range(50)]
    p, c, n = pillarize(_pts(pts), VOX, RNG, 100, 8)
    assert len(c) == 1 and n[0] == 8 and p.shape[1] == 8


def test_max_pillars_keeps_densest():
    pts = [(0.0, 0.0, 1.0)] * 5 + [(10.0, 0.0, 1.0)]
    p, c, n = pillarize(_pts(pts), VOX, RNG, 1, 32)
    assert len(c) == 1 and n[0] == 5


def test_coords_are_x_y_grid_indices():
    p, c, n = pillarize(_pts([(0.0, 0.0, 1.0)]), VOX, RNG, 10, 4)
    assert c[0, 0] == 0
    assert c[0, 1] == int(np.floor(30.0 / 0.16))
    assert c[0, 2] == int(np.floor(25.0 / 0.16))
