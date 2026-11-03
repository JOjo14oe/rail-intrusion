import math

import numpy as np
import pytest

from railintrude.track import center_y, fit_track, on_track

HALF_GAUGE = 0.7175


def _two_rails(n=400, a=0.0, b=0.0, c=0.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(4.0, 40.0, n)
    y_c = a + b * x + c * x * x
    side = rng.choice([-1.0, 1.0], n)
    y = y_c + side * HALF_GAUGE + rng.normal(0.0, noise, n)
    z = rng.normal(0.0, 0.02, n)
    return np.column_stack([x, y, z])


def test_fit_track_recovers_straight_centreline():
    p = fit_track(_two_rails(a=0.3, b=0.01))
    assert p is not None
    assert abs(center_y(20.0, p) - (0.3 + 0.01 * 20.0)) < 0.05


def test_fit_track_recovers_curve():
    p = fit_track(_two_rails(a=0.0, b=0.0, c=0.002))
    assert p is not None
    assert abs(center_y(30.0, p) - 0.002 * 900.0) < 0.10


def test_fit_track_ignores_off_rail_points():
    rng = np.random.default_rng(7)
    junk = np.column_stack([
        rng.uniform(4.0, 40.0, 300),
        rng.uniform(3.0, 6.0, 300),
        rng.uniform(-0.3, 0.3, 300),
    ])
    p = fit_track(np.vstack([_two_rails(a=-0.2), junk]))
    assert p is not None
    assert abs(center_y(20.0, p) + 0.2) < 0.10


def test_fit_track_needs_enough_points():
    assert fit_track(np.zeros((3, 3))) is None


def test_on_track_respects_half_gauge():
    p = fit_track(_two_rails(a=0.0, b=0.0))
    assert p is not None
    assert on_track(20.0, 0.0, p)
    assert on_track(20.0, 0.6, p)
    assert not on_track(20.0, 0.95, p)
    assert not on_track(20.0, -1.4, p)


def test_on_track_uses_local_centre_for_curves():
    p = fit_track(_two_rails(b=0.05))
    assert p is not None
    # centre is at 0.05*30 = 1.5 m at x=30, so a target at y=1.5 is on the track
    assert on_track(30.0, 1.5, p)
    assert not on_track(30.0, 0.0, p)
