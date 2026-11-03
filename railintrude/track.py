"""Fit the running track's centreline from lidar points.

The demo used to decide "is this target on the track" with a trapezoid drawn on
the image.  That is picture geometry, not track geometry: it neither follows a
curve nor knows the gauge.  The rail top plane is the vehicle XY plane in
OSDaR23 (``Z = 0``), and the gauge is a known constant, so the centreline can be
fitted from the returns instead:

  * keep points close to the rail plane inside a forward window,
  * assume the two rails sit at ``centre(x) +/- half_gauge``,
  * assign every point to its nearer rail, refit ``centre(x) = a + b x + c x^2``
    on the de-offset points, and shrink the inlier tolerance as it converges.

Starting tolerance is deliberately loose (1.5 m) because the first guess is a
straight line through the origin; the looser gate still rejects the neighbouring
track and the platform, while letting a curve's outer rail in.  A track whose
points are mostly outside the gate ends up with too few inliers and the caller
falls back to the image trapezoid instead of trusting a bad fit.
"""

import numpy as np

HALF_GAUGE_STANDARD = 0.7175
_TOLERANCES = (1.5, 1.0, 0.6, 0.35, 0.25, 0.2)
_MIN_INLIERS = 12


def center_y(x: float, params: dict) -> float:
    """Lateral offset of the track centre at forward distance ``x``."""
    return params["a"] + params["b"] * x + params["c"] * x * x


def fit_track(
    points,
    half_gauge: float = HALF_GAUGE_STANDARD,
    x_range=(4.0, 40.0),
    z_max: float = 0.4,
) -> dict | None:
    """Fit ``centre(x) = a + b x + c x^2``; ``None`` when the points do not support it."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None

    x_all, y_all, z_all = arr[:, 0], arr[:, 1], arr[:, 2]
    keep = (x_all >= x_range[0]) & (x_all <= x_range[1]) & (np.abs(z_all) <= z_max)
    x, y = x_all[keep], y_all[keep]
    if x.size < _MIN_INLIERS:
        return None

    # Near-field first, then extend: the first guess is a straight line through
    # the origin, which is only valid where the curve has not bent away yet.
    coef = np.array([0.0, 0.0, 0.0])  # [c, b, a]
    for x_max in (12.0, 20.0, 30.0, x_range[1]):
        sel = x <= x_max
        if sel.sum() < _MIN_INLIERS:
            continue
        xs, ys = x[sel], y[sel]
        for tol in (1.0, 0.6, 0.35, 0.25, 0.2):
            centre = coef[0] * xs * xs + coef[1] * xs + coef[2]
            d_plus = ys - (centre + half_gauge)
            d_minus = ys - (centre - half_gauge)
            use_plus = np.abs(d_plus) <= np.abs(d_minus)
            resid = np.where(use_plus, d_plus, d_minus)
            inliers = np.abs(resid) <= tol
            if inliers.sum() < _MIN_INLIERS:
                break
            target = ys[inliers] - np.where(use_plus[inliers], half_gauge, -half_gauge)
            coef = np.polyfit(xs[inliers], target, 2)

    centre = coef[0] * x * x + coef[1] * x + coef[2]
    resid = np.minimum(np.abs(y - (centre + half_gauge)), np.abs(y - (centre - half_gauge)))
    inl = resid <= _TOLERANCES[-1]
    if inl.sum() < _MIN_INLIERS:
        return None
    return {
        "a": float(coef[2]),
        "b": float(coef[1]),
        "c": float(coef[0]),
        "half_gauge": float(half_gauge),
        "n_inliers": int(inl.sum()),
        "rmse": float(np.sqrt(np.mean(resid[inl] ** 2))),
    }


def on_track(x: float, y: float, params: dict | None, margin: float = 0.0) -> bool:
    """True when ``(x, y)`` lies between the two rails (plus optional margin)."""
    if params is None:
        return False
    half = params.get("half_gauge", HALF_GAUGE_STANDARD) + margin
    return abs(y - center_y(x, params)) <= half
