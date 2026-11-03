"""Lidar <-> image geometry for OSDaR23.

The chain is the one documented in data/osdar23/seq_1.2/readme.md (lines 180-196)
and validated visually by out/probe/probe_lidar_proj.py: projected points land on
platform edges, people, poles and buildings only when the calibration 4x4 is
*inverted* first. So the calibration file stores camera -> parent ("pose_wrt_parent"),
and lidar -> camera is its inverse.

  p_nono = inv(T_calib) @ p_lidar        non-optical frame: x fwd, y left, z up
  p_opt  = (-y, -z, x)                   optical frame: x right, y down, z fwd
  u = fx * x_opt / z_opt + cx,  v = fy * y_opt / z_opt + cy

dist_coeffs in the release are all zero, so no undistortion is needed.
"""
import numpy as np


def read_pcd(path):
    """Read an OSDaR23 .pcd (ascii or binary) and return an (N, 3) float array."""
    with open(path, "rb") as f:
        header = {}
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"truncated pcd header: {path}")
            line = raw.decode("ascii", "ignore").strip()
            if not line:
                continue
            key, _, val = line.partition(" ")
            header[key.upper()] = val
            if key.upper() == "DATA":
                break
        mode = header["DATA"].strip()
        if mode == "ascii":
            data = np.loadtxt(f, usecols=(0, 1, 2), ndmin=2)
            return data[:, :3].astype(np.float64)
        if mode != "binary":
            raise ValueError(f"unsupported DATA={mode}")

        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header["COUNT"].split()] if "COUNT" in header else [1] * len(fields)
        dtype = []
        for name, size, typ, cnt in zip(fields, sizes, types, counts):
            code = {"F": "f", "U": "u", "I": "i"}[typ]
            if cnt != 1:
                dtype.append((name, f"<{code}{size}", (cnt,)))
            else:
                dtype.append((name, f"<{code}{size}"))
        n = int(header["POINTS"])
        pts = np.frombuffer(f.read(), dtype=np.dtype(dtype), count=n)
        return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)


def to_optical(p_nono):
    """non-optical (x fwd, y left, z up) -> optical (x right, y down, z fwd)."""
    return np.stack([-p_nono[:, 1], -p_nono[:, 2], p_nono[:, 0]], axis=1)


def from_optical(p_opt):
    """optical -> non-optical, the inverse of to_optical."""
    return np.stack([p_opt[:, 2], -p_opt[:, 0], -p_opt[:, 1]], axis=1)


def camera_from_calibration(T_calib):
    """The calibration file stores camera -> parent; lidar -> camera is the inverse."""
    return np.linalg.inv(np.asarray(T_calib, dtype=np.float64))


def project_points(xyz, T_l2c, K):
    """Project lidar points into the image.

    Returns (uv, depth, ok) where ok marks points in front of the camera.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    T_l2c = np.asarray(T_l2c, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if len(xyz) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool)
    homo = np.hstack([xyz, np.ones((len(xyz), 1))])
    p_nono = (T_l2c @ homo.T).T[:, :3]
    p_opt = to_optical(p_nono)
    z = p_opt[:, 2]
    ok = z > 1e-6
    u = K[0, 0] * p_opt[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * p_opt[ok, 1] / z[ok] + K[1, 2]
    return np.stack([u, v], axis=1), z[ok], ok


def unproject(uv, depth, T_l2c, K):
    """Back-project pixels at a given optical depth into the lidar frame."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    p_opt = np.stack(
        [
            (uv[:, 0] - K[0, 2]) * depth / K[0, 0],
            (uv[:, 1] - K[1, 2]) * depth / K[1, 1],
            depth,
        ],
        axis=1,
    )
    p_nono = from_optical(p_opt)
    homo = np.hstack([p_nono, np.ones((len(p_nono), 1))])
    return (np.linalg.inv(np.asarray(T_l2c, dtype=np.float64)) @ homo.T).T[:, :3]


def pick_depth(uv, depth, box, min_points=3, quantile=0.2):
    """Depth of a detection read off the lidar returns inside its box.

    Takes a near-side quantile rather than the median: a detection usually has the
    background behind it inside the same box, and the object is the nearer surface.
    Returns ``(depth, n_points)``; ``(None, n)`` when fewer than ``min_points``
    returns land in the box, because one or two stray rays are not a measurement.
    """
    if len(uv) == 0:
        return None, 0
    x0, y0, x1, y1 = box
    inside = (
        (uv[:, 0] >= x0)
        & (uv[:, 0] <= x1)
        & (uv[:, 1] >= y0)
        & (uv[:, 1] <= y1)
    )
    n = int(np.count_nonzero(inside))
    if n < min_points:
        return None, n
    d = np.sort(depth[inside])
    idx = int(round(quantile * (n - 1)))
    return float(d[idx]), n


def depth_at_foot(uv, depth, box, pad_frac=0.0, band=0.5):
    """Median optical depth of the lidar points in the lower part of a box.

    `band` is the fraction of the box height, measured up from the bottom edge,
    that counts as "near the foot". The default 0.5 keeps the original lower-half
    behaviour; a small band (e.g. 0.15) is better for tall thin targets, where the
    lower half is mostly background several tens of metres behind the object.

    Returns None when no lidar point hits that band (sky, glass, long range).
    """
    if len(uv) == 0:
        return None
    x0, y0, x1, y1 = box
    pad = pad_frac * (x1 - x0)
    mid = y1 - band * (y1 - y0)
    inside = (
        (uv[:, 0] >= x0 - pad)
        & (uv[:, 0] <= x1 + pad)
        & (uv[:, 1] >= mid)
        & (uv[:, 1] <= y1)
    )
    if not np.any(inside):
        return None
    return float(np.median(depth[inside]))
