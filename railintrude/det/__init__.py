"""Pillar 化：把 (N,4) 点云编码成 (P,M,C) pillar 张量 + (P,3) 坐标。

对齐 PointPillars (Lang et al., CVPR 2019) 的编码方式，为 8GB 显存做两处收敛：
- 每个点只用 4 维特征 [dx, dy, dz, intensity]（原文另加 dcx, dcy, dzc）；
- pillar 数量由配置封顶，超限按点数从多到少截断。
"""
from __future__ import annotations

import numpy as np


def pillarize(points, voxel_size, point_range, max_pillars, max_points):
    """点云 → (pillars(P,max_points,4), coords(P,3)=[batch, ix, iy], num_points(P,))."""
    vx, vy, _vz = (float(v) for v in voxel_size)
    xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in point_range)
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return (np.zeros((0, max_points, 4), np.float32),
                np.zeros((0, 3), np.int32),
                np.zeros((0,), np.int32))
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be (N, >=3)")
    if pts.shape[1] == 3:
        pts = np.concatenate([pts, np.zeros((len(pts), 1), np.float32)], axis=1)

    x, y, z, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    keep = ((x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            & (z >= zmin) & (z < zmax))
    if not keep.any():
        return (np.zeros((0, max_points, 4), np.float32),
                np.zeros((0, 3), np.int32),
                np.zeros((0,), np.int32))
    x, y, z, inten = x[keep], y[keep], z[keep], inten[keep]

    H = int(np.ceil((xmax - xmin) / vx))
    W = int(np.ceil((ymax - ymin) / vy))
    ix = np.floor((x - xmin) / vx).astype(np.int64)
    iy = np.floor((y - ymin) / vy).astype(np.int64)

    order = np.argsort(ix * W + iy, kind="stable")
    x, y, z, inten = x[order], y[order], z[order], inten[order]
    ix, iy = ix[order], iy[order]

    _uniq, inv = np.unique(ix * W + iy, return_inverse=True)
    counts = np.bincount(inv)
    if len(counts) > max_pillars:
        top = np.argsort(-counts)[:max_pillars]
        remap = np.full(len(counts), -1, np.int64)
        remap[top] = np.arange(len(top))
        sel = remap[inv] >= 0
        x, y, z, inten = x[sel], y[sel], z[sel], inten[sel]
        ix, iy = ix[sel], iy[sel]
        inv = remap[inv[sel]]

    n_pillars = int(inv.max()) + 1
    cx = xmin + (ix + 0.5) * vx
    cy = ymin + (iy + 0.5) * vy
    feats = np.stack([x - cx, y - cy, z, inten], axis=1).astype(np.float32)

    pillars = np.zeros((n_pillars, max_points, 4), np.float32)
    num_points = np.zeros(n_pillars, np.int32)
    coords = np.full((n_pillars, 3), -1, np.int32)
    for p, f, xi, yi in zip(inv, feats, ix, iy):
        if num_points[p] < max_points:
            pillars[p, num_points[p]] = f
            num_points[p] += 1
        if coords[p, 0] < 0:
            coords[p, 0], coords[p, 1], coords[p, 2] = 0, xi, yi
    return pillars, coords, num_points
