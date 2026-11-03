"""10 维 pillar 编码（对齐 Rail-PillarNet / CMC 2024 的点编码）：

    [x, y, z, r,  x-xm, y-ym, z-zm,  x-xp, y-yp, z-zp]

其中 (xm,ym,zm) 为该 pillar 内所有点的均值，(xp,yp) 为 pillar 中心、zp 沿用 z 偏移。
原有 `pillarize` 只给 4 维 [x-xp, y-yp, z, r]，本模块是它的加宽版，接口一致。
"""
from __future__ import annotations

import numpy as np


def pillarize10(points, voxel_size, point_range, max_pillars, max_points):
    """点云 → (pillars(P,M,10), coords(P,3)=[batch, ix, iy], num_points(P,))."""
    vx, vy, _vz = (float(v) for v in voxel_size)
    xmin, ymin, zmin, xmax, ymax, zmax = (float(v) for v in point_range)
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return (np.zeros((0, max_points, 10), np.float32),
                np.zeros((0, 3), np.int32), np.zeros(0, np.int32))

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = pts[:, 3] if pts.shape[1] > 3 else np.zeros(len(pts), np.float32)
    keep = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) & (z >= zmin) & (z < zmax)
    x, y, z, r = x[keep], y[keep], z[keep], r[keep]

    ix = ((x - xmin) / vx).astype(np.int32)
    iy = ((y - ymin) / vy).astype(np.int32)
    nx = int((xmax - xmin) / vx)
    vy_n = int((ymax - ymin) / vy)
    key = ix.astype(np.int64) * vy_n + iy

    order = np.argsort(key, kind="stable")
    key_s = key[order]
    new_p = np.zeros(len(key_s), bool)
    new_p[0] = True
    new_p[1:] = key_s[1:] != key_s[:-1]
    starts = np.nonzero(new_p)[0]
    pid_raw = np.cumsum(new_p) - 1

    counts = np.diff(np.append(starts, len(key_s)))
    keep_p = counts >= 1
    if keep_p.sum() == 0:
        return (np.zeros((0, max_points, 10), np.float32),
                np.zeros((0, 3), np.int32), np.zeros(0, np.int32))

    # 按点数从多到少截断
    order_p = np.argsort(-counts, kind="stable")[:max_pillars]
    order_p = order_p[keep_p[order_p]]
    remap = np.full(len(counts), -1, np.int64)
    remap[order_p] = np.arange(len(order_p))
    sel = remap[pid_raw] >= 0

    idx = order[sel]
    p = remap[pid_raw[sel]]
    xs, ys, zs, rs = x[idx], y[idx], z[idx], r[idx]
    ixs, iys = ix[idx], iy[idx]

    # pillar 内均值（按 pillar 聚合）
    n_p = len(order_p)
    sums = np.zeros((n_p, 3), np.float64)
    np.add.at(sums, p, np.stack([xs, ys, zs], 1))
    cnt = np.bincount(p, minlength=n_p).astype(np.float64)
    mean = sums / np.maximum(cnt, 1)[:, None]
    cx = xmin + (ixs + 0.5) * vx
    cy = ymin + (iys + 0.5) * vy

    feats = np.stack([xs, ys, zs, rs,
                      xs - mean[p, 0], ys - mean[p, 1], zs - mean[p, 2],
                      xs - cx, ys - cy, zs - mean[p, 2]], axis=1).astype(np.float32)

    pillars = np.zeros((n_p, max_points, 10), np.float32)
    num_points = np.zeros(n_p, np.int32)
    coords = np.full((n_p, 3), -1, np.int32)
    for f, pi, xi, yi in zip(feats, p, ixs, iys):
        if num_points[pi] < max_points:
            pillars[pi, num_points[pi]] = f
            num_points[pi] += 1
        if coords[pi, 0] < 0:
            coords[pi, 0], coords[pi, 1], coords[pi, 2] = 0, int(xi), int(yi)
    return pillars, coords, num_points
