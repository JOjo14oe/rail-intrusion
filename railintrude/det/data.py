"""KITTI 格式的 OSDaR23 LiDAR 数据集读取（对应 tools/build_kitti_osdar23.py 的产物）。

标签行是标准 KITTI 15 列：type trunc occ alpha x1 y1 x2 y2 h w l x y z ry，
本项目只用到 type 与后 7 个数值；box 顺序统一为 [x, y, z, l, w, h, yaw]。

注意内存：一帧点云约 3.4 MB（21 万点 × 4 float），1069 帧全读入约 3.6 GB。
训练必须用 load_ids + read_frame 按帧惰性读取，不要用 load_split。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASS_NAMES = ("buffer_stop", "person", "road_vehicle", "train")

# 原文 Table 4.2 给出的 anchor 尺寸（l, w, h），顺序与 CLASS_NAMES 一致
ANCHOR_DIMS = {
    "buffer_stop": (1.78, 2.85, 2.50),
    "person": (0.88, 0.85, 1.89),
    "road_vehicle": (4.29, 3.06, 2.78),
    "train": (62.95, 3.99, 4.28),
}


def read_label(path: Path):
    """解析一个 KITTI 标签文件，返回 (names, boxes[N,7])。"""
    names, boxes = [], []
    text = Path(path).read_text(encoding="utf-8").splitlines()
    for line in text:
        parts = line.split()
        if len(parts) < 15:
            continue
        name = parts[0]
        if name not in ANCHOR_DIMS:
            continue
        h, w, l = (float(parts[8]), float(parts[9]), float(parts[10]))
        x, y, z, yaw = (float(parts[11]), float(parts[12]),
                        float(parts[13]), float(parts[14]))
        names.append(name)
        boxes.append([x, y, z, l, w, h, yaw])
    arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    return names, arr


def read_points(path: Path) -> np.ndarray:
    """读取 velodyne .bin（N x 4，xyz + intensity）。"""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


def load_ids(root: Path, split: str):
    """只读帧号列表（不碰点云）。"""
    root = Path(root)
    return (root / "ImageSets" / f"{split}.txt").read_text(encoding="utf-8").split()


def read_frame(root: Path, fid: str):
    """按帧惰性读取：返回 (points, names, boxes)。"""
    root = Path(root)
    pts = read_points(root / "training" / "velodyne" / f"{fid}.bin")
    names, boxes = read_label(root / "training" / "label" / f"{fid}.txt")
    return pts, names, boxes


def load_split(root: Path, split: str):
    """一次性读入整个 split（仅供小数据与测试；训练请用 load_ids + read_frame）。"""
    return [(fid, *read_frame(root, fid)) for fid in load_ids(root, split)]
