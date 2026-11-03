"""可见光/红外晚融合：IR 框经单应映射到 RGB 像平面，再按类做并集合并。

融合层级的选择依据见 docs/osdar23-getting-started.md §8.4：IR 视场只有 640x480、
只覆盖 RGB 中央约 35%，两相机不同位置有视差，因此不做像素级早融合；晚融合
（各模态各自检测 + 对象级关联）是唯一站得住的做法。

映射矩阵有两条来源：`fit_homography` 用成对标注拟合（受配对错误影响大），
`ground_homography` 用 OSDaR23 官方 calibration.txt 的内外参解析推导（推荐，
不依赖标注质量）。
"""

from __future__ import annotations

import re

import numpy as np

Box = tuple[float, float, float, float]
Detection = tuple[int, float, Box]


def _matrix_after(block: str, key: str, rows: int, cols: int) -> np.ndarray:
    """从 calibration.txt 的一个 CAMERA 段落里取出 key 后面的矩阵字面量。"""
    pos = block.find(key)
    if pos < 0:
        raise ValueError(f"missing {key!r} in CALIBRATION block")
    start = block.find("[", pos)
    end = block.find("]", start)
    if start < 0 or end < 0:
        raise ValueError(f"malformed matrix for {key!r}")
    values = [float(v) for v in re.split(r"[,;\s]+", block[start + 1:end].strip()) if v]
    if len(values) != rows * cols:
        raise ValueError(f"{key!r} has {len(values)} numbers, expected {rows * cols}")
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def parse_calibration(text: str) -> dict[str, dict]:
    """解析 calibration.txt：返回 {data_folder: {"K": 3x3 内参, "T": 4x4 相机->车体}}。

    坐标系原文：X=前进方向，Y=左，Z=上，XY 平面即轨面（Z=0）。
    """
    cams: dict[str, dict] = {}
    for block in text.split("CAMERA")[1:]:
        name = re.search(r"data_folder:\s*([^\s]+)", block)
        if name is None:
            continue
        cams[name.group(1)] = {
            "K": _matrix_after(block, "camera_matrix", 3, 3),
            "T": _matrix_after(block, "homogeneous transform", 4, 4),
        }
    if not cams:
        raise ValueError("no CAMERA block found in calibration text")
    return cams


def _plane_matrix(cam: dict) -> np.ndarray:
    """把一个地面点 (X, Y, 1) 映射到该相机的齐次像点。

    OSDaR23 的相机坐标系与车体坐标系同轴（x 前进、y 左、z 上），因此像点是
    u = cx - fx * y/x、v = cy - fy * z/x，而不是 OpenCV 的 z 朝前模型。
    """
    K = np.asarray(cam["K"], dtype=np.float64)
    T = np.asarray(cam["T"], dtype=np.float64)
    Rt = T[:3, :3].T
    t = T[:3, 3]
    M = np.column_stack([Rt[:, 0], Rt[:, 1], -Rt @ t])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    proj = np.array([[cx, -fx, 0.0],
                     [cy, 0.0, -fy],
                     [1.0, 0.0, 0.0]])
    return proj @ M


def project_ground(cam: dict, x: float, y: float) -> tuple[float, float]:
    """把车体坐标系里的轨面点 (x, y, 0) 投到该相机像平面。"""
    p = _plane_matrix(cam) @ np.array([x, y, 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def ground_homography(src_cam: dict, dst_cam: dict) -> np.ndarray:
    """由官方标定推导轨面单应矩阵（src 像素 -> dst 像素），不依赖任何标注。"""
    H = _plane_matrix(dst_cam) @ np.linalg.inv(_plane_matrix(src_cam))
    return H / H[2, 2]


def frame_key(image_name: str) -> str:
    """把 `<序列>__<帧号>_<该流自己的时间戳>.jpg` 归一成 `<序列>__<帧号>`。

    同一个物理帧在不同流里时间戳不同（rgb ...460.900000056 / ir ...460.900464720），
    所以配对键必须只取序列名与帧号。
    """
    stem = image_name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if stem.lower().endswith((".jpg", ".jpeg", ".png")):
        stem = stem.rsplit(".", 1)[0]
    if "__" not in stem:
        return stem
    seq, rest = stem.split("__", 1)
    return f"{seq}__{rest.split('_', 1)[0]}"


def fit_homography(src, dst) -> np.ndarray:
    """用至少 4 组对应点拟合单应矩阵（IR 像素 -> RGB 像素）。"""
    import cv2

    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 1, 2)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 1, 2)
    if len(src_arr) < 4:
        raise ValueError(f"need at least 4 point pairs, got {len(src_arr)}")
    H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 3.0)
    if H is None:
        raise ValueError("homography estimation failed")
    return H


def map_boxes_h(boxes, H: np.ndarray) -> list[Box]:
    """把框的四个角经 H 变换后取外接矩形。"""
    H = np.asarray(H, dtype=np.float64)
    out: list[Box] = []
    for box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]])
        proj = corners @ H.T
        w = proj[:, 2:3]
        w[np.abs(w) < 1e-12] = 1e-12
        xy = proj[:, :2] / w
        out.append((float(xy[:, 0].min()), float(xy[:, 1].min()),
                    float(xy[:, 0].max()), float(xy[:, 1].max())))
    return out


def fit_similarity(src, dst) -> np.ndarray:
    """RANSAC 拟合 IR->RGB 的相似变换（尺度+平移，旋转自由度极小），返回 2x3。

    用相似变换而不是 8 自由度的单应：同类对象在两侧数量不等、排序错位时，
    单应会退化成带反射的畸形矩阵把错误配对也拟合进去（实测 H 行列式为负）；
    4 自由度的相似变换则能把错配对当外点扔掉（实测外点率 56%）。
    """
    import cv2

    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src_arr) < 2:
        raise ValueError(f"need at least 2 point pairs, got {len(src_arr)}")
    A, _ = cv2.estimateAffinePartial2D(src_arr, dst_arr, method=cv2.RANSAC,
                                       ransacReprojThreshold=8.0, maxIters=5000)
    if A is None:
        raise ValueError("similarity estimation failed")
    return A


def apply_affine(boxes, A: np.ndarray) -> list[Box]:
    """用 2x3 仿射矩阵变换框的四个角，取外接矩形。"""
    A = np.asarray(A, dtype=np.float64)
    out: list[Box] = []
    for box in boxes:
        x0, y0, x1, y1 = (float(v) for v in box)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        pts = corners @ A[:, :2].T + A[:, 2]
        out.append((float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())))
    return out


def merge_by_affine(rgb: list[Detection], ir: list[Detection], A: np.ndarray,
                    max_centre_px: float = 40.0) -> list[Detection]:
    """RGB 为主、IR 补盲：IR 框中心映射到 RGB 后与同类 RGB 框中心距离够近即判重复。"""
    merged: list[Detection] = list(rgb)
    for cls, conf, box in ir:
        mapped = apply_affine([box], A)[0]
        mx = (mapped[0] + mapped[2]) / 2.0
        my = (mapped[1] + mapped[3]) / 2.0
        duplicate = None
        for idx, (r_cls, _r_conf, r_box) in enumerate(merged):
            if r_cls != cls:
                continue
            rx = (r_box[0] + r_box[2]) / 2.0
            ry = (r_box[1] + r_box[3]) / 2.0
            if np.hypot(mx - rx, my - ry) <= max_centre_px:
                duplicate = idx
                break
        if duplicate is None:
            merged.append((cls, conf, mapped))
        elif conf > merged[duplicate][1]:
            merged[duplicate] = (cls, conf, merged[duplicate][2])
    return sorted(merged, key=lambda d: -d[1])
