"""OSDaR23 openlabel cuboid → KITTI 3D 标签。

复现对象是 D3 Rail-PillarNet（CMC 2024）：它用 OSDaR23 的 LiDAR 点云做 4 类 3D 检测，
1356 帧按 8:2 分成 1084/272，检测范围 X[-30,354] Y[-25,39] Z[-4,8]（见 docs/readings/D3.md）。
本模块只做“标签侧”的纯粹的几何与格式换算，不做读写盘（盘上流程在 tools/build_kitti_osdar23.py）。
"""

import math

# 论文选的 4 类
CATEGORIES = ("person", "road_vehicle", "train", "buffer_stop")

# 论文 4.2 节设定的检测范围（米），按目标中心过滤
RANGE_X = (-30.0, 354.0)
RANGE_Y = (-25.0, 39.0)
RANGE_Z = (-4.0, 8.0)


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """四元数绕 Z 轴的偏航角（OSDaR23 的 cuboid 用 z-up 右手系，与 KITTI 的 ry 同义）。"""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def within_range(x: float, y: float, z: float) -> bool:
    """检测范围含端点。"""
    return (RANGE_X[0] <= x <= RANGE_X[1]
            and RANGE_Y[0] <= y <= RANGE_Y[1]
            and RANGE_Z[0] <= z <= RANGE_Z[1])


def frame_labels(frame: dict, classes=None) -> list:
    """从一帧 openlabel 里取目标类的 cuboid。

    返回 [(cls, x, y, z, l, w, h, yaw), ...]；l/w/h 对应 cuboid 的 sx/sy/sz。
    cuboid.val 的字段序是 x,y,z,qx,qy,qz,qw,sx,sy,sz（本仓库实测，见 docs/readings/D3.md）。
    """
    want = set(CATEGORIES) if classes is None else set(classes)
    out = []
    for obj in (frame.get("objects") or {}).values():
        for cub in ((obj.get("object_data") or {}).get("cuboid") or []):
            cls = (cub.get("name") or "").split("__")[-1]
            if cls not in want:
                continue
            val = cub.get("val") or []
            if len(val) < 10:
                continue
            x, y, z, qx, qy, qz, qw, sx, sy, sz = (float(v) for v in val[:10])
            out.append((cls, x, y, z, sx, sy, sz, yaw_from_quaternion(qx, qy, qz, qw)))
    return out


def to_kitti_line(cls: str, x: float, y: float, z: float,
                  l: float, w: float, h: float, yaw: float) -> str:
    """一行 KITTI 3D 标签：cls trunc occ alpha bbox4 h w l x y z ry（图像平面字段填占位值）。

    OpenPCDet 读这一点时按 [8:11]=dimensions(h,w,l)、[11:14]=location、[14]=rotation_y 取用，
    所以必须严格用 KITTI 的 h,w,l 顺序，不能按 l,w,h 写。
    """
    return (f"{cls} 0.00 0 0.00 -1.0 -1.0 -1.0 -1.0 "
            f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {yaw:.2f}")

def frame_id_key(name: str):
    """Extract the numeric frame id from a pcd file name or an openlabel frame id.

    OSDaR23 zero pads them inconsistently (labels say "65", files say "065_1631....pcd"),
    so both go through int() before matching.
    """
    head = str(name).split("_", 1)[0]
    return int(head) if head.isdigit() else None
