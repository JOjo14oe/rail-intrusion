"""几何：轨道 ROI 判定、单目距离标定、TTC。纯数学，不依赖 OpenCV。"""

from __future__ import annotations

from dataclasses import dataclass


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """射线法。poly 顶点顺序无关。"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class RoiPolygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError("ROI 多边形至少需要 3 个顶点")

    @classmethod
    def from_pairs(cls, pairs) -> "RoiPolygon":
        return cls(tuple((float(a), float(b)) for a, b in pairs))

    def contains(self, x: float, y: float) -> bool:
        return _point_in_polygon(x, y, list(self.points))


@dataclass(frozen=True)
class DistanceCalibration:
    """地面平面假设下的反比模型 d(y) = a / (y - y_h)。"""

    y_h: float
    a: float

    @classmethod
    def from_two_points(cls, y1: float, d1: float, y2: float, d2: float) -> "DistanceCalibration":
        if d1 <= 0 or d2 <= 0:
            raise ValueError("标定距离必须为正")
        if y1 == y2 or d1 == d2:
            raise ValueError("两个标定点必须不同")
        y_h = (d2 * y2 - d1 * y1) / (d2 - d1)
        a = d1 * (y1 - y_h)
        if a <= 0:
            raise ValueError("标定结果无效：a<=0，检查 y 与 d 的对应关系（越远 y 越小）")
        return cls(y_h=y_h, a=a)

    def distance(self, y: float) -> float:
        if y <= self.y_h:
            return float("inf")
        return self.a / (y - self.y_h)


def ttc(distance_m: float, speed_kmh: float) -> float:
    if speed_kmh <= 0 or distance_m == float("inf"):
        return float("inf")
    return distance_m / (speed_kmh / 3.6)
