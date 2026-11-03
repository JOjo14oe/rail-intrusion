"""切片推理的几何工具：把大图切成带重叠的块，再把块内坐标映射回原图。

为什么需要：接触网立柱、信号杆这类细长目标在 640 输入下只剩几十个像素，
切片后同样的目标占据更大比例，检测器才有机会看见。
"""

from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple[float, float, float, float]


def tile_rects(width: int, height: int, nx: int = 2, ny: int = 2, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """返回 nx×ny 个 (x0, y0, x1, y1)，保证覆盖整幅图且相邻块重叠 overlap 比例。"""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    def axis(total: int, n: int) -> list[tuple[int, int]]:
        tile = total / (n * (1.0 - overlap) + overlap)
        step = tile * (1.0 - overlap)
        out = []
        for i in range(n):
            start = round(i * step)
            end = total if i == n - 1 else round(start + tile)
            out.append((min(start, total - 1), min(end, total)))
        return out

    xs = axis(width, nx)
    ys = axis(height, ny)
    return [(x0, y0, x1, y1) for (x0, x1) in xs for (y0, y1) in ys]


def remap_boxes(
    boxes: Iterable[Sequence[float]],
    x0: int,
    y0: int,
    width: int | None = None,
    height: int | None = None,
) -> list[Box]:
    """把块内坐标平移回原图；给了 width/height 就顺带裁到图内。"""
    out: list[Box] = []
    for bx0, by0, bx1, by1 in boxes:
        nx0, ny0, nx1, ny1 = bx0 + x0, by0 + y0, bx1 + x0, by1 + y0
        if width is not None:
            nx0, nx1 = max(0.0, min(nx0, width)), max(0.0, min(nx1, width))
        if height is not None:
            ny0, ny1 = max(0.0, min(ny0, height)), max(0.0, min(ny1, height))
        out.append((nx0, ny0, nx1, ny1))
    return out
