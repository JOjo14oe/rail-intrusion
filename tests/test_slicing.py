"""切片推理的几何部分（先写测试，后写实现）。"""

from railintrude.slicing import remap_boxes, tile_rects


def test_tile_rects_covers_whole_image():
    rects = tile_rects(1920, 1080, nx=2, ny=2, overlap=0.2)

    assert len(rects) == 4
    for x0, y0, x1, y1 in rects:
        assert 0 <= x0 < x1 <= 1920
        assert 0 <= y0 < y1 <= 1080

    # 左上角与右下角都必须被某个切片覆盖
    assert any(x0 == 0 and y0 == 0 for x0, y0, _, _ in rects)
    assert any(x1 == 1920 and y1 == 1080 for _, _, x1, y1 in rects)

    # 中心点必须被覆盖
    cx, cy = 960, 540
    assert any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in rects)


def test_tile_rects_overlap_when_requested():
    no_overlap = tile_rects(1920, 1080, nx=2, ny=2, overlap=0.0)
    with_overlap = tile_rects(1920, 1080, nx=2, ny=2, overlap=0.2)

    # 无重叠时左右两块的宽度之和正好是 1920
    assert no_overlap[0][2] - no_overlap[0][0] + no_overlap[1][2] - no_overlap[1][0] == 1920
    # 有重叠时总覆盖宽度大于 1920
    width_sum = with_overlap[0][2] - with_overlap[0][0] + with_overlap[1][2] - with_overlap[1][0]
    assert width_sum > 1920


def test_tile_rects_rejects_bad_grid():
    import pytest

    with pytest.raises(ValueError):
        tile_rects(100, 100, nx=0, ny=2, overlap=0.0)


def test_remap_boxes_shifts_to_original_frame():
    boxes = [(10.0, 10.0, 50.0, 50.0)]

    assert remap_boxes(boxes, 100, 200) == [(110.0, 210.0, 150.0, 250.0)]


def test_remap_boxes_clips_to_image():
    # 块内框很大，平移后越过原图右/下边界
    boxes = [(10.0, 10.0, 500.0, 500.0)]

    # 切片原点 (100,200)、图幅 200x300 → 右/下越界被裁到 200/300
    assert remap_boxes(boxes, 100, 200, width=200, height=300) == [(110.0, 210.0, 200.0, 300.0)]
