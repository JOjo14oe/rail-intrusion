"""数据读取的惰性契约：load_ids 不碰点云，read_frame 才读一帧。

这条测试守住的是内存：全量读入 1069 帧约 3.6 GB，训练进程会被系统杀掉。
"""
from pathlib import Path

import numpy as np

from railintrude.det.data import (CLASS_NAMES, load_ids, load_split,
                                 read_frame)

DATA = Path(__file__).resolve().parents[1] / "data/pp_kitti"


def test_load_ids_returns_frame_ids_without_points():
    ids = load_ids(DATA, "val")
    assert len(ids) > 0
    assert all(isinstance(i, str) and i.isdigit() for i in ids)


def test_read_frame_reads_one_frame():
    fid = load_ids(DATA, "val")[0]
    pts, names, boxes = read_frame(DATA, fid)
    assert pts.ndim == 2 and pts.shape[1] == 4
    assert all(n in CLASS_NAMES for n in names)
    assert boxes.shape[1] == 7 or boxes.size == 0


def test_load_split_is_the_eager_version_of_the_same_ids():
    ids = load_ids(DATA, "val")[:2]
    eager = load_split(DATA, "val")[:2]
    assert [e[0] for e in eager] == ids
    assert np.isclose(eager[0][1].shape[0], read_frame(DATA, ids[0])[0].shape[0])
