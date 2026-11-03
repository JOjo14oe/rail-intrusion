"""解压器的路径匹配：两种 zip 内部布局都要认。

回归来源：早期只认 `<序列>/<流>/xxx.png`，扁平布局的 17 个序列解出 0 个文件却报“命中 14 个”。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from extract_osdar23 import in_stream, is_meta  # noqa: E402


def test_nested_layout_matches():
    assert in_stream("14_signals_station_14.1/rgb_center/091_1631449460.200000056.png", ["rgb_center"])


def test_flat_layout_matches():
    assert in_stream("rgb_center/224_1631703821.500000008.png", ["rgb_center"])


def test_other_stream_does_not_match():
    assert not in_stream("14_signals_station_14.1/ir_center/091.png", ["rgb_center"])


def test_split_does_not_match_partial_name():
    assert not in_stream("rgb_highres_center/091.png", ["rgb_center"])


def test_metadata_recognised():
    assert is_meta("14_signals_station_14.1_labels.json")
    assert is_meta("14_signals_station_14.1/calibration.txt")
    assert not is_meta("rgb_center/091.png")
