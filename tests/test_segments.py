"""时序报警事件切分的单元测试（C2 指标的地基）。

C2 要比的是"逐帧阈值"与"跟踪+滞回"两种报警策略。两者都产出一串逐帧布尔量，
所以必须先有一个可信的"把逐帧布尔量切成事件"的函数，否则事件数/持续时长都是错的。
"""

from __future__ import annotations

from railintrude.evaluate import segments


def test_segments_splits_on_false():
    assert segments([True, True, False, True, False, False, True]) == [(0, 1), (3, 3), (6, 6)]


def test_segments_handles_all_false_and_all_true():
    assert segments([False, False]) == []
    assert segments([True, True, True]) == [(0, 2)]


def test_segments_handles_empty():
    assert segments([]) == []


def test_segments_last_flag_true_is_closed():
    assert segments([False, True]) == [(1, 1)]
