"""类别选择：从标注统计里挑出要训练的类别。

为什么要测：OSDaR23 扩数据后冒出了 animal(30) / bicycle(24) / signal_bridge(20)
这类只有几十个框的稀有类。让它们进数据集会污染 mAP 而且和上一轮实验不可比——
上一轮是 5 类。所以类别选择要能被显式钉住，并且顺序稳定（类别索引写进数据集就不再变）。
"""

from __future__ import annotations

from railintrude.dataset import select_classes


def test_select_classes_keeps_only_frequent_classes() -> None:
    # animal 30 个框其实过得了 15 的门槛，所以这里用真正过不了门槛的 8 个框
    counter = {"person": 1919, "signal_pole": 1263, "signal_bridge": 8}

    assert select_classes(counter, min_instances=15) == ["person", "signal_pole"]


def test_select_classes_honours_required_list_and_order() -> None:
    # 钉死类别时按给定顺序返回，保证类别索引前后一致（0 buffer_stop / 1 catenary_pole ...）
    counter = {"person": 1919, "signal_pole": 1263, "buffer_stop": 320, "animal": 30}
    required = ["buffer_stop", "catenary_pole", "person", "signal_pole"]

    assert select_classes(counter, min_instances=15, required=required) == [
        "buffer_stop",
        "person",
        "signal_pole",
    ]


def test_select_classes_returns_empty_when_nothing_qualifies() -> None:
    assert select_classes({"animal": 3}, min_instances=15) == []
