"""帧级（presence）指标的单元测试。

C1 的第二个版本不走「跨模态几何配准」——那条路已证明不可靠（同一批配对
拟合出的尺度一会 1.28 一会 0.63）。改成帧级：一帧里有没有某类目标，
RGB 与 IR 各自判断，再按融合规则合并。这类指标不依赖像素配准，
但必须先证明聚合逻辑本身是对的。
"""

from __future__ import annotations

from railintrude.evaluate import evaluate_presence


def test_evaluate_presence_perfect():
    preds = [{0}, {1}, set()]
    gts = [{0}, {1}, set()]
    out = evaluate_presence(preds, gts)
    assert out["tp"] == 2
    assert out["fp"] == 0
    assert out["fn"] == 0
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0


def test_evaluate_presence_counts_false_positive_and_negative():
    preds = [{0}, set(), {0}]
    gts = [{0}, {0}, set()]
    out = evaluate_presence(preds, gts)
    assert out["tp"] == 1
    assert out["fn"] == 1
    assert out["fp"] == 1
    assert out["precision"] == 0.5
    assert out["recall"] == 0.5
    assert abs(out["f1"] - 0.5) < 1e-9


def test_evaluate_presence_empty_gt_and_empty_pred_is_zero_not_error():
    out = evaluate_presence([set(), set()], [set(), set()])
    assert out["tp"] == 0
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert out["f1"] == 0.0


def test_evaluate_presence_rejects_length_mismatch():
    import pytest

    with pytest.raises(ValueError, match="same length"):
        evaluate_presence([{0}], [{0}, {1}])
