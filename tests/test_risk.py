import pytest

from railintrude.risk import Hysteresis, level_from_ttc


def test_level_boundaries():
    assert level_from_ttc(20.0) == "notice"
    assert level_from_ttc(8.1) == "notice"
    assert level_from_ttc(8.0) == "alert"
    assert level_from_ttc(5.1) == "alert"
    assert level_from_ttc(5.0) == "warn"
    assert level_from_ttc(3.1) == "warn"
    assert level_from_ttc(3.0) == "emerg"
    assert level_from_ttc(0.5) == "emerg"
    assert level_from_ttc(float("inf")) == "notice"


def test_hysteresis_upgrades_immediately():
    h = Hysteresis(down_frames=5)
    assert h.update("notice") == "notice"
    assert h.update("emerg") == "emerg"


def test_hysteresis_holds_level_for_down_frames():
    h = Hysteresis(down_frames=5)
    h.update("emerg")
    assert [h.update("warn") for _ in range(4)] == ["emerg"] * 4
    assert h.update("warn") == "warn"


def test_hysteresis_resets_counter_on_reupgrade():
    h = Hysteresis(down_frames=3)
    h.update("emerg")
    h.update("warn")
    h.update("warn")
    h.update("emerg")
    assert h.update("warn") == "emerg"  # 计数器已清零，重新数


def test_bad_down_frames_is_rejected():
    with pytest.raises(ValueError):
        Hysteresis(down_frames=0)
