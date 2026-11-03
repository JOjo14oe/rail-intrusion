import pytest

from railintrude.vision import Detection, FakeDetector, IouTracker, iou

BOX = (10.0, 10.0, 50.0, 90.0)


def test_iou_identical_is_one():
    assert iou(BOX, BOX) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou(BOX, (200.0, 200.0, 260.0, 280.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_bottom_center_and_height():
    d = Detection(BOX, "person", 0.9)
    assert d.bottom_center == (30.0, 90.0)
    assert d.height == pytest.approx(80.0)


def test_tracker_keeps_id_for_overlapping_box():
    tracker = IouTracker()
    first = tracker.update([Detection(BOX, "person", 0.9)])
    second = tracker.update([Detection((12.0, 12.0, 52.0, 92.0), "person", 0.8)])
    assert first[0].track_id == 0
    assert second[0].track_id == 0


def test_tracker_assigns_new_id_for_far_box():
    tracker = IouTracker()
    tracker.update([Detection(BOX, "person", 0.9)])
    far = tracker.update([Detection((300.0, 300.0, 340.0, 380.0), "person", 0.9)])
    assert far[0].track_id == 1


def test_track_expires_after_max_age():
    tracker = IouTracker(max_age=2)
    tracker.update([Detection(BOX, "person", 0.9)])
    for _ in range(3):
        tracker.update([])
    revived = tracker.update([Detection(BOX, "person", 0.9)])
    assert revived[0].track_id == 1  # 旧轨迹已过期，不复用 0


def test_tracker_counts_hits_per_track():
    """同一目标连续出现时 hits 递增，供时序门控使用（单帧误检不得升级报警）。"""
    tracker = IouTracker()
    seen = []
    for _ in range(3):
        out = tracker.update([Detection(BOX, "person", 0.9)])
        seen.append(out[0].hits)
    assert seen == [1, 2, 3]


def test_fake_detector_cycles_scripted_frames():
    a = Detection(BOX, "person", 0.9)
    fake = FakeDetector([[a], []])
    assert len(fake(None, 0)) == 1
    assert len(fake(None, 1)) == 0
    assert len(fake(None, 2)) == 1
