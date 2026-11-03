import csv

import cv2
import numpy as np
import pytest

from railintrude.pipeline import Config, evaluate_frame, run
from railintrude.risk import Hysteresis
from railintrude.vision import Detection, FakeDetector, IouTracker

W, H = 160, 120


def make_video(path, n=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.zeros((H, W, 3), np.uint8))
    writer.release()


def person_at(y2):
    return Detection((60.0, y2 - 40.0, 100.0, y2), "person", 0.9)


def test_evaluate_frame_reports_level_and_distance():
    cfg = Config(min_hits=1)
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)
    assert len(states) == 1
    assert states[0].distance_m == pytest.approx(6.0, rel=1e-6)
    assert states[0].ttc_s == pytest.approx(0.54, rel=1e-2)   # 6 m / (40 km/h)
    assert states[0].raw_level == "emerg"
    assert level == "emerg"


def test_single_frame_false_positive_cannot_escalate():
    """单帧误检（微调模型在隧道里凭空报 train）不得升级报警。"""
    cfg = Config(min_hits=3)
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)
    assert len(states) == 1                      # 仍然画框
    assert states[0].raw_level == "notice"       # 但不报警
    assert level == "notice"


def test_persistent_track_escalates_after_min_hits():
    cfg = Config(min_hits=3)
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    levels = [
        evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)[1]
        for _ in range(3)
    ]
    assert levels == ["notice", "notice", "emerg"]


def test_detection_outside_roi_is_ignored():
    cfg = Config()
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.2 * H)], cfg, tracker, hyst, W, H)
    assert states == []
    assert level == "notice"


def test_run_writes_video_and_events(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video, n=8)
    out_mp4, out_csv = tmp_path / "demo.mp4", tmp_path / "events.csv"
    detector = FakeDetector([[person_at(0.95 * H)]] * 8)
    summary = run(video, out_mp4, out_csv, Config(), detector=detector, log=lambda *_: None)
    assert out_mp4.exists() and out_csv.exists()
    assert summary["frames"] >= 7          # 合成视频偶有丢尾帧，与 A1 的 ±1 一致
    assert summary["fps"] > 0
    cap = cv2.VideoCapture(str(out_mp4))
    assert abs(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - summary["frames"]) <= 1
    cap.release()


def test_events_only_written_on_level_change(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video, n=8)
    out_csv = tmp_path / "events.csv"
    detector = FakeDetector([[person_at(0.95 * H)]] * 8)
    run(video, tmp_path / "demo.mp4", out_csv, Config(min_hits=1), detector=detector,
        log=lambda *_: None)
    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1                      # 首帧 notice→emerg 一次，之后等级不变不再写
    assert rows[0]["level"] == "emerg"
    assert rows[0]["cls"] == "person"


def test_render_draws_level_color_bar():
    from railintrude.pipeline import OUT_H, OUT_W, render

    frame = np.zeros((H, W, 3), np.uint8)
    cfg = Config()
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)
    canvas = render(frame, states, level, cfg, frame_idx=0, fps=10.0)
    assert canvas.shape == (OUT_H, OUT_W, 3)
    assert canvas[:24, :, :].max() > 0            # 顶部报警条被填色
    assert canvas.sum() > frame.sum()             # 至少画了点东西
