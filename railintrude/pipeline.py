"""管线：检测 → 跟踪 → 距离 → TTC → 分级 → 记录 → 写视频。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field

import cv2

from .geometry import DistanceCalibration, RoiPolygon, ttc
from .risk import Hysteresis, level_from_ttc, severity
from .vision import Detection, IouTracker

OUT_W, OUT_H = 1280, 720
DEFAULT_ROI = ((0.12, 0.50), (0.88, 0.50), (1.0, 1.0), (0.0, 1.0))
DEFAULT_CALIB = {"y1": 0.95, "d1": 6.0, "y2": 0.62, "d2": 50.0}
CSV_FIELDS = ["frame", "time_s", "level", "track_id", "cls", "distance_m", "ttc_s"]

COLORS = {
    "notice": (150, 150, 150),
    "alert": (0, 210, 255),
    "warn": (0, 140, 255),
    "emerg": (0, 0, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Config:
    roi: RoiPolygon = field(default_factory=lambda: RoiPolygon.from_pairs(DEFAULT_ROI))
    calib: DistanceCalibration = field(
        default_factory=lambda: DistanceCalibration.from_two_points(**DEFAULT_CALIB)
    )
    speed_kmh: float = 40.0
    conf: float = 0.5
    min_hits: int = 3          # 轨迹需连续命中这么多帧才允许升级报警（压单帧误检）
    down_frames: int = 5
    max_age: int = 15
    iou_thresh: float = 0.3


@dataclass
class TrackState:
    det: Detection
    distance_m: float
    ttc_s: float
    raw_level: str


def evaluate_frame(dets, cfg: Config, tracker: IouTracker, hysteresis: Hysteresis, width, height):
    """返回（本帧命中 ROI 的目标状态列表，经过滞回后的显示等级）。"""
    states: list[TrackState] = []
    for det in tracker.update(dets):
        x, y = det.bottom_center
        if not cfg.roi.contains(x / width, y / height):
            continue
        distance = cfg.calib.distance(y / height)
        time_to_collision = ttc(distance, cfg.speed_kmh)
        raw = level_from_ttc(time_to_collision)
        if det.hits < cfg.min_hits:
            raw = "notice"      # 新轨迹先观察，不升级；框照画
        states.append(
            TrackState(det=det, distance_m=distance, ttc_s=time_to_collision,
                       raw_level=raw)
        )
    states.sort(key=lambda s: severity(s.raw_level), reverse=True)
    raw_level = states[0].raw_level if states else "notice"
    return states, hysteresis.update(raw_level)


def _event_row(frame_idx, fps, level, states):
    lead = states[0] if states else None
    return {
        "frame": frame_idx,
        "time_s": round(frame_idx / fps, 3) if fps else 0.0,
        "level": level,
        "track_id": lead.det.track_id if lead else "",
        "cls": lead.det.cls_name if lead else "",
        "distance_m": round(lead.distance_m, 2) if lead and lead.distance_m != float("inf") else "",
        "ttc_s": round(lead.ttc_s, 2) if lead and lead.ttc_s != float("inf") else "",
    }


def _fmt(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.1f}"


def render(frame, states, level: str, cfg: Config, frame_idx: int, fps: float):
    """把原帧缩放成 1280x720，叠加框、HUD 与顶部报警条。"""
    canvas = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    sx, sy = OUT_W / frame.shape[1], OUT_H / frame.shape[0]
    color = COLORS[level]
    for st in states:
        x1, y1, x2, y2 = st.det.xyxy
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(canvas, p1, p2, COLORS[st.raw_level], 2)
        label = f"{st.det.cls_name}#{st.det.track_id} {st.det.conf:.2f} d={_fmt(st.distance_m)}m T={_fmt(st.ttc_s)}s"
        cv2.putText(canvas, label, (p1[0], max(p1[1] - 6, 12)), FONT, 0.45, COLORS[st.raw_level], 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (OUT_W, 24), color, -1)
    bar = f"{level.upper()}  speed={cfg.speed_kmh:.0f}km/h  dets={len(states)}"
    cv2.putText(canvas, bar, (8, 17), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"frame {frame_idx}  {fps:.1f}fps"
    if states:
        hud += f"  nearest d={_fmt(states[0].distance_m)}m TTC={_fmt(states[0].ttc_s)}s"
    cv2.putText(canvas, hud, (8, OUT_H - 10), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def run(video_in, out_mp4, out_csv, cfg: Config | None = None, detector=None,
        max_frames: int | None = None, log=print) -> dict:
    if detector is None:
        raise ValueError("必须传入 detector（真实模型或 FakeDetector）")
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer_v = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    if not writer_v.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{out_mp4}")

    tracker = IouTracker(cfg.iou_thresh, cfg.max_age)
    hysteresis = Hysteresis(cfg.down_frames)
    rows: list[dict] = []
    prev_level: str | None = None

    t0 = time.time()
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            states, level = evaluate_frame(detector(frame, idx), cfg, tracker, hysteresis, w, h)
            if level != prev_level:
                rows.append(_event_row(idx, fps, level, states))
                prev_level = level
            writer_v.write(render(frame, states, level, cfg, idx, fps))
            idx += 1
            if idx % 100 == 0:
                log(f"  已处理 {idx} 帧 / {time.time() - t0:.1f}s")
            if max_frames and idx >= max_frames:
                break
    finally:
        cap.release()
        writer_v.release()

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    summary = {"frames": idx, "fps": fps, "elapsed_s": round(elapsed, 2),
               "proc_fps": round(idx / elapsed, 2) if elapsed else 0.0, "events": len(rows)}
    log(f"完成：{summary}")
    return summary
