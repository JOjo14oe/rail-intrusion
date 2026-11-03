"""C2: temporal comparison of two alarm policies (per-frame threshold vs tracked+TTC).

C2 claims "temporal risk estimation beats per-frame thresholding", so the difference
has to be measured. Both policies consume the SAME per-frame detections:
  A frame_only: alarm whenever any detection lands in the ROI and is closer than --near-m
  B tracked   : IouTracker + min_hits gate + TTC levels + hysteresis (railintrude.pipeline)

What is compared is not "who is right" but alarm jitter and earliness on one clip:
  events            number of alarm events (jitter inflates it)
  flicker           events shorter than 0.5 s
  alarm_seconds     total alarmed duration
  duration_median   median event length
  lead_median_s     per event, time from first alarm to the closest-approach frame

Usage:
    python tools/eval_temporal_c2.py --weights runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from railintrude.evaluate import segments  # noqa: E402
from railintrude.geometry import DistanceCalibration, RoiPolygon  # noqa: E402
from railintrude.pipeline import DEFAULT_CALIB, DEFAULT_ROI, Config, evaluate_frame  # noqa: E402
from railintrude.risk import Hysteresis  # noqa: E402
from railintrude.vision import IouTracker, YoloDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def summarise(flags: list[bool], fps: float, nearest: list[float]) -> dict:
    events = segments(flags)
    durations = [(e - s + 1) / fps for s, e in events]
    lead = []
    for s, e in events:
        window = nearest[s:e + 1]
        if not window:
            continue
        best = min(range(len(window)), key=lambda i: window[i])
        lead.append(best / fps)
    return {
        "alarm_frames": int(sum(flags)),
        "alarm_seconds": round(sum(durations), 2),
        "events": len(events),
        "flicker": sum(1 for d in durations if d < 0.5),
        "duration_median_s": round(statistics.median(durations), 2) if durations else 0.0,
        "duration_max_s": round(max(durations), 2) if durations else 0.0,
        "lead_median_s": round(statistics.median(lead), 2) if lead else 0.0,
        "lead_max_s": round(max(lead), 2) if lead else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/platform-arrival-1080p.webm")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--speed", type=float, default=40.0)
    ap.add_argument("--near-m", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="out/osdar23_yolo/temporal_c2.json")
    args = ap.parse_args()

    roi = RoiPolygon.from_pairs(DEFAULT_ROI)
    calib = DistanceCalibration.from_two_points(**DEFAULT_CALIB)
    detector = YoloDetector(str(ROOT / args.weights), conf=args.conf, device=args.device, imgsz=args.imgsz)

    cfg = Config(roi=roi, calib=calib, speed_kmh=args.speed)
    tracker = IouTracker(iou_thresh=cfg.iou_thresh, max_age=cfg.max_age)
    hyst = Hysteresis(cfg.down_frames)

    cap = cv2.VideoCapture(str(ROOT / args.video))
    if not cap.isOpened():
        print(f"[error] cannot open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_flags: list[bool] = []
    tracked_flags: list[bool] = []
    nearest: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        dets = detector(frame, idx)

        # policy A: per-frame threshold, no tracking, no hysteresis
        near = []
        for d in dets:
            x, y = d.bottom_center
            if not roi.contains(x, y):
                continue
            dist = calib.distance(y)
            if not math.isnan(dist):
                near.append(dist)
        nearest.append(min(near) if near else float("inf"))
        frame_flags.append(bool(near) and min(near) <= args.near_m)

        # policy B: tracker + min_hits + TTC levels + hysteresis
        _, level = evaluate_frame(dets, cfg, tracker, hyst, w, h)
        tracked_flags.append(level in ("warn", "emerg"))

        idx += 1
        if args.max_frames and idx >= args.max_frames:
            break
    cap.release()

    out = {
        "video": args.video,
        "frames": idx,
        "fps": round(fps, 2),
        "seconds": round(idx / fps, 2),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "near_m": args.near_m,
        "frame_only": summarise(frame_flags, fps, nearest),
        "tracked": summarise(tracked_flags, fps, nearest),
    }
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[video] {args.video}  frames={idx}  fps={fps:.2f}  seconds={idx / fps:.1f}")
    for arm in ("frame_only", "tracked"):
        m = out[arm]
        print(f"{arm:11s} events={m['events']:4d} flicker={m['flicker']:4d} "
              f"alarm_s={m['alarm_seconds']:7.2f} dur_med={m['duration_median_s']:5.2f} "
              f"dur_max={m['duration_max_s']:5.2f} lead_med={m['lead_median_s']:5.2f} "
              f"lead_max={m['lead_max_s']:5.2f}")
    print(f"[out] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
