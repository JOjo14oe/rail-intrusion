"""Run the intrusion demo on an OSDaR23 sequence with lidar-measured distance.

Compared with railintrude.demo (which turns a pixel row into metres through a hand
fitted calibration), this entry point reads the range off the lidar returns that
land inside each detection box, so the printed metres come from a measurement.
Frames without a lidar return inside the box fall back to no distance at all
rather than to an invented one.

Usage:
  python tools/demo_osdar23_lidar.py --seq 2_station_berliner_tor_2.1 --out out/demo_lidar
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from railintrude.fusion import parse_calibration
from railintrude.ins import read_ins_speeds, speed_for_frame
from railintrude.lidar import (
    camera_from_calibration,
    pick_depth,
    project_points,
    read_pcd,
    unproject,
)
from railintrude.risk import Hysteresis, level_from_ttc, severity
from railintrude.track import center_y, fit_track, on_track
from railintrude.vision import YoloDetector

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "osdar23"
FPS = 10.0
COLORS = {
    "notice": (170, 170, 170),
    "alert": (0, 210, 255),
    "warn": (0, 140, 255),
    "emerg": (0, 0, 255),
}


def frame_id_of(path: pathlib.Path) -> int:
    return int(path.name.split("_")[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="2_station_berliner_tor_2.1")
    ap.add_argument("--weights", default="runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt")
    ap.add_argument("--out", default="out/demo_lidar")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--speed", type=float, default=40.0,
                    help="fallback ego speed in km/h when no INS stream is available")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gate-track", action="store_true",
                    help="only escalate targets lying between the lidar-fitted rails "
                         "(experimental: the fit locks onto the wrong track at stations)")
    args = ap.parse_args()

    seq_dir = DATA / args.seq
    if not seq_dir.is_dir():
        raise SystemExit(f"missing sequence {seq_dir}")
    weights = ROOT / args.weights if not pathlib.Path(args.weights).is_absolute() else pathlib.Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"missing weights {weights}")

    cams = parse_calibration((DATA / "seq_1.2" / "calibration.txt").read_text(encoding="utf-8"))
    k_rgb = cams["rgb_center"]["K"]
    t_rgb = camera_from_calibration(cams["rgb_center"]["T"])

    images = sorted((seq_dir / "rgb_center").glob("*.png"), key=frame_id_of)
    pcds = {frame_id_of(p): p for p in (seq_dir / "lidar").glob("*.pcd")}
    ins_dir = seq_dir / "novatel_oem7_inspva"
    speeds = read_ins_speeds(ins_dir) if ins_dir.is_dir() else {}
    speed_source = "ins" if speeds else "constant"
    if args.max_frames:
        images = images[: args.max_frames]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(str(weights), conf=args.conf, device=args.device, imgsz=args.imgsz)
    hyst = Hysteresis(down_frames=2)
    rows = []
    writer = None
    level_counts = {lv: 0 for lv in COLORS}
    track_frames = 0
    last_track = None
    t0 = time.time()
    overlay_points = True

    for i, img_path in enumerate(images):
        fid = frame_id_of(img_path)
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        dets = detector(frame, frame_idx=fid)

        uvi = di = None
        track = None
        pcd = pcds.get(fid)
        if pcd is not None:
            xyz = read_pcd(pcd)
            uvi, di, _ = project_points(xyz, t_rgb, k_rgb)
            track = fit_track(xyz)
            if overlay_points:
                for (u, v), d in zip(uvi, di):
                    if 0 <= u < w and 0 <= v < h:
                        t = min(d / 40.0, 1.0)
                        cv2.circle(frame, (int(u), int(v)), 1,
                                   (int(40 + 60 * t), int(220 - 120 * t), 255), -1)
        if track is not None and args.gate_track:
            track_frames += 1
            last_track = track
            for sign in (-1.0, 1.0):
                rail = np.array([
                    [x, center_y(x, track) + sign * track["half_gauge"], 0.0]
                    for x in np.linspace(4.0, 40.0, 120)
                ])
                uv_r, _, ok_r = project_points(rail, t_rgb, k_rgb)
                for (u, v), good in zip(uv_r, ok_r):
                    if good and 0 <= u < w and 0 <= v < h:
                        cv2.circle(frame, (int(u), int(v)), 2, (0, 255, 0), -1)

        v_frame = (speed_for_frame(speeds, fid) or args.speed) / 3.6
        best = None
        for det in dets:
            x0, y0, x1, y1 = det.xyxy
            depth, n = pick_depth(uvi, di, (x0, y0, x1, y1)) if uvi is not None else (None, 0)
            if depth is None:
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), COLORS["notice"], 2)
                cv2.putText(frame, f"{det.cls_name} {det.conf:.2f} no-depth",
                            (int(x0), max(16, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            COLORS["notice"], 2)
                continue
            foot = np.array([[(x0 + x1) / 2.0, y1]])
            p3 = unproject(foot, np.array([depth]), t_rgb, k_rgb)[0]
            fwd = float(p3[0])
            ttc = fwd / v_frame if v_frame > 0 else float("inf")
            inside = (on_track(fwd, float(p3[1]), track)
                      if (args.gate_track and track is not None) else True)
            level = level_from_ttc(ttc) if inside else "notice"
            if inside and (best is None or severity(level) > severity(best[0])):
                best = (level, det.cls_name, fwd, ttc)
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[level], 2)
            mark = "TRACK" if inside else "off"
            cv2.putText(frame, f"{det.cls_name} {fwd:.1f}m T={ttc:.1f}s {mark}",
                        (int(x0), max(16, int(y0) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[level], 2)

        frame_level = hyst.update(best[0] if best else "notice")
        level_counts[frame_level] += 1
        cv2.rectangle(frame, (0, 0), (w, 8), COLORS[frame_level], -1)
        if best:
            head = (f"{frame_level.upper()}  {best[1]}  {best[2]:.1f} m  TTC {best[3]:.1f} s"
                    f"  (lidar + {v_frame * 3.6:.0f} km/h {speed_source})")
            rows.append([fid, f"{fid / FPS:.1f}", frame_level, best[1], f"{best[2]:.2f}", f"{best[3]:.2f}"])
        else:
            head = f"{frame_level.upper()}  no target"
        cv2.rectangle(frame, (0, 10), (w, 46), (0, 0, 0), -1)
        cv2.putText(frame, head, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[frame_level], 2)

        if writer is None:
            writer = cv2.VideoWriter(str(out_dir / "demo_lidar.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        writer.write(frame)
        if i % 25 == 0:
            print(f"[{i + 1}/{len(images)}] frame {fid} dets={len(dets)} level={frame_level}")

    if writer is not None:
        writer.release()
    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "level", "cls", "distance_m", "ttc_s"])
        w.writerows(rows)
    summary = {
        "sequence": args.seq,
        "frames": len(images),
        "weights": str(weights),
        "level_counts": level_counts,
        "elapsed_s": round(time.time() - t0, 1),
        "distance_source": "lidar",
        "speed_source": speed_source,
        "track_fit_frames": track_frames,
        "track_rmse_m": round(last_track["rmse"], 3) if last_track else None,
        "speed_kmh": {
            "min": round(min(speeds.values()), 1) if speeds else args.speed,
            "max": round(max(speeds.values()), 1) if speeds else args.speed,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
