"""Build the 30 s answer-pitch video: title card + clips + end card."""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
W, H = 1280, 720

TITLE = [
    ("基于激光雷达测距与 INS 测速的车端侵限报警原型", 34),
    ("", 10),
    ("YOLOv8n@1280 检测  →  点云反投影测距  →  INS 航向测速  →  TTC 四级报警", 20),
    ("数据：OSDaR23（DZSF / Digitale Schiene Deutschland）  ·  实车多传感器，10 Hz", 18),
]

CLIPS = [
    ("14", "14_signals_station_14.1", "信号机站：接触网杆 / 信号杆 / 站台行人"),
    ("9", "9_station_ruebenkamp_9.1", "站台：行人密集，距离由点云给出"),
    ("4", "4_station_pedestrian_bridge_4.3", "人行天桥区段：结构物与远处行人"),
    ("21", "21_station_wedel_21.1", "站台进站：目标最靠近时触发紧急级"),
]

END = [
    ("已验证", 26),
    ("点云↔图像几何吻合；距离米级实测；速度为 INS 实测；104 项单测通过", 19),
    ("", 10),
    ("未验证（材料中必须写明）", 26),
    ("「是否站在运行轨内」为画面 ROI 近似；公开数据无侵限事件真值；融合未超纯可见光", 19),
    ("", 10),
    ("检测 mAP50 0.703（516 张 / 155 张验证集，研究性质）", 19),
]


def card(lines: list[tuple[str, int]], bg: tuple[int, int, int] = (18, 20, 26)) -> np.ndarray:
    font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    def font(size: int) -> ImageFont.FreeTypeFont:
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(FONT, size)
        return font_cache[size]

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    heights = [size * 1.6 if text else size for text, size in lines]
    y = (H - sum(heights)) / 2
    for (text, size), line_h in zip(lines, heights):
        if text:
            f = font(size)
            tw = draw.textlength(text, font=f)
            draw.text(((W - tw) / 2, y), text, font=f, fill=(238, 240, 245))
        y += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def label_strip(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    draw = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 26)
    draw.rectangle([(0, H - 46), (W, H)], fill=(0, 0, 0))
    draw.text((16, H - 40), text, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def read_clip(path: pathlib.Path, want: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != W or fr.shape[0] != H:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        return []
    if len(frames) >= want:
        idx = np.linspace(0, len(frames) - 1, want).round().astype(int)
        return [frames[i] for i in idx]
    out = list(frames)
    while len(out) < want:
        out.extend(frames)
    return out[:want]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/demo_show")
    ap.add_argument("--out", default="materials/demo-30s.mp4")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--title-seconds", type=float, default=4.0)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--end-seconds", type=float, default=5.0)
    ap.add_argument("--extra", default="2_station_berliner_tor_2.1")
    ap.add_argument("--extra-file", default="out/demo_final/demo_lidar.mp4")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    jobs: list[tuple[pathlib.Path, str, int]] = [
        (pathlib.Path(args.extra_file), "站台：行人 16.5 m / 车速 19.2 km/h（本次演示主序列）",
         int(args.clip_seconds * args.fps))
    ]
    for tag, seq, label in CLIPS:
        jobs.append((root / tag / "demo_lidar.mp4", f"{seq} — {label}",
                     int(args.clip_seconds * args.fps)))

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("cannot open writer")

    n_title = int(args.title_seconds * args.fps)
    for _ in range(n_title):
        writer.write(card(TITLE))

    used = 0
    for path, label, want in jobs:
        if not path.is_file():
            print(f"[skip] {path}")
            continue
        frames = read_clip(path, want)
        if not frames:
            print(f"[skip-empty] {path}")
            continue
        for fr in frames:
            writer.write(label_strip(fr, label))
        used += len(frames)
        print(f"[clip] {path.name} {len(frames)} frames")

    for _ in range(int(args.end_seconds * args.fps)):
        writer.write(card(END))
    writer.release()
    print(f"[out] {out_path} clips_frames={used} total_s={(n_title + used + int(args.end_seconds * args.fps)) / args.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
