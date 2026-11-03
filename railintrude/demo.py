"""V0 演示入口：视频进，带框视频 + 事件表出。

用法：
    python -m railintrude.demo --video data/raw/x.webm --out out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import Config, run
from .vision import YoloDetector

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="车端侵限 V0 演示（YOLO 预训练，无需训练）")
    p.add_argument("--video", required=True, help="输入视频路径")
    p.add_argument("--out", default="out", help="输出目录（默认 out/）")
    p.add_argument("--weights", default=str(ROOT / "models" / "yolov8n.pt"))
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--speed", type=float, default=40.0, help="自车速度 km/h")
    p.add_argument("--max-frames", type=int, default=None, help="只跑前 N 帧（调试用）")
    p.add_argument("--device", default="cpu", help="cpu 或 0（显卡序号）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(conf=args.conf, speed_kmh=args.speed)
    detector = YoloDetector(args.weights, conf=args.conf, device=args.device)
    print(f"模型 {args.weights} → 视频 {args.video}")
    summary = run(args.video, out_dir / "demo.mp4", out_dir / "events.csv",
                  cfg, detector=detector, max_frames=args.max_frames)
    print(f"视频：{out_dir / 'demo.mp4'}")
    print(f"事件：{out_dir / 'events.csv'}")
    return 0 if summary["frames"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
