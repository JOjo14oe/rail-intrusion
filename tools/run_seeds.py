"""Run the same training several times with different seeds, one after another.

Why this exists (G axis): single-seed differences of ~0.02 mAP50 are inside
training noise in this project (v8n vs v11n differed by 0.019 under identical
settings).  So any "we improved it" claim needs a mean and spread over >=3 seeds,
not one lucky run.

Each seed writes its own out dir, so nothing overwrites anything:

  out/g_seed0/metrics.json   runs/detect/osdar23-yolov8n-1359img-seed0/
  out/g_seed1/metrics.json   runs/detect/osdar23-yolov8n-1359img-seed1/
  out/g_seed2/metrics.json   runs/detect/osdar23-yolov8n-1359img-seed2/

Example:
  python tools/run_seeds.py --seeds 0 1 2 --wait-pid 10828
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def pid_alive(pid: int) -> bool:
    """True while the process exists (Windows; no psutil dependency)."""
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def wait_pid(pid: int, poll_s: float = 20.0, timeout_h: float = 6.0) -> None:
    t0 = time.time()
    while pid_alive(pid):
        if time.time() - t0 > timeout_h * 3600:
            print(f"[queue] gave up waiting for pid {pid} after {timeout_h} h", flush=True)
            return
        time.sleep(poll_s)
    print(f"[queue] pid {pid} finished", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/osdar23_yolo43/dataset.yaml")
    ap.add_argument("--weights", default="models/yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers; >4 crashes at the train->val handover (EOFError)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--wait-pid", type=int, default=0, help="wait for this process to exit first (GPU is 8 GB)")
    args = ap.parse_args()

    if args.wait_pid:
        print(f"[queue] waiting for pid {args.wait_pid} to exit before starting", flush=True)
        wait_pid(args.wait_pid)

    failures: list[int] = []
    for seed in args.seeds:
        out = ROOT / f"out/g_seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", str(ROOT / "tools" / "train_osdar23.py"),
            "--data", args.data,
            "--weights", args.weights,
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--workers", str(args.workers),
            "--device", args.device,
            "--seed", str(seed),
            "--name", f"osdar23-yolov8n-1359img-seed{seed}",
            "--out", f"out/g_seed{seed}",
        ]
        print(f"[seed {seed}] start", flush=True)
        t0 = time.time()
        with (out / "train.log").open("w", encoding="utf-8", errors="replace") as fh:
            code = subprocess.call(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = round(time.time() - t0, 1)
        print(f"[seed {seed}] exit={code} elapsed={elapsed}s", flush=True)
        if code != 0:
            failures.append(seed)

    print(f"[queue] done; failed seeds: {failures or 'none'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
