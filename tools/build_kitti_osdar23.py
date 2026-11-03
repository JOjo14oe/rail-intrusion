"""Build a KITTI-format 3D detection set from OSDaR23 openlabel + lidar.

Reproduction target: D3 Rail-PillarNet (CMC 2024), which trains on OSDaR23 LiDAR with
4 classes (person / road_vehicle / train / buffer_stop), 1356 frames split 1084/272 (8:2),
detection range X[-30,354] Y[-25,39] Z[-4,8]. See docs/readings/D3.md.

Layout written (OpenPCDet kitti style):
  <out>/training/velodyne/000000.bin   float32 (N,4) x,y,z,intensity
  <out>/training/label/000000.txt      KITTI label lines
  <out>/ImageSets/train.txt, val.txt
  <out>/stats.json

Deviation to record: intensity is written as 0 (read_pcd returns xyz only), and the
train/val split is a seeded random permutation, because the paper does not publish
its split.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from railintrude.kitti import frame_labels, to_kitti_line, within_range  # noqa: E402
from railintrude.lidar import read_pcd  # noqa: E402


def find_label_json(seq_dir: Path):
    hits = sorted(seq_dir.glob("*_labels.json"))
    return hits[0] if hits else None


def frame_to_pcd(seq_dir: Path):
    """Map frame id -> pcd path.

    PCD names look like 148_1631705117.900464580.pcd, and openlabel frame ids can be
    zero padded differently (labels say "65", file says "065"), so both sides are
    normalised through int().
    """
    out = {}
    lidar = seq_dir / "lidar"
    if not lidar.is_dir():
        return out
    for p in lidar.glob("*.pcd"):
        head = p.name.split("_", 1)[0]
        if head.isdigit():
            out[int(head)] = p
    return out


def collect(args):
    seqs = sorted(p for p in Path(args.data).iterdir() if p.is_dir())
    frames = []
    per_seq = {}
    for seq_dir in seqs:
        label_json = find_label_json(seq_dir)
        pcd_map = frame_to_pcd(seq_dir)
        if label_json is None or not pcd_map:
            continue
        doc = json.loads(label_json.read_text(encoding="utf-8"))
        openlabel = doc.get("openlabel", doc)
        kept = 0
        for fid, frame in (openlabel.get("frames") or {}).items():
            key = int(fid) if str(fid).isdigit() else None
            if key is None or key not in pcd_map:
                continue
            boxes = [b for b in frame_labels(frame) if within_range(b[1], b[2], b[3])]
            if not boxes:
                continue
            frames.append((seq_dir.name, fid, pcd_map[key], boxes))
            kept += 1
        per_seq[seq_dir.name] = kept
        if args.limit_seqs and len(per_seq) >= args.limit_seqs:
            break
    return frames, per_seq


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "osdar23"))
    ap.add_argument("--out", default=str(ROOT / "data" / "kitti" / "osdar23"))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-seqs", type=int, default=0, help="smoke test: stop after N sequences")
    args = ap.parse_args(argv)

    frames, per_seq = collect(args)
    if not frames:
        print("[kitti] no frames collected; is lidar extracted?")
        return 1

    out = Path(args.out)
    velodyne = out / "training" / "velodyne"
    labels = out / "training" / "label"
    for d in (velodyne, labels, out / "ImageSets"):
        d.mkdir(parents=True, exist_ok=True)

    idx = list(range(len(frames)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(round(len(frames) * args.val_ratio))
    val_set = set(idx[:n_val])

    instances = {}
    for i, (seq, fid, pcd, boxes) in enumerate(frames):
        pts = read_pcd(str(pcd))
        arr = np.zeros((pts.shape[0], 4), dtype=np.float32)
        arr[:, :3] = pts.astype(np.float32)
        arr.tofile(str(velodyne / f"{i:06d}.bin"))
        lines = []
        for cls, x, y, z, l, w, h, yaw in boxes:
            lines.append(to_kitti_line(cls, x, y, z, l, w, h, yaw))
            instances[cls] = instances.get(cls, 0) + 1
        (labels / f"{i:06d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if (i + 1) % 100 == 0:
            print(f"[kitti] {i + 1}/{len(frames)} frames", flush=True)

    train_ids = [i for i in range(len(frames)) if i not in val_set]
    val_ids = sorted(val_set)
    (out / "ImageSets" / "train.txt").write_text(
        "\n".join(f"{i:06d}" for i in train_ids) + "\n", encoding="utf-8")
    (out / "ImageSets" / "val.txt").write_text(
        "\n".join(f"{i:06d}" for i in val_ids) + "\n", encoding="utf-8")

    stats = {
        "frames": len(frames),
        "train": len(train_ids),
        "val": len(val_ids),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "instances": instances,
        "per_seq_frames": per_seq,
        "note": "intensity written as 0; split is seeded random (paper does not publish its split)",
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[kitti] frames={len(frames)} train={len(train_ids)} val={len(val_ids)}")
    print(f"[kitti] instances={instances}")
    print(f"[kitti] out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
