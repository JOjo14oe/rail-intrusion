"""把 OSDaR23 三个已落地序列转成 YOLO 数据集（同一个相机视角）。

用法：
  python tools/build_osdar23_yolo.py --out data/osdar23_yolo --camera rgb_center

设计选择（为什么这样做）：
- 只用一个相机（默认 rgb_center）：YOLO 训练/评估要在同一视角内比，混相机等于混分布。
- 类别只保留实例数 >= --min-instances 的：OSDaR23 类别长尾严重，留 1-2 个框的类会让 mAP 变成噪声。
- 按序列内部时间切分（前 70% 训练 / 后 30% 验证）：同一场景都要出现在两边，
  否则 120 帧里留一个 10 帧的整段验证集，指标方差会大到没有意义。
- 图像统一转 jpg：原图 2464x1600 PNG 每张 ~4.6MB，120 张就 550MB；jpg 体积降一个数量级，YOLO 反正要 letterbox 到 640。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from railintrude.dataset import merge_classes, select_classes, to_yolo_lines  # noqa: E402
from railintrude.osdar23 import discover_sequences, parse_openlabel  # noqa: E402
from railintrude.vision import Detection  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "osdar23"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/osdar23_yolo")
    ap.add_argument("--camera", default="rgb_center")
    ap.add_argument("--min-instances", type=int, default=15)
    ap.add_argument("--classes", default="",
                    help="逗号分隔的类别，钉死类别与顺序（同一次实验前后可比）；空=按实例数自动选")
    ap.add_argument("--merge", default="",
                    help="类别合并规格，如 catenary_pole,signal_pole>pole；空=不合并")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-seqs", default="",
                    help="逗号分隔：只允许这些序列产生 val，其余序列全部进 train。"
                         "用来在「同一份验证集」上比「数据量」这一个变量")
    ap.add_argument("--holdout-seqs", default="",
                    help="逗号分隔：这些序列整条只进 val（一帧都不进训练），其余整条只进 train。"
                         "用来度量「训练线路 != 上线线路」时的掉幅（C3 跨序列泛化）")
    ap.add_argument("--max-side", type=int, default=0, help=">0 时等比缩放，0=保持原分辨率")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    import cv2

    merge = merge_classes(args.merge)

    def canon(cls: str) -> str:
        return merge.get(cls, cls)

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out

    # ---- 第一遍：只解析标注，统计类别 ----
    sequences = discover_sequences(DATA)
    if not sequences:
        print(f"在 {DATA} 下没找到任何 *_labels.json，先解压子序列")
        return 2
    print(f"[discover] 找到 {len(sequences)} 个序列")
    per_seq = []
    class_counter: Counter[str] = Counter()
    for name, seq_dir, label_json in sequences:
        img_dir = seq_dir / args.camera
        if not img_dir.is_dir():
            print(f"[skip] 缺 {args.camera} 图像目录: {img_dir}")
            continue
        ol = json.loads(label_json.read_text(encoding="utf-8"))["openlabel"]
        frames = parse_openlabel(ol, args.camera)
        present = [f for f in frames if (img_dir / f.image).is_file()]
        for f in present:
            for cls, _box in f.boxes:
                class_counter[canon(cls)] += 1
        per_seq.append((name, img_dir, present))
        print(f"[scan] {name}: 帧 {len(frames)}，图像在盘 {len(present)}")

    required = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    keep = select_classes(class_counter, args.min_instances, required)
    dropped = {c: n for c, n in class_counter.items() if c not in keep}
    if not keep:
        print("没有类别达到 --min-instances，退出")
        return 2
    classes = {c: i for i, c in enumerate(keep)}
    print(f"[classes] 保留 {len(keep)} 类: {classes}")
    print(f"[classes] 丢弃 {len(dropped)} 类（实例过少）: {dropped}")

    # ---- 第二遍：落盘 ----
    # 重建前清空 images/ 与 labels/：否则上一轮的切分文件会残留在同一目录，
    # 两份 val 混在一起（2026-10-27 实际踩过：旧 holdout 的 21_* 帧留在新 val 里）。
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"camera": args.camera, "classes": classes, "dropped": dropped, "sequences": {}}
    val_allowed = {s.strip() for s in args.val_seqs.split(",") if s.strip()} or None
    # ponytail: --holdout-seqs 的整条切分只靠 stats.json 的输出核对，没有单测覆盖
    # （tools/ 不在 pytest 范围内）；一旦这段逻辑变复杂，就把它挪进 railintrude.dataset。
    holdout = {s.strip() for s in args.holdout_seqs.split(",") if s.strip()} or None
    if holdout:
        stats["holdout_seqs"] = sorted(holdout)
    inst_total: Counter[str] = Counter()
    for name, img_dir, present in per_seq:
        n = len(present)
        if holdout is not None:
            buckets = {"train": [], "val": present} if name in holdout else {"train": present, "val": []}
        elif val_allowed is not None and name not in val_allowed:
            buckets = {"train": present, "val": []}
        else:
            cut = max(1, int(round(n * args.train_ratio)))
            buckets = {"train": present[:cut], "val": present[cut:]}
        seq_stat = {"frames": n, "train": len(buckets["train"]), "val": len(buckets["val"]), "empty_val": 0}
        for split, frames in buckets.items():
            for f in frames:
                src = img_dir / f.image
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                dets = [
                    Detection((x1, y1, x2, y2), canon(cls), 1.0)
                    for cls, (x1, y1, x2, y2) in f.boxes
                    if canon(cls) in classes
                ]
                lines = to_yolo_lines(dets, w, h, classes)
                stem = f"{name}__{src.stem}"
                if args.max_side and max(w, h) > args.max_side:
                    scale = args.max_side / max(w, h)
                    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
                )
                if split == "val" and not lines:
                    seq_stat["empty_val"] += 1
                for cls, _ in f.boxes:
                    if canon(cls) in classes:
                        inst_total[canon(cls)] += 1
        stats["sequences"][name] = seq_stat
        print(f"[write] {name}: train {seq_stat['train']} / val {seq_stat['val']}")
        if n > 0:
            stats["sequences"][name]["first_frame"] = present[0].frame_id
            stats["sequences"][name]["last_frame"] = present[-1].frame_id

    stats["instances"] = dict(inst_total.most_common())
    names_block = "\n".join(f"  {i}: {c}" for c, i in classes.items())
    (out / "dataset.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
