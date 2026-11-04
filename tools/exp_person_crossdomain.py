"""修正版小实验：先用已知 person 框反查真实类别 id，重建 person-only val，再算跨域 AP。

已知（m05109 的探针记录）：帧 14_signals_station_14.1__098 里有一个 person 框
RGB 像素 (1361.1, 852.6)-(1384.4, 915.2) → 归一化 cx/w 可反查标签行。
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "osdar23_yolo43"
OUT = ROOT / "data" / "person_val"
W, H = 2464.0, 1600.0
cx, cy = (1361.1 + 1384.4) / 2 / W, (852.6 + 915.2) / 2 / H
bw, bh = (1384.4 - 1361.1) / W, (915.2 - 852.6) / H

lab = SRC / "labels" / "val" / "14_signals_station_14.1__098_1631449460.900000056.txt"
pid = None
for line in lab.read_text().splitlines():
    c, a, b, w_, h_ = map(float, line.split())
    if abs(a - cx) < 0.002 and abs(b - cy) < 0.002 and abs(w_ - bw) < 0.002:
        pid = int(c)
        print(f"[0] 命中 person 框 -> class id = {pid}  (line={line})")
        break
if pid is None:
    print("[0] 未命中，打印该帧所有行"); print(lab.read_text()); sys.exit(1)

# 重建 person-only val（用真实 id）
for sub in ("images/val", "labels/val"):
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        f.unlink()
n_img = n_box = 0
for labp in sorted((SRC / "labels" / "val").glob("*.txt")):
    keep = []
    for line in labp.read_text().splitlines():
        p = line.split()
        if p and int(p[0]) == pid:
            keep.append("0 " + " ".join(p[1:]) + "\n")
    cands = list((SRC / "images" / "val").glob(f"{labp.stem}.*"))
    if not cands:
        continue
    (OUT / "images/val" / cands[0].name).write_bytes(cands[0].read_bytes())
    (OUT / "labels/val" / f"{labp.stem}.txt").write_text("".join(keep), encoding="utf-8")
    n_img += 1
    n_box += len(keep)
print(f"[1] person-only val rebuilt: {n_img} 图 / {n_box} 框 (class id {pid})")

# AP
from ultralytics import YOLO  # noqa: E402

def evaluate(weights, cls_ids, imgsz):
    m = YOLO(weights)
    P, G = [], []
    for img in sorted((OUT / "images/val").glob("*")):
        r = m.predict(str(img), imgsz=imgsz, conf=0.01, iou=0.6, device=0, verbose=False,
                      classes=cls_ids)[0]
        P.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
        h, w = r.orig_shape
        g = []
        for line in (OUT / "labels/val" / f"{img.stem}.txt").read_text().splitlines():
            _, a, b, bw_, bh_ = map(float, line.split())
            g.append([(a - bw_ / 2) * w, (b - bh_ / 2) * h, (a + bw_ / 2) * w, (b + bh_ / 2) * h])
        G.append(np.array(g, float).reshape(-1, 4))

    def iou(A, B):
        xx1 = np.maximum(A[:, None, 0], B[None, :, 0]); yy1 = np.maximum(A[:, None, 1], B[None, :, 1])
        xx2 = np.minimum(A[:, None, 2], B[None, :, 2]); yy2 = np.minimum(A[:, None, 3], B[None, :, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)

    def ap_at(thr):
        scores, tp, npos = [], [], sum(len(g) for g in G)
        for (bx, sc), g in zip(P, G):
            if len(bx) == 0:
                continue
            M = iou(bx, g) if len(g) else np.zeros((len(bx), 0))
            used = np.zeros(len(g), bool)
            for i in np.argsort(-sc):
                scores.append(sc[i])
                if len(g) == 0:
                    tp.append(0); continue
                j = int(np.argmax(M[i])); hit = M[i, j] >= thr and not used[j]
                if hit:
                    used[j] = True
                tp.append(1 if hit else 0)
        if npos == 0 or not scores:
            return 0.0, 0.0, 0.0, 0
        o = np.argsort(-np.array(scores)); tp = np.array(tp)[o]
        ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
        rec = ctp / npos; prec = ctp / np.maximum(ctp + cfp, 1e-9)
        grid = np.linspace(0, 1, 101)
        interp = np.array([prec[rec >= t].max() if np.any(rec >= t) else 0.0 for t in grid])
        return float(interp.mean()) * 100, float(prec[-1]) * 100, float(rec[-1]) * 100, len(scores)

    a50, p, r, n = ap_at(0.5)
    a95 = float(np.mean([ap_at(t)[0] for t in np.arange(0.5, 1.0, 0.05)]))
    print(f"[2] {Path(weights).parent.parent.name}: AP50={a50:.2f} AP50-95={a95:.2f} "
          f"P={p:.2f} R={r:.2f} preds={n} GT={sum(len(g) for g in G)}")
    return a50

evaluate(str(ROOT / "runs/detect/osdar23-yolov8n-1359img-1280/weights/best.pt"), [pid], 1280)
evaluate(str(ROOT / "weights_ext/rail_person.pt"), [0], 640)
