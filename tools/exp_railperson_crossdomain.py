"""小实验：把迁移包里的 rail_person.pt（合成铁路行人单类）拿到真实 OSDaR23 上测 person。

步骤：
1) 从 C:\\Users\\jojo as\\Desktop\\yolo11_full.zip 解出 rail_person/weights/best.pt → rail-intrusion/weights_ext/
2) 用 OSDaR23 的 val 划分，只保留 person 类，重映射成 0，写成 data/person_val
3) ultralytics 直接 val 该权重，输出 mAP50 / mAP50-95 / P / R
"""
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from railintrude.det.data import CLASS_NAMES, load_ids

ZIP = Path(r"C:\Users\jojo as\Desktop\yolo11_full.zip")
SRC = ROOT / "data" / "osdar23_yolo43"
OUT = ROOT / "data" / "person_val"
WEXT = ROOT / "weights_ext"
WEXT.mkdir(exist_ok=True)

# 1) 解权重
member = next(n for n in zipfile.ZipFile(ZIP).namelist()
              if n.endswith("rail_person/weights/best.pt"))
wpath = WEXT / "rail_person.pt"
if not wpath.exists():
    with zipfile.ZipFile(ZIP) as z, open(wpath, "wb") as f:
        f.write(z.read(member))
print(f"[1] weights -> {wpath} ({wpath.stat().st_size/1e6:.1f} MB)")

# 2) 只留 person 的 val
pid = CLASS_NAMES.index("person")
for sub in ("images/val", "labels/val"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
n_img = n_box = 0
for lab in sorted((SRC / "labels" / "val").glob("*.txt")):
    fid = lab.stem
    keep = []
    if lab.exists():
        for line in io.open(lab, encoding="utf-8"):
            p = line.split()
            if p and int(p[0]) == pid:
                keep.append("0 " + " ".join(p[1:]) + "\n")
    img = next((p for p in (SRC / "images" / "val").glob(f"{fid}.*")), None)
    if img is None:
        continue
    (OUT / "images/val" / img.name).exists() or (OUT / "images/val" / img.name).write_bytes(img.read_bytes())
    io.open(OUT / "labels/val" / f"{fid}.txt", "w", encoding="utf-8").writelines(keep)
    n_img += 1
    n_box += len(keep)
io.open(OUT / "dataset.yaml", "w", encoding="utf-8").write(
    yaml.safe_dump({"path": str(OUT).replace("\\", "/"), "train": "images/val",
                    "val": "images/val", "names": {0: "person"}}, allow_unicode=True))
print(f"[2] person-only val: {n_img} 图 / {n_box} 个 person 框 -> {OUT}")

# 3) 评测
from ultralytics import YOLO          # noqa: E402
m = YOLO(str(wpath))
r = m.val(data=str(OUT / "dataset.yaml"), imgsz=1280, batch=4, device=0, plots=False, verbose=False)
d = r.results_dict
print("[3] rail_person.pt on OSDaR23 person:")
for k, v in d.items():
    if "metrics/" in k or "fitness" in k:
        print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")
