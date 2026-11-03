"""复算验证集的每类指标，供报告引用（不训练，只 val）。

注意：Windows 下 ultralytics 的 val 会用 DataLoader 起子进程，
必须把入口包在 `if __name__ == "__main__":` 里，否则 spawn 会重新导入本文件并报
"An attempt has been made to start a new process before the current process has finished its bootstrapping phase"。
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(msg: str) -> None:
    print(f"[step] {msg}", flush=True)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    step("ultralytics imported")

    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "runs" / "detect" / "osdar23-yolov8n-v2" / "weights" / "best.pt")
    imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    data = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data" / "osdar23_yolo" / "dataset.yaml"
    if not data.is_absolute():
        data = ROOT / data
    step(f"weights exists={weights.is_file()} data exists={data.is_file()} imgsz={imgsz}")

    model = YOLO(str(weights))
    step("model loaded")

    r = model.val(data=str(data), split="val", imgsz=imgsz, device=0, verbose=False, plots=False)
    step("val done")

    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        per_class[r.names[int(c)]] = {
            "precision": round(float(r.box.p[i]), 4),
            "recall": round(float(r.box.r[i]), 4),
            "mAP50": round(float(r.box.ap50[i]), 4),
            "mAP50-95": round(float(r.box.ap[i]), 4),
        }

    nt = getattr(r.box, "nt_per_class", None)
    payload = {
        "overall": {k: round(float(v), 4) for k, v in r.results_dict.items()},
        "per_class": per_class,
        "val_instances": int(sum(nt)) if nt is not None else None,
    }
    dst = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "out" / "osdar23_yolo" / "per_class_metrics.json"
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload["overall"], indent=2, ensure_ascii=False), flush=True)
    print("val instances:", payload["val_instances"], flush=True)
    for name, m in per_class.items():
        print(f"{name:16s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"mAP50={m['mAP50']:.3f} mAP50-95={m['mAP50-95']:.3f}", flush=True)
    print("saved:", dst, flush=True)


if __name__ == "__main__":
    main()
