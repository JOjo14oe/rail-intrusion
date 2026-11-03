"""Balanced VLM probe: same questions, but on class-balanced frames.

The first probe (tools/eval_vlm.py) scored a 0.5B VLM on the raw val split, where
92% of frames contain a person -- a constant "yes" predictor beat it. This script
removes that artefact: for every class it takes an equal number of frames where the
class is present and absent, so chance is 0.5 and an always-yes predictor scores
F1 = 0.667. It also asks for a count, because "how many" cannot be answered by
guessing a single label.

Usage:
  python tools/eval_vlm_balanced.py --limit 24
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "osdar23_yolo" / "images" / "val"
LBL_DIR = ROOT / "data" / "osdar23_yolo" / "labels" / "val"
CLASS_IDS = {"buffer_stop": 0, "catenary_pole": 1, "person": 2, "road_vehicle": 3}

PRESENCE = {
    "buffer_stop": "Is there a buffer stop (a rail end stop) visible in this image? Answer with one word: yes or no.",
    "catenary_pole": "Is there a catenary pole or an overhead line mast visible in this image? Answer with one word: yes or no.",
    "person": "Is there a person visible in this image? Answer with one word: yes or no.",
    "road_vehicle": "Is there a road vehicle such as a car or truck visible in this image? Answer with one word: yes or no.",
}
COUNT = {
    "person": "How many people are visible in this image? Answer with a single number.",
    "road_vehicle": "How many road vehicles (cars, trucks, vans) are visible in this image? Answer with a single number.",
}


def read_labels(label_path: Path) -> list[int]:
    """Class id per box, so counts are available alongside presence."""
    if not label_path.exists():
        return []
    return [int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_yes_no(text: str) -> bool | None:
    low = text.strip().lower()[:24]
    if "yes" in low:
        return True
    if "no" in low:
        return False
    return None


def parse_count(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group())


def balanced_frames(limit: int) -> dict[str, list[Path]]:
    images = sorted(IMG_DIR.glob("*.jpg"))
    labels = {img: read_labels(LBL_DIR / (img.stem + ".txt")) for img in images}
    picked: dict[str, list[Path]] = {}
    for cls_name, cls_id in CLASS_IDS.items():
        pos = [img for img in images if cls_id in labels[img]]
        neg = [img for img in images if cls_id not in labels[img]]
        n = min(limit, len(pos), len(neg))
        if n == 0:
            print(f"[skip] {cls_name}: pos={len(pos)} neg={len(neg)}")
            continue
        # Deterministic, evenly spaced subsets so the run is reproducible.
        step_pos = max(1, len(pos) // n)
        step_neg = max(1, len(neg) // n)
        picked[cls_name] = pos[::step_pos][:n] + neg[::step_neg][:n]
        print(f"[pick] {cls_name}: {n} present + {n} absent = {2 * n} frames "
              f"(pool pos={len(pos)} neg={len(neg)})")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--limit", type=int, default=24, help="frames per side per class")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="out/vlm/vlm_balanced.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[env] device={device} dtype={dtype}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    picked = balanced_frames(args.limit)
    if not picked:
        print("[fail] nothing to evaluate")
        return 1

    def ask(image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=chat, images=[image], return_tensors="pt")
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**moved, max_new_tokens=args.max_new_tokens, do_sample=False)
        return processor.batch_decode(out[:, moved["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    rows = []
    total = sum(len(v) for v in picked.values())
    done = 0
    for cls_name, images in picked.items():
        cls_id = CLASS_IDS[cls_name]
        for img_path in images:
            image = Image.open(img_path).convert("RGB")
            counts = read_labels(LBL_DIR / (img_path.stem + ".txt"))
            truth_count = counts.count(cls_id)
            presence_raw = ask(image, PRESENCE[cls_name])
            row = {
                "class": cls_name,
                "image": img_path.name,
                "truth": truth_count > 0,
                "truth_count": truth_count,
                "presence_raw": presence_raw,
                "presence": parse_yes_no(presence_raw),
            }
            if cls_name in COUNT:
                count_raw = ask(image, COUNT[cls_name])
                row["count_raw"] = count_raw
                row["count"] = parse_count(count_raw)
            rows.append(row)
            done += 1
            if done % 20 == 0:
                print(f"[run] {done}/{total}")

    def pct(x: float) -> float:
        return round(100.0 * x, 1)

    metrics: dict[str, dict] = {}
    for cls_name in picked:
        subset = [r for r in rows if r["class"] == cls_name and r["presence"] is not None]
        tp = sum(1 for r in subset if r["presence"] and r["truth"])
        fp = sum(1 for r in subset if r["presence"] and not r["truth"])
        fn = sum(1 for r in subset if not r["presence"] and r["truth"])
        tn = sum(1 for r in subset if not r["presence"] and not r["truth"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / len(subset) if subset else 0.0
        entry = {
            "frames": len(subset),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "unparsed": sum(1 for r in rows if r["class"] == cls_name and r["presence"] is None),
            # Balanced subset: guessing "yes" everywhere yields F1 = 0.667, accuracy 0.5.
            "always_yes_f1": 0.6667,
        }
        counted = [r for r in subset if r.get("count") is not None]
        if counted:
            errors = [abs(r["count"] - r["truth_count"]) for r in counted]
            entry["count_frames"] = len(counted)
            entry["count_mae"] = round(sum(errors) / len(errors), 3)
            entry["count_exact_pct"] = pct(sum(1 for e in errors if e == 0) / len(errors))
            entry["count_mean_truth"] = round(sum(r["truth_count"] for r in counted) / len(counted), 2)
            entry["count_mean_pred"] = round(sum(r["count"] for r in counted) / len(counted), 2)
        metrics[cls_name] = entry

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "limit_per_side": args.limit, "metrics": metrics, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out_path}")
    for name, m in metrics.items():
        line = (f"[metric] {name:15s} frames={m['frames']:3d} acc={m['accuracy']:.3f} "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"(always-yes F1 0.667)")
        if "count_mae" in m:
            line += (f" | count: MAE={m['count_mae']} exact={m['count_exact_pct']}% "
                     f"truth_mean={m['count_mean_truth']} pred_mean={m['count_mean_pred']}")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
