"""Ask a small VLM what it sees on OSDaR23 frames, then score it as a detector.

The point is a fair, cheap comparison against the trained YOLOv8n: presence level
(is there a person / a pole in this frame?) rather than box mAP, because a 0.5B VLM
cannot output reliable boxes. Frame-level truth comes from the YOLO-format labels.

Usage:
  python tools/eval_vlm.py --model HuggingFaceTB/SmolVLM-500M-Instruct --limit 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "osdar23_yolo" / "images" / "val"
LBL_DIR = ROOT / "data" / "osdar23_yolo" / "labels" / "val"
NAMES = ["buffer_stop", "catenary_pole", "person", "road_vehicle", "signal_pole"]

# One closed question per class keeps the answer parseable and the comparison honest.
# signal_pole is excluded: it is present in 155/155 val frames, so "always yes" already
# scores 1.000 and the question measures nothing.
PROMPTS = {
    "buffer_stop": "Is there a buffer stop (a rail end stop) visible in this image? Answer with one word: yes or no.",
    "catenary_pole": "Is there a catenary pole or an overhead line mast visible in this image? Answer with one word: yes or no.",
    "person": "Is there a person visible in this image? Answer with one word: yes or no.",
    "road_vehicle": "Is there a road vehicle such as a car or truck visible in this image? Answer with one word: yes or no.",
}
CLASS_IDS = {
    "buffer_stop": 0,
    "catenary_pole": 1,
    "person": 2,
    "road_vehicle": 3,
    "signal_pole": 4,
}


def frame_truth(label_path: Path) -> set[int]:
    classes: set[int] = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.add(int(line.split()[0]))
    return classes


def parse_yes_no(text: str) -> bool | None:
    low = text.strip().lower()
    head = low[:20]
    if "yes" in head:
        return True
    if "no" in head:
        return False
    # Fall back to a looser scan: some checkpoints answer in a full sentence.
    if low.startswith("yes") or "there is a" in low:
        return True
    if low.startswith("no") or "there is no" in low:
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="out/vlm/vlm_presence.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[env] device={device} dtype={dtype}")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"[model] {args.model} loaded")

    images = sorted(IMG_DIR.glob("*.jpg"))[: args.limit]
    if not images:
        print(f"[fail] no images under {IMG_DIR}")
        return 1

    rows = []
    for idx, img_path in enumerate(images, 1):
        image = Image.open(img_path).convert("RGB")
        truth = frame_truth(LBL_DIR / (img_path.stem + ".txt"))
        row = {"image": img_path.name, "truth": sorted(truth), "answers": {}}
        for cls_name, prompt in PROMPTS.items():
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=chat, images=[image], return_tensors="pt")
            moved = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()
            }
            with torch.no_grad():
                out = model.generate(**moved, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer = processor.batch_decode(
                out[:, moved["input_ids"].shape[1] :], skip_special_tokens=True
            )[0]
            row["answers"][cls_name] = {
                "raw": answer.strip(),
                "yes": parse_yes_no(answer),
            }
        rows.append(row)
        if idx % 10 == 0:
            print(f"[run] {idx}/{len(images)} frames done")

    # Presence-level P/R/F1 per class, same convention as railintrude.evaluate.evaluate_presence.
    metrics = {}
    for cls_name, cls_id in (("person", 2), ("catenary_pole", 1), ("road_vehicle", 3)):
        tp = fp = fn = 0
        skipped = 0
        for row in rows:
            pred = row["answers"][cls_name]["yes"]
            if pred is None:
                skipped += 1
                continue
            truth = cls_id in row["truth"]
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif truth and not pred:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[cls_name] = {
            "tp": tp, "fp": fp, "fn": fn, "skipped_unparsed": skipped,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "frames": len(rows), "metrics": metrics, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out_path}")
    for name, m in metrics.items():
        print(f"[metric] {name}: P={m['precision']} R={m['recall']} F1={m['f1']} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']} unparsed={m['skipped_unparsed']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
