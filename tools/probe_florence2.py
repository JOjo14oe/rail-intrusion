"""Smoke-test native Florence-2 (transformers>=5) on one OSDaR23 val frame.

transformers 5.x ships its own Florence2ForConditionalGeneration, so we must NOT
pass trust_remote_code=True -- the hub copy of configuration_florence2.py touches
Florence2LanguageConfig.forced_bos_token_id, which no longer exists and crashes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "osdar23_yolo" / "images" / "val"
MODEL_ID = "microsoft/Florence-2-base-ft"


def main() -> int:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[env] torch {torch.__version__} device={device} dtype={dtype}")

    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        print(f"[fail] no images under {IMG_DIR}")
        return 1
    img_path = images[0]
    print(f"[data] {img_path.name}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    model.eval()
    print(f"[model] loaded {MODEL_ID}")

    image = Image.open(img_path).convert("RGB")
    for task in ("<OD>", "<CAPTION>"):
        inputs = processor(text=task, images=image, return_tensors="pt")
        moved = {}
        for key, val in inputs.items():
            if not torch.is_tensor(val):
                moved[key] = val
            elif val.is_floating_point():
                moved[key] = val.to(device, dtype)
            else:
                moved[key] = val.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=moved["input_ids"],
                pixel_values=moved["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
            )
        text = processor.batch_decode(out, skip_special_tokens=False)[0]
        print(f"\n[{task}] raw={text!r}")
        try:
            parsed = processor.post_process_generation(
                text, task=task, image_size=(image.width, image.height)
            )
            for key, val in parsed.items():
                if task == "<OD>":
                    print(f"[{task}] labels={val.get('labels')}")
                    print(f"[{task}] n_boxes={len(val.get('bboxes', []))}")
                else:
                    print(f"[{task}] {val}")
        except Exception as exc:
            print(f"[{task}] post_process failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
