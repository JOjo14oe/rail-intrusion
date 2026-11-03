"""Print per-class frame counts on the OSDaR23 val split, for judging base rates."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LBL_DIR = ROOT / "data" / "osdar23_yolo" / "labels" / "val"
NAMES = ["buffer_stop", "catenary_pole", "person", "road_vehicle", "signal_pole"]


def main() -> int:
    label_files = sorted(LBL_DIR.glob("*.txt"))
    n = len(label_files)
    present: Counter[str] = Counter()
    for path in label_files:
        classes = {int(line.split()[0]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        for cls_id, name in enumerate(NAMES):
            if cls_id in classes:
                present[name] += 1
    print(f"[frames] {n}")
    for name in NAMES:
        pos = present[name]
        neg = n - pos
        # Frame-level accuracy of the two trivial constant predictors.
        always_yes = pos / n
        always_no = neg / n
        print(f"[class] {name:15s} present={pos:3d} absent={neg:3d}  "
              f"always-yes acc={always_yes:.3f}  always-no acc={always_no:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
