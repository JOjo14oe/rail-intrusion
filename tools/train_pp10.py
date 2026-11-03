"""用 10 维 pillar 编码跑同一套训练（对齐 Rail-PillarNet 的点编码），不改动主脚本。

runpy 返回的就是 train_pp.py 的模块 globals，所以直接替换其中的 pillarize / PointPillars
两个名字即可让 main() 用上 10 维编码。
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from railintrude.det.pillar10 import pillarize10  # noqa: E402

ns = runpy.run_path(str(ROOT / "tools" / "train_pp.py"), run_name="train_pp_mod")
_orig = ns["PointPillars"]


class PointPillars10(_orig):
    def __init__(self, *a, **kw):
        kw["pillar_dim"] = 10
        super().__init__(*a, **kw)


ns["pillarize"] = pillarize10
ns["PointPillars"] = PointPillars10
sys.argv[0] = str(ROOT / "tools" / "train_pp.py")
ns["main"]()
