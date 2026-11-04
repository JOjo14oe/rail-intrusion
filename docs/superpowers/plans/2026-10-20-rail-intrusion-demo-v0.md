# 车端侵限检测 V0 演示 · 实施计划

> **给执行者（我自己或新会话的代理）**：本计划用 `writing-plans` 的格式写，配套 `executing-plans` 使用。
> 每个任务都是"先写失败测试 → 跑到红 → 写最小实现 → 跑到绿 → 提交"。

**目标**：把一段列车前视视频，变成一段带检测框、估计距离、TTC、四级报警叠加的视频 + 一张事件表。

**架构**：`视频 → ROI 多边形判定 → 检测器(可替换插槽) → IoU 跟踪 → 两点标定单目距离 → TTC → 分级+滞回 → 渲染/CSV`。
检测器是唯一的重依赖，藏在一个 `Detector` 协议后面；管线其余部分纯数学、可离线单测。

**技术栈**：Python 3.12、PyTorch + ultralytics YOLOv8（权重离线加载 `models/yolov8n.pt`）、OpenCV、pytest。

**规模**：5 个源文件（`geometry/risk/vision/pipeline/demo`）+ 4 个测试文件，约 600 行，预计 3 小时。

**上游**：`docs/superpowers/specs/2026-10-20-rail-intrusion-demo-v0-design.md`

**工作目录**：`rail-intrusion`

---

## 全局约束

- V0 只做推理、不做训练；训练在紧随其后的 V0.5 单独做（同一套代码，只换权重）。
- 检测器固定 ultralytics YOLOv8，权重从 `models/yolov8n.pt` 本地加载；**加载与推理全程不许联网**（`YOLO(path)` 传本地文件即可，不要传模型名）。
- 中文只出现在文档、注释、CSV 里；**视频叠加一律英文**（OpenCV Hershey 字体画不了中文）。
- 每个任务结束都要 `git commit`。
- 任何"跑通了"的判断，必须在终端里看到实际输出，不许凭印象。

---

## Task 0：仓库骨架

**文件**：`.gitignore`、`requirements.txt`、`pytest.ini`、`railintrude/__init__.py`

- [ ] **步骤 1：建目录并初始化 git**

```powershell
cd "rail-intrusion"
git init
New-Item -ItemType Directory -Force -Path railintrude, tests, data\raw, models, out | Out-Null
```

- [ ] **步骤 2：写 `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
models/
out/
data/raw/*
!data/raw/_source_url.txt
```

- [ ] **步骤 3：写 `requirements.txt`**

```
ultralytics
opencv-python
numpy
pytest
```

- [ ] **步骤 4：写 `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **步骤 5：写 `railintrude/__init__.py`**

```python
"""车端侵限检测 V0：零训练通用检测器 + 轨道 ROI + 时序风险分级。"""

__version__ = "0.0.1"
```

- [ ] **步骤 6：验证 pytest 能跑（0 个测试不算失败）**

```powershell
python -m pytest
```

期望：`no tests ran`，退出码 5 属于正常（此时还没有测试文件）。这一步只确认解释器与 pytest 可用。

- [ ] **步骤 7：提交**

```powershell
git add -A
git commit -m "chore: 初始化 rail-intrusion V0 骨架"
```

---

## Task 1：几何模块（ROI / 单目距离 / TTC）

**文件**：`railintrude/geometry.py`、`tests/test_geometry.py`

**接口**：

```python
RoiPolygon(points: tuple[tuple[float, float], ...]).contains(x: float, y: float) -> bool
DistanceCalibration.from_two_points(y1, d1, y2, d2) -> DistanceCalibration
DistanceCalibration.distance(y: float) -> float          # y 为归一化坐标；地平线之上返回 inf
ttc(distance_m: float, speed_kmh: float) -> float        # 秒
```

- [ ] **步骤 1：写失败测试 `tests/test_geometry.py`**

```python
import pytest

from railintrude.geometry import DistanceCalibration, RoiPolygon, ttc

ROI = RoiPolygon.from_pairs([(0.1, 0.5), (0.9, 0.5), (0.98, 1.0), (0.02, 1.0)])


def test_calibration_reproduces_anchor_points():
    cal = DistanceCalibration.from_two_points(y1=0.95, d1=6.0, y2=0.62, d2=50.0)
    assert cal.distance(0.95) == pytest.approx(6.0, rel=1e-9)
    assert cal.distance(0.62) == pytest.approx(50.0, rel=1e-9)


def test_distance_increases_as_point_moves_up():
    cal = DistanceCalibration.from_two_points(0.95, 6.0, 0.62, 50.0)
    assert cal.distance(0.90) < cal.distance(0.70) < cal.distance(0.63)


def test_distance_above_horizon_is_infinite():
    cal = DistanceCalibration.from_two_points(0.95, 6.0, 0.62, 50.0)
    assert cal.distance(cal.y_h) == float("inf")


def test_bad_calibration_is_rejected():
    with pytest.raises(ValueError):
        # 近处的点反而画在更高的位置 -> a<=0，物理上不可能
        DistanceCalibration.from_two_points(0.62, 6.0, 0.95, 50.0)


def test_roi_contains_bottom_center_and_rejects_sky():
    assert ROI.contains(0.5, 0.9)
    assert not ROI.contains(0.5, 0.2)
    assert not ROI.contains(0.005, 0.95)   # 梯形左边界在 y=0.95 处约 x=0.028


def test_ttc_is_distance_over_speed():
    assert ttc(11.111, 40.0) == pytest.approx(1.0, rel=1e-3)
    assert ttc(50.0, 0.0) == float("inf")
    assert ttc(float("inf"), 40.0) == float("inf")
```

- [ ] **步骤 2：跑测试，确认红**

```powershell
python -m pytest tests/test_geometry.py
```

期望：`ModuleNotFoundError: No module named 'railintrude.geometry'`

- [ ] **步骤 3：实现 `railintrude/geometry.py`**

```python
"""几何：轨道 ROI 判定、单目距离标定、TTC。纯数学，不依赖 OpenCV。"""

from __future__ import annotations

from dataclasses import dataclass


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """射线法。poly 顶点顺序无关。"""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class RoiPolygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self):
        if len(self.points) < 3:
            raise ValueError("ROI 多边形至少需要 3 个顶点")

    @classmethod
    def from_pairs(cls, pairs) -> "RoiPolygon":
        return cls(tuple((float(a), float(b)) for a, b in pairs))

    def contains(self, x: float, y: float) -> bool:
        return _point_in_polygon(x, y, list(self.points))


@dataclass(frozen=True)
class DistanceCalibration:
    """地面平面假设下的反比模型 d(y) = a / (y - y_h)。"""

    y_h: float
    a: float

    @classmethod
    def from_two_points(cls, y1: float, d1: float, y2: float, d2: float) -> "DistanceCalibration":
        if d1 <= 0 or d2 <= 0:
            raise ValueError("标定距离必须为正")
        if y1 == y2 or d1 == d2:
            raise ValueError("两个标定点必须不同")
        y_h = (d2 * y2 - d1 * y1) / (d2 - d1)
        a = d1 * (y1 - y_h)
        if a <= 0:
            raise ValueError("标定结果无效：a<=0，检查 y 与 d 的对应关系（越远 y 越小）")
        return cls(y_h=y_h, a=a)

    def distance(self, y: float) -> float:
        if y <= self.y_h:
            return float("inf")
        return self.a / (y - self.y_h)


def ttc(distance_m: float, speed_kmh: float) -> float:
    if speed_kmh <= 0 or distance_m == float("inf"):
        return float("inf")
    return distance_m / (speed_kmh / 3.6)
```

- [ ] **步骤 4：跑测试，确认绿**

```powershell
python -m pytest tests/test_geometry.py
```

期望：`6 passed`

- [ ] **步骤 5：提交**

```powershell
git add -A
git commit -m "feat(geometry): 轨道 ROI 判定、两点标定单目距离、TTC"
```

---

## Task 2：风险分级与滞回

**文件**：`railintrude/risk.py`、`tests/test_risk.py`

- [ ] **步骤 1：写失败测试 `tests/test_risk.py`**

```python
import pytest

from railintrude.risk import Hysteresis, level_from_ttc


def test_level_boundaries():
    assert level_from_ttc(20.0) == "notice"
    assert level_from_ttc(8.1) == "notice"
    assert level_from_ttc(8.0) == "alert"
    assert level_from_ttc(5.1) == "alert"
    assert level_from_ttc(5.0) == "warn"
    assert level_from_ttc(3.1) == "warn"
    assert level_from_ttc(3.0) == "emerg"
    assert level_from_ttc(0.5) == "emerg"
    assert level_from_ttc(float("inf")) == "notice"


def test_hysteresis_upgrades_immediately():
    h = Hysteresis(down_frames=5)
    assert h.update("notice") == "notice"
    assert h.update("emerg") == "emerg"


def test_hysteresis_holds_level_for_down_frames():
    h = Hysteresis(down_frames=5)
    h.update("emerg")
    assert [h.update("warn") for _ in range(4)] == ["emerg"] * 4
    assert h.update("warn") == "warn"


def test_hysteresis_resets_counter_on_reupgrade():
    h = Hysteresis(down_frames=3)
    h.update("emerg")
    h.update("warn")
    h.update("warn")
    h.update("emerg")
    assert h.update("warn") == "emerg"  # 计数器已清零，重新数


def test_bad_down_frames_is_rejected():
    with pytest.raises(ValueError):
        Hysteresis(down_frames=0)
```

- [ ] **步骤 2：跑测试，确认红**

```powershell
python -m pytest tests/test_risk.py
```

期望：`ModuleNotFoundError: No module named 'railintrude.risk'`

- [ ] **步骤 3：实现 `railintrude/risk.py`**

```python
"""四级报警与降级滞回。内部枚举英文，中文映射只在这里定义一次。"""

from __future__ import annotations

LEVELS = ("notice", "alert", "warn", "emerg")
LEVEL_ZH = {"notice": "注意", "alert": "提示", "warn": "警告", "emerg": "紧急"}
_SEVERITY = {name: i for i, name in enumerate(LEVELS)}

# TTC 阈值（秒），从宽到严
T_EMERG = 3.0
T_WARN = 5.0
T_ALERT = 8.0


def level_from_ttc(ttc_s: float) -> str:
    if ttc_s <= T_EMERG:
        return "emerg"
    if ttc_s <= T_WARN:
        return "warn"
    if ttc_s <= T_ALERT:
        return "alert"
    return "notice"


def severity(level: str) -> int:
    return _SEVERITY[level]


class Hysteresis:
    """升级立即生效；降级需连续 down_frames 帧低等级。"""

    def __init__(self, down_frames: int = 5):
        if down_frames < 1:
            raise ValueError("down_frames 至少为 1")
        self.down_frames = down_frames
        self._current = "notice"
        self._below = 0

    @property
    def current(self) -> str:
        return self._current

    def update(self, new_level: str) -> str:
        if severity(new_level) >= severity(self._current):
            self._current = new_level
            self._below = 0
            return self._current
        self._below += 1
        if self._below >= self.down_frames:
            self._current = new_level
            self._below = 0
        return self._current
```

- [ ] **步骤 4：跑测试，确认绿**

```powershell
python -m pytest tests/test_risk.py
```

期望：`5 passed`

- [ ] **步骤 5：提交**

```powershell
git add -A
git commit -m "feat(risk): TTC 四级分级与降级滞回"
```

---

## Task 3：检测数据结构与 IoU 跟踪

**文件**：`railintrude/vision.py`、`tests/test_vision.py`

**接口**：

```python
Detection(xyxy, cls_name, conf, track_id=-1)      # frozen dataclass
Detection.bottom_center -> (x, y)
Detection.height -> float
iou(a, b) -> float
IouTracker(iou_thresh=0.3, max_age=15).update(detections) -> list[Detection]   # 回填 track_id
FakeDetector(per_frame).__call__(frame, frame_idx) -> list[Detection]
```

- [ ] **步骤 1：写失败测试 `tests/test_vision.py`**

```python
import pytest

from railintrude.vision import Detection, FakeDetector, IouTracker, iou

BOX = (10.0, 10.0, 50.0, 90.0)


def test_iou_identical_is_one():
    assert iou(BOX, BOX) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou(BOX, (200.0, 200.0, 260.0, 280.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_bottom_center_and_height():
    d = Detection(BOX, "person", 0.9)
    assert d.bottom_center == (30.0, 90.0)
    assert d.height == pytest.approx(80.0)


def test_tracker_keeps_id_for_overlapping_box():
    tracker = IouTracker()
    first = tracker.update([Detection(BOX, "person", 0.9)])
    second = tracker.update([Detection((12.0, 12.0, 52.0, 92.0), "person", 0.8)])
    assert first[0].track_id == 0
    assert second[0].track_id == 0


def test_tracker_assigns_new_id_for_far_box():
    tracker = IouTracker()
    tracker.update([Detection(BOX, "person", 0.9)])
    far = tracker.update([Detection((300.0, 300.0, 340.0, 380.0), "person", 0.9)])
    assert far[0].track_id == 1


def test_track_expires_after_max_age():
    tracker = IouTracker(max_age=2)
    tracker.update([Detection(BOX, "person", 0.9)])
    for _ in range(3):
        tracker.update([])
    revived = tracker.update([Detection(BOX, "person", 0.9)])
    assert revived[0].track_id == 1  # 旧轨迹已过期，不复用 0


def test_fake_detector_cycles_scripted_frames():
    a = Detection(BOX, "person", 0.9)
    fake = FakeDetector([[a], []])
    assert len(fake(None, 0)) == 1
    assert len(fake(None, 1)) == 0
    assert len(fake(None, 2)) == 1
```

- [ ] **步骤 2：跑测试，确认红**

```powershell
python -m pytest tests/test_vision.py
```

期望：`ModuleNotFoundError: No module named 'railintrude.vision'`

- [ ] **步骤 3：实现 `railintrude/vision.py`（只写本任务需要的部分，检测器实现留到 Task 6）**

```python
"""检测与跟踪：Detection 数据结构、IoU 跟踪器、可替换的检测器。"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    cls_name: str
    conf: float
    track_id: int = -1

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.xyxy
        return y2 - y1


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """逐检测贪心匹配 + 轨迹老化。检测数 <50 时 O(n^2) 完全够用。"""

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections) -> list[Detection]:
        detections = list(detections)
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for di, det in enumerate(detections):
            best_id, best_score = None, 0.0
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                s = iou(det.xyxy, tr["xyxy"])
                if s > best_score:
                    best_id, best_score = tid, s
            if best_id is not None and best_score >= self.iou_thresh:
                assigned[di] = best_id
                used.add(best_id)

        out: list[Detection] = []
        for di, det in enumerate(detections):
            tid = assigned.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
            used.add(tid)
            self._tracks[tid] = {"xyxy": det.xyxy, "age": 0}
            out.append(replace(det, track_id=tid))

        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        return out


class FakeDetector:
    """按帧号返回预设检测，供测试与无权重环境使用。"""

    def __init__(self, per_frame):
        self.per_frame = per_frame

    def __call__(self, frame, frame_idx: int) -> list[Detection]:
        if isinstance(self.per_frame, dict):
            return list(self.per_frame.get(frame_idx, []))
        return list(self.per_frame[frame_idx % len(self.per_frame)])
```

- [ ] **步骤 4：跑测试，确认绿**

```powershell
python -m pytest tests/test_vision.py
```

期望：`8 passed`

- [ ] **步骤 5：提交**

```powershell
git add -A
git commit -m "feat(vision): Detection、IoU 跟踪器与 FakeDetector"
```

---

## Task 4：管线核心（逐帧评估 + 事件表）

**文件**：`railintrude/pipeline.py`、`tests/test_pipeline.py`

**接口**：

```python
Config()                                            # roi / calib / speed_kmh / conf / down_frames / max_age / iou_thresh
TrackState(det, distance_m, ttc_s, raw_level)
evaluate_frame(dets, cfg, tracker, hysteresis, width, height) -> (list[TrackState], displayed_level)
run(video_in, out_mp4, out_csv, cfg=None, detector=None, max_frames=None, log=print) -> dict
```

- [ ] **步骤 1：写失败测试 `tests/test_pipeline.py`（用合成视频，不碰模型）**

```python
import csv

import cv2
import numpy as np
import pytest

from railintrude.pipeline import Config, evaluate_frame, run
from railintrude.risk import Hysteresis
from railintrude.vision import Detection, FakeDetector, IouTracker

W, H = 160, 120


def make_video(path, n=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.zeros((H, W, 3), np.uint8))
    writer.release()


def person_at(y2):
    return Detection((60.0, y2 - 40.0, 100.0, y2), "person", 0.9)


def test_evaluate_frame_reports_level_and_distance():
    cfg = Config()
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)
    assert len(states) == 1
    assert states[0].distance_m == pytest.approx(6.0, rel=1e-6)
    assert states[0].ttc_s == pytest.approx(0.54, rel=1e-2)   # 6 m / (40 km/h)
    assert states[0].raw_level == "emerg"
    assert level == "emerg"


def test_detection_outside_roi_is_ignored():
    cfg = Config()
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.2 * H)], cfg, tracker, hyst, W, H)
    assert states == []
    assert level == "notice"


def test_run_writes_video_and_events(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video, n=8)
    out_mp4, out_csv = tmp_path / "demo.mp4", tmp_path / "events.csv"
    detector = FakeDetector([[person_at(0.95 * H)]] * 8)
    summary = run(video, out_mp4, out_csv, Config(), detector=detector, log=lambda *_: None)
    assert out_mp4.exists() and out_csv.exists()
    assert summary["frames"] >= 7          # 合成视频偶有丢尾帧，与 A1 的 ±1 一致
    assert summary["fps"] > 0
    cap = cv2.VideoCapture(str(out_mp4))
    assert abs(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - summary["frames"]) <= 1
    cap.release()


def test_events_only_written_on_level_change(tmp_path):
    video = tmp_path / "in.mp4"
    make_video(video, n=8)
    out_csv = tmp_path / "events.csv"
    detector = FakeDetector([[person_at(0.95 * H)]] * 8)
    run(video, tmp_path / "demo.mp4", out_csv, Config(), detector=detector, log=lambda *_: None)
    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1                      # 首帧 notice→emerg 一次，之后等级不变不再写
    assert rows[0]["level"] == "emerg"
    assert rows[0]["cls"] == "person"
```

- [ ] **步骤 2：跑测试，确认红**

```powershell
python -m pytest tests/test_pipeline.py
```

期望：`ModuleNotFoundError: No module named 'railintrude.pipeline'`

- [ ] **步骤 3：实现 `railintrude/pipeline.py`（本任务先不画图，`render` 留到 Task 5）**

```python
"""管线：检测 → 跟踪 → 距离 → TTC → 分级 → 记录 → 写视频。"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field

import cv2

from .geometry import DistanceCalibration, RoiPolygon, ttc
from .risk import Hysteresis, level_from_ttc, severity
from .vision import Detection, IouTracker

OUT_W, OUT_H = 1280, 720
DEFAULT_ROI = ((0.12, 0.50), (0.88, 0.50), (1.0, 1.0), (0.0, 1.0))
DEFAULT_CALIB = {"y1": 0.95, "d1": 6.0, "y2": 0.62, "d2": 50.0}
CSV_FIELDS = ["frame", "time_s", "level", "track_id", "cls", "distance_m", "ttc_s"]


@dataclass
class Config:
    roi: RoiPolygon = field(default_factory=lambda: RoiPolygon.from_pairs(DEFAULT_ROI))
    calib: DistanceCalibration = field(
        default_factory=lambda: DistanceCalibration.from_two_points(**DEFAULT_CALIB)
    )
    speed_kmh: float = 40.0
    conf: float = 0.5
    down_frames: int = 5
    max_age: int = 15
    iou_thresh: float = 0.3


@dataclass
class TrackState:
    det: Detection
    distance_m: float
    ttc_s: float
    raw_level: str


def evaluate_frame(dets, cfg: Config, tracker: IouTracker, hysteresis: Hysteresis, width, height):
    """返回（本帧命中 ROI 的目标状态列表，经过滞回后的显示等级）。"""
    states: list[TrackState] = []
    for det in tracker.update(dets):
        x, y = det.bottom_center
        if not cfg.roi.contains(x / width, y / height):
            continue
        distance = cfg.calib.distance(y / height)
        time_to_collision = ttc(distance, cfg.speed_kmh)
        states.append(
            TrackState(det=det, distance_m=distance, ttc_s=time_to_collision,
                       raw_level=level_from_ttc(time_to_collision))
        )
    states.sort(key=lambda s: severity(s.raw_level), reverse=True)
    raw_level = states[0].raw_level if states else "notice"
    return states, hysteresis.update(raw_level)


def _event_row(frame_idx, fps, level, states):
    lead = states[0] if states else None
    return {
        "frame": frame_idx,
        "time_s": round(frame_idx / fps, 3) if fps else 0.0,
        "level": level,
        "track_id": lead.det.track_id if lead else "",
        "cls": lead.det.cls_name if lead else "",
        "distance_m": round(lead.distance_m, 2) if lead and lead.distance_m != float("inf") else "",
        "ttc_s": round(lead.ttc_s, 2) if lead and lead.ttc_s != float("inf") else "",
    }


def run(video_in, out_mp4, out_csv, cfg: Config | None = None, detector=None,
        max_frames: int | None = None, log=print) -> dict:
    if detector is None:
        raise ValueError("必须传入 detector（真实模型或 FakeDetector）")
    cfg = cfg or Config()
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"打不开视频：{video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    tracker = IouTracker(cfg.iou_thresh, cfg.max_age)
    hysteresis = Hysteresis(cfg.down_frames)
    rows: list[dict] = []
    prev_level: str | None = None

    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        states, level = evaluate_frame(detector(frame, idx), cfg, tracker, hysteresis, w, h)
        if level != prev_level:
            rows.append(_event_row(idx, fps, level, states))
            prev_level = level
        idx += 1
        if idx % 100 == 0:
            log(f"  已处理 {idx} 帧 / {time.time() - t0:.1f}s")
        if max_frames and idx >= max_frames:
            break
    cap.release()

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # 视频先写一个空的占位，Task 5 换成真实渲染
    writer_v = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    writer_v.release()

    elapsed = time.time() - t0
    summary = {"frames": idx, "fps": fps, "elapsed_s": round(elapsed, 2),
               "proc_fps": round(idx / elapsed, 2) if elapsed else 0.0, "events": len(rows)}
    log(f"完成：{summary}")
    return summary
```

- [ ] **步骤 4：跑测试，确认绿**

```powershell
python -m pytest tests/test_pipeline.py
```

期望：`4 passed`

- [ ] **步骤 5：提交**

```powershell
git add -A
git commit -m "feat(pipeline): 逐帧风险评估与事件表输出"
```

---

## Task 5：渲染与命令行

**文件**：`railintrude/pipeline.py`（加 `render`）、`railintrude/demo.py`

- [ ] **步骤 1：写失败测试（追加到 `tests/test_pipeline.py`）**

```python
def test_render_draws_level_color_bar():
    from railintrude.pipeline import OUT_H, OUT_W, render

    frame = np.zeros((H, W, 3), np.uint8)
    cfg = Config()
    tracker, hyst = IouTracker(), Hysteresis(cfg.down_frames)
    states, level = evaluate_frame([person_at(0.95 * H)], cfg, tracker, hyst, W, H)
    canvas = render(frame, states, level, cfg, frame_idx=0, fps=10.0)
    assert canvas.shape == (OUT_H, OUT_W, 3)
    assert canvas[:24, :, :].max() > 0            # 顶部报警条被填色
    assert canvas.sum() > frame.sum()             # 至少画了点东西
```

- [ ] **步骤 2：跑测试，确认红**

```powershell
python -m pytest tests/test_pipeline.py::test_render_draws_level_color_bar
```

期望：`ImportError: cannot import name 'render'`

- [ ] **步骤 3：实现 `render`（写进 `railintrude/pipeline.py`）**

```python
COLORS = {
    "notice": (150, 150, 150),
    "alert": (0, 210, 255),
    "warn": (0, 140, 255),
    "emerg": (0, 0, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fmt(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.1f}"


def render(frame, states, level: str, cfg: Config, frame_idx: int, fps: float):
    """把原帧缩放成 1280x720，叠加框、HUD 与顶部报警条。"""
    canvas = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    sx, sy = OUT_W / frame.shape[1], OUT_H / frame.shape[0]
    color = COLORS[level]
    for st in states:
        x1, y1, x2, y2 = st.det.xyxy
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(canvas, p1, p2, COLORS[st.raw_level], 2)
        label = f"{st.det.cls_name}#{st.det.track_id} {st.det.conf:.2f} d={_fmt(st.distance_m)}m T={_fmt(st.ttc_s)}s"
        cv2.putText(canvas, label, (p1[0], max(p1[1] - 6, 12)), FONT, 0.45, COLORS[st.raw_level], 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (OUT_W, 24), color, -1)
    bar = f"{level.upper()}  speed={cfg.speed_kmh:.0f}km/h  dets={len(states)}"
    cv2.putText(canvas, bar, (8, 17), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    hud = f"frame {frame_idx}  {fps:.1f}fps"
    if states:
        hud += f"  nearest d={_fmt(states[0].distance_m)}m TTC={_fmt(states[0].ttc_s)}s"
    cv2.putText(canvas, hud, (8, OUT_H - 10), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas
```

- [ ] **步骤 4：改 `run()`，用 `render` 真正写视频**

把 Task 4 里"写空占位视频"的两行替换成：逐帧 `canvas = render(...)` + `writer.write(canvas)`。完整循环体：

```python
    writer_v = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    if not writer_v.isOpened():
        raise RuntimeError(f"无法创建输出视频：{out_mp4}")
    cap = cv2.VideoCapture(str(video_in))
    ...
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        states, level = evaluate_frame(detector(frame, idx), cfg, tracker, hysteresis, w, h)
        if level != prev_level:
            rows.append(_event_row(idx, fps, level, states))
            prev_level = level
        writer_v.write(render(frame, states, level, cfg, idx, fps))
        idx += 1
        ...
    writer_v.release()
```

- [ ] **步骤 5：跑全部测试，确认绿**

```powershell
python -m pytest
```

期望：`24 passed`（6+5+8+5）

- [ ] **步骤 6：写 `railintrude/demo.py`**

```python
"""命令行入口：python -m railintrude.demo --video ... """

from __future__ import annotations

import argparse
import sys

from .pipeline import Config, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="车端侵限检测 V0 演示")
    p.add_argument("--video", default="data/raw/platform-arrival-1080p.webm")
    p.add_argument("--out-mp4", default="out/demo.mp4")
    p.add_argument("--out-csv", default="out/events.csv")
    p.add_argument("--weights", default="models/yolov8n.pt", help="本地 YOLOv8 权重路径")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--speed", type=float, default=40.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-model", action="store_true", help="空检测器冒烟测试，不加载权重")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_model:
        from .vision import FakeDetector

        detector = FakeDetector({})
    else:
        from .vision import YoloDetector

        detector = YoloDetector(weights=args.weights, conf=args.conf)

    cfg = Config(speed_kmh=args.speed, conf=args.conf)
    summary = run(args.video, args.out_mp4, args.out_csv, cfg, detector=detector,
                  max_frames=args.max_frames)
    print(f"输出：{args.out_mp4} / {args.out_csv}  事件 {summary['events']} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **步骤 7：冒烟测试（不加载模型）**

```powershell
python -m railintrude.demo --video data/raw/platform-arrival-1080p.webm --max-frames 20 --no-model --out-mp4 out/smoke.mp4 --out-csv out/smoke.csv
```

期望：打印 `完成：{...}` 与 `输出：...`，`out/smoke.mp4` 存在且能播放（黑框视频，因为空检测器）。

- [ ] **步骤 8：提交**

```powershell
git add -A
git commit -m "feat(pipeline): 渲染叠加与命令行入口"
```

---

## Task 6：YOLOv8 检测器

**文件**：`railintrude/vision.py`（追加 `YoloDetector`）

- [ ] **步骤 1：实现 `YoloDetector`（追加到 `railintrude/vision.py`）**

```python
COCO_KEEP = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "train"}


class YoloDetector:
    """ultralytics YOLOv8 的 COCO 检测器。权重必须是本地文件，避免加载时联网。"""

    def __init__(self, weights: str = "models/yolov8n.pt", conf: float = 0.5,
                 classes: set[str] | None = None, device: str = "cpu", imgsz: int = 640):
        from pathlib import Path

        from ultralytics import YOLO

        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"找不到本地权重：{weights}。不要传模型名（yolov8n.pt 会触发联网下载）。"
            )
        self.model = YOLO(str(weights))
        self.conf = conf
        self.classes = classes or COCO_KEEP
        self.device = device
        self.imgsz = imgsz
        self._names = dict(self.model.names)

    def __call__(self, frame, frame_idx: int = 0) -> list[Detection]:
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            verbose=False, plots=False,
        )[0]
        dets: list[Detection] = []
        if result.boxes is None:
            return dets
        for xyxy, cls_id, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            name = self._names[int(cls_id)]
            if name not in self.classes:
                continue
            dets.append(Detection(tuple(float(v) for v in xyxy), name, float(score)))
        return dets
```

- [ ] **步骤 2：单帧冒烟脚本验证真实推理**

```powershell
python -c "
import cv2, time
from railintrude.vision import YoloDetector
cap = cv2.VideoCapture('data/raw/platform-arrival-1080p.webm'); ok, frame = cap.read(); cap.release()
assert ok, '读不到帧'
t=time.time(); det = YoloDetector(weights='models/yolov8n.pt', conf=0.5); print('模型加载', round(time.time()-t,1), 's')
t=time.time(); out = det(frame, 0); print('推理', round(time.time()-t,2), 's', len(out), '个框')
print([(d.cls_name, round(d.conf,2), [round(v) for v in d.xyxy]) for d in out[:5]])
"
```

期望：加载数秒（本地权重，不联网），单帧推理 0.1–1s，打印至少 1 个 `person`。

- [ ] **步骤 3：提交**

```powershell
git add -A
git commit -m "feat(vision): YOLOv8 本地权重检测器（可替换插槽）"
```

---

## Task 7：真视频端到端与验收

- [ ] **步骤 1：全量跑**

```powershell
python -m railintrude.demo --video data/raw/platform-arrival-1080p.webm --out-mp4 out/demo.mp4 --out-csv out/events.csv
```

- [ ] **步骤 2：逐条取证据（A1–A5）**

```powershell
# A1 帧数与分辨率
python -c "
import cv2
for p in ['data/raw/platform-arrival-1080p.webm','out/demo.mp4']:
    c=cv2.VideoCapture(p); print(p, int(c.get(cv2.CAP_PROP_FRAME_COUNT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))); c.release()
"
# A3 事件表
Get-Content out\events.csv -TotalCount 10
```

A2（person 帧占比）用一次带统计的跑法确认，把数字抄进交付说明。

- [ ] **步骤 3：抽三帧截图给用户看**

```powershell
python -c "
import cv2
c=cv2.VideoCapture('out/demo.mp4'); i=0
while True:
    ok, f = c.read()
    if not ok: break
    if i in (60, 600, 1200): cv2.imwrite(f'out/still_{i}.jpg', f)
    i+=1
c.release(); print('done', i)
"
```

- [ ] **步骤 4：提交并汇报**

```powershell
git add -A
git commit -m "feat: V0 演示端到端跑通（真实视频 + 事件表）"
```

---

## 验收对照

| 编号 | 标准 | 在哪个任务被验证 |
|---|---|---|
| A1 | 输出 1280×720，帧数一致（±1） | Task 7 步骤 2 |
| A2 | person 帧占比 > 30% | Task 7 步骤 2 |
| A3 | 至少一条 warn/emerg | Task 7 步骤 2 |
| A4 | `pytest` 全绿 | Task 5 步骤 5（24 passed） |
| A5 | 57 s 视频 < 10 分钟 | Task 7 步骤 1 的 `elapsed_s` |

## 执行完之后的下一步（不在本计划内）

- 把 `out/demo.mp4` 与三张截图交给用户，决定是否进入 V1。
- V1（OSDaR23 微调 + 红外分支 + C1/C2 消融）另写 spec 与计划。
