"""检测与跟踪：Detection 数据结构、IoU 跟踪器、可替换的检测器。"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    cls_name: str
    conf: float
    track_id: int = -1
    hits: int = 1

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
                hits = 1
            else:
                hits = self._tracks[tid].get("hits", 1) + 1
            used.add(tid)
            self._tracks[tid] = {"xyxy": det.xyxy, "age": 0, "hits": hits}
            out.append(replace(det, track_id=tid, hits=hits))

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
