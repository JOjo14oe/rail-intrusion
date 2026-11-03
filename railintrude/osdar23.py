"""OSDaR23 openlabel 标注解析：把每帧每个相机的 bbox 取出来，供转 YOLO 用。

openlabel 结构（实测于 1_calibration_1.2 / 7.1 / 2.1）：
  openlabel.frames["<frame_id>"]
    .frame_properties.streams["<camera>"].uri = "/rgb_center/<frame>_<timestamp>.png"
    .objects["<uuid>"].object_data.bbox[] = [{"name": "<camera>__bbox__<class>",
                                              "val": [x, y, w, h]}, ...]
其中 val 是像素坐标、左上角 + 宽高。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameBoxes:
    """一帧里某个相机的所有框。"""

    frame_id: int
    image: str
    boxes: list[tuple[str, Box]]


def parse_openlabel(ol: dict, camera: str = "rgb_center") -> list[FrameBoxes]:
    """返回按 frame_id 升序的 FrameBoxes 列表。

    ol 是 labels json 里的 "openlabel" 对象。没有该相机图像的帧会被跳过。
    坐标从 (x, y, w, h) 转成 (x1, y1, x2, y2)。
    """
    prefix = f"{camera}__bbox__"
    frames = ol.get("frames", {})
    out: list[FrameBoxes] = []
    for fid in sorted(frames, key=int):
        frame = frames[fid]
        stream = frame.get("frame_properties", {}).get("streams", {}).get(camera)
        uri = (stream or {}).get("uri")
        if not uri:
            continue
        boxes: list[tuple[str, Box]] = []
        for obj in frame.get("objects", {}).values():
            for bb in obj.get("object_data", {}).get("bbox") or []:
                name = bb.get("name", "")
                if not name.startswith(prefix):
                    continue
                x, y, w, h = bb["val"][:4]
                boxes.append((name[len(prefix):], (x, y, x + w, y + h)))
        out.append(FrameBoxes(int(fid), PurePosixPath(uri.replace("\\", "/")).name, boxes))
    return out


_LABEL_SUFFIX = "_labels.json"


def discover_sequences(root: Path) -> list[tuple[str, Path, Path]]:
    """扫描已解压的 OSDaR23 根目录，返回 [(序列名, 图像目录, 标注文件), ...]（按名字排序）。

    为什么这样找：每个子序列单独一个 zip，标注文件名叫 `<序列名>_labels.json`，
    但 1_calibration 的标注解到 `seq_1.2/` 子目录、图像解到 `1_calibration_1.2/`。
    所以按标注文件名反推图像目录 `root/<序列名>`；图像没解出来的序列也照实返回，
    由调用方打印 skip（静默丢序列会让人以为数据白下了）。

    去重：同一份标注可能既在 `seq_1.2/` 又在 `<序列名>/` 里（解压工具复制过），
    不去重就会把同一序列处理两遍、图像同名互相覆盖。优先选父目录名等于序列名的那份。
    """
    chosen: dict[str, tuple[Path, Path]] = {}
    for label_json in sorted(root.rglob(f"*{_LABEL_SUFFIX}")):
        name = label_json.name[: -len(_LABEL_SUFFIX)]
        if not name:
            continue
        img_dir = root / name
        prev = chosen.get(name)
        if prev is None or (label_json.parent.name == name and prev[1].parent.name != name):
            chosen[name] = (img_dir, label_json)
    return [(name, img_dir, label_json) for name, (img_dir, label_json) in sorted(chosen.items())]

