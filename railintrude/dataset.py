"""把 COCO 检测结果转成 YOLO 训练标签，并按时间切分训练/验证集。"""

CLASS_MAP = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
             "bus": 4, "truck": 5, "train": 6}


def to_yolo_lines(detections, width, height, class_map=None):
    """检测框 -> YOLO 行（cls cx cy w h，归一化）。非目标类别丢弃。

    class_map 默认用 COCO 的 7 类；换数据集时必须传自己的映射，否则
    类别名对不上的框会被静默丢掉（OSDaR23 曾因此把 4 类缩成 1 类）。
    """
    mapping = CLASS_MAP if class_map is None else class_map
    lines = []
    for det in detections:
        cls = mapping.get(det.cls_name)
        if cls is None:
            continue
        x1, y1, x2, y2 = det.xyxy
        x1, x2 = max(0.0, min(float(x1), width)), max(0.0, min(float(x2), width))
        y1, y2 = max(0.0, min(float(y1), height)), max(0.0, min(float(y2), height))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
    return lines


def select_frames(labelled, empty, neg_stride=5):
    """有目标的帧全收；无目标帧每 neg_stride 个取 1 个当负样本。

    负样本是必需的：只喂"有目标"的帧，模型学不到"空隧道=什么都没有"，
    实测微调后会把隧道里对面车头灯幻觉成 60% 画面的 train 框。
    """
    kept = dict(labelled)
    if neg_stride > 0:
        for idx in empty[::neg_stride]:
            kept[idx] = []
    return kept


def select_classes(counter, min_instances=15, required=None):
    """从 {类别: 实例数} 里挑出要训练的类别，返回有序列表（下标即类别 id）。

    required 给定时按它给的顺序返回、并丢掉数据集里没有的类；否则按实例数从多到少。
    钉住类别是为了让两次实验的类别索引一致，否则 mAP 表格没法前后对比。
    """
    if required:
        return [name for name in required if name in counter]
    return [name for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= min_instances]


def temporal_split(frames, train_ratio=0.7):
    """按时间顺序切分，验证集整体落在训练集之后，避免同场景泄漏。"""
    ordered = sorted(frames)
    if not ordered:
        raise ValueError("frames must not be empty")
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]


def merge_classes(spec):
    """把类别映射规格解析成 {原名: 新名}。

    规格形如 "catenary_pole,signal_pole>pole;animal,crotchet>misc"。
    存在的理由：接触网柱与信号柱在画面里都是一根细竖条，两者的 mAP 都低
    （0.53 / 0.48），把它们并成"杆状物"能让 mAP 反映"能否找到杆"而不是
    "能否分清杆的种类"，而后者对侵限报警没有意义。
    """
    mapping = {}
    for group in spec.split(";"):
        if ">" not in group:
            continue
        names, target = group.split(">", 1)
        target = target.strip()
        if not target:
            continue
        for name in names.split(","):
            name = name.strip()
            if name:
                mapping[name] = target
    return mapping
