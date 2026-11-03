# OSDaR23 上的 YOLOv8n 训练评估报告

日期：2026-10-22　　代码：`tools/build_osdar23_yolo.py`、`tools/train_osdar23.py`
数据：OSDaR23 官方标注（CC0 1.0）　权重：`models/yolov8n.pt`（COCO 预训练）　硬件：RTX 4060 Laptop 8 GB

---

## 一、一句话结论

**零样本的 COCO 模型在这份数据上完全失效（mAP50 = 0）；用 84 张图微调 60 个 epoch 后 mAP50 到 0.52、mAP50-95 到 0.25。**
其中**车辆类（road_vehicle 0.88）和止冲挡（buffer_stop 0.90）已经可用**，但**接触网立柱（catenary_pole 0.06）和信号杆（signal_pole 0.06）基本没学会**——原因是细长目标 + 训练图太少，不是标注问题。

---

## 二、数据是怎么来的

| 项目 | 内容 |
|---|---|
| 来源 | OSDaR23（德国 DZSF 发布），已落地三个序列 |
| 序列 | `1_calibration_1.2`（100 帧）、`7_approach_underground_station_7.1`（10 帧）、`2_station_berliner_tor_2.1`（10 帧） |
| 相机 | 只用 `rgb_center`（2464×1600）。**混相机等于混分布**，评估必须在同一视角内 |
| 标注解析 | OpenLABEL 1.0.0 → `bbox` 的 `val=[x,y,w,h]`（左上角+宽高，不是 xyxy） |
| 类别筛选 | 只保留实例数 ≥ 15 的 5 类；长尾类留 1–2 个框会让 mAP 变成噪声 |
| 切分 | 每个序列内部按时间前 70% 训练 / 后 30% 验证，**不按文件名随机切**（相邻帧几乎相同，随机切会泄漏） |
| 总量 | 120 帧 → 84 训练 / 36 验证；标注 2,143 个框（验证集 630 个） |

类别与实例数：

| 类别 | 实例数（全部） | 验证集实例 |
|---|---|---|
| catenary_pole（接触网立柱） | 887 | 252 |
| road_vehicle（道路车辆） | 400 | 120 |
| person（人员） | 398 | 120 |
| buffer_stop（止冲挡） | 310 | 93 |
| signal_pole（信号杆） | 148 | 45 |

---

## 三、实验设置

| 项 | 值 |
|---|---|
| 模型 | YOLOv8n（3.15 M 参数，8.7 GFLOPs） |
| 输入 | 640×640 letterbox |
| 预训练 | COCO `yolov8n.pt` |
| 训练 | 60 epoch，batch 16，单卡 RTX 4060，AdamW 自动学习率 |
| 耗时 | 训练 115 s + 评估 32 s（合计 147 s） |
| 评估 | 与训练同一 `data/osdar23_yolo/dataset.yaml` 的 val 划分，`imgsz=640` |

---

## 四、结果

### 4.1 零样本 vs 微调

| 指标 | 零样本（COCO 权重） | 微调后 | 变化 |
|---|---|---|---|
| mAP50 | 0.000 | **0.520** | +0.520 |
| mAP50-95 | 0.000 | **0.247** | +0.247 |
| Precision | 0.000 | 0.787 | +0.787 |
| Recall | 0.000 | 0.467 | +0.467 |

零样本的 per-class 表里出现 `person / bicycle / car / motorcycle / airplane` ——**这些是 COCO 的类名，不是数据的类名**，说明模型的类别空间与数据根本对不上，所以全 0。

### 4.2 每类指标（微调后，验证集 36 帧 / 630 框）

| 类别 | 验证框数 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| buffer_stop | 93 | 0.766 | 0.892 | **0.900** | 0.382 |
| road_vehicle | 120 | 0.872 | 0.758 | **0.876** | 0.408 |
| person | 120 | 0.674 | 0.636 | **0.703** | 0.398 |
| catenary_pole | 252 | 0.621 | 0.016 | 0.059 | 0.019 |
| signal_pole | 45 | 1.000 | 0.033 | 0.060 | 0.026 |

**读法**：`catenary_pole` 和 `signal_pole` 的 Precision 高（0.62 / 1.00）但 Recall 极低（0.016 / 0.033）——模型只要报就报对，但绝大多数漏掉了。这是典型的**细长小目标 + 训练样本不足**，不是把框画错了。

---

## 五、这次踩到并修掉的一个真 bug（值得写进答辩）

第一次训练拿到的是 **mAP50 = 0.844** 的好看数字。但验证日志里 `Instances = 120`、且**只出现 buffer_stop 一类**，而数据明明有 5 类 630 个框。

根因：构建数据集时复用了 V0 demo 的 `to_yolo_lines()`，它把类别**硬编码在 COCO 的 7 类**里：

```python
cls = CLASS_MAP.get(det.cls_name)   # CLASS_MAP = {person:0, bicycle:1, ...}
if cls is None:
    continue                        # ← 静默丢弃
```

于是 `catenary_pole / road_vehicle / signal_pole / buffer_stop` 全被丢掉，只剩 `person` 拿到索引 0；而 `dataset.yaml` 的 0 号是 `buffer_stop`。**模型实际学的是“把行人框叫成 buffer_stop”**，0.844 是纯假的。

修复（TDD）：

1. 先加失败测试 `test_to_yolo_lines_accepts_custom_class_map` → `TypeError: to_yolo_lines() takes 3 positional arguments but 4 were given`
2. `to_yolo_lines(detections, width, height, class_map=None)`，默认仍是 COCO 表，**换数据集必须传自己的映射**
3. `build_osdar23_yolo.py` 改为 `to_yolo_lines(dets, w, h, classes)`
4. 重建后验证：val 标签类分布 `{0:93, 1:252, 2:120, 3:120, 4:45}`（5 类齐全），全套 **40 个单测通过**
5. 重训 → mAP50 从假的 0.844 降到真的 0.520

**教训**：类别映射失败必须是「报错」，不能是「静默丢弃」。任何 `continue` 式的静默过滤，都要在统计里留下计数。

---

## 六、局限（必须说清，否则数字会被误用）

1. **验证集只有 36 帧、630 个框**，来自 3 个序列的尾段。指标方差大，别把 0.52 当成稳定性能。
2. **每类只有 30–36 帧覆盖**，`signal_pole` 只有 45 个框，它的 0.06 更像“没学会”而不是“精确的 0.06”。
3. **训了两个序列的 7 帧就进验证**（7.1 和 2.1 各 7 训练 / 3 验证），等于验证集里也混了跨场景，泛化结论要谨慎。
4. **只有 84 张训练图**，远低于 YOLO 正常需要的量级；这份数字说明“流程通了、简单目标能学”，不能代表工业部署性能。
5. **未做数据增强消融**（默认增强开启）、未做多模型对比（只跑了 n 版），没调超参。
6. 类别口径来自 OSDaR23（德国），与国内 TB/T 术语不通用。

---

## 七、产物在哪

| 产物 | 路径 |
|---|---|
| 权重 | `runs/detect/osdar23-yolov8n-v2/weights/best.pt`（6.5 MB） |
| 指标 json | `out/osdar23_yolo/metrics.json` |
| 训练日志 | `out/osdar23_yolo/train_v2.log` |
| 每轮曲线 | `runs/detect/osdar23-yolov8n-v2/results.png`、`results.csv` |
| PR / F1 / P / R 曲线 | `runs/detect/val-16/BoxPR_curve.png` 等 |
| 混淆矩阵 | `runs/detect/val-16/confusion_matrix.png`、`confusion_matrix_normalized.png` |
| 验证集标注 vs 预测对照 | `runs/detect/val-16/val_batch{0,1,2}_{labels,pred}.jpg` |
| 数据集配置 | `data/osdar23_yolo/dataset.yaml`、`stats.json` |

---

## 八、复现命令

```powershell
# 1. 从 zip 里按需解出 rgb_center 流（8 GB 的 zip 只取需要的 100 张图）
python tools/extract_osdar23.py "C:\Users\jojo as\Desktop\1_calibration_1.2.zip" --streams rgb_center

# 2. 解析 openlabel 标注 → YOLO 数据集
python tools/build_osdar23_yolo.py --camera rgb_center --min-instances 15

# 3. 零样本基线 + 微调 + 微调后评估
python tools/train_osdar23.py --epochs 60 --batch 16 --device 0 --name osdar23-yolov8n-v2
```

---

## 九、下一步（按性价比排序）

1. **再解 2–3 个序列**（如 `9_station_ruebenkamp_9.1`、`14_signals_station_14.1`）把训练图从 84 提到 200+，`catenary_pole` 的 recall 是最有希望先救回来的。
2. **针对性切片推理** —— 已试，见第十节：在“整图 640 训练”的模型上直接原生尺度切片**反而更差**，这条路要先改训练方式才有意义。
3. **类别重定义**：把 `catenary_pole` + `signal_pole` 合成「杆状物」一类，先验证“能不能检出”，再谈细分。
4. ~~加入 `imgsz=1280` 对照~~ —— 已做，见第十节：mAP50 +0.02、细杆类翻倍。

---

## 十、两次提升尝试（2026-10-22 增补）

动机：`catenary_pole` 0.059 / `signal_pole` 0.060 太低，按常规套路有两个解法——把大图切开送推理，或者直接提高训练分辨率。两个都试了，结论是**只有一个有效**。

### 10.1 尝试 A：原生尺度切片推理 —— 失败，且失败原因值得记

做法：2464×1600 切成 640×640（重叠 20%，每图 15 块），按原图坐标尺度送入 `imgsz=640`，块内坐标平移回原图后再做跨块 NMS，最后用自写 AP 评测器（`railintrude/evaluate.py`）评分。

结果（验证集前 6 张，同一子集对照）：

| 推理方式 | mAP50 | mAP50-95 |
|---|---|---|
| 整图 640 | 0.5475 | 0.3336 |
| 切片 640（原生尺度） | **0.0018** | 0.0003 |

**为什么反而更差**：这版模型的训练输入是「整张 2464×1600 缩到 640」，也就是说模型学到的目标是“很小的”——一根立柱在训练图里只有十几像素。切片后同一根立柱变成七八十像素，**尺度直接超出训练分布**，模型于是给出大量置信度虽低、但数量极多的错框，AP 被淹没（每图约 2900 个候选框）。

**结论（答辩可以直接讲，这是个真知识点）**：切片不是“推理阶段的免费午餐”。切片要和训练尺度对齐才有收益——要么训练时也用切块，要么把训练分辨率提上去。否则只是把图像从“看起来大”变成“模型没见过”。

### 10.2 尝试 B：`imgsz=1280` 重训 —— 有效

配置：`--imgsz 1280 --batch 8 --epochs 60`，数据与类别不变（84 训练 / 36 验证，630 个验证框）。

统一复测（同一份 `out/osdar23_yolo/eval1280.py`、batch 4、各自训练尺度）：

| 指标 | 640 训练 | 1280 训练 | 变化 |
|---|---|---|---|
| mAP50 | 0.5195 | **0.5399** | +0.0204 |
| mAP50-95 | 0.2463 | **0.2932** | +0.0469 |
| Precision | 0.7865 | 0.7182 | −0.0683 |
| Recall | 0.4672 | 0.4927 | +0.0255 |
| buffer_stop | 0.9002 | 0.8533 | −0.0469 |
| catenary_pole | 0.0591 | **0.1590** | **+0.0999（×2.7）** |
| person | 0.7025 | 0.7307 | +0.0282 |
| road_vehicle | 0.8758 | 0.8994 | +0.0236 |
| signal_pole | 0.0600 | 0.0572 | −0.0028 |

读法：**只有“细长杆状物”这一类吃到了分辨率的红利**（catenary_pole ×2.7），其余类别基本持平，回报是精度换召回（P↓R↑）。这与“细目标在 640 下被抹平”的假设一致，也说明 1280 不是万能钥匙。

### 10.3 两个必须交代的瑕疵

- **1280 那次训练没跑完**：epoch 55/60 崩在 `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`。事后查到前一次崩溃的训练进程残留占着约 1 GB 显存没释放，属于本机 GPU 状态问题而非代码问题。用的是 epoch≈54 的 `best.pt`，所以 1280 的数字**还有约 5 个 epoch 的余量没吃到**。
- **训练时 val 与统一复测一致**：训练脚本记录的 `out/osdar23_yolo/metrics.json` 为 mAP50 0.5196 / mAP50-95 0.2466 / P 0.7865 / R 0.4672，`tools/eval_slicing.py` 统一复测（batch 4）为 0.5195 / 0.2463 / 0.7865 / 0.4672。两组差异仅在小数第四位（batch 对 NMS 的微小影响），**不存在评测口径漂移**。本报告第四节与第 10.2 节一律引用这一套实测值。

### 10.4 复现

```powershell
# 统一复测（权重, 尺寸…）
python -u out/osdar23_yolo/eval1280.py runs/detect/osdar23-yolov8n-v2/weights/best.pt 640
python -u out/osdar23_yolo/eval1280.py runs/detect/osdar23-yolov8n-1280/weights/best.pt 1280

# 1280 重训
python tools/train_osdar23.py --epochs 60 --batch 8 --imgsz 1280 --device 0 --name osdar23-yolov8n-1280

# 切片对照（含自写评测器正确性闸门：整图自评应≈0.52）
python tools/eval_slicing.py --limit 6
```

单元测试：`python -m pytest` → 50 passed（新增 `tests/test_slicing.py` 5 个、`tests/test_evaluate.py` 5 个）。

## 十一、扩数据实验：3 序列 → 18 序列（目前唯一被证明有效的杠杆）

### 11.1 动机

第 10.2 节的 1280 只是“分辨率红利”，修不了根本问题：84 张训练图、只有 3 个场景。本轮把 OSDaR23 的序列数从 3 扩到 18。

### 11.2 数据

用户用浏览器逐个下载了 18 个序列（共 40.47 GB），每个 zip 只解 `rgb_center` 一支：

| 序列 | 帧数 | 备注 |
|---|---|---|
| 1_calibration_1.2 | 100 | 标定序列，白天站台 |
| 2_station_berliner_tor_2.1 | 10 | |
| 4_station_pedestrian_bridge_4.1–4.5 | 10/10/100/100/76 | 4.3、4.4 各 100 帧 |
| 7_approach_underground_station_7.1–7.3 | 各 10 | 地下站进站 |
| 9_station_ruebenkamp_9.1–9.5 | 各 10 | |
| 14_signals_station_14.1–14.3 | 各 10 | |

产出数据集 `data/osdar23_yolo`：**516 张**（train 361 / val 155），训练框 5,101 / 验证框 2,154。

类别**钉死**为与上一轮完全相同的 5 类、同一顺序（0 buffer_stop / 1 catenary_pole / 2 person / 3 road_vehicle / 4 signal_pole），唯一目的是让两次实验的 mAP 表可以直接对比。扩数据后新出现的稀有类被排除：animal 30 框、bicycle 24 框、signal_bridge 20 框（用 `--classes` 显式指定，选择逻辑在 `railintrude/dataset.py::select_classes`，有 3 个单测）。

### 11.3 受控对比（同一个验证集）

验证集也换了，所以**老模型必须在新验证集上重测**，否则是拿旧考试跟新考试比。下表：同一 155 张验证集、同为 imgsz=1280、同一脚本 `tools/verify_per_class.py`。

| 类别 | 120 图模型 mAP50 | 516 图模型 mAP50 | 变化 |
|---|---|---|---|
| buffer_stop | 0.825 | 0.813 | −0.012 |
| catenary_pole | 0.071 | 0.533 | **+0.462（×7.5）** |
| person | 0.145 | 0.729 | **+0.584（×5.0）** |
| road_vehicle | 0.190 | 0.962 | **+0.772（×5.1）** |
| signal_pole | 0.007 | 0.476 | **+0.469（×68）** |
| **整体 mAP50** | **0.2477** | **0.7027** | **+0.455（×2.8）** |
| 整体 mAP50-95 | 0.1306 | 0.4493 | +0.319 |
| Precision / Recall | 0.53 / 0.36 | 0.78 / 0.63 | — |

### 11.4 一个必须讲清楚的事实

上一轮报告里 120 图模型的 0.5399，是在“旧验证集”（36 张，来自训练用的同 3 个序列的后 30%）上测的。换到 18 序列的 155 张验证集，**同一份权重掉到 0.2477**。

说明当时的 0.5399 有相当一部分是“同场景近邻帧”撑起来的，泛化能力被高估。这不是把数字改小，而是换了覆盖更广的考试之后的真实成绩。答辩引用时应以 0.2477 作为“3 序列模型的跨场景表现”。

### 11.5 同时修掉一个评测口径 bug

`tools/train_osdar23.py` 的 `metrics_of()` 原来把 `imgsz` 写死为 640，而本轮训练是 1280——于是 `metrics.json` 里记的 0.5356 实际是“1280 训练的模型按 640 评估”，与 1280 复测的 0.7027 差 0.167。已改为透传 `args.imgsz`，并在 docstring 里记下这个坑。

### 11.6 训练成本

60 epoch、batch 4、imgsz 1280、RTX 4060 Laptop（8 GB）：**758 s（12.6 分钟）**，峰值显存 2.31 GB。

### 11.7 结论

- **数据量与场景多样性是本项目目前唯一被证明有效的杠杆**：mAP50 0.248 → 0.703，其中 road_vehicle ×5.1、signal_pole ×68。
- 分辨率（640→1280）的红利只对细长杆状物明显（10.2）。
- 切片推理在未对齐训练尺度时是负收益（10.1）。
- 剩下的短板仍是 buffer_stop 之外的三类 P/R 不平衡（precision 0.78 vs recall 0.63），以及验证集只有 155 张、单次训练无多种子重复。

### 11.8 复现

```powershell
# 解压 18 个序列的 rgb_center
python tools/extract_osdar23.py "<zip>" --streams rgb_center

# 重建 5 类数据集（类别钉死，保证可比）
python tools/build_osdar23_yolo.py --classes "buffer_stop,catenary_pole,person,road_vehicle,signal_pole"

# 训练
python tools/train_osdar23.py --epochs 60 --batch 4 --imgsz 1280 --device 0 --name osdar23-yolov8n-516img-1280

# 受控复测（老/新模型都在同一份验证集上）
python -u tools/verify_per_class.py "runs/detect/osdar23-yolov8n-1280/weights/best.pt" 1280
python -u tools/verify_per_class.py "runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt" 1280
```

单元测试：`python -m pytest` → 58 passed（新增 `tests/test_select_classes.py` 3 个）。

