# 提升检测精度：四条手段的实测记录

> 目标：在 OSDaR23 上把 YOLOv8n 的 mAP50 从 0.7027 往上推。
> 验证集固定为 **155 帧**（18 个"有验证帧"的序列，按时间后 30% 切出），
> 所有对照都在这一份验证集上、同一评测脚本（`tools/verify_per_class.py`）算出。
> 单一数字出自哪一个实验、哪一份 json，本文逐行标注。

## 0. 为什么先钉验证集

数据从 516 张扩到 1359 张时，**验证集必须不动**，否则"涨了多少"会被切分变化污染。
本文件夹的所有对照都跑在 `data/osdar23_yolo43/dataset.yaml` 的 val 上。
已核对：这份 val 的 155 个文件名与基线那次的 155 个**完全相同**（比对命令见第 5 节），
并且用基线权重在新 val 上复测，得到 `0.7027 / 0.4493`，与当时记录逐位一致 —— 说明可比。

## 1. A 轴：加数据（1359 图 / 43 序列）

数据侧改动：
- 修 `tools/extract_osdar23.py`：它原先只认 `<序列名>/<流>/` 嵌套布局，
  而部分 zip（如 `10_station_suelldorf_10.1.zip`）把 `<流>/` 放在 zip 根部，
  导致这些序列"解出了 0 张图却报命中 14 个文件"。现在两种布局都认。
- 给 `tools/build_osdar23_yolo.py` 加 `--val-seqs`：**验证集只从指定序列切**，
  其余序列的帧全部进训练集，避免验证帧与训练帧来自同一段视频。
- 结果：43 个序列都能解出 `rgb_center`，训练图 466 → **1359**，验证图仍是 155。

同验证集、同 imgsz 1280、同 60 epoch、同 `yolov8n.pt` 起点：

| 训练图数 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 516（基线） | 0.7755 | 0.6243 | **0.7027** | **0.4493** |
| **1359（A 轴）** | 0.7521 | 0.6584 | **0.7348** | **0.4721** |
| 变化 | −0.023 | **+0.034** | **+0.032** | **+0.023** |

每类 mAP50（基线 → A 轴）：

| 类别 | mAP50 | mAP50-95 |
|---|---|---|
| buffer_stop | 0.813 → **0.879** | 0.551 → **0.608** |
| catenary_pole | 0.533 → **0.610** | 0.262 → **0.300** |
| person | 0.729 → 0.742 | 0.416 → 0.421 |
| road_vehicle | 0.962 → 0.964 | 0.716 → 0.728 |
| signal_pole | 0.476 → 0.479 | 0.301 → 0.304 |

读法：
- 收益集中在**数据本来就少的两类**（buffer_stop、catenary_pole）与召回率，符合"加数据"的预期。
- 精度（P）下降 0.023，是典型的"多召回换来的误报"，不是变差。
- 这仍是**单种子**结果。历史上两个模型差 0.019 就落在抖动内（见 `yolo-version-compare.md`），
  所以 +0.032 只能说"很可能真实"，要断言必须做 G 轴多种子。

产物：
- 权重 `runs/detect/osdar23-yolov8n-1359img-1280/weights/best.pt`
- 指标 `out/osdar23_yolo43/metrics.json`、`out/osdar23_yolo43/per_class_1359img.json`
- 基线复测 `out/osdar23_yolo43/per_class_516img_on_newval.json`

## 2. C1 轴：合并两根杆状类

- 动机：catenary_pole（2825 实例）与 signal_pole（1752 实例）在 1280 图上都是细长竖杆，
  模型反复把两者搞混（基线 recall 0.447 / 0.400，precision 却 0.62 / 0.82 —— "报得准、漏得多"）。
  合成一类可以合并样本量、消除互错，代价是失去"接触网柱 vs 信号柱"的区分。
- 实现：`railintrude/dataset.py::merge_classes`（`"a,b>new;c;d>other"` 语法），
  `tools/build_osdar23_yolo.py --merge`，产出 `data/osdar23_yolo43_merge`（4 类：buffer_stop / person / pole / road_vehicle）。
- 状态：训练在 **epoch 50/60 被外部中断**（日志停在 50、进程已消失、没有 metrics.json），
  但 `best.pt` 已落盘。用它与 A 轴完全相同的 155 帧 val、同样的 1280 分辨率单独评测：

  `python -u tools/verify_per_class.py runs/detect/osdar23-yolov8n-1359merge-1280/weights/best.pt 1280 data/osdar23_yolo43_merge/dataset.yaml out/osdar23_yolo43_merge/per_class.json`

- 结果（4 类，155 图 / 2154 实例）：**P 0.747 / R 0.7194 / mAP50 0.7821 / mAP50-95 0.5045**

| 类别（4 类版） | mAP50 | mAP50-95 | 对照（A 轴 5 类版） |
|---|---|---|---|
| buffer_stop | **0.943** | 0.599 | 0.879 / 0.608 |
| person | 0.697 | 0.398 | 0.742 / 0.421 |
| pole（合并） | 0.513 | 0.275 | catenary 0.610 + signal 0.479 → 平均 **0.545** / 0.302 |
| road_vehicle | **0.974** | 0.746 | 0.964 / 0.728 |

- **0.7821 不能写成"比 A 轴 +0.047"**。类别数从 5 变 4，mAP 是先按类算 AP 再平均：
  把两个 0.610 / 0.479 的低分类换成一个 0.513 的类，平均值天然上升。
  分量看：**杆状类本身反而是降的**（0.545 → 0.513），buffer_stop +0.064、road_vehicle +0.010、person −0.045。
- 结论：C1 不是"提精度"的手段，它是**标签空间的重定义**——牺牲"接触网柱 / 信号柱"的区分，
  换一个类、少一处混淆。工程视角（只要判断"轨面有没有障碍物"）可接受；需要按柱型区分的场景不可用。
- 顺带证据：**没改标签的类也在跳**（buffer_stop +0.064、person −0.045），
  再次说明单次训练抖动不可忽略——这是 G 轴多种子的存在理由。
- 产物：`out/osdar23_yolo43_merge/per_class.json`；权重 `runs/detect/osdar23-yolov8n-1359merge-1280/weights/best.pt`（中断前最佳）。
- 备注：该读数来自"近似收敛"的检查点。要写进正式材料，应重跑到 60 epoch 再测一次。

## 3. D 轴：类路由双模态

- 想法：红外在自己 640 分辨率上 mAP50 0.7674，而可见光 1280 是 0.7027；
  但红外框搬到可见光会被视差毁掉（已证否，见 `c1-fusion-report.md`）。
- 所以不搬框，而是**按类路由**：杆类/近处大目标交给红外分支，其余交给可见光分支，各自在自己像平面出框。
- 状态：未开始（前置的两份权重都在手）。

## 4. G 轴：多种子复跑

- 目的：把"抖动"量出来，才能判断 A 与 C1 的差距是否真实。
- 计划：`yolov8n` 在 1359 图上跑 3 个种子（train 相同、`seed`/数据顺序不同），报 mAP50 均值与极差。
- 状态：未开始。成本约 3 × 37 min。

## 5. 复现命令

```powershell
cd rail-intrusion
# 验证集是否与基线一致（应输出 only in new: 0 / only in old: 45）
$new = Get-ChildItem data\osdar23_yolo43\images\val -File | % { $_.Name }
$old = Get-ChildItem data\osdar23_yolo\images\val -File | % { $_.Name }
Compare-Object $new $old | Group-Object SideIndicator | % { "$($_.Name) $($_.Count)" }

# 基线权重在新 val 上复测（应得 0.7027 / 0.4493）
python -u tools/verify_per_class.py runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt 1280 data/osdar23_yolo43/dataset.yaml out/osdar23_yolo43/per_class_516img_on_newval.json

# A 轴训练 37 分钟
python -u tools/train_osdar23.py --data data/osdar23_yolo43/dataset.yaml --out out/osdar23_yolo43 --weights models/yolov8n.pt --epochs 60 --imgsz 1280 --batch 4 --device 0 --name osdar23-yolov8n-1359img-1280
```
