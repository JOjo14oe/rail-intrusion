# 微调实验报告（V0.5）

日期：2026-10-20　硬件：RTX 4060 Laptop 8GB / i9-13980HX / 15.6GB RAM
数据：Wikimedia 莫斯科地铁进站前视视频（1431 帧 / 57.3 s / 1920×1080 / 25 fps，仅此一段）
软件：torch 2.14.1+cu130 / torchvision 0.29.1+cu130 / ultralytics 8.4.183

## 1. 结论先行

**用"自己出的伪标签"微调自己，在结构上不可能变好，只会退化。** 三个变体全部验证了这一点：mAP50 基本不动，mAP50-95 一致下降 0.09–0.36，precision 下降 0.15–0.28，recall 涨到 0.99–1.00（因为模型趋向"能报就报"）。所以**没有把微调权重设为演示默认模型**，演示继续用 yolov8n 原权重。

同时暴露一个**必须写进答辩**的失效模式：微调后模型在隧道段把"大片暗区 + 车头灯光"识别成 `train`，给出覆盖画面 60% 的巨型框；该框底边落在标定近端 → 距离 5.6 m → TTC 0.5 s → 直接 `emerg`。原权重在同样三帧上是 0 检测。

## 2. 三个变体一览

| 变体 | 数据集 | 关键设置 | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| v3 解冻 | dataset_v3（245 目标帧 + 47 背景） | 60 epoch，freeze=0，lr 由 ultralytics 自动选 0.000909 | 0.6315（−0.3576） | 0.4839（−0.3637） | 0.9911（+0.0749） | 0.4652（−0.5128） |
| v4 冻结+马赛克 | dataset_v3 | 25 epoch，freeze=10，lr0=0.002，mosaic=1 | 0.9882（−0.0009） | 0.7499（−0.0977） | 0.7638（−0.1524） | 0.9915（+0.0135） |
| v5 冻结+无马赛克+密集背景 | dataset_v4（245 目标帧 + **116 背景**，背景占 32%） | 25 epoch，freeze=10，lr0=0.002，**mosaic=0** | 0.9921（+0.0002） | 0.7674（−0.0856） | 0.6378（−0.2803） | 1.0000（+0.0261） |

括号内为相对同数据集 val 上**原权重**的差值（原权重：v3 val 0.9891 / 0.8476 / 0.9162 / 0.978；v4 val 0.9919 / 0.8530 / 0.9181 / 0.9739）。

三个变体都**在第 1 epoch 就取到 best**，随后 15 epoch 无提升被早停——这是"起点已在最优点、之后每一步都是偏移"的直接证据。

## 3. 隧道误报的作用机制（可复现）

| 模型 | 第 120 帧 | 第 149 帧 | 第 1079 帧 |
|---|---|---|---|
| yolov8n 原权重 | 0 检测 | 0 检测 | 0 检测 |
| v4 微调 | `train` conf 0.811，框 (1121,383,1914,1035) | 0 检测 | `train` conf 0.974，框 (0,0,913,1063) |
| v5 微调 | 0 检测 | 0 检测 | `train` conf 0.974，框 (1,30,903,1053)，底边 y=1054 → d=5.6 m |

加背景帧 + 关马赛克把 3 帧里的 2 帧压下去了，最坏那帧（1079）仍是 0.974。**结论：这个误报不是抖动，是权重偏移产生的场景级伪目标，靠加空帧和增强设置治不干净。**

## 4. 为什么"必然不涨"（结构性原因）

1. **教师就是学生**：val 标签由同一权重生成，原权重在该 val 上已接近满分（mAP50 0.989–0.992）。学生的初始化已经在这个函数的最优点上，损失在起点≈0，梯度只会把它推离最优点。所以"微调后更高"在数学上不可能，任何显示"上涨"的数字都是噪声或过拟合。
2. **只能学到"更爱报"**：freeze=10 冻结骨干后，可训练的只有检测头；检测头能在 25 epoch 内改变的，主要是分类置信度的整体偏移——于是 recall 被推到 1.0、precision 腰斩。这不是"学到了铁路特征"，是"降低了报警门槛"。
3. **解冻则直接崩**（v3）：245 帧、2180 个框、类别严重不均衡（person 2070 / truck 4 / car 2 / train 104），全网络解冻后 lr 一旦不够小就把预训练表征冲掉——第 1 epoch 之后 mAP 一路下滑到早停。
4. **伪标签带不进新信息**：教师漏检的东西（隧道里真正该报的、夜间目标、远处小目标）标签里没有，学生再怎么也学不会。

## 5. 已保留的工程加固（不随实验回退）

- **时序门控** `Config.min_hits=3`：一条轨迹连续命中 3 帧才允许升级报警（`railintrude/pipeline.py`，测试 `tests/test_pipeline.py`）。`tests/test_dataset.py::test_select_frames_*` 覆盖背景帧抽样。
- **背景帧抽样** `railintrude/dataset.py::select_frames(labelled, empty, neg_stride=5)`：保留全部目标帧，背景帧按 `neg_stride` 抽稀；`tools/build_dataset.py --neg-stride`。
- 测试总数 **35 passed**。
- 注意：门控**不能**解决隧道误报——那框连续存在十几帧，不是单帧抖动。

## 6. 下一步（按"能否真的涨"排序）

1. **换真标签**（唯一能抬天花板的路）：人工抽检 200–300 帧修框，或下载 OSDaR23 官方标注。伪标签路线的上限已经撞到了，不值得再调参。
2. **换数据源**：本实验只有一段莫斯科地铁视频——单场景、单车型、单天气。任何跨场景结论都无从谈起，这也正是 V1（可见光 + 红外融合、夜间/逆光子集）必须依赖 OSDaR23 的原因。
3. **把这次负结果写进材料**：它是"为什么不能自举伪标签"的干净证据（三个变体 + 三帧对照），比一个假的 +0.5 mAP 更有说服力。
4. 若必须继续在本视频上做训练实验：先在**留出的整段（不是随机帧）**上做人评基线，否则所有 mAP 都是自证。

## 7. 复现命令

```powershell
# 数据集（v4 = 密集背景帧）
python tools/build_dataset.py --video data/raw/platform-arrival-1080p.webm --out data/dataset_v4 --stride 3 --conf 0.35 --neg-stride 2 --device 0

# 三个变体
python tools/train_experiment.py --data data/dataset_v3/dataset.yaml --weights models/yolov8n.pt --epochs 60 --freeze 0 --device 0 --name finetune-v3-unfrozen --out out/experiment_v3
python tools/train_experiment.py --data data/dataset_v3/dataset.yaml --weights models/yolov8n.pt --epochs 25 --freeze 10 --lr0 0.002 --device 0 --name finetune-v4-frozen --out out/experiment_v4
python tools/train_experiment.py --data data/dataset_v4/dataset.yaml --weights models/yolov8n.pt --epochs 25 --freeze 10 --lr0 0.002 --mosaic 0 --device 0 --name finetune-v5-nomosaic --out out/experiment_v5

# 逐帧探针（三帧对照）
python out/probe/probe_f149.py runs/detect/runs/finetune-v5-nomosaic/weights/best.pt v5
```
