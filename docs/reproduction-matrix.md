# 复现矩阵（文献 × 可复现条目）

目的：把 `refs/` 里的**每篇论文**落成一行，明确"数据集 / 指标 / 超参 / 消融"四项是否齐全，从而知道哪篇能被我复现、哪篇只能引用。这是 m04987 要求的"文献 × 铁路项目闭环"的记账表。

标签沿用 `docs/report-conventions.md`：**【基线】【复现】【消融】【自研】**。

## 一、总表

| 编号 | 论文（`refs/extracted-text/` 下文件名） | 数据集 | 主指标 | 四项齐全？ | 状态 |
|---|---|---|---|---|---|
| A3 | `2023_Tagiew_OSDaR23-dataset_ICRAE` | OSDaR23（自建） | 无数值基线 | 数据描述完整、无基线 | **数据来源**（已用） |
| A5 | `2024_RailFOD23-dataset_ScientificData` | RailFOD23（自建） | 检测 mAP | 数据完整、基线在正文 | 待评估 |
| D3 | `2024_Rail-PillarNet-LiDAR-3D-FOD_CMC` | **OSDaR23 LiDAR** | mAP@0.5（4 类） | **四项齐全** | **复现中（见下）** |
| B1 | `2024_Liu_IR-Vis-fusion-data-compatibility-to-task-adaption_TPAMI` | KAIST / 自采 | 检测/分割 | 部分（依赖 KAIST） | 待评估 |
| B2 | `2019_Zhang_Weakly-aligned-cross-modal-learning_ICCV` | KAIST | mAP | 部分 | 仅引用 |
| B3 | `2021_Zhang_Weakly-aligned-feature-fusion_TNNLS` | KAIST | mAP | 部分 | 仅引用 |
| B4 | `2024_MS-DETR-loosely-coupled-fusion_TITS` | M3FD / 自采 | mAP | 部分 | 待评估（显存门槛高） |
| B6 | `2022_Wang_Unsupervised-misaligned-IR-Vis-fusion-registration_IJCAI` | KAIST | 配准误差 | 部分 | 仅引用 |
| E3 | `2022_Liu_TarDAL-target-aware-dual-adversarial_CVPR` | 多域 | mAP | 部分 | 仅引用 |
| C3 | `2014_Motion-prediction-risk-assessment-survey_ROBOMECH` | — | — | 综述 | 仅引用 |

> "部分"= 论文给了方法但数据集需申请/涉密，或超参不全，无法在本机复现。

## 二、D3 复现规格（唯一四项齐全的）

- **数据集**：OSDaR23 LiDAR，论文 45 序列 / 1534 帧 → 取 4 类（vehicle / pedestrian / buffer_stop / trains）→ 1356 帧，8:2 切分 = **1084 train / 272 val**。
- **本机对齐**：我有 43 序列 / 1437 帧（少 2 序列）→ **1336 帧**，与论文差 20 帧（1.5%），必须写进偏差表。
- **超参**：OpenPCDet；Adam lr 0.03 / wd 0.01 / momentum 0.9；voxel [0.16, 0.16, 12]，最多 16000 pillar × 32 点；点云范围 X[-30,354] Y[-25,39] Z[-4,8]；anchor 尺寸按 4 类 GT 均值（vehicle [4.29,3.06,2.78]、pedestrian [0.88,0.85,1.89]、buffer_stop [1.78,2.85,2.50]、trains [62.95,3.99,4.28]）。
- **指标**：IoU 0.5、40 个 recall 阈值下的 AP；论文报告 PointPillars（Model A，无 PA/CA/backbone/TL）**mAP@0.5 = 45.68**。
- **消融**：Table 2 的 Model A–G（PA、CA、自研 backbone、迁移学习四项开关）。

### 本机化后的已知偏差（必须写进论文"复现差异"段）

| 项 | 论文 | 本机 | 影响 |
|---|---|---|---|
| 框架 | OpenPCDet + spconv（CUDA 扩展） | 纯 PyTorch 重实现（spconv 在镜像源无 wheel） | 算子实现不同 |
| 显存 | TITAN RTX 24 GB | RTX 4060 Laptop 8 GB | batch 2 → **batch 1** |
| 点特征 | 9 维（含 pillar 中心偏移） | 4 维（x y z intensity） | 表达能力略弱 |
| backbone | 3 级下采样至 1/8，1/4+1/8 融合 | 只降 2 倍，输出 1/4 | 感受野更小 |
| 每格 anchor | 4 类 × 2 朝向 | 同（A=8） | 一致 |
| 帧数 | 1356 | 1336 | −1.5% |
| 学习率 | 0.03 / batch 2 | 见 `tools/train_pp.py` 实参 | 需记录 |

## 三、复现流程（每次照做）

1. 读论文 → 填本表该行（四项）。
2. 数据能拿到才动手；拿不到 → 标"仅引用"，不硬造。
3. 跑通 → 在 `docs/` 写复现报告，含"本机数字 / 论文数字 / 偏差表 / 差异原因"。
4. 达标才进入消融；不达标先查实现，不换模型。
5. 汇报一律按 `docs/report-conventions.md` 三段式（复现了什么 / 消融 / 突破）。

## 四、当前进度

- **D3**：数据侧完成（`data/pp_kitti`，1336 帧，KITTI 格式）；模型侧完成 `railintrude/det/`（pillarize / PointPillars / anchors / loss / pipeline / align）；训练入口 `tools/train_pp.py` 冒烟通过（loss 24327 → 3.94）；**20 epoch 基线正在跑**（`out/pp_train.log`）。
- 待办：等基线出 mAP → 与 45.68 对比 → 写 `docs/d3-reproduction-report.md` → 再做 B–G 消融。
