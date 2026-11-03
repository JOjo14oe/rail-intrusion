# rail-intrusion

轨道交通车端侵限检测：数据构建、2D 基线、车端报警原型，以及在 OSDaR23 LiDAR 上对 Rail-PillarNet（CMC 2024）PointPillars 基线的纯 PyTorch 复现。

> **单位约定**：2D（YOLO）指标是 0–1 比例；3D（PointPillars）指标是百分制。下表逐行标注。

## 结果（全部实测，可按"复现"一节重跑）

### 2D 检测（YOLOv8n，OSDaR23 rgb_center，155 帧固定验证集）

| 实验 | 类型 | mAP50 | 说明 |
|---|---|---|---|
| 516 图 / 640 | 基线 | 0.7027 | 5 类 |
| 1359 图 / 1280 | 自研 | 0.7348 | +0.032，落在种子抖动（0.0399）内 |
| 三种子 | 自研 | 0.7348 / 0.6982 / 0.6949 | 均值 0.7093 |
| 640 → 1280 | 消融 | 0.5195 → 0.5399 | 细杆类 catenary_pole 0.0591 → 0.1590（×2.7） |
| 合并两类杆 | 消融 | 0.7024 | 劣于 0.7348 |
| 原生尺度切片推理 | 消融 | 0.5475 → 0.0018 | 训练/推理尺度失配 |
| 跨站留出（9 条整序列 / 178 帧） | 自研 | 0.7348 → 0.6683 | 真实域偏移代价 |
| 帧级红外融合 | 自研 | F1 0.975 | 与纯 RGB 完全一致，无增益 |

### 3D 检测（PointPillars，OSDaR23 LiDAR，4 类，1069/267 帧）

| 实验 | 类型 | mAP50（百分制） | 说明 |
|---|---|---|---|
| Rail-PillarNet 论文 PointPillars | 基线 | 45.68 | 1084/272 帧，OpenPCDet |
| 本项目 12 轮 | 复现 | **0.42** | 未收敛，见"已知问题" |

## 车端报警原型

输入前视视频 → YOLOv8n 检测 → IoU 跟踪 → 框底边透视反比测距（正式评测改用 LiDAR 测距）→ INS 测速 → TTC 四级报警（notice >8s / alert ≤8s / warn ≤5s / emerg ≤3s，升即刻降需 5 帧）→ 带框视频 + `events.csv`。

```powershell
python -m railintrude.demo --video data/raw/platform-arrival-1080p.webm --out out/full
```

## 目录

```
railintrude/     核心库：2D 管线、lidar 反投影、det（PointPillars）模块
tools/           数据构建 / 训练 / 评测 / 文档转换
docs/            设计与实验报告、文献复现规格、夜间记录
tests/           单元测试
refs/            论文 PDF、提取文本、文献综述、复现矩阵
```

## 复现

```powershell
python tools/build_osdar23_yolo.py --camera rgb_center          # OSDaR23 -> YOLO
python tools/train_osdar23.py --data data/osdar23_yolo/dataset.yaml --epochs 60
python tools/build_pp_kitti.py                                  # OSDaR23 -> KITTI（LiDAR）
python tools/train_pp.py --epochs 60 --device cuda
python -m pytest -q                                             # 单元测试
```

自检（判断实现是否正确，不靠猜）：小批过拟合——24 帧训练集与验证集相同，跑完后 mAP50 必须接近满分，否则说明实现有 bug。

```powershell
python tools/train_pp.py --dense --overfit --limit 24 --epochs 60 --lr 2e-3
```

## 已知问题（诚实清单）

1. **3D 分支尚未收敛**：12 轮全量训练 mAP50 = 0.42（百分制），论文是 45.68（同为百分制）。当前正在做小批过拟合自检以区分"欠训练"与"实现 bug"；主因很可能是训练步数只有论文的约 4%。
2. 3D 复现与原论文存在明确偏差：OpenPCDet → 纯 PyTorch（本机 spconv 无 win+cp312 轮子），backbone 降采样次数、intensity 填 0、1336 帧 vs 1356 帧；明细见 `docs/readings/D3.md` 第 8 节。
3. 红外融合、跨站泛化、时序风险三项均为**负结果**，已写入 `docs/`，不作为卖点。

## 数据与许可

数据与权重均不入库。OSDaR23 需自行从 DZSF 官方渠道下载（下载站有 Anubis 反爬，须用浏览器）；标注许可 CC0，传感器数据 CC BY-SA 3.0 DE。
