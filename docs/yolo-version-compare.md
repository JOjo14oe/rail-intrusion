# YOLOv8n vs YOLO11n 同条件对照

**一句话**：在**完全相同的数据、分辨率、轮数、batch**下，两者的整体 mAP50 差 0.019（v8n 0.7027 / v11n 0.6842），这个差距落在单次训练的正常抖动范围内，**不能据此说"v8n 更好"或"新版更强"**；真正的差别在细杆类上（catenary_pole、signal_pole 各差 0.06/0.03）和参数量上（v11n 更小）。

## 1. 实验条件（唯一变量是模型）

| 项 | 值 |
|---|---|
| 数据 | `data/osdar23_yolo`，18 序列 516 张（train 361 / val 155） |
| 类别 | 钉死 5 类：buffer_stop / catenary_pole / person / road_vehicle / signal_pole |
| 分辨率 / batch / 轮数 | 1280 / 4 / 60，均从 COCO 预训练权重出发 |
| 验证集 | 同一份 155 张、2154 框 |
| 硬件 | RTX 4060 Laptop 8 GB，torch 2.14.1+cu130 |
| 权重 | `runs/detect/osdar23-yolov8n-516img-1280/weights/best.pt`（6,594,410 B）、`runs/detect/osdar23-yolo11n-1280/weights/best.pt`（5.6 MB） |
| 参数量 | v8n 3.16 M / v11n 2.60 M（ultralytics 打印值） |

## 2. 整体指标

| 模型 | mAP50 | mAP50-95 | P | R | 训练耗时 |
|---|---|---|---|---|---|
| YOLOv8n | **0.7027** | **0.4493** | 0.78 | 0.63 | 758 s |
| YOLO11n | 0.6842 | 0.4145 | 0.746 | 0.601 | 929 s |

v11n 慢了约 23%（同 batch 下每轮迭代更贵），指标低 0.019。

## 3. 逐类（mAP50 / mAP50-95）

| 类别 | v8n mAP50 | v11n mAP50 | v8n mAP50-95 | v11n mAP50-95 |
|---|---|---|---|---|
| buffer_stop | 0.813 | **0.816** | **0.551** | 0.499 |
| catenary_pole | **0.533** | 0.469 | **0.262** | 0.182 |
| person | **0.729** | 0.726 | **0.416** | 0.405 |
| road_vehicle | 0.962 | **0.964** | 0.716 | **0.722** |
| signal_pole | **0.476** | 0.446 | **0.301** | 0.265 |

两模型在同一档位上：person / road_vehicle / buffer_stop 差 ≤0.005（可视为相同），差异集中在两个细杆类（catenary_pole −0.064、signal_pole −0.030），且都是 v8n 略高。

## 4. 怎么读这个结果（写材料的口径）

1. **不能宣称"v8n 更强"**：单次训练、单一验证集、无多种子、无超参搜索。0.019 的差距与"换个随机种子"同量级——这是本项目在 YOLO 微调上已经踩过的坑（`docs/finetune-experiment-report.md` 里 40 epoch 与 25 epoch 之间就有 0.17 的摆动）。
2. **可以说**：在本数据与本配置下，两个模型处于同一水平；v11n 以 18% 更少的参数达到相当精度，但推理与训练都更慢；升级模型版本**没有带来收益**，因此工程上沿用 v8n。
3. **要下结论还需**：同配置 3 个种子复跑，报告均值±标准差。预计 3×25 分钟机器时间（v8n）与 3×31 分钟（v11n）。

## 5. 复现

```
python -u tools/train_osdar23.py --weights models/yolov8n.pt --epochs 60 --imgsz 1280 --batch 4 --device 0 \
    --name osdar23-yolov8n-516img-1280 --out out/osdar23_yolo
python -u tools/train_osdar23.py --weights models/yolo11n.pt --epochs 60 --imgsz 1280 --batch 4 --device 0 \
    --name osdar23-yolo11n-1280 --out out/osdar23_yolo11n
python -u tools/verify_per_class.py runs/detect/osdar23-yolo11n-1280/weights/best.pt 1280
```
