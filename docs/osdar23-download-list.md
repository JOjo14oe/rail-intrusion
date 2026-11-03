# OSDaR23 扩数据下载清单（当前唯一有效的一步）

## 为什么现在做这件事

`imgsz=1280` 重训把 `catenary_pole` 的 mAP50 从 0.059 提到 0.159（×2.7），**但 `signal_pole` 从 0.060 掉到 0.057**——分辨率救不回它。

验证集里 `signal_pole` 只有 **45 个框**、训练集只有 84 张图。这是数据量问题，不是分辨率问题、也不是标注问题（Precision 1.000 说明报出来的都对）。所以下一步只有一条：**把训练图从 84 张扩到 300 张以上**。

## 现状与目标

| 项 | 现在 | 目标 |
|---|---|---|
| 序列数 | 3（1_calibration / 2.1 / 7.1） | 20+ |
| 训练图 | 84 | 300+ |
| 验证图 | 36 | 100+ |
| `signal_pole` 训练框 | 极少数 | 数百 |

## 怎么下载（重要：不能用 curl）

OSDaR23 下载服务器有 **Anubis 机器人验证**，`curl`/下载器/右键另存为拿到的都是 24 KB 的验证页 HTML，不是 zip。

正确做法：**用浏览器打开下面的链接**，等它自己过验证（几秒），浏览器就会开始下载真正的 zip。下完的 zip 丢到桌面即可，不用解压。

数据集主页（可核对）：<https://data.fid-move.de/dataset/osdar23> · DOI: 10.57806/9mv146r0

## 第一批（15 个 zip ≈ 150 帧，优先下这批）

挑的都是对现有 5 类（buffer_stop / catenary_pole / person / road_vehicle / signal_pole）补充最直接的序列。

### 1. 信号类 —— 直击最弱的 `signal_pole`

- <https://download.data.fid-move.de/dzsf/osdar23/14_signals_station_14.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/14_signals_station_14.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/14_signals_station_14.3.zip>

### 2. 地下站进站 —— 最贴近"车端侵限"场景

- <https://download.data.fid-move.de/dzsf/osdar23/7_approach_underground_station_7.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/7_approach_underground_station_7.3.zip>

### 3. 人行天桥车站 —— 行人最多（person 现为 0.703 mAP50）

- <https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.3.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.4.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/4_station_pedestrian_bridge_4.5.zip>

### 4. 大站系列（类别最全）

- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.3.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.4.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.5.zip>

## 第二批（16 个 zip ≈ 160 帧，第一批跑通后再下）

- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.6.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/9_station_ruebenkamp_9.7.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/8_station_altona_8.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/8_station_altona_8.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/8_station_altona_8.3.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/21_station_wedel_21.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/21_station_wedel_21.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/21_station_wedel_21.3.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/13_station_ohlsdorf_13.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/11_main_station_11.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/5_station_bergedorf_5.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/5_station_bergedorf_5.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/6_station_klein_flottbek_6.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/6_station_klein_flottbek_6.2.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/17_signal_bridge_17.1.zip>
- <https://download.data.fid-move.de/dzsf/osdar23/10_station_suelldorf_10.1.zip>

## 这批不做

`3_fire_site_3.x`（flame/smoke 不在 5 类）、`12/18/19/20_vegetation_*`（植被，对 5 类无用）、`15_construction_vehicle_15.1`、`16_under_bridge_16.1`（类别不在 5 类内）、`1_calibration_1.1`（已有的 1.2 是超集）。

## 磁盘与体积

每个车站类 zip 约 **0.5–1.5 GB**，31 个约 **25–40 GB**。C 盘现在 380 GB 空闲，够。

解压时**只取 `rgb_center` 一个流**（8.77 GB 的 calibration 包只解出约 480 MiB 图像就是靠这个），所以解压后占用远小于 zip 体积。

## 拿到 zip 之后我这边做什么（不需要你操作）

1. `python tools/extract_osdar23.py "<zip>" --streams rgb_center` —— 只解需要的流 + 标注 + 标定
2. `python tools/build_osdar23_yolo.py` —— **序列现在是自动发现的**（扫 `data/osdar23/**/*_labels.json`，按文件名反推图像目录，重复副本自动去重），加多少个序列都不用改代码
3. `python tools/train_osdar23.py --epochs 60 --imgsz 1280 --batch 4 --device 0` —— 重训并出零样本/微调对照
4. 复跑 `tools/eval_slicing.py` 看 `signal_pole` 是否被救回来

## 提醒（写进报告时要用）

- 这批数据是**德国汉堡车站/线路**场景，类别口径是 OSDaR23 的德标定义，答辩时要说清"公开基准上的结果"，不等于国内线路实测性能。
- 训练/验证仍按**序列内时间切分**（前 70% 训练 / 后 30% 验证），避免近邻帧泄漏。
- 解压后 `data/osdar23/` 体积会明显变大，但已在 `.gitignore` 里，不会进仓库。
