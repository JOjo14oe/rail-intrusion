# OSDaR23 获取指南：免申请，但下载要过浏览器验证

**两条纠正（都是我先前写错的）**：

1. 之前说「需要申请 / 等审批」——**错的**。DOI 解析到数据集页，页面直接列出 45 个 zip 直链，无账号、无表单。
2. 后来改口说「`curl` 直接就能下、无反爬」——**也错了**。`curl` 拿到的是 24 KB 的 HTML 验证页（Anubis 反爬），不是 zip。实测证据见第 4 节。

**结论：在浏览器里打开数据集页、点链接下载**（浏览器会自动完成 Anubis 的 JS 验证）；程序化 curl / wget / requests 一律拿不到文件。

---

## 1. 入口

| 用途 | 地址 |
|---|---|
| 数据集页（清单 + 许可说明） | `https://data.fid-move.de/dataset/osdar23` |
| DOI | `https://doi.org/10.57806/9mv146r0` → 302 到上面那个页面 |
| 文件直链模板 | `https://download.data.fid-move.de/dzsf/osdar23/<序列名>.zip` |
| 研究报告 + 标注指南 | `https://www.dzsf.bund.de/SharedDocs/Standardartikel/DZSF/Projekte/Projekt_70_Reale_Datensaetze.html` |
| 标注格式 devkit | `github.com/dbinfrago/raillabel`（Apache-2.0；本机 GitHub 不可达，但 `raillabel` 是 PyPI 包，用清华源可装） |

已把 45 条直链存到 `out/osdar23/urls.txt`（会话工作区），页面快照在 `out/osdar23/page.html`。

## 2. 许可（必须写进答辩材料）

| 内容 | 许可 |
|---|---|
| 标注数据（`.json`） | **CC0 1.0**（公共领域，无署名义务） |
| 传感器数据（`.png` / `.pcd` / `.csv`） | **CC BY-SA 3.0 DE** |

实务含义：

- **可商用、可修改**，没有非商用限制——比 RAWPED（仅非商业、禁再分发）宽松得多。
- 传感器数据**必须署名**：DZSF（德国联邦铁路局铁路交通研究中心）、Digitale Schiene Deutschland / DB Netz AG、FusionSystems GmbH。
- **ShareAlike**：如果你把裁出来的子集 / 转换过格式的标注**作为数据集再发布**，要以同样许可发布；论文、答辩 PPT、演示视频属于引用，不属于衍生数据集，正常署名即可。
- 数据里含车站行人与车辆，公开发布截图/视频时注意个人信息。

## 3. 45 个文件与推荐子集

| 序列 | 文件 | 与本项目的相关性 |
|---|---|---|
| `1_calibration` | 1.1, 1.2 | **先下这个**：标定参数决定融合能做特征级还是只能做决策级 |
| `7_approach_underground_station` | 7.1, 7.2, 7.3 | **最贴合**：地下站进站，隧道 + 站台过渡，正是 V0 里误报的场景 |
| `2_station_berliner_tor` | 2.1 | 站台，行人与列车 |
| `8_station_altona` | 8.1, 8.2, 8.3 | 大站站台 |
| `9_station_ruebenkamp` | 9.1–9.7 | 站台多序列，适合做跨序列泛化（C3） |
| `5_station_bergedorf` / `6_station_klein_flottbek` / `10_station_suelldorf` / `11_main_station` / `13_station_ohlsdorf` / `21_station_wedel` | 各 1–3 个 | 更多站点，扩样本用 |
| `4_station_pedestrian_bridge` | 4.1–4.5 | 天桥/人行通道 |
| `3_fire_site` | 3.1–3.4 | 火灾现场，非主线 |
| `12/19/20_vegetation_*` | 各 1 个 | 植被侵限（可选，正好是"侵限"类） |
| `14_signals_station` / `17_signal_bridge` | 14.1–14.3 / 17.1 | 信号机 |
| `15_construction_vehicle` | 15.1 | 施工车辆（异物类） |
| `16_under_bridge` | 16.1 | 桥下弱光 |

**建议第一批只下三个**：`1_calibration_1.2`、`7_approach_underground_station_7.1`、`2_station_berliner_tor_2.1`。够验证格式、够跑通融合实验的第一步。

## 4. 一个硬坑：下载服务器有 Anubis 反爬

下载域名 `download.data.fid-move.de` 前面挡着 **Anubis**（techaro 的反爬代理，Linux 发行版镜像常用）。任何非浏览器请求都会被它回一张 HTML 验证页：

| 实测 | 结果 |
|---|---|
| `curl -I <zip 直链>` | `HTTP/1.1 200 OK`、`Content-Type: text/html`、`Set-Cookie: techaro.lol-anubis-*`、**没有 `Content-Length`** |
| `curl <zip 直链>` | 24,467 字节，`<!doctype html><title>Making sure you're not a bot!</title>` |
| 换 Chrome UA 再试 | 同样是验证页（24,567 字节，只是 cookie 不同） |
| `Expand-Archive` 该文件 | `找不到中央目录结尾记录` —— 因为它根本不是 zip |

所以：

- **不能用 curl / wget / requests 下载**，拿到的永远是 HTML；
- 第 52 节原本写的「服务器不返回 content-length、不支持 Range」其实是同一个原因：你看到的是验证页，不是文件；
- **正确做法**：在浏览器里打开数据集页，点 zip 链接，浏览器跑完 JS 验证后会自动开始下载；
- 下完必须校验是不是真 zip：看前 2 字节是否为 `50 4B`（`PK`），或直接 `Expand-Archive` 试解压。

## 5. M0 验收步骤（每步留证据）

| # | 步骤 | 判据 |
|---|---|---|
| 1 | 下 `1_calibration_1.2.zip` 并解压 | 能看到各相机内参/外参文件；记录有哪些相机（RGB / IR 各自的分辨率与位置） |
| 2 | 下 `7_approach_underground_station_7.1.zip` | zip 完整（`Expand-Archive` 无报错），目录里同时有 RGB 与 IR 的 png + raillabel json |
| 3 | 解析一份 json | 用 `raillabel` 读出 frame 列表与类别分布，输出 `out/osdar23/class_stats.json` |
| 4 | ~~确认 RGB 与 IR 是否可配准~~ **已完成，见第 8 节** | 用 1230 对同目标标注验证：单个单应 IR→RGB 中位残差 **11.1 px**（画面宽 0.45%）→ 可做**对象级对齐**，不到像素级；结论已写进第 8 节 |
| 5 | 划夜间/隧道子集 | 按序列（7 号就是地下站）或按时间戳，产出子集索引 |
| 6 | 裁训练子集 | 8 GB 显存 + 15.6 GB 内存 → 单次 ≤5k 帧、imgsz 640、batch ≤8 |

**第 4 步之前不要按"特征级融合"写死设计。**

## 6. 环境准备

```powershell
# 代理变量必须用 Remove-Item 清（赋值空串无效）
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue

python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple raillabel
```

下载**不用 curl**（会被 Anubis 拦，见第 4 节）：浏览器打开数据集页 → 点链接 → 下完先验 `50 4B` 文件头再解压。

## 7. 与本项目主张的对应

| 主张 | 依赖 OSDaR23 的什么 |
|---|---|
| C1 融合 vs 单模态 | 同序列成对的 RGB + IR（7 号地下站序列最理想） |
| C2 时序风险 vs 逐帧阈值 | 连续帧 + 真值标注（raillabel json 带逐帧框） |
| C3 跨序列泛化 | 多个 station 序列，留出整条序列不参与训练 |

即：**这个数据源能同时支撑三条主张**，而且没有审批等待，随时可以开工。

## 8. 真包实测（`1_calibration_1.2`，8.77 GB / 1426 个文件）

以下全部来自真包内的 `readme.md`、`calibration.txt`、`1_calibration_1.2_labels.json`，不是查来的。

### 8.1 传感器（9 相机 + 3 激光雷达 + 毫米波 + INS，全部 10 Hz 硬件同步）

| 模态 | 型号 | 分辨率 | 目录 |
|---|---|---|---|
| 12 MP 彩色 ×3 | Teledyne GenieNano 5GigE C4040 | 4112×2504 | `rgb_highres_left/center/right` |
| 5 MP 彩色 ×3 | Teledyne GenieNano C2420 | 2464×1600 | `rgb_left/center/right` |
| 红外 ×3 | Teledyne Calibir DXM640 | 640×480（8 bit 灰度） | `ir_left/center/right` |
| 长距激光 ×3 | Livox Tele-15 | 5 万–8.4 万点/帧 | `lidar/`（合并点云） |
| 毫米波 | — | 2856×1428 | `radar/` |
| INS/IMU | NovAtel OEM7 | CSV | `novatel_oem7_{corrimu,inspva,insstdev}/` |

坐标系：`X=前进, Y=左, Z=上`，**XY 平面就是轨面，Z=0**。每台相机都有 `position` / 旋转矩阵 / `camera_matrix` / `dist_coeffs`（红外与激光的畸变为 0）。

### 8.2 标注结构（OpenLABEL 1.0.0，不是 raillabel 的私有格式）

```
openlabel
├── metadata         # 许可、导出器版本（这里写清了 CC0 / CC BY-SA 3.0 DE）
├── streams          # 每个流的 intrinsics_pinhole（含 width_px / height_px）
├── coordinate_systems
├── objects          # 39 个对象，含 type 与生命周期 frame_intervals
└── frames           # {"0".."99"} 每帧：frame_properties.{timestamp,streams[uri]} + objects
```

取框的方式：`frames[fid].objects[uuid].object_data.bbox[*]`，其中 `name` 形如 `"ir_center__bbox__catenary_pole"`（**相机名__bbox__类别**），`val = [x, y, w, h]`（像素，左上角 + 宽高，不是 xyxy）。

### 8.3 本序列的标注分布（100 帧 / 15,126 个标注）

`catenary_pole 4041`、`road_vehicle 2867`、`signal 2250`、`track 1782`、`buffer_stop 1598`、`person 1155`、`switch 1025`、`signal_pole 379`、`group_of_bicycles 29`；
**`train`、`wagons`、`bicycle`、`motorcycle`、`animal`、`flame`、`smoke` 全为 0** —— 这是车辆段标定序列，没有列车。

### 8.4 RGB 与 IR 能不能配准：能到「对象级」，不到「像素级」

用同一对象在同一帧里**同时**有的 `ir_center` 与 `rgb_center` 框（共 **1230 对**：person 100 / road_vehicle 245 / buffer_stop 300 / catenary_pole 585），拟合一个全局单应 IR→RGB：

- 残差（RGB 2464×1600 像素）：**中位 11.1 px、均值 12.4 px、p90 21.7 px、最大 62.5 px**，占画面宽 **0.45%**；
- 逐类残差：road_vehicle 9.7 / catenary_pole 10.1 / buffer_stop 12.1 / person 13.6 px。

FOV 不对等是根因：IR 640×480 @ fx=3535 → **10.35°×7.77°**；RGB 5MP 2464×1600 @ fx=4609 → **29.96°×19.73°**。实测 IR 的全部横向范围只映射到 RGB 归一化 **x∈[0.345, 0.697]**，即**中央约 35%**。两者还是不同位置的相机（IR 高 3.52 m，5MP RGB 高 3.50 m，12MP RGB 高 2.08 m），存在视差。

**因此融合层级这样定（有数据依据，答辩可以直接讲）**：

| 层级 | 可行性 | 依据 |
|---|---|---|
| 早融合（IR 上采样拼进 RGB） | **不建议** | FOV 差 3 倍 + 视差，贴上去就是错位 |
| 中融合（共享特征） | **需深度** | 需要逐目标深度（用 LiDAR）才能把两个视角投到同一处 |
| 晚融合（各模态各检，再用单应做对象级关联） | **推荐主线** | 11 px 的关联精度足够做 ROI/目标级融合与消融 |

### 8.5 体积与预算

单个序列约 **8.8–9.2 GB**、100 帧（10 秒）。做 C1 只需要 `ir_center` + `rgb_center` + 标注 json，解压后可删 zip。
45 个文件全下 ≈ 400 GB，**不现实**；按需下 3–5 个序列即可。

### 8.6 子序列的真实规模（容易误判，务必先看）

**每个 zip 不是一条长序列，而是一个短片段**。实测三个（各自 zip 里只有 10 个文件/流的帧数）：

| 序列 | 帧数 | zip 体积 | 解压后（只取 ir_center + rgb_center + 标注） | 有 `train` 标注 |
|---|---|---|---|---|
| `1_calibration_1.2` | **100** | 8.77 GB | 元数据 + 标注 23 MB（图像未解） | 0 |
| `7_approach_underground_station_7.1` | **10** | 0.64 GiB | 57 MB | **435** |
| `2_station_berliner_tor_2.1` | **10** | 0.80 GiB | 61 MB | 0（但 `person` 949） |

即：一个 zip = 1 秒（10 帧）或 10 秒（100 帧）的多传感器片段。45 个文件加起来也只有千帧量级 —— **够做评测与轻量微调，不够从头训练**。这跟 V0 的"预训练权重 + 少量微调"路线正好吻合。

### 8.7 这两个序列暗不暗（C1 的夜间子集能不能立住）

对 rgb_center 与 ir_center 逐帧算灰度统计（各 10 帧）：

| 序列 | 模态 | 平均灰度 | 灰度 < 40 的像素占比 |
|---|---|---|---|
| `7_approach_underground_station_7.1` | rgb_center | 107.4 | **42.3%** |
| 同上 | ir_center | 127.9 | **2.1%** |
| `2_station_berliner_tor_2.1` | rgb_center | 102.7 | **30.4%** |
| 同上 | ir_center | 127.9 | **1.0%** |

结论不是"夜景"，而是**高动态范围 + 大面积压暗**：RGB 有 30–42% 的像素被压在 40 灰度以下（隧道口、地下站台、强逆光），红外只有 1–2%。所以 C1 的主张应当写成 **"弱光/大暗区子集"**，而不是"夜间"——用 `灰度<40 像素占比 ≥30%` 就能把帧自动划进这个子集，判据可复现。
