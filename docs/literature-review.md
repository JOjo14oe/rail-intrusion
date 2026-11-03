# rail-intrusion 文献清单（按项目卡点组织）

> 目标读者：写本项目的 technical report / 中期答辩稿的人。
> 组织方式不是"关键词堆一排"，而是按本项目**已经踩到的三个卡点**（C1 配准、C2 时序、v1 在轨判定）分组：
> 每条都写清"它解决什么问题 / 能支撑项目里的哪句话 / 不能支撑哪句话"。
> 全部 26 篇的标题、作者、年份、卷期页、DOI 已逐条用 OpenAlex 元数据核对；可下载的 10 篇 PDF 放在 `refs/`。

生成日期：2026-10-26

---

## 0. 这份清单怎么来的（可复现）

| 环节 | 做法 |
|---|---|
| 检索 | OpenAlex Works API（`api.openalex.org/works?search=...&sort=relevance_score:desc`），约 20 组查询，覆盖「铁路侵限/铁路障碍/OSDaR23/铁路数据集/轨道区域分割/轨道点云/RGB-T 检测/多光谱错配/跨模态对齐/红外可见光配准/TTC/前撞预警风险/单目测距/恶劣天气域适应/小目标/驾驶场景 VLM」 |
| 核对 | 每条用 `api.openalex.org/works/doi:<DOI>` 单独拉 `display_name / authorships / biblio / primary_location`，逐条比对标题与卷期页；arXiv 版本再抓 `arxiv.org/abs/<id>` 的 `citation_title` 复核，防止张冠李戴 |
| 摘要 | 取 OpenAlex `abstract_inverted_index` 重建成原文摘要，下文每条的"它做了什么"都据此写，未取到摘要的会明确标注 |
| 取全文 | 仅取开放获取（arXiv / IJCAI / Nature / Tech Science / SpringerOpen）；MDPI 与 ACM 对脚本访问返回 403，需浏览器打开，已在第 5 节注明 |

**一个更正**：项目大纲第十一节把综述写成「轨道交通障碍物侵入检测综述（IEEE 2025，document/11175219）」。核对结果是
Xu, Wang, Liu, Ni, *Advancements in Obstacle Intrusion Detection Methods for Rail Transit: A Comprehensive Review*, **IEEE TIM 74:1–34, 2025**, DOI [10.1109/TIM.2025.3612624](https://doi.org/10.1109/TIM.2025.3612624)。
引用时请用 DOI 而不是 document id，否则知网/GB-T 7714 格式会缺卷期页。

---

## 1. 速览表

| # | 文献 | 年/出处 | 对应项目问题 | PDF |
|---|---|---|---|---|
| A1 | Cao et al., Railway Intrusion Detection Based on Machine Vision: A Survey | T-ITS 25(7):6427–6448, 2024 | 选题定位、评测缺口 | — |
| A2 | Xu et al., Advancements in Obstacle Intrusion Detection Methods for Rail Transit | TIM 74:1–34, 2025 | 相关工作骨架 | — |
| A3 | Tagiew et al., OSDaR23: Open Sensor Data for Rail 2023 | ICRAE 2023:270–276 | **数据来源**、模态同步 | ✅ |
| A4 | Shi et al., A Survey on Multi-Sensor Fusion Perimeter Intrusion Detection in HSR | Sensors 24(17):5463, 2024 | 多传感器互补的必要性 | — |
| A5 | Chen et al., RailFOD23 | Sci Data 11:72, 2024 | 备选数据、异常稀有 | ✅ |
| A6 | Zhou et al., Machine Vision-Based Intrusion Detection in Low-Light Environments | Appl Sci 16(6):2848, 2026 | 夜间/低照度单模态对照 | — |
| B1 | Liu et al., IR and Visible Image Fusion: From Data Compatibility to Task Adaption | TPAMI 47(4):2349–2369, 2025 | **C1 根因的权威依据** | ✅ |
| B2 | Zhang et al., Weakly Aligned Cross-Modal Learning (AR-CNN) | ICCV 2019:5126–5136 | 区域级对齐而非全局配准 | ✅ |
| B3 | Zhang et al., Weakly Aligned Feature Fusion for Multimodal Object Detection | TNNLS 36(3):4145–4159 | 同上（期刊扩写版） | ✅ |
| B4 | Xing et al., MS-DETR | T-ITS 25(12):20628–20642, 2024 | 松耦合融合、模态不平衡 | ✅ |
| B5 | Song et al., Misaligned Visible-Thermal Detection (DVTOD) | T-IV 9(11):7449–7460, 2024 | 错配 benchmark、报告错配程度 | — |
| B6 | Wang et al., Unsupervised Misaligned IR-Vis Fusion | IJCAI 2022:3508–3515 | 免真值配准路线 | ✅ |
| B7 | Li et al., IR-Vis Fusion Without Strict Registration | IJCV 132(5):1625–1644, 2024 | 弱配准即可融合 | — |
| B8 | Liu et al., TarDAL + M3FD benchmark | CVPR 2022:5792–5801 | 融合以检测为目标 | ✅ |
| C1 | Lefèvre et al., A Survey on Motion Prediction and Risk Assessment | ROBOMECH J 1:1, 2014 | TTC 的定位与局限 | ✅ |
| C2 | Park & Hwang, Robust Range Estimation for VFCW | Sci World J 2014:923632 | 单目测距的工程修正 | — |
| C3 | Wang et al., Near-Incident Detection: Lateral Distance Estimation | CVPRW 2025:2027–2036 | **C2 最近的同类工作** | — |
| C4 | Vajgl et al., Dist-YOLO | Appl Sci 12(3):1354, 2022 | 检测头直接回归距离 | — |
| D1 | Wang et al., Adaptive Track Segmentation for Intrusion Detection | Sensors 19(11):2594, 2019 | 轨道报警区域自动提取 | — |
| D2 | Gézero & Antunes, 3D Linear Elements from Mobile LiDAR | Infrastructures 4(3):46, 2019 | 钢轨点云提取（方案 D 对照） | — |
| D3 | Li et al., Rail-PillarNet | CMC 80(3):3819–3833, 2024 | 点云 3D 检测、在轨判定 | ✅ |
| D4 | Song et al., Unsupervised Point-Cloud FOID | ICIEA 2024:1–6 | 无侵限真值的训练策略 | — |
| D5 | Nan et al., LiDAR-Camera Joint Obstacle Detection for Track Area | ESWA 275:127089, 2025 | 与 v1 同构的同类工作 | — |
| D6 | Bai et al., Lightweight Camera-LiDAR Fusion for Rail Transit | CACML 2024:303–308 | 与 v1 同构的工程实现 | — |
| E1 | Zhang et al., TTSDA-YOLO | TIM 74:1–13, 2024 | C3 微调可恢复的支撑 | — |
| E2 | Zhang et al., Multi-Weather Adaptation via Causal Intervention | TPAMI 46(3):1742–1756, 2024 | C3 更强的域适应方案 | — |

---

## 2. A 组：铁路侵限检测的现状与评测缺口

**A1. Railway Intrusion Detection Based on Machine Vision: A Survey, Challenges, and Perspectives**
Zhiwei Cao, Yong Qin, Limin Jia, Zhengyu Xie, Yang Gao, Yaguan Wang, Ping Li, Zujun Yu.
*IEEE T-ITS* 25(7):6427–6448, 2024. DOI: [10.1109/TITS.2024.3412170](https://doi.org/10.1109/TITS.2024.3412170)
- 做了什么：把机器视觉侵限检测分成**地面监控 / 车载巡检 / 无人机巡检**三条技术线，系统回顾近二十年工作，再归纳"阻碍落地的挑战"并给出展望。
- 支撑项目的哪句话：report 的"尚无公开数据与统一评测协议"、以及为什么本项目选择**车载前向**而不是地面监控——三条线里车载线对"提前量"最敏感。
- 不能支撑：它不会替你说"融合瓶颈是配准"。这是它的 general challenges，不是你的具体测量。

**A2. Advancements in Obstacle Intrusion Detection Methods for Rail Transit: A Comprehensive Review**
Xin-Yue Xu, Su-Mei Wang, Wenqiang Liu, Yi-Qing Ni.
*IEEE TIM* 74:1–34, 2025. DOI: [10.1109/TIM.2025.3612624](https://doi.org/10.1109/TIM.2025.3612624)
- 做了什么：从传感设备特性（相机、LiDAR…）出发，梳理**传统视觉 + AI** 两条 OID 路线，明确指出"与道路交通相比，铁路 OIDS 的研究投入相对不足"，且缺少对 SOTA 的系统总结。
- 用处：related work 的主骨架 + 选题必要性（"研究不足"是可引用的原话级结论）。
- 提示：它比 A1 新，且更贴"车载障碍物"而非"侵限"，两篇一起引可以覆盖"广"与"专"。

**A3. OSDaR23: Open Sensor Data for Rail 2023** ✅PDF
Rustam Tagiew, Pavel Klasek, Roman Tilly, et al.
*ICRAE* 2023:270–276. DOI: [10.1109/ICRAE59816.2023.10458449](https://doi.org/10.1109/ICRAE59816.2023.10458449)｜arXiv: [2305.03001](https://arxiv.org/abs/2305.03001)
- 做了什么：汉堡采集的多传感器数据集，45 个子序列；**多路已标定、已同步的 IR 与 RGB 相机** + LiDAR + Radar + 位置/加速度。
- 支撑：数据来源引用；更重要的是——"已标定、已同步"不等于"像素级对齐"，这正好解释为什么我们在 C1 里必须自己做几何投影，而不能指望数据集直接给配准好的 RGB-IR 对。
- 关键限制（写进 limitations）：标注是**铁路专用目标类别**，没有"侵限事件"真值 → 直接推出我们"事件定义缺失、C2 提前量指标无 ground truth"的未验证项。

**A4. A Survey on Multi-Sensor Fusion Perimeter Intrusion Detection in High-Speed Railways**
Tianyun Shi, Pengyue Guo, Rui Wang, et al.
*Sensors* 24(17):5463, 2024. DOI: [10.3390/s24175463](https://doi.org/10.3390/s24175463)
- 做了什么：高速铁路**周界**侵限的多传感器融合综述，核心论断是"单一传感器很难满足全场景、全天气、全时段的应用需求"，图像与点云优势互补。
- 支撑：C1 的"双模态互补"假设本身有文献基础（雾/霾/雨等恶劣天气下尤甚）。
- 注意：它是**周界（车站/线路边界）**而非车头前向，场景语义不同，引用时必须点明，否则容易被审稿人抓"类比不当"。

**A5. RailFOD23: A Dataset for Foreign Object Detection on Railroad Transmission Lines** ✅PDF
Zhichao Chen, Jie Yang, Zhicheng Feng, Hao Zhu.
*Scientific Data* 11:72, 2024. DOI: [10.1038/s41597-024-02918-9](https://doi.org/10.1038/s41597-024-02918-9)
- 做了什么：14,615 张图 / 40,541 个标注目标，**用 ChatGPT + 文生图合成**了一批异物数据补真实数据之不足；并评测主流检测模型。
- 支撑：项目大纲把它列为备选数据是对的——但要写清它是**接触网/输电线路异物**，不是轨道面侵限；它更适合作"异常稀有 + 合成补数据"的方法论引用（对应 A4 的异常缺样本问题）。
- 额外价值：它示范了"公开铁路数据少 → 用合成补齐"这一招，可以作为 C3 小样本微调之外的备选叙事。

**A6. Machine Vision-Based Intelligent Intrusion Detection Method for Obstacles on Open Railways in Low-Light Environments**
Heng Zhou, Fengkui Chen, Xinyao Dong, Jikang Sun, Qing Yang, Dexin Gao.
*Applied Sciences* 16(6):2848, 2026. DOI: [10.3390/app16062848](https://doi.org/10.3390/app16062848)
- 做了什么：改进 RT-DETR（SWC-DETR）：先 SCINet 做低照度增强，再把 WTConv 融进 RepC3、把 CloFormer 双分支塞进 AIFI，压背景噪声、强化小目标边缘；面向**开放线路**低光场景。
- 支撑：C1 的对照基线逻辑——低照度下"先把图修好再检测"是主流单模态路线，可用来论证"为什么还要引入红外"（增强是事后补偿，红外是原生另一物理通道）。
- 也提醒一件事：单纯提升可见光低光性能，可能就吃掉一部分融合的收益；如果 C1 最终要证明"融合优于单模态"，A6 这类单模态 SOTA 是必须打的靶子。

---

## 3. B 组：C1 卡点——跨模态错配与配准（最重要的一组）

> 结论先说：**这一组的共识是"错配是 RGB-T 检测的固有一等问题，像素级特征融合在错配下会失效，主流解法是区域级/对象级对齐或松耦合融合"**。
> 我们 C1 的负结果（fusion −0.0545 mAP50、相似变换残差 p90 579 px）落在这一共识区间里 —— 但注意：**文献能说明"这个现象合理"，不能说明"我们的数字对"**，数字只能自证。

**B1. Infrared and Visible Image Fusion: From Data Compatibility to Task Adaption** ✅PDF
Jinyuan Liu, Guanyao Wu, Zhu Liu, Di Wang, et al.
*IEEE TPAMI* 47(4):2349–2369, 2025（early access 2024）. DOI: [10.1109/TPAMI.2024.3521416](https://doi.org/10.1109/TPAMI.2024.3521416)｜arXiv: [2501.10761](https://arxiv.org/abs/2501.10761)
- 做了什么：综述，明确把 **data compatibility（数据兼容性）** 与感知精度、效率并列为不可忽视的三大议题，并强调从"视觉好看"转向"任务适配"。
- 为什么是 C1 的头号依据：它给了你一个**领域公认的措辞**——配准/兼容性是任务适配的前置条件。写 report 时"融合的瓶颈是配准，不是检测器"这句话，前面挂 B1 做立场背书，后面挂自己的残差统计做证据。这是最稳的写法。

**B2. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection (AR-CNN)** ✅PDF
Lu Zhang, Xiangyu Zhu, Xiangyu Chen, Xu Yang, Zhen Lei, Zhiyong Liu.
*ICCV* 2019:5126–5136. DOI: [10.1109/ICCV.2019.00523](https://doi.org/10.1109/ICCV.2019.00523)｜arXiv: [1901.02645](https://arxiv.org/abs/1901.02645)
- 做了什么：直指"真实多光谱数据存在位置偏移，同一目标在两模态里位置不同，导致特征图难以融合、CNN 训练受扰"；提出 AR-CNN，用 **Region Feature Alignment (RFA)** 捕捉位移并自适应对齐区域特征，再做特征重加权融合。
- 对我们的直接含义：**对齐的粒度应该是"目标/区域"，不是"整幅图像的一个全局变换"**。我们拟合的全局相似变换（p90 579 px 残差）正是被 B2 否定的那类做法；下一步方案①（用官方标定做真投影）与方案②（对象级关联）都能在这里找到依据。

**B3. Weakly Aligned Feature Fusion for Multimodal Object Detection** ✅PDF
Lu Zhang, Zhiyong Liu, Xiangyu Zhu, Zhanping Song, Xu Yang, Zhen Lei, Hong Qiao.
*IEEE TNNLS* 36(3):4145–4159, 2025（early access 2021）. DOI: [10.1109/TNNLS.2021.3105143](https://doi.org/10.1109/TNNLS.2021.3105143)｜arXiv: [2204.09848](https://arxiv.org/abs/2204.09848)
- 做了什么：B2 的期刊扩写版。RF 对齐模块加了**相邻相似性约束**来稳定位移预测，并提出 RoI jitter 策略提升鲁棒性。
- 用处：如果 C1 走"可学习对齐"路线，这篇是必引的方法源头；也说明"对齐模块需要正则/约束，否则位移预测会抖"——这正是我们靠单一全局矩阵做不到的。

**B4. MS-DETR: Multispectral Pedestrian Detection Transformer With Loosely Coupled Fusion and Modality-Balanced Optimization** ✅PDF
Yinghui Xing, Shuo Yang, Song Wang, et al.
*IEEE T-ITS* 25(12):20628–20642, 2024. DOI: [10.1109/TITS.2024.3450584](https://doi.org/10.1109/TITS.2024.3450584)｜arXiv: [2302.00290](https://arxiv.org/abs/2302.00290)
- 做了什么：把多光谱检测的两个核心问题点名为 **misalignment + modality imbalance**；双模态各自 backbone + encoder，在多模态 Transformer decoder 里做**松耦合融合**：从两模态特征里各自稀疏采样关键点，再用自适应注意力融合。
- 对我们最有价值的一句：**"不需要逐像素对齐，也能融合"**。这就是我们方案②（放弃像素级融合、改对象级关联）的文献依据；同时"modality imbalance"提醒我们：ir_only 在 RGB 平面评分只有 0.159，很可能是"评分平面错 + 模态不平衡"双重叠加，不能简单解读成"红外没用"。

**B5. Misaligned Visible-Thermal Object Detection: A Drone-Based Benchmark and Baseline (DVTOD / CMA-Det)**
Kechen Song, Xiaotong Xue, Hongwei Wen, Yingying Ji, Yunhui Yan, Qinggang Meng.
*IEEE T-IV* 9(11):7449–7460, 2024. DOI: [10.1109/TIV.2024.3398429](https://doi.org/10.1109/TIV.2024.3398429)
- 做了什么：指出**现有多光谱数据集都是人工对齐的，无法代表真实错配场景**；构建无人机采的错配 VT 数据集 DVTOD（16 种挑战属性 / 54 个采集场景），并提出 CMA-Det：先由对齐网络估计 visible→thermal 的位移。
- 为什么重要：它是"**把错配程度本身当作评测维度**"的先例。我们报告 p90 残差像素数、两相机横向 0.54 m 错位、近远端点不能同时对准——这正是 B5 主张的报告方式，不再是"训练崩了"这种描述。

**B6. Unsupervised Misaligned Infrared and Visible Image Fusion via Cross-Modality Image Generation and Registration** ✅PDF
Di Wang, Jinyuan Liu, Xin Fan, Risheng Liu.
*IJCAI* 2022:3508–3515. DOI: [10.24963/ijcai.2022/487](https://doi.org/10.24963/ijcai.2022/487)｜arXiv: [2205.11876](https://arxiv.org/abs/2205.11876)
- 做了什么：预配准数据上训练的融合方法在错配数据上会出现**鬼影**；提出 CPSTN 把可见光生成"伪红外"，从而把**跨模态配准问题转成单模态配准**（结构保持好，配准更稳）。
- 用处：当官方标定不可信/不可得时的替代路线；也可作为"为什么我们坚持先用标定做真投影"的对照（有更好解，但代价是要训练一个生成网络）。

**B7. A Deep Learning Framework for Infrared and Visible Image Fusion Without Strict Registration**
Huafeng Li, Junyu Liu, Yafei Zhang, Yu Liu.
*IJCV* 132(5):1625–1644, 2024. DOI: [10.1007/s11263-023-01948-x](https://doi.org/10.1007/s11263-023-01948-x)
- 做了什么：按标题与摘要可得信息，主张融合框架**不依赖严格配准**即可工作（OpenAlex 无摘要，未取到全文，引用时只按标题转述，不要展开细节）。
- 用处：支撑"配准不必追求像素级完美"这一折中立场——但要小心，它与 B1/B2 的"兼容性优先"并不矛盾：不要求严格配准 ≠ 错配可以任意大。

**B8. Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse IR and Visible for Object Detection (TarDAL / M3FD)** ✅PDF
Jinyuan Liu, Xin Fan, Zhanbo Huang, Guanyao Wu, Risheng Liu, Wei Zhong, Zhongxuan Luo.
*CVPR* 2022:5792–5801. DOI: [10.1109/CVPR52688.2022.00571](https://doi.org/10.1109/CVPR52688.2022.00571)｜arXiv: [2203.16220](https://arxiv.org/abs/2203.16220)
- 做了什么：指出"为视觉质量而融合"会丢掉两模态差异里的互补信息；用 bilevel 优化把融合与检测联合起来（TarDAL：单生成器 + 双判别器），并给出多场景多模态 benchmark M3FD。
- 用处：C1 的**评价哲学**来源——融合要用下游检测指标衡量，不是看图像好看。这也是我们"三组对照 + 夜间/逆光子集"指标设计的依据之一。
- 注意：M3FD/多光谱行人这类数据集是**手持/车载平行光轴近似共视**采集，与铁路车头 IR/RGB 的 0.54 m 横向错位 + 视差不在一个量级，跨到我们场景要明说差异。

---

## 4. C 组：C2 卡点——时序风险、提前量与距离

**C1. A Survey on Motion Prediction and Risk Assessment for Intelligent Vehicles** ✅PDF
Stéphanie Lefèvre, Dizan Vasquez, Christian Laugier.
*ROBOMECH Journal* 1:1, 2014. DOI: [10.1186/s40648-014-0001-z](https://doi.org/10.1186/s40648-014-0001-z)
- 做了什么：按"运动模型语义"和"风险度量语义"对智能车风险估计方法分类；明确指出**模型完整性与实时约束之间的权衡**，且"风险度量方法的选择受运动模型选择影响"。
- 支撑：TTC 不是唯一风险度量，它等价于"恒定速度 + 沿视线接近"的强假设。我们用 INS 测速 + 点云前向距离算 TTC，等价于把"沿轨向接近"这个假设摆到台面上——引用 C1 说明这一点，比硬套 TTC 更站得住。
- 也顺便解释了我们 C2 只有 3 次报警（0.92 s）：在静止障碍 + TTC≈2.5 s 的情形下，TTC 度量本身对"检测抖动"极其敏感，这不是实现 bug。

**C2. Robust Range Estimation with a Monocular Camera for Vision-Based Forward Collision Warning System**
Ki-Yeong Park, Sun-Young Hwang.
*The Scientific World Journal* 2014:923632, 2014. DOI: [10.1155/2014/923632](https://doi.org/10.1155/2014/923632)
- 做了什么：单目前撞预警的测距；针对车身俯仰/路面坡度导致相机 pitch 漂移的问题，用画面中车辆尺寸与位置**在线估计虚拟地平线**，不依赖车道线。
- 用处：我们 v0 的"透视反比测距"（d = f·H / (v − v_h)）本质就是这个模型；这篇给了"地平线必须在线估计"的工程修正。项目现在靠点云+INS 绕开了这个问题，但写"方法演进"时可以引它说明我们为什么不走纯单目。
- 诚实边界：它没有"提前量(s)"指标，也没有事件级误报统计。

**C3. Near-Incident Detection in Railroad Environments: Lateral Distance Estimation from Train-Mounted Monocular Camera**
Yilei Wang, Giacomo D'Ambicantonio, Egor Bondarev.
*CVPRW* 2025:2027–2036. DOI: [10.1109/CVPRW67362.2025.00191](https://doi.org/10.1109/CVPRW67362.2025.00191)
- 做了什么：NMD 框架——车头单目，检测 + 实例分割 + 深度估计，构建 3D 场景视图；用**列车与目标的横向距离**衡量事故风险；并提出基于相机标定的深度校准机制以适配实车。
- **跟 C2 最贴的一篇**：同场景（铁路）、同传感器（车头单目）、同目标（提前发现危险事件而非只做检测）。差异点要写清：它用横向距离 + 单目深度，我们用前向 TTC + LiDAR 深度 + INS 测速；它没有事件级误报率/提前量统计，我们有（虽然样本还少）。这正是一个"可对标、但不重叠"的引用位置。
- 风险提示：CVPRW 的 workshop 论文，方法深度有限，作为"相关工作的最近邻"引用合适，别当 baseline 结论引用。

**C4. Dist-YOLO: Fast Object Detection with Distance Estimation**
Marek Vajgl, Petr Hurtík, Tomáš Nejezchleba.
*Applied Sciences* 12(3):1354, 2022. DOI: [10.3390/app12031354](https://doi.org/10.3390/app12031354)
- 做了什么：把绝对距离回归塞进 YOLO 预测向量，与 bbox 回归共享 backbone，改损失；class-agnostic 比 class-aware 更省更好；KITTI 上 8 类全距离段平均相对误差 11%。
- 用处：如果相机—点云外参标定不稳（我们现在正卡在这），端到端回归距离是一条退路；也给了"11%"这个量级的参照，说明单目测距的误差天花板大概在哪。
- 注意：KITTI 是道路近距场景，铁路远距 + 小目标下这个 11% 不可直接迁移。

---

## 5. D 组："目标是否在运行轨内"与轨道区域（v1 未验证项）

**D1. An Adaptive Track Segmentation Algorithm for a Railway Intrusion Detection System**
Yang Wang, Liqiang Zhu, Zujun Yu, Baoqing Guo.
*Sensors* 19(11):2594, 2019. DOI: [10.3390/s19112594](https://doi.org/10.3390/s19112594)
- 做了什么：为巡检视频自动划出"轨道报警区域"的轻量分割算法（图像先分成碎块再合并边界），专治"摄像头太多、人工画报警区不现实"，包括会动的 PTZ 相机。
- 用处：我们 v1 用"画面梯形 ROI"近似轨内区域，是最土的做法；D1 是最便宜的正规替代（无需训练、算力极小），刚好补上"是否在轨内未验证"这一条。
- 边界：它是**地面监控视角**（高处俯视），车头前向视角的轨道区域收敛形状不同，迁移时要重新定参。

**D2. Automated Three-Dimensional Linear Elements Extraction from Mobile LiDAR Point Clouds in Railway Environments**
Luis Gézero, Carlos Antunes.
*Infrastructures* 4(3):46, 2019. DOI: [10.3390/infrastructures4030046](https://doi.org/10.3390/infrastructures4030046)
- 做了什么：用点云自带的**扫描角属性**先把候选点限制住（因为线性要素大致平行于轨道与采集轨迹），再套简单几何算法提取 3D 线性要素。
- 直接对应我们的坑：方案 D 用"|z|<0.15 m 近平面 + 1.435 m 轨距分离"去拟合钢轨，在柏林站场锁到邻轨、走廊偏右 3 m。D2 的思路是**先用采集几何约束候选点**，而不是只靠高度/间距阈值——这正是我们该补的一步。
- 注意：它是移动测量车（MLS，高位多线）点云，我们是车头低位单雷达，遮挡与视角完全不同。

**D3. Rail-PillarNet: A 3D Detection Network for Railway Foreign Object Based on LiDAR** ✅PDF
Fan Li, Shuyao Zhang, Jie Yang, Zhicheng Feng, Zhichao Chen.
*Computers, Materials & Continua* 80(3):3819–3833, 2024. DOI: [10.32604/cmc.2024.054525](https://doi.org/10.32604/cmc.2024.054525)
- 做了什么：针对 2D 图像法的三个短板（探测距离短、受环境影响大、缺距离信息），提出基于 LiDAR 的铁路异物 3D 检测网络。
- 用处：把"在轨内"从 2D ROI 变成**3D 空间判定**的现成论据；也支持"相机负责语义、点云负责几何"的分工。
- 边界：它是点云单模态，没有 RGB-T 问题；引用时定位成"另一条技术线的代表"。

**D4. Unsupervised Deep Learning-Based Point Cloud Detection for Railway Foreign Object Intrusion**
Haifeng Song, Xiying Song, Min Zhou, Ling Liu, Hairong Dong.
*ICIEA* 2024:1–6. DOI: [10.1109/ICIEA61579.2024.10664865](https://doi.org/10.1109/ICIEA61579.2024.10664865)
- 做了什么：点云自监督/无监督框架——掩码点预测 + 异常检测，绕开点云逐点标注；因为真实数据里没有异常样本，还设计了**伪异常样本生成**算法造测试集。
- 为什么对我们特别有用：我们同样"没有侵限事件真值"。D4 的伪异常生成思路，可以直接搬到 C2：**用合成逼近片段人为制造事件**，让"提前量"指标有定义——这和我们 c2 报告里"合成逼近片段"的下一步建议是同一招，引它就有了先例。

**D5. LiDAR-Camera Joint Obstacle Detection Algorithm for Railway Track Area**
Zongliang Nan, Wenlong Liu, Guoan Zhu, Hongwei Zhao, Wentao Xia, Xuechun Lin, Yingying Yang.
*Expert Systems with Applications* 275:127089, 2025. DOI: [10.1016/j.eswa.2025.127089](https://doi.org/10.1016/j.eswa.2025.127089)
- 做了什么：按标题与期刊定位，做**轨道区域内的 LiDAR-相机联合障碍检测**（OpenAlex 与出版社均未给开放摘要，只按标题转述，勿展开方法细节）。
- 用处：与我们 v1 架构几乎同构（相机检测 + 点云测距 + 轨道区域约束），是**必须引用的同类工作**，用来划清"我们与它的差异"（我们多红外模态、多事件级时序与分级预警）。
- 建议：写之前先想办法拿到它的摘要/正文（学校图书馆或邮件索取），否则别在 report 里对它做技术性对比。

**D6. A Lightweight Camera and Lidar Fusion Framework for Railway Transit Obstacle Detection**
Ri Bai, Zhenxin Wu, Tao Xu.
*CACML* 2024:303–308. DOI: [10.1145/3654823.3654879](https://doi.org/10.1145/3654823.3654879)
- 做了什么：改进 YOLOv5 提 2D 位置（双特征融合通道 + 两阶段串行融合），再用 LiDAR 与图像坐标变换把障碍物点云提出来，得到距离信息，服务紧急避障。
- 用处：**与我们 v1 的管线流程几乎一致**（2D 检测 → 坐标变换 → 点云取距），是证明"这条工程路线成立"的引用；同时也说明这套做法在 2024 年已属常规，我们的增量必须落在红外融合与事件级预警上，不能停在"相机+LiDAR 融合"。

---

## 6. E 组：C3 跨域退化

**E1. TTSDA-YOLO: A Two Training Stage Domain Adaptation Framework for Object Detection in Adverse Weather**
Mengmeng Zhang, Qiyu Rong, Hongyuan Jing.
*IEEE TIM* 74:1–13, 2024. DOI: [10.1109/TIM.2024.3497132](https://doi.org/10.1109/TIM.2024.3497132)
- 做了什么：基于 YOLOv7 的两阶段训练，利用**辅助域**把源域知识迁到目标域，兼顾正常与恶劣天气。
- 支撑 C3：公开集→自采小样本的退化与恢复，本质就是域偏移 + 少量目标域样本适应；"两阶段、先辅助域再目标域"是可参考的训练配方，也是"微调可恢复"的文献级支持。

**E2. Multiple Adverse Weather Conditions Adaptation for Object Detection via Causal Intervention**
Hua Zhang, Liqiang Xiao, Xiaochun Cao, Hassan Foroosh.
*IEEE TPAMI* 46(3):1742–1756, 2024. DOI: [10.1109/TPAMI.2022.3166765](https://doi.org/10.1109/TPAMI.2022.3166765)
- 做了什么：把恶劣天气看作 confounder，用记忆网络维护"混淆字典"（各场景目标特征原型）+ 动态原型提取，学**天气不变特征**，做多天气联合适应。
- 用处：C3 若想走出"只报衰减/恢复幅度"，往"为什么会退化"走一步（是天气混淆还是传感器差异），这篇是因果视角的入口。
- 注意：需要目标域数据较多，与我们"自采小样本"的约束不匹配，引用定位是"更强的方案/未来工作"，不是当下可实施路线。

---

## 7. 哪些话现在有文献撑、哪些还不能说

| 项目里的说法 | 文献支撑 | 能不能这么写 |
|---|---|---|
| 「公开铁路侵限数据少、缺统一评测」 | A1, A2, A5 | ✅ 可写，A2 有"研究不足"的明确表述 |
| 「融合的瓶颈是配准，不是检测器」 | B1, B2, B3, B4, B5 | ⚠️ 前半句（错配是固有一等问题、像素级融合会失效）可引文献；**后半句是我们自己的实验结论**，只能由 p90 579 px、横向 0.54 m 错位这些自测数据支撑，不能写成"文献指出我们的融合会失败" |
| 「指标要用夜间/逆光子集漏检率，不用总 mAP」 | B8（以检测为目标评价融合）、B5（按错配/挑战属性分层）、A3（数据有光照条件信息） | ✅ 可写；B5 还给了"按错配程度分层报告"的做法 |
| 「时序 + 提前量比逐帧置信度阈值有意义」 | C1, C3 | ✅ 可写（时间量纲的风险度量有分类学与先例）；但没有同行数字可比 |
| 「跨域退化可量化、微调可恢复」 | E1, E2 | ✅ 可写；具体衰减/恢复幅度必须自测 |
| 「目标是否在轨内」应当用几何判定 | D1, D3, D5 | ✅ 可写；我们的梯形 ROI 只是临时近似 |
| 「IR 单独就能打」/「IR 没用」 | B4（modality imbalance） | ❌ 都不能。ir_only 0.1593 是在 RGB 平面评分得到的，评分平面本身错了，这个数字不能用来下任何关于红外的结论；只有"IR 在自身像平面 640 下 mAP50 0.7674"才是有效观察 |
| 「我们的 0.7027 mAP50 达到了 SOTA」 | — | ❌ 无同协议可比对象。A1/A2 都指出缺统一 benchmark，这既是机会也是限制 |

---

## 8. 建议的引用位置（对应 technical report 章节）

- **§1 引言 / 研究背景**：A1（三条技术线、挑战）、A2（铁路投入不足）、A4（单一传感器不够）
- **§2 相关工作**：分四段写——铁路侵限与障碍检测（A1/A2/D3/D5/D6）、多模态融合与错配（B1–B8）、时序风险与预警（C1–C4）、铁路数据集与评测（A3/A5）
- **§4 C1 实验**：
  - 方法动机段引 B2/B4（区域级对齐、松耦合）；
  - 负结果解释段引 B1/B5（兼容性前置、错配必须量化）；
  - 「下一步用官方标定做真投影」引 A3（数据自带标定）+ B6（无监督配准是替代路线）；
  - 「改对象级关联」引 B4（稀疏关键点、松耦合）与 B8（融合以检测为目标）。
- **§5 C2 实验**：C1（风险度量分类）、C3（铁路近失事件最近的同类）、D4（伪异常合成补事件真值）
- **§6 C3 实验**：E1（两阶段域适应）、E2（因果视角/未来工作）
- **§7 局限**：A3（无事件真值）、B5（我们没做错配敏感性扫描）、D5（同类工作未取到全文，未做定量对比）

---

## 9. 阅读顺序与预算

按"先补最卡的，再补背景"排，每篇 30–40 分钟（正文为主，实验表格可跳）：

1. **B2 (ICCV'19)** — 30 min，理解"区域级对齐"到底怎么做的。这是修 C1 的钥匙。
2. **B4 (MS-DETR)** — 30 min，看松耦合融合怎么绕开逐像素对齐。
3. **C3 (CVPRW'25)** — 20 min，看同场景的人怎么把"提前量"落到距离上。
4. **A3 (OSDaR23)** — 20 min（只读数据描述 + 标注类别），确认没有事件真值。
5. **B1 (TPAMI 综述)** — 40 min，只读 data compatibility 与 task adaption 两节。
6. **D1 + D4** — 各 20 min，一个补"轨道区域"，一个补"无真值怎么造事件"。
7. **A1 + A2** — 各 30 min，写 related work 时再读，跳过大部分方法细节。
8. 其余（B3/B5/B6/B7/B8/C1/C2/C4/D2/D3/D5/D6/E1/E2/A4/A5/A6）按需查阅，写哪段读哪篇。

---

## 10. 全文下载与许可

已下载到 `refs/`（10 篇，均为开放获取版本）：

| 文件 | 来源 | 许可/说明 |
|---|---|---|
| `2023_Tagiew_OSDaR23-dataset_ICRAE.pdf` | arXiv 2305.03001 | arXiv 非独占许可，作者版 |
| `2024_Liu_IR-Vis-fusion-data-compatibility-to-task-adaption_TPAMI.pdf` | arXiv 2501.10761 | 同上（TPAMI early access 版） |
| `2019_Zhang_Weakly-aligned-cross-modal-learning_ICCV.pdf` | arXiv 1901.02645 | 同上 |
| `2021_Zhang_Weakly-aligned-feature-fusion_TNNLS.pdf` | arXiv 2204.09848 | 同上 |
| `2024_MS-DETR-loosely-coupled-fusion_TITS.pdf` | arXiv 2302.00290 | 同上 |
| `2022_Wang_Unsupervised-misaligned-IR-Vis-fusion-registration_IJCAI.pdf` | ijcai.org 会议集 | IJCAI 开放获取 |
| `2022_Liu_TarDAL-target-aware-dual-adversarial_CVPR.pdf` | arXiv 2203.16220 | 作者版 |
| `2024_RailFOD23-dataset_ScientificData.pdf` | nature.com | CC BY 4.0 |
| `2024_Rail-PillarNet-LiDAR-3D-FOD_CMC.pdf` | Tech Science Press | 开放获取 |
| `2014_Motion-prediction-risk-assessment-survey_ROBOMECH.pdf` | SpringerOpen | CC BY |

**未下载（需浏览器打开）**：MDPI（A4/A6/C4/D1/D2 —— `www.mdpi.com/...` 对脚本访问一律 403）、ACM DL（D6）、以及全部 IEEE 订阅刊（A1/A2/B5/C3/E1/E2/D3 的部分卷期、B7 的 IJCV）。

另有 `docs/refs.bib`，含 26 篇的 BibTeX，`author/year/journal/volume/pages/doi` 均已核对，可直接 `\cite{}`。

> 版权提醒：`refs/` 里的 PDF 仅供个人研究阅读，不要再分发或上传到公开仓库。
