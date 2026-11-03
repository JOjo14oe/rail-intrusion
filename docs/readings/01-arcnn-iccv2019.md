# 01 · AR-CNN：弱对齐跨模态学习（ICCV 2019）

> 读于 2026-10-26。原文：`refs/2019_Zhang_Weakly-aligned-cross-modal-learning_ICCV.pdf`
> 纯文本层：`refs/extracted-text/2019_Zhang_Weakly-aligned-cross-modal-learning_ICCV.txt`
> 引用键：`\cite{zhang2019arccnn}`（见 `docs/refs.bib`）
> 为什么第一篇读它：**它把 C1 的失败解释清楚了** —— 我们拟合的那个全局相似变换，正是这篇论文在 2019 年就明确否定掉的做法。

---

## 一、一句话结论

多光谱融合的错配**不是仿射变换能吸收的**，它随像素位置变化（中心小、边缘大），因此对齐必须**按区域（目标级）预测位移**，而不是求一个全局矩阵。作者用"人工模拟位移"实验量化了代价：**6 像素的错位就能让漏检率恶化 65%**。

对我们：C1 的负结果不是检测器不行，也不是红外没用，而是**对齐粒度选错了**。这条结论现在有原文依据，可以写进 report 的 §4。

---

## 二、原文关键段落（逐句核对过 PDF 文本层，页码为 PDF 页序）

### Q1 摘要（p.1）——问题定义

> "real multispectral data suffers from **the position shift problem**, i.e. the color-thermal image pairs are **not strictly aligned**, making one object has different positions in different modalities. In deep learning based methods, this problem makes it difficult to fuse the feature maps from both modalities and **puzzles the CNN training**."

译文：真实多光谱数据存在"位置偏移"问题，即彩色—热红外图像对并非严格对齐，同一目标在两模态中位置不同；在深度学习方法中，这让两模态特征图难以融合，并扰乱 CNN 训练。

**对我们**：这就是我们 C1 的病灶。注意它的措辞是"not strictly aligned"（弱对齐），而不是"没对齐"——意味着**不需要完美配准，但需要处理偏移**。

### Q2 成因清单（p.1，§1 Challenges）——为什么我们会错得比 KAIST 离谱

> "This position shift problem can be caused by **physical properties of different sensors (e.g. parallax, mismatched resolutions and field-of-views)**, **imperfection of alignment algorithms**, external disturbance, and hardware aging. Moreover, the calibration for color-thermal cameras are **tortuous**, generally require particular hardware as well as special heated calibration board."

**对我们逐条对号**：
- parallax（视差）→ 我们两相机横向错开 **0.54 m**（IR Z≈0.52 m / y=+0.2527，RGB Z≈0.50 m / y=−0.2859）
- mismatched resolutions → **2464 vs 640** 宽
- mismatched field-of-views → cy **239.5 vs 820.05**
- imperfection of alignment algorithms → 我们拟合的相似变换，残差 p90 **579.4 px**

四个成因里我们中了三个半。**这就是"为什么我们的错配比 KAIST 一个量级"的文献级解释**，写在方法学讨论里很稳。

### Q3 偏移的两条伤害路径（p.1–2）

> "The position shift problem degrades the detector in two aspects. First, features from different modalities are **mismatched in the corresponding positions**, which puzzles the inference. Second, it is **difficult to cover the objects in both modalities with a single bounding box**."

**对我们**：第二条常被忽略，但它解释了我们 ir_only 评分那张表的荒谬——我们把 IR 的检出框映射到 RGB 平面去算 AP，而一个目标在两平面上的框根本对不上（中位 18.8 px 还好，p90 579 px 就完全错位），AP 自然崩到 0.1593。**这个数字因此不是"红外弱"的证据，而是"用一个错位的框去算 AP"的证据。**

### Q4 最重要的一句（p.4，§4.2）——为什么全局变换注定失败

> "Note that the position shift is **not simply affine transformation** and depends on the cameras. Furthermore, **the shift distance varies from pixels to pixels, always small in the center and large in the edge**. As a result, the shift prediction and alignment process is **performed in a region-wise way**."

三个分句，句句打在我们的方案上：
1. 不是仿射变换 → 我们用的相似变换（旋转+尺度+平移）是仿射的子集，越界了；
2. 位移**逐像素变化，中心小、边缘大** → 我们的残差中位数只有 18.8 px，但 p90 飙到 579 px，正是"端点/边缘剧烈放大"的典型形态；
3. 因此对齐要**按区域做** → 这就是"对象级关联"的原文出处。

> **写作提示**：report 里"融合瓶颈是配准"这句话，前面挂 `\cite{liu2025ivif}`（兼容性是任务适配前置），后面挂这句 + 我们自己的残差统计。三者层次分明，不会被质疑"你自己的结论凭什么"。

### Q5 量化代价（p.3，§3.2）——6 像素 = 65% 恶化

> "the performance **dramatically drops** as the absolute shift values increase. Especially, the worst case (∆x, ∆y) = (−6, 6) suffers **∼65.3% relative performance decrement**, i.e. from **15.18 MR to 25.10 MR**."

实验设置：固定热红外图，把彩色图在 x/y 各 ±6 px 内平移，共 169 种偏移；基线是改造过的 Faster R-CNN + halfway fusion（KAIST reasonable 集 15.18 MR）。

还有一个反直觉的细节值得记住：

> "when the image is shifted to a specific direction (∆x = 1, ∆y = −1), a **better** result is achieved (15.18 MR to 14.68 MR), which indicates that we can **improve the performance by appropriately handling the position shift problem**."

即：偏移不只有害，**对齐本身就能白赚约 0.5 MR**。

### Q6 成对标注的稀缺性（p.3，§3.1）

> "Since the image pair of two modalities may have different field-of-views due to bad camera synchronization and calibration, some pedestrians exist in one modality but are **truncated/lost in another**... Specifically, **12.5% (2,245 of 18,017)** of bounding boxes are **unpaired** in the CVC-14 dataset."

**对我们**：OSDaR23 的标注是 3D 目标投影到各相机，天然"每台相机一套框"。这个特性在 C1 里是负担（框对不上），但换个角度——**它正好是 RFA 训练所需的"逐模态框对"**（见第五节建议 1）。

### Q7 结论（p.8）

> "considering this weakly aligned characteristic is **widespread and hard to completely avoid when multimodal inputs are required**."

**对我们**：这句话可以用来预先回应"你是不是应该先把相机标定做好"的质询——作者的态度是：多模态输入的错配无法彻底消除，所以要在算法层面处理它。

---

## 三、方法骨架（三件套，各管一件事）

1. **RFA（Region Feature Alignment）**——管"对齐"
   对每个候选框，把两模态的上下文区域特征池化成 7×7，拼接后用两层 FC 预测**该区域的**位移目标 `(t_x, t_y)`，再按新坐标对 sensed 模态重新池化。位移的监督信号就是逐模态框对的中心差：
   `t*_x = (x_s − x_r) / w_r`，`t*_y = (y_s − y_r) / h_r`（Eq.1），用 smooth L1 回归。
   关键点：`(x_s − x_r)/w_r` **用框宽归一化** → 远目标（小框）与近目标（大框）的位移被拉到同一尺度。这正好治我们"误差随距离放大"的病。

2. **RoI jitter**——管"泛化到未见过的位移模式"
   训练时对 sensed 侧 RoI 加 `N(0, σ²)` 扰动并同步修正位移目标，人为扩充偏移分布。

3. **Confidence-aware fusion**——管"融合时别信错模态"
   对两模态特征做重加权，挑更可靠的特征、压低干扰特征（对应 MS-DETR 后来讲的 modality imbalance）。

---

## 四、实验结果（原文数字）

**主结果（Table 2, KAIST）**，指标为 log-average miss rate（MR，越低越好）：

| 方法 | Reasonable Day | Night | All |
|---|---|---|---|
| Adapted Halfway Fusion（基线） | 15.36 | 14.99 | 15.18 |
| MSDS-RCNN | 10.60 | 13.73 | 11.63 |
| **AR-CNN** | **9.94** | **8.38** | **9.34** |

**配对标注下的分模态指标**：MR_C **9.86** vs 基线 11.28；MR_T **8.26** vs 基线 12.51。

**CVC-14（Table 3）**：全时 22.1，**夜间 18.1 vs 次优 30.8** —— 夜间收益最大，这正是红外模态该起作用的场景。

**鲁棒性（Table 4，四个方向各 21 种偏移的均值/标准差）**：基线 Adapted Halfway 在 S45° 上 σ=11.55；仅加 RFA 后 σ=**2.81**，作者报"标准差降低 8.53（9.77→1.24）"（该处指 RFA+RoIJ 的组合行）。也就是说，**对齐模块主要买到的是"稳定性"，不只是均值**。

---

## 五、和我们的数字对照（写 report 用）

| 项目 | AR-CNN / KAIST | rail-intrusion / OSDaR23 | 折算 |
|---|---|---|---|
| 图像宽度 | 640 | RGB 2464 | — |
| 实测/典型偏移 | 多数框位移 **0–10 px**（≈0–1.6% 宽） | 中位 **18.8 px**（0.76% 宽） | 中位数**同一量级**，甚至更好 |
| 尾部偏移 | 模拟最坏 **±6 px**（0.94% 宽） | p90 **579.4 px**（23.5% 宽）≈ KAIST 尺度下 **150 px** | 尾部是它的 **25 倍** |
| 偏移代价 | 6 px → MR 15.18→25.10（**+65.3%**） | fusion vs rgb mAP50 0.6457 vs 0.7002（**−7.8%**） | 我们反而"没那么惨" |
| 对齐方式 | 逐区域学出来的位移 | 全局相似变换 1 个矩阵 8 个自由度 | — |

**这张表里最重要的一行是最后一行**：我们的偏移尾部大了 25 倍，性能却只掉了 7.8%，说明**融合分支很可能没真正起作用**（网络学会了忽略红外）。

→ 这给了一个 30 分钟就能做完的验证实验，见下面建议 2。

---

## 六、对 C1 的三条具体建议（按性价比排序）

**建议 1（最关键，也最省人工）：用 OSDaR23 的 3D 标注自动生成"逐模态框对"，喂 RFA 式的区域对齐。**
原作者为了拿到框对，**人工重标了 KAIST 的 20,025 帧 / 59,812 个行人**（p.2）。我们不需要：OSDaR23 的每个目标在 3D 里有唯一身份，投影到 IR 与 RGB 各得一个框 → 框对**自动生成**，且带真实视差。
代价：依赖投影精度；但投影误差本身可测（用同一目标两模态框的差做统计，就是我们已经在算的残差分布）。
产出：一个 `pairs.json`（目标 id → ir_bbox, rgb_bbox, 距离），既是对齐模块的监督，也是"错配程度"的评测集。

**建议 2（30 分钟，先把结论钉死）：做"IR 通道消融"，证明融合分支确实被网络忽略了。**
训练好的 fusion 模型上，把 IR 输入替换为 (a) 全零、(b) 高斯噪声、(c) 跨帧错位样本，看 mAP50 掉多少。
- 若几乎不掉 → 融合是"名义上的"，C1 负结果的原因就完整了（对齐错 → 学不到可用互补信息 → 网络自动关掉该分支）；
- 若掉得多 → 说明融合有效但被错配拖累，那么修对齐的收益上限值得投入。
这个实验能把"融合无效"从主观判断变成可复现证据，成本极低。

**建议 3（方法层面）：如果走可学习对齐，必须抄它的"归一化 + 扰动"两个细节。**
位移用框宽/高归一化（`/w_r, /h_r`），否则远小目标的位移被近大目标淹没；训练时加 RoI jitter，否则实车换镜头/调焦后偏移分布一变就崩。这两点在我们的场景里比在 KAIST 更致命——因为我们的偏移本来就是"随距离放大"的。

---

## 七、可直接用的引用句（英文，report 用）

- 定义错配：`Following [\cite{zhang2019arccnn}], the RGB–IR pair is only weakly aligned: one object occupies different positions in the two modalities, which prevents naive feature-map fusion and confuses detector training.`
- 否定全局变换：`As noted in [\cite{zhang2019arccnn}], the shift is "not simply affine transformation" and "varies from pixels to pixels, always small in the center and large in the edge"; hence a single global homography cannot align the pair, and alignment must be region-wise.`
- 量化代价的锚点：`Even a 6-pixel misalignment was reported to degrade the miss rate by 65.3% on KAIST [\cite{zhang2019arccnn}], whereas our p90 residual reaches 579 px on a 2464-px-wide RGB image (150 px in KAIST scale), i.e. 25× their worst simulated case.`

⚠️ 第三句里"150 px 折算"是**我们的换算**，不是原文结论，写作时要标明 `(scaled to KAIST resolution for comparison)`，别让读者误以为是论文数据。

---

## 八、读后自测（答不上来就回去翻原文）

1. RFA 的位移监督为什么必须用框宽归一化？不用会怎样？
2. 为什么"对齐做好"能让性能比原始（无偏移）更好？（提示：Q5 里 14.68 < 15.18）
3. RoI jitter 提升的是均值还是标准差？作者自己怎么说？
4. 这篇为什么必须重新标注 KAIST？我们的数据集为什么不需要？

---

## 九、与下一篇的衔接

下一篇读 **MS-DETR（B4）**：它给出另一条路——不学位移，而是**干脆不做像素级融合**，在 Transformer 解码器里稀疏采关键点后融合。两篇合起来，正好对应我们 C1 的两个候选方向（对象级对齐 vs 松耦合融合）。
