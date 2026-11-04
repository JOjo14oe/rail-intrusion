# refs/ — 参考文献原文（开放获取版）

配套清单与逐篇解读见 [`../docs/literature-review.md`](../docs/literature-review.md)，BibTeX 见 [`../docs/refs.bib`](../docs/refs.bib)。
下载日期：2026-10-26。以下 10 篇均为**开放获取**版本（arXiv 作者版 / 会议或期刊 OA 版），仅供个人研究阅读，不要二次分发。

| 文件 | 大小 | 来源链接 | 对应条目 | 版权 |
|---|---|---|---|---|
| `2023_Tagiew_OSDaR23-dataset_ICRAE.pdf` | 5.3 MB | [arXiv:2305.03001](https://arxiv.org/abs/2305.03001) | A3 | arXiv 作者版 |
| `2024_Liu_IR-Vis-fusion-data-compatibility-to-task-adaption_TPAMI.pdf` | 10.4 MB | [arXiv:2501.10761](https://arxiv.org/abs/2501.10761) | B1 | arXiv 作者版（TPAMI early access） |
| `2019_Zhang_Weakly-aligned-cross-modal-learning_ICCV.pdf` | 3.5 MB | [arXiv:1901.02645](https://arxiv.org/abs/1901.02645) | B2 | arXiv 作者版 |
| `2021_Zhang_Weakly-aligned-feature-fusion_TNNLS.pdf` | 19.1 MB | [arXiv:2204.09848](https://arxiv.org/abs/2204.09848) | B3 | arXiv 作者版（TNNLS 期刊版） |
| `2024_MS-DETR-loosely-coupled-fusion_TITS.pdf` | 17.6 MB | [arXiv:2302.00290](https://arxiv.org/abs/2302.00290) | B4 | arXiv 作者版 |
| `2022_Wang_Unsupervised-misaligned-IR-Vis-fusion-registration_IJCAI.pdf` | 3.4 MB | [ijcai.org/…/0487.pdf](https://www.ijcai.org/proceedings/2022/0487.pdf) | B6 | IJCAI 开放获取 |
| `2022_Liu_TarDAL-target-aware-dual-adversarial_CVPR.pdf` | 14.0 MB | [arXiv:2203.16220](https://arxiv.org/abs/2203.16220) | B8 | arXiv 作者版 |
| `2024_RailFOD23-dataset_ScientificData.pdf` | 4.6 MB | [nature.com](https://www.nature.com/articles/s41597-024-02918-9) | A5 | CC BY 4.0 |
| `2024_Rail-PillarNet-LiDAR-3D-FOD_CMC.pdf` | 2.0 MB | [techscience.press](https://cdn.techscience.press/files/cmc/2024/TSP_CMC-80-3/TSP_CMC_54525/TSP_CMC_54525.pdf) | D3 | 期刊 OA |
| `2014_Motion-prediction-risk-assessment-survey_ROBOMECH.pdf` | 2.3 MB | [springeropen](https://robomechjournal.springeropen.com/track/pdf/10.1186/s40648-014-0001-z) | C1 | CC BY |

合计约 82 MB。

## 未下载的 16 篇（需浏览器 / 机构订阅打开）

脚本访问被站点拦截（MDPI 与 ACM DL 对本机返回 403），或属于 IEEE / Springer 订阅刊。直接用 DOI 打开即可：

| 条目 | DOI |
|---|---|
| A1 Cao et al., T-ITS 2024 综述 | https://doi.org/10.1109/TITS.2024.3412170 |
| A2 Xu et al., TIM 2025 综述 | https://doi.org/10.1109/TIM.2025.3612624 |
| A4 Shi et al., Sensors 2024 周界融合综述 | https://doi.org/10.3390/s24175463 |
| A6 Zhou et al., Appl Sci 2026 低照度 | https://doi.org/10.3390/app16062848 |
| B5 Song et al., T-IV 2024 错配 benchmark | https://doi.org/10.1109/TIV.2024.3398429 |
| B7 Li et al., IJCV 2024 免严格配准 | https://doi.org/10.1007/s11263-023-01948-x |
| C2 Park & Hwang, Sci World J 2014 | https://doi.org/10.1155/2014/923632 |
| C3 Wang et al., CVPRW 2025 近失事件 | https://doi.org/10.1109/CVPRW67362.2025.00191 |
| C4 Vajgl et al., Appl Sci 2022 Dist-YOLO | https://doi.org/10.3390/app12031354 |
| D1 Wang et al., Sensors 2019 轨道区域 | https://doi.org/10.3390/s19112594 |
| D2 Gézero & Antunes, Infrastructures 2019 | https://doi.org/10.3390/infrastructures4030046 |
| D4 Song et al., ICIEA 2024 无监督点云 | https://doi.org/10.1109/ICIEA61579.2024.10664865 |
| D5 Nan et al., ESWA 2025 激光-相机联合 | https://doi.org/10.1016/j.eswa.2025.127089 |
| D6 Bai et al., CACML 2024 | https://doi.org/10.1145/3654823.3654879 |
| E1 Zhang et al., TIM 2024 TTSDA-YOLO | https://doi.org/10.1109/TIM.2024.3497132 |
| E2 Zhang et al., TPAMI 2024 因果干预 | https://doi.org/10.1109/TPAMI.2022.3166765 |

> 想补齐 MDPI 那 5 篇：用浏览器打开 DOI 页面，右上角 "Download PDF" 即可；本机之所以失败，是 MDPI 对非浏览器 UA 一律 403，不是链接失效（已用 HEAD 复核过页面可访问性）。

## 怎么跑（检索 → 定位 → 下载 → 摘要 → 作者 → 抽正文）

这是一个**按需跑一次的流水线**，不是常驻服务——没人启动它的时候它当然"不运行"。

```powershell
cd "rail-intrusion"
python tools/run_lit.py            # 六步全跑；产物已是最新的步骤自动跳过
python tools/run_lit.py --list     # 只看步骤顺序与各步产物
python tools/run_lit.py --steps search --force   # 只重跑检索
```

- 六步脚本在 `..\out\lit\`（工作区临时目录，不入库），产物：`results.json` / `locations.json` / `abstracts.json` / `authors.json`。
- 每步日志写 `..\out\lit\logs\`，末 8 行会打到控制台。
- 代理：脚本自己剥掉 `HTTP(S)_PROXY`/`ALL_PROXY`（本机那个 127.0.0.1:7897 已失效，带着它一切请求都失败）。
- 已知失败面：MDPI / ACM DL / Hindawi / TechScience 对脚本 UA 返回 403（见上表 16 篇）。
- 抽正文需要 `pymupdf`；缺依赖时 `extract` 步会报错退出，其余步骤不受影响。
