# rail_uav_inspection_image_dataset 数据使用申请（邮件草稿）

实测结论：该数据集托管在鲸智社区（中国信通院），**git 仓库公开可克隆，但仓库里只有说明文档，6.5 GB 图像与标注不在仓库中**。
说明文档第七节写明「使用方需经授权，签订正式数据使用协议后方可使用」——所以拿数据只能走授权。

- 发布组织：`pci_tech`（企业命名空间），联系人邮箱 `liyang12@pcitech.com`（从平台 API `api/v1/datasets/pci_tech/rail_uav_inspection_image_dataset` 读取）
- 平台方：中国信通院鲸智社区，联系邮箱 `wuyinqian@caict.ac.cn`（从页面脚本读取）
- 数据集页：https://aihub.caict.ac.cn/datasets/pci_tech/rail_uav_inspection_image_dataset

---

## 邮件一：发给发布方（主送）

**收件人**：liyang12@pcitech.com
**抄送**：wuyinqian@caict.ac.cn
**主题**：数据使用申请：rail_uav_inspection_image_dataset（pci_tech）

您好：

我是【学校名称】【学院/专业】的【姓名】，指导教师为【导师姓名】老师。我们正在开展**轨道交通车端与无人机巡检场景下的侵限/异物目标检测**研究，拟参加 2026 年首届全国大学生轨道交通科技创新大赛（中国铁道学会主办）。在鲸智社区检索到贵方发布的 `rail_uav_inspection_image_dataset`，内容与我们的研究场景高度吻合，特此申请数据使用授权。

**申请使用的子集**（如可分批提供，请优先第一批）：
1. `scene_add/track_area_person`（轨道区域人员，209 张）
2. `scene1`（轨道区域人员入侵，593 张，1920×1080）
3. `scene_add/foreign_object`（异物，225 张）
4. 对应的 labelme JSON 标注文件

**用途**：仅用于学术研究与竞赛作品的模型训练、微调与评测，具体为 YOLO 系列目标检测模型的训练与精度评估。

**我方承诺**：
- 不将数据用于任何商业用途，不对外共享、转让或再分发；
- 遵守《网络安全法》《数据安全法》《个人信息保护法》及贵方数据使用协议的全部条款；
- 研究成果中明确标注数据来源为贵方发布的数据集；
- 如协议要求，可签署正式数据使用协议并提供单位/导师的在读证明或项目说明；
- 研究结束后按贵方要求删除或归还数据副本。

不知需要提交哪些材料、走什么流程？如方便，烦请告知数据获取方式（下载链接/网盘/其他），我们将尽快完成协议签署。

感谢！

【姓名】
【学校/院系】
【联系电话】
【邮箱】
【日期】

---

## 邮件二：发给平台方（若无回应再发，或同时抄送）

**收件人**：wuyinqian@caict.ac.cn
**主题**：关于鲸智社区数据集 rail_uav_inspection_image_dataset 获取流程的咨询

您好：

我是一名在校学生，拟使用贵平台托管的数据集 `pci_tech/rail_uav_inspection_image_dataset`（https://aihub.caict.ac.cn/datasets/pci_tech/rail_uav_inspection_image_dataset）开展轨道交通巡检目标检测的学术研究与竞赛作品开发。

我注意到该数据集的 git 仓库中仅有说明文档，而说明文档第七节载明需经授权并签署数据使用协议后方可使用。想请教：

1. 该数据集的申请入口是否在平台内的某个页面（我未找到明显的「申请」按钮）？
2. 需要向发布方（pci_tech）直接申请，还是由平台统一受理？
3. 是否接受在校学生个人申请，或需由指导教师/学院出具证明？

如需我补充任何材料，请告知。

感谢！

【姓名】【学校/院系】【联系电话】【邮箱】

---

## 备选联系人线索（若上面两个邮箱都无回应）

- 数据集 API 返回的发布组织邮箱：`liyang12@pcitech.com`（`user.email` 字段，已实测可取）
- 平台页面「联系我们」复制出的邮箱：`wuyinqian@caict.ac.cn`
- 平台社区需求广场：`https://aihub.caict.ac.cn/issues`（可在平台上公开提问）
