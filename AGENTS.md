# rail-intrusion — 给下一位 agent 的须知

车端侵限报警原型（OSDaR23 数据 + YOLOv8n 检测 + lidar 测距 + INS 测速 + TTC 四级报警）。
本文件只写「不做就会出错」的事。

## 0. 长任务的时间约束（用户 2026-10-31 定的规矩，选 B）

**不夜跑。** 任何可能超过 1 小时的任务（训练/多种子/批量推理），只在用户明确说「跑」的时候启动；
哪怕跑起来后跨到了他睡觉的时间，也照跑不误——但要满足两条：

1. 启动前先说明预计耗时与结束时刻；
2. 跑完把结论写进 `docs/nightly/<日期>.md`（跑了什么、数字、怎么自查、成功/失败判据），
   让用户早上第一屏就能看到，**不依赖我在场汇报**。

理由：他的反馈是「你每次都 23 点后跑，而我已经睡觉了」——夜间产出如果没人读，等于无效产出。
固定交流窗口：**每晚 20:00 一篇论文（或进展）**。

## 1. 汇报必须先读 docs/report-conventions.md

用户明确要求（2026-10-26）：**复现论文内容、做消融实验、并在汇报时标出每个数字的来源类型**。
四类标签：**【基线】/【复现】/【消融】/【自研】**。每个数字后必须跟来源路径，表格必须有「来源类型」列。
没标来源的数字不要写进汇报。

## 2. 常跑的命令

```powershell
python -m pytest -q                                   # 113 项单测，改代码前后各跑一次
python tools/verify_per_class.py <best.pt> 1280 <dataset.yaml> <out.json>   # 统一口径评测
python tools/train_osdar23.py --data <yaml> --out <dir> --epochs 60 --imgsz 1280 --batch 4 --device 0 --seed 0 --name <name>
python tools/run_seeds.py --seeds 0 1 2 --wait-pid <pid>                    # G 轴多种子
python tools/md2docx.py docs/<x>.md                    # 生成 materials/<x>.docx
```

## 3. 会咬人的坑（都踩过）

- **别用 PowerShell 的 `Set-Content` / `Out-File` / `-replace` 改含中文的 UTF-8 文件**：会变乱码，
  甚至吃掉收尾引号导致语法错误（`tools/train_osdar23.py` 就这样被毁过一次）。改文件用 edit/write 工具。
- **`.py` 脚本不要放 `%TEMP%` 跑**：系统 TEMP 里有 `inspect.py` 之类同名文件，会遮蔽标准库。
- **`tools/` 下的脚本保持 ASCII**（注释/输出都用英文），避免再发生编码往返。
- **`data/`、`runs/`、`out/` 不入 git**（`.gitignore` 已含），`materials/*.docx` 入 git。
- **评测口径必须统一**：所有 mAP 都用 `tools/verify_per_class.py` 在同一 val 上算，
  不要混用 ultralytics 训练期日志与独立评测（历史上差过 0.02）。
- **机器是 RTX 4060 Laptop 8 GB**：训练 `--batch 4 --imgsz 1280` 是上限，两个训练进程并行会 OOM。
- **`--workers` 不要超过 4**：默认 8 时 dataloader 子进程会在 train→val 交接处崩
  （`EOFError: Ran out of input`，父进程 exit 1、日志停在某个 epoch 末尾且无 Python traceback）。
  2026-10 起 `tools/train_osdar23.py` 显式暴露 `--workers`，跑长训练一律 `--workers 4` 并重定向到文件。

## 4. 当前进度指针

- 精度实验（A/C1/G 轴）：`docs/accuracy-improvements.md`
- 融合（C1 三轮全部否证）：`docs/c1-fusion-report.md`
- 时序（C2 数据缺口）：`docs/c2-temporal-report.md`
- YOLO 版本对照：`docs/yolo-version-compare.md`
- VLM 探测（否证）：`docs/vlm-probe-report.md`
- 文献：`refs/README.md`（编号体系）、`docs/literature-review.md`、`docs/reading-plan.md`
- 总账本：`C:\Users\jojo as\Desktop\AI项目账本\projects\rail-intrusion.md`
