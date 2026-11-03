# 把本项目发布到 GitHub

本机**没有** `gh` CLI，仓库也**没有**配置远程（`git remote -v` 为空），而且没有你的 GitHub 凭据 —— 所以最后一步"推送"必须由你提供一次授权。以下是两种做法，二选一。

## 现状（已由我准备好）

- 已存在干净的发布分支 `publish`：只含代码与文档，不含 `data/`、`runs/`、`out/`、`*.pt`、`refs/*.pdf`（主分支 `main` 保留完整历史，不受影响）。
- `README.md` 已写好：2D/3D 全部实测数字、复现命令、诚实的已知问题。
- `.gitignore` 已覆盖数据、权重、runs、materials、refs PDF。

## 做法 A：你在 GitHub 网页上建空仓库，然后把地址给我

1. 打开 https://github.com/new ，仓库名建议 `rail-intrusion`，**不要**勾选 README / .gitignore / license（保持空仓库）。
2. 建好后把仓库地址（形如 `https://github.com/<你的用户名>/rail-intrusion.git`）发我。
3. 我执行推送（需要你在此之前完成下面的"凭据"步骤，或改用做法 B）。

## 做法 B：你给我一个 PAT，我一条命令推完

1. 打开 https://github.com/settings/tokens → Generate new token (classic) → 勾选 **repo** 权限 → 生成。
2. 把 token 和仓库地址一起发我。
3. 我会执行（token 只用于这一次推送，不会写进任何文件）：

```powershell
git push https://<token>@github.com/<user>/rail-intrusion.git publish:main
```

推完建议你立刻在 GitHub 上删除该 token（或用细粒度 token 只授权这一个仓库）。

## 凭据步骤（做法 A 用）

不想用 token 的话，用 Git 自带的凭据管理器登录一次：

```powershell
git config --global credential.helper manager
git push https://github.com/<user>/rail-intrusion.git publish:main
```

首次会弹窗要求登录（浏览器授权），之后同一台机器上的推送都不用再输。

## 推送前我自己会先做的检查

```powershell
git ls-files | Measure-Object                    # 只应有代码与文档
git ls-files | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum   # 总量应 < 50 MB
```

单文件 GitHub 上限 100 MB，仓库建议 < 1 GB。当前发布分支已排除所有大文件。

## 不需要推送的东西（已在 .gitignore）

`data/`（含 OSDaR23 与派生数据集）、`runs/`（训练权重）、`out/`（实验产物）、`materials/`（Word 交付物）、`refs/*.pdf`（论文原文，走 DOI 更合适）、`*.pt`、`*.zip`、`*.png`。
