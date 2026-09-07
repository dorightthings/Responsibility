# 新主力机开发与分支

新电脑作为主力开发环境；GitHub 同步代码、配置、环境文件和说明。
数据、历史参考表、新结果、checkpoint、prediction 和日志默认都不提交。
不要求两台机器同步训练，也不要求结果相同。

## 首次开始

用 HTTPS 克隆即可下载公开仓库；下载不需要复制服务器的 SSH 私钥。
按照 [README](../README.md) 创建新机环境、下载个人网盘数据并开始训练。
新电脑以自己重新跑出的结果为参考。

需要推送时，在新电脑独立配置 GitHub 身份或专用 SSH key，
不要从服务器拷贝已有私钥，也不要把任何凭据放入项目目录。

## 代码开发

开始前先检查工作区：

```text
git status --short
git pull --ff-only origin main
```

有尚未处理的修改时，先确认其归属并保存到自己的分支，不执行覆盖式更新。
从当前主线创建新实验分支，例如：

```text
git switch -c codex/rtx5070ti-development
```

完成一次确定要保留的代码修改后：

```text
git status --short
git add src configs scripts docs
git commit -m "Describe the model change"
git push origin HEAD:refs/heads/codex/rtx5070ti-development
```

提交前逐项检查文件；上面的 `git add` 范围应按本次实际修改缩小，
环境文件另行明确添加。推送指定仓库远端和分支，不修改其他项目的远端设置。

## 本地结果

训练、回测和汇总自动写入 `outputs/`，已被 Git 忽略。
`results/` 同样忽略，当前版本不附带旧服务器结果。

需要为个人网盘整理轻量摘要时，可使用 `scripts/export_progress.py`，
默认输出到 `outputs/progress/<experiment-id>/`，仅作本地归档。
不要按照旧版教程执行 `git add results/...`。
历史提交未改写，旧代码和当时发布的参考文件仍留在 Git 历史中。
