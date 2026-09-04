# 两台机器联动

GitHub 只同步代码、配置、数据构建脚本和轻量结果摘要。数据、checkpoint、
prediction 和完整运行日志都保存在各自机器或网盘中，不进入 Git 历史。

## 首次配置另一台机器

1. 在新机器生成一把 SSH key，并把公钥添加为本 GitHub 仓库的可写
   Deploy Key。两台机器应使用各自的 key。
2. 克隆仓库并创建环境：

```bash
git clone git@github.com:dorightthings/Responsibility.git
cd Responsibility
conda env create -f environment.yml
conda activate responsibility
python -m pip install -e .
```

3. 从个人网盘下载数据包，核对 SHA-256，再解压到仓库的 `data/`。
   `data/` 已被 `.gitignore` 排除，不会在 `git push` 时误传。

两台机器不要求得到逐位相同的模型权重或指标。不同 GPU、CUDA 和驱动会带来正常
数值波动；联动时只需确认代码 commit、`data/manifest.json`、YAML 配置和 seed 记录
正确。

## 日常同步

每次开始修改前：

```bash
git pull --rebase origin main
```

完成一次代码、配置或轻量结果摘要更新后：

```bash
git status --short
git add <本次确定要同步的文件>
git commit -m "Describe this update"
git push origin main
```

然后另一台机器再执行 `git pull --rebase origin main`。最简单的用法是同一时间只在
一台机器修改同一个文件；如果两端需同时开发，再为其中一端创建独立分支。

## 应该同步与不应该同步的内容

应该提交：

- `src/`、`scripts/`、`configs/`中的模型、数据处理和实验配置；
- `docs/`中的方法与运行说明；
- `results/experiments/`中的小型 CSV/JSON/Markdown 结果摘要。

不应该提交：

- `data/`中的数据、Qlib provider 和 sampler pickle；
- `outputs/`、`runs/`、checkpoint、prediction 和回测二进制产物；
- SSH 私钥、Token、Cookie、`.env` 或带凭据的下载链接。

完整运行目录需要跨机器迁移时，放网盘；GitHub 只保留能说明进度和结论的轻量摘要。

## 把本机进度同步到 Git

正式任务汇总后，可以导出不含本机绝对运行路径的小型摘要：

```bash
python scripts/export_progress.py \
  --summary-dir outputs/<run>/summary_<UTC时间> \
  --experiment-id exp_001 \
  --machine server_a \
  --config configs/csi300.yaml \
  --config configs/csi800_direct.yaml \
  --note "本机完成 seeds 0-4"

git add results/experiments/exp_001
git commit -m "Add exp_001 summary"
git push origin main
```

另一台机器拉取后能看到配置、逐 seed 指标与汇总，但不会下载 checkpoint 和完整日志。
若两台机器同时做不同实验，使用不同的 `experiment-id`，避免修改同一个摘要目录。
