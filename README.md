# Responsibility

本仓库作为新主力机继续开发的代码起点，当前版本为 **v0.4.0：分组学习率版 CRFR＋RRCA**。
模型和优化器分组恢复自 `e7bcbb9`，不包含 v0.3.0 新增的四机制时间记忆。

仓库只提供代码、配置、环境说明和数据构建工具，不附带历史实验结果。
后续比较以新电脑实际训练、评价和回测生成的结果为准，不要求与服务器数值相同。

## 1. 当前模型与学习率

```text
Alpha158 个股因子＋Market63 市场特征＋14维股票状态与规则证据
    → CRFR：共识责任因子路由
    → 单股时间注意力
    → 基础截面注意力
    → RRCA：个股责任路由的跨股票聚合
    → 时间汇聚与未来收益预测
```

Alpha158 是已有因子输入；CRFR 生成的是对这些因子的权重。
本版参数量为 846229；不使用路由 detach，不加入额外记忆模块。

整个模型使用一个 Adam 优化器、两个参数组。CSI300 和 CSI800 均为：

| 参数组 | 学习率 |
| --- | ---: |
| 主干＋RRCA 适配器 | `1e-5` |
| CRFR＋RRCA 条件系数 rho | `1e-4` |

配置保留原来的四个字段：`base_learning_rate`、`crfr_learning_rate`、
`rrca_adapter_learning_rate`、`rrca_condition_learning_rate`；
它们对应上述两组实际更新，不是四个独立优化器，也没有学习率调度器。
每次训练会把实际参数分组写入本机运行目录的 `optimizer.json`。

## 2. 新电脑安装（RTX 5070 Ti）

默认 [environment.yml](environment.yml) 面向 Python 3.10、
PyTorch 2.7.1＋CUDA 12.8 的 Windows x64 / WSL2 Linux x64 环境。
保留 NumPy 1.23.5、pandas 1.5.3、SciPy 1.10.1、scikit-learn 1.3.2、
PyYAML 6.0.2 和 PyQlib 0.9.7，减少数据读取与评价组件的变化。

先准备 Git、Conda 和支持本机显卡的 NVIDIA 驱动，然后执行：

```text
git clone https://github.com/dorightthings/Responsibility.git
cd Responsibility
conda env create -f environment.yml
conda activate responsibility-rtx50
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_environment.py
```

以上命令均为单行，可在 Windows 的 Anaconda Prompt / 已初始化 Conda 的 PowerShell，
或 WSL2 终端中执行。`check_environment.py` 会实际运行模型前向、反向与一次 Adam 更新；
没有可用 CUDA 时默认报错，不会把 CPU 检查误当作显卡检查。

5070 Ti 环境目前是迁移配置，尚未在目标 Windows 电脑实测；它不是服务器环境的复制品。
原服务器实际使用的 Python 3.8.20／PyTorch 2.1.2／CUDA 12.1 配置单独记录在
[envs/server-legacy.yml](envs/server-legacy.yml)，不要在 5070 Ti 上照搬它。
版本来源、CPU 检查方式与常见问题见 [环境说明](docs/ENVIRONMENT.md)。

## 3. 数据准备

使用此前的个人网盘数据包即可，不需要因为恢复分组学习率而重建数据。
包内应包含两个数据集的 sampler、14维股票状态、规则分量、训练段方向尺度，
以及冻结回测所需的 Qlib provider。数据不要提交到 GitHub。

下载归档后，可以用实际的本机文件 URI 解压，例如 Windows：

```text
python scripts/download_data.py --url "file:///D:/Downloads/你的数据包.tar.gz" --destination data
python scripts/verify_data.py --data-root data
```

`file:///D:/...` 是示例，替换成真实下载位置。有已知归档 SHA-256 时可给
`download_data.py` 增加 `--sha256`；它只检查文件完整性，不要求训练结果相同。
只加载自己生成或可信来源的 pickle 数据。

数据目录约定见 [data/README.md](data/README.md)。
需要以后重新生成数据时，使用 [数据构建说明](docs/DATA_BUILD.md) 中的现有脚本。
首次升级环境需要确认原 pickle 能正确读取；不要为解决兼容问题擅自改股票池、样本或标签。

## 4. 开始训练

先进行一次最小数据与训练检查：

```text
python scripts/train.py --config configs/csi300.yaml --data-root data --mode smoke --seed 0 --gpu 0 --output-dir outputs/smoke_grouped_csi300_seed0
```

Smoke 固定两轮、跳过回测，仅说明链路可运行，不是正式结果。
输出目录必须尚不存在；失败重试使用新目录，不覆盖旧产物。

随后在一张显卡上排队训练 CSI300、CSI800，各五个种子：

```text
python scripts/run_batch.py --configs configs/csi300.yaml configs/csi800_direct.yaml --data-root data --mode formal --seeds 0 1 2 3 4 --gpus 0 --jobs-per-gpu 1 --output-root outputs/rtx5070ti_grouped_baseline
```

默认会继续执行评价、Top30/Drop30 回测和结果汇总。先使用单任务并发；显存不足时可加
`--cpu-data-cache`，仅把数据缓存放在内存中，不拆分完整日截面，也不改变模型输入。
不要同时照搬服务器的多 GPU 命令。

需要单独补回测或重新汇总时：

```text
python scripts/evaluate.py --run-root outputs/rtx5070ti_grouped_baseline --data-root data
python scripts/summarize.py --run-root outputs/rtx5070ti_grouped_baseline
```

所有结果以本机生成的文件为准，保存在 `outputs/`，默认不会进入 Git。

## 5. 固定实验口径

- 数据集：CSI300、CSI800（内部标识为 `short_csi300`、`short_csi800_direct`）。
- 训练段：2008-01-02—2020-03-31；验证段：2020-04-01—2020-06-30。
- 测试段：2020-07-01—2022-12-30；lookback 8；no-purge。
- 原始标签：`Ref($close, -5) / Ref($close, -1) - 1`。
- 仅训练时按日去除上下各 2.5% 极端标签，再按截面样本标准差做 z-score。
- CSI300 的市场门温度 beta 为 5，CSI800 为 2；不是股票状态里的收益回归 beta。
- 最多 150 epochs；首次日均 TrainMSE <= 0.95 停止；seeds 0—4。
- 六项指标：IC、ICIR、RankIC、RankICIR、AR、IR。
- 回测：Qlib Top30/Drop30；AR/IR 使用不计手续费的超额收益。
- 同一测试期已用于开发；更换电脑重新训练不使它变成未见测试集。

完整参数与运行记录要求见 [实验约定](docs/REPRODUCIBILITY.md)。

## 6. 项目目录与开发

```text
Responsibility/
├── configs/          # 两个数据集的分组学习率配置
├── src/              # CRFR、RRCA、训练、评价与数据处理
├── scripts/          # 环境检查、数据准备、训练、回测、汇总
├── envs/             # 旧服务器环境记录
├── docs/             # 方法、数据构建、环境与开发说明
├── tests/            # 必要的代码检查
├── data/             # 数据文件留本机，Git仅跟踪说明与示例清单
├── outputs/          # 本机权重、预测、日志与结果，不提交
├── environment.yml   # 5070 Ti 新电脑环境
└── pyproject.toml
```

代码与配置通过 GitHub 更新；数据与实验产物留本机或个人网盘。
参见 [主力机开发与分支说明](docs/TWO_MACHINE_WORKFLOW.md)。

此前分组版来源为 `e7bcbb9`，无记忆统一学习率版为 `60e5a58`，
统一学习率＋四机制记忆版为 `9f70c7d`。历史代码与结果仍可在 Git 历史中查看，
但不作为当前版本附带的参考表；记忆版 checkpoint 不能直接加载到本版。

本次发布仅整理当前方法；独立 MASTER 和各对比方法的一键运行工程尚未接入本仓库，
不要把它们在服务器上的可运行状态当作本仓库已完成的功能。

## 7. 许可

方法见 [METHOD](docs/METHOD.md)，代码许可见 [LICENSE](LICENSE)，
上游来源见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)，引用元数据见 [CITATION](CITATION.cff)。
代码许可不自动覆盖行情和指数成分数据。本项目用于研究，不构成投资建议。
