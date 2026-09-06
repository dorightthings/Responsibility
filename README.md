# Responsibility

`ResponsibilityModel` 是一套面向股票收益预测的责任机制模型。它用同一组动态机制责任，同时回答两个问题：当前应重点采用哪些因子证据，以及一只股票应从哪些机制群体中获取跨股票信息。

本仓库发布的是当前选定的统一学习率开发底座，便于在不同机器上从相同代码和数据开始各自的改进实验：

```text
Alpha158 个股因子 + Market63 市场状态 + 个股/规则证据
    -> CRFR：共识责任因子路由
    -> 单股时序证据编码
    -> 基础截面信息编码
    -> RRCA：责任路由截面聚合
    -> 时间汇聚与未来收益预测
```

模型使用四类条件预测责任：共同延续、共同反转、个股延续和个股反转。这里的“责任”表示当前条件下不同经济机制对预测证据的相对分配强度，不表示已经识别出真实因果责任。

## 1. 当前发布边界

- 公开模型名：`ResponsibilityModel`。
- 核心模块：CRFR（Consensus Responsibility Factor Routing）和 RRCA（Responsibility-Routed Cross-sectional Aggregation）。
- 保留原有 stock-specific RRCA 路由及初始化，不使用路由梯度分离（detach），不加入新的初始化方案。
- 整个模型使用一个 Adam 参数组；唯一学习率配置字段为 `training.learning_rate`，CSI300 为 `2e-5`，CSI800 为 `1e-5`，训练期间固定不变。
- 数据集：`short_csi300` 与 `short_csi800_direct`。
- `short_csi800_direct` 沿用冻结 provider 的作者兼容直接 CSI800 口径，并保留其已知
  的早期历史成分截断；它不是重新合成的 CSI300+CSI500 股票池。
- 正式矩阵：两个数据集分别运行 seeds `0,1,2,3,4`，共 10 个任务。
- 默认回测：Qlib `TopkDropoutStrategy(topk=30, n_drop=30)`，headline 为不计手续费的 `excess_return_without_cost` AR/IR。
- 当前结果属于开发期重复测试证据（`development/repeated-test`），不应描述为从未查看过的独立测试集验证。

本项目仅用于研究，不构成任何投资建议。

## 2. 安装

推荐 Linux 或 Windows 11 + WSL2，并使用 NVIDIA GPU。冻结环境为：

- Python 3.8.20
- PyTorch 2.1.2
- CUDA 11.8
- NumPy 1.23.5
- pandas 1.5.3
- SciPy 1.10.1
- scikit-learn 1.3.2
- PyQlib 0.9.7

```bash
git clone https://github.com/dorightthings/Responsibility.git
cd Responsibility
conda env create -f environment.yml
conda activate responsibility
python -m pip install -e .
```

若机器的驱动或 CUDA 条件不同，请先按 PyTorch 官方方式安装与本机兼容的 PyTorch 2.1.2，再安装其余依赖。当前正式实验在 24 GiB RTX 4090 上验证；新机器首次运行建议每张卡只放一个任务。

## 3. 准备数据

训练数据、责任特征和 Qlib provider 合计约 3.5 GB，不直接存入普通 Git 历史。
在两台个人机器之间，推荐把数据包放在自己的网盘，GitHub 只同步代码与配置。

下载一个数据归档：

```bash
python scripts/download_data.py \
  --url "<DATA_ARCHIVE_URL>" \
  --sha256 "<OPTIONAL_ARCHIVE_SHA256>" \
  --destination data
```

如果数据被拆成多个互不重叠的归档，可对每个 URL 重复执行上述命令。归档内部应直接包含 `manifest.json`、`samplers/`、`responsibility/`、`direction_scales/` 或 `qlib/` 等相对于 `data/` 的路径，不要再套一层绝对机器目录。

完成后校验：

```bash
python scripts/verify_data.py --data-root data
```

只检查布局和文件大小、暂时跳过 SHA-256 时可使用：

```bash
python scripts/verify_data.py --data-root data --quick
```

预期数据布局及 manifest 约定见 [data/README.md](data/README.md)。

仓库也包含完整数据构建代码。需要从冻结 Qlib provider 重新生成 sampler、股票状态、
规则特征和方向尺度时，按 [数据获取与重建](docs/DATA_BUILD.md) 操作。直接使用已生成
数据包更快；重新生成适合以后修改数据区间、股票池或特征定义。

## 4. Smoke test

Smoke test 固定使用 CSI300、seed 0、两轮训练并跳过回测，用于确认环境、数据读取、模型前向和反向传播能够工作：

```bash
python scripts/train.py \
  --config configs/csi300.yaml \
  --data-root data \
  --mode smoke \
  --seed 0 \
  --gpu 0 \
  --output-dir outputs/smoke_csi300_seed0
```

输出目录必须是一个尚不存在的新目录。Smoke 成功只说明程序能够运行，不用于报告正式性能。

## 5. 正式十任务

```bash
python scripts/run_batch.py \
  --configs configs/csi300.yaml configs/csi800_direct.yaml \
  --data-root data \
  --mode formal \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 2 3 \
  --jobs-per-gpu 1 \
  --output-root outputs/formal_responsibility_v1
```

这条命令启动：

- `short_csi300` × 5 seeds；
- `short_csi800_direct` × 5 seeds；
- 合计 10 个相互独立的训练任务。

24 GiB 显卡可以在确认显存余量后提高并发，但不同显卡应从 `--jobs-per-gpu 1` 开始。不要复用已有输出目录，也不要覆盖已有 checkpoint、prediction 或回测文件。

每个训练任务会写入 `optimizer.json`，记录实际学习率、参数组数量和作用范围，便于确认整个模型使用同一个学习率。

若训练入口与评价入口分开，使用：

```bash
python scripts/evaluate.py \
  --run-root outputs/formal_responsibility_v1 \
  --data-root data

python scripts/summarize.py \
  --run-root outputs/formal_responsibility_v1
```

## 6. 参考结果

当前统一学习率开发底座的五 seed 均值 ± 总体标准差（population standard deviation，`ddof=0`）如下。两组均保留 CRFR＋原 stock-specific RRCA，不使用 detach；Top30/Drop30 的 AR/IR 均采用 `excess_return_without_cost`，AR 使用小数形式（例如 `0.297308` 表示约 `29.73%`）：

| 数据集 | IC | ICIR | RankIC | RankICIR | AR | IR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSI300 | 0.057674 ± 0.005655 | 0.383250 ± 0.044031 | 0.069350 ± 0.003849 | 0.452649 ± 0.034287 | 0.297308 ± 0.056426 | 2.522038 ± 0.420316 |
| CSI800 | 0.047661 ± 0.003485 | 0.395893 ± 0.045613 | 0.061988 ± 0.002760 | 0.489023 ± 0.026603 | 0.293114 ± 0.052997 | 2.250111 ± 0.401712 |

完整精度的均值、标准差、逐 seed 数值和协议记录位于 [results/reference_uniform_lr/](results/reference_uniform_lr/)，seeds 为 `0,1,2,3,4`。这些是 `development/repeated-test` 开发参考结果，不是未见测试集上的独立验证，也不是异机运行必须达到的数值目标。

原 [results/reference/](results/reference/) 保留历史分组学习率路线及其对照结果，文件和数值未改动；它不再是当前统一学习率实现的参考结果。两台机器只需从相同代码、配置和数据开始，可以分别改进并将有效版本推到各自分支，不要求最终结果相同。

## 7. 冻结实验口径

| 项目 | CSI300 | CSI800 |
| --- | --- | --- |
| `beta` | 5 | 2 |
| `training.learning_rate` | `2e-5` | `1e-5` |
| benchmark | `SH000300` | `SH000906` |
| train | 2008-01-02 至 2020-03-31 | 同左 |
| valid | 2020-04-01 至 2020-06-30 | 同左 |
| test | 2020-07-01 至 2022-12-30 | 同左 |
| bridge | 2020-06-30 | 2020-06-30 |

共同设置：

- lookback `T=8`；输入为 158 个股因子、63 个市场特征和 1 个标签字段；
- 标签为 `Ref($close, -5) / Ref($close, -1) - 1`；
- 训练时按日去除上下各 2.5% 极端标签，再做横截面 z-score；
- Adam 单参数组：主干、CRFR、RRCA context body 与 condition 共用当前数据集的 `training.learning_rate`，无模块级学习率和学习率调度器；
- 责任选择温度从 `1.0` 线性退火到 `0.5`，前 10 epochs 完成；
- 梯度裁剪绝对值 `3.0`；最多 150 epochs；首次日均 TrainMSE `<=0.95` 时停止；
- 正式回测使用 Top30/Drop30，headline 为无手续费 AR/IR。

具体机器、CUDA 和底层算子不同可能造成最终指标差异。这里提供共同的代码和数据起点；后续实验可各自调整配置和方法，只需记录版本、数据、配置与结果。学习率修改只使用 `training.learning_rate`，不要沿用旧的四个模块级学习率字段。

## 8. 仓库结构

```text
Responsibility/
├── configs/                 # 两个冻结数据集的配置
├── data/                    # 下载后的数据；Git 只保留说明文件
├── docs/                    # 方法、数据构建、双机联动和许可说明
├── results/experiments/     # 可提交 Git 的轻量实验摘要
├── scripts/                 # 数据构建、训练、评价、汇总和打包入口
├── src/                     # 模型、训练组件与 preprocessing
├── environment.yml
├── pyproject.toml
└── README.md
```

## 9. 两台机器联动

两台机器分别配置本仓库的 SSH key，然后都通过同一 GitHub 地址拉取和推送。代码、
YAML 和轻量结果摘要进 Git；数据、checkpoint、prediction 和完整日志放本地或个人
网盘。具体命令见 [两台机器联动](docs/TWO_MACHINE_WORKFLOW.md)。不同机器不要求跑出
逐位相同的结果，只需记录各自使用的 commit、data manifest、配置和 seed。

## 10. 引用与许可

代码许可见 [LICENSE](LICENSE)，复用组件和第三方项目说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，引用元数据见 [CITATION.cff](CITATION.cff)。代码许可不自动覆盖数据、指数成分、模型权重或第三方内容；分发和使用数据前请单独确认其许可。
