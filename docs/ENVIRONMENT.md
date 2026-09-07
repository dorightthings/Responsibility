# 运行环境

## 新主力机：RTX 5070 Ti

默认 `environment.yml` 是新机器的环境配置，不是服务器的环境备份：

| 组件 | 配置 |
| --- | --- |
| 系统目标 | Windows x64 或 WSL2 Ubuntu/Linux x64 |
| Python | 3.10 |
| PyTorch | 2.7.1，CUDA 12.8 wheel |
| NumPy / pandas | 1.23.5 / 1.5.3 |
| SciPy / scikit-learn | 1.10.1 / 1.3.2 |
| PyYAML / PyQlib | 6.0.2 / 0.9.7 |
| MLflow / CVXPY | 2.17.2 / 1.5.2（沿用服务器依赖版本） |

选择固定的 2.7.1＋cu128 是为了提供一个支持新显卡架构的明确安装起点，
不表示它是最新版本，也不表示已经在目标机验证通过。
PyTorch 官方介绍了 2.7 的 Blackwell 支持，并提供 2.7.1 的 Linux/Windows CUDA 12.8 安装命令：
[Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、
[历史版本安装](https://pytorch.org/get-started/previous-versions/)。
PyQlib 0.9.7 提供 CPython 3.10 的 Windows 和 Linux wheel：
[PyQlib 发行文件](https://pypi.org/project/pyqlib/0.9.7/#files)。

环境文件固定了关键直接依赖；完整传递依赖及驱动应以新机实际安装记录为准，
它不是跨平台的逐包构建锁文件。安装成功后运行：

```text
python -m pip install -e . --no-deps
python -m pip check
python scripts/check_environment.py
```

环境检查默认要求 CUDA，并用当前模型实际完成一次合成输入的前向、反向和分组 Adam 更新。
检查不会读真实股票数据，也不产生实验结果或 checkpoint。
只做 CPU 代码检查时显式使用 `python scripts/check_environment.py --device cpu`；
CPU 通过不代表 5070 Ti 已通过。

随后用 README 的一次数据 smoke 验证旧 pickle、责任特征与训练入口，再运行正式任务。
暂未在目标 Windows/5070 Ti 机器运行过本项目，不能预先保证驱动、GPU 运算和原 pickle 全部兼容。

## 驱动与平台

先在 Windows 安装适用于 5070 Ti 的 NVIDIA 驱动。`nvidia-smi` 显示的是驱动所支持的
CUDA 能力，训练使用的 PyTorch CUDA runtime 版本见检查脚本输出的 `torch_cuda`。
默认方案使用官方预编译 wheel，不需要为本模型单独编译 CUDA 扩展。

原生 Windows 可使用 Anaconda Prompt 或已初始化 Conda 的 PowerShell；
WSL2 是另一种选择，不要求重装 Windows。WSL2 使用 Windows 主机驱动，
不要在 WSL 中再安装 Linux 显卡驱动：
[NVIDIA WSL 文档](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

若已有适配 5070 Ti 的环境，先检查版本和依赖；不要直接覆盖现有环境。
希望采用不同 PyTorch/Python 版本时，应在独立环境记录改动，
同步调整项目依赖约束，再在该环境重跑本机的所有比较。

## 旧服务器环境

实际配置记录在 `envs/server-legacy.yml`：
Python 3.8.20、PyTorch 2.1.2、CUDA 12.1，科学计算依赖和 PyQlib 同上。
旧发布模板曾写 CUDA 11.8；本次读取 `torch.version.cuda` 和 Conda 安装记录均为
12.1（PyTorch 构建为 `py3.8_cuda12.1_cudnn8.9.2_0`），因此按实际情况更正，
没有修改服务器安装环境。旧模板仍可在历史提交中查看。
该文件记录旧版的运行起点，不作为 5070 Ti 安装入口。
本次服务器上的 CPU 单元检查使用原环境，不会安装或替换服务器依赖。

## 数据与结果

保留同一数据包、划分、标签、特征和回测 provider。
先确认加载器能读取原 sampler；若需要文件格式兼容转换，应另外保留原文件，
验证样本索引、数值和缺失掩码未变，不能用重新下载的数据静默替换。

正式任务在本机记录 Python/PyTorch、GPU、参数配置和命令。
新机器第一次运行后，可将 `python -m pip freeze`、
`conda list` 和 `nvidia-smi` 输出保存在本机运行记录中，检查凭据后再按需归档。
不把旧服务器指标当作目标机的运行结果。
