# 数据获取与重建

本项目支持两种数据准备方式。对于两台个人机器联动，推荐第一种；第二种用于以后
修改数据区间、股票池或责任特征时重新生成。

## 方式一：直接使用已生成的数据包

把服务器生成的 `.tar.gz` 上传到个人网盘。在另一台机器下载后执行：

```bash
python scripts/download_data.py \
  --url "file:///本机下载目录/责任机制股票预测_完整复现实验数据包_20260904.tar.gz" \
  --sha256 "8a046e91573eb36794a75550dc3c7ea90aba85951c8498d5d6613a9e38c73acb" \
  --destination data

python scripts/verify_data.py --data-root data
```

如果网盘提供可直接下载的 HTTP(S) 地址，可以把 `file:///...` 换成该地址。SHA-256
只用于确认两台机器拿到的是同一个完整数据包，不要求两台机器的模型结果逐位相同。

这个方式最快，因为包内已经包含：

- 两个数据集的 train/valid/test sampler；
- 股票状态与规则特征；
- 训练集方向尺度；
- 冻结回测所需的精简 Qlib provider。

## 方式二：从原始 Qlib provider 重建

### 1. 下载冻结 provider

```bash
python scripts/download_data.py \
  --url "https://github.com/chenditc/investment_data/releases/download/2024-12-07/qlib_bin.tar.gz" \
  --sha256 "5f6ec2448070d73834dcbba27a778462451359334757876815ff2795d2358b93" \
  --destination data/source \
  --keep-archive
```

解压后的 provider 应位于 `data/source/qlib_bin/`。构建脚本不会把服务器绝对路径写入
生成产物。

### 2. 一键生成全部派生数据

```bash
python scripts/prepare_data.py \
  --provider data/source/qlib_bin \
  --data-root data \
  --datasets short_csi300 short_csi800_direct
```

该命令依次完成：

1. `build_samplers.py`：构建 Alpha158 + Market63 + label 的 8 日窗口 sampler；
2. `build_responsibility_features.py`：由价量数据生成 14 维股票状态和 8 维规则分量；
3. `build_direction_scales.py`：只使用 train 段计算四个责任方向的尺度；
4. `build_data_manifest.py`：生成相对路径、文件大小、数据形状和 SHA-256 清单；
5. `verify_data.py`：检查最终数据合同。

所有输出都写入 `data/`，并被 Git 忽略。脚本拒绝覆盖已有 sampler、特征、尺度或
`manifest.json`；要重新构建时请使用新的 `--data-root`，保留旧数据作为独立版本。

其中 `short_csi800_direct` 严格沿用冻结 provider 的作者兼容直接 CSI800 定义，保留
已知的早期历史成分截断，不在构建时改成 CSI300 与 CSI500 的重新合成股票池。

若中间某一步失败，已有部分会保留。可查看各子命令的 `--help`，从失败步骤在一个
新的输出目录重新执行，而不必修改模型代码。

## 派生数据的含义

- `stock_state.npy`：每个股票—日期样本的 14 维状态，包括共同/残差收益分量、
  市场模型统计量、残差波动与成交量状态；其中连续字段只用 train 段统计量标准化。
- `rule_components.npy`：共同与残差收益在 1、5、20、60 日窗口下的 8 个规则分量。
- `direction_scale.json`：四类责任方向在 train 段、逐日截面中心化后的 RMS 尺度。

它们不是人工标签，也不是必须从旧机器复制的模型结果；它们都能由同一 provider 和
本仓库脚本重新生成。直接放进数据包只是为了让另一台机器省去一次耗时重建。

## 打包给另一台机器

数据构建并校验完成后，可生成新的私有迁移包：

```bash
python scripts/package_data.py \
  --data-root data \
  --output ../责任机制股票预测_实验数据包_v2.tar.gz
```

命令会同时生成 `.sha256` 文件。数据包和 SHA 文件上传个人网盘即可；不要把数据包、
provider、sampler 或访问凭据提交到 GitHub。

## 结果一致性的边界

两台机器只需要保持代码 commit、数据 manifest 和实验 YAML 一致。不同 GPU、驱动、
CUDA 或底层算子可能让训练轨迹和最终指标略有差异，因此不要求模型结果逐位相同。
应分别记录每台机器的 seed 结果，再在同一评价口径下汇总比较。

代码许可不自动覆盖行情或指数成分数据。这里的源 URL 和校验值用于个人研究环境迁移；
向公众再分发原始或派生数据前，应另行确认数据许可。
