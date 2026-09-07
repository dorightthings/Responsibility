# Third-party notices

本文件说明 Responsibility 发布候选中可能复用或依赖的第三方内容。它不是第三方许可证原文的替代品；发布前应以实际纳入仓库的文件为准再次核对。

## MASTER

模型主干和部分训练接口沿用了或改写自 MASTER（Market-Guided Stock Transformer for Stock Price Forecasting）的公开实现。

- 原项目：<https://github.com/SJTU-DMTai/MASTER>
- 论文：Tong Li et al., “MASTER: Market-Guided Stock Transformer for Stock Price Forecasting,” AAAI 2024.
- 原公开代码许可证：MIT License。
- 原许可证版权声明：Copyright (c) 2025 Data Management Technology and AI。

凡直接保留或改写的上游代码，应继续保留其原始版权和许可证通知。Responsibility 新增代码的 MIT 许可不会取消上游作者的权利与署名要求。

## Microsoft Qlib

训练数据对象兼容和投资组合回测依赖 Microsoft Qlib / PyQlib。

- 项目：<https://github.com/microsoft/qlib>
- 当前冻结版本：PyQlib 0.9.7。
- Qlib 由其权利人按自身许可证发布；请以安装版本随附许可证为准。

## Python 科学计算依赖

运行环境还依赖 PyTorch、NumPy、pandas、SciPy、scikit-learn 和 PyYAML。各依赖分别受其自身许可证约束，新电脑的关键版本见 `environment.yml` 和 `pyproject.toml`，旧服务器配置见 `envs/server-legacy.yml`。完整安装版本以本机环境记录为准。

## 数据与模型产物

本仓库的 MIT License 仅覆盖明确由本项目发布并有权许可的软件代码，不自动覆盖：

- 股票行情、指数成分和基准指数数据；
- 从外部数据源构建的 sampler、责任特征或 Qlib provider；
- 第三方预训练权重、图表、论文文本或商标；
- 用户在本地生成的预测、checkpoint 和回测结果。

数据包发布者必须在上传前确认原始数据来源、再分发权、署名要求和适用地域限制，并在数据包内附独立的数据来源与许可记录。来源或再分发权不明确的数据不应随公开 Release 上传。
