# 新机器实验约定

本仓库提供分组学习率版的开发起点，不要求异机结果相同。
当前发布不包含旧结果；比较以新主力机实际生成的结果为准。

## 当前方法身份

- 模型/优化器来源：`e7bcbb9` 的 CRFR＋stock-specific RRCA。
- 不含四机制时间记忆，不使用路由 detach，不改选择头初始化。
- 总参数量 846229。
- 一个 Adam，两个实际参数组：主干＋RRCA adapters 为 `1e-5`；
  CRFR＋RRCA rho 为 `1e-4`。两个数据集设置相同，没有学习率调度器。
- 配置保留原四个学习率字段，加载器拒绝统一学习率版的 `training.learning_rate`，
  避免错误配置被静默忽略。
- 当前协议标识：`responsibility-crfr-rrca-grouped-lr-portable-v4`。
- `60e5a58` 和 `9f70c7d` 是不同版本，不混用配置或记忆版 checkpoint。

## 数据与训练

| 项目 | 约定 |
| --- | --- |
| 数据集 | CSI300、CSI800 |
| 内部标识 | short_csi300、short_csi800_direct |
| train | 2008-01-02—2020-03-31 |
| valid | 2020-04-01—2020-06-30 |
| test | 2020-07-01—2022-12-30，共611日 |
| bridge | 2020-06-30 |
| 边界 / lookback | no-purge / 8 |
| seeds | 0、1、2、3、4 |
| 原始标签 | Ref($close, -5) / Ref($close, -1) - 1 |
| 训练标签处理 | 按日去除上下各2.5%后，按截面样本标准差z-score |
| 市场门beta | CSI300为5，CSI800为2 |
| 停止方式 | 最多150 epochs，首次日均TrainMSE <= 0.95停止 |
| 梯度裁剪 | clip value 3.0 |
| 责任选择温度 | 从1.0线性退火至0.5，前10 epochs完成 |

验证和测试保留股票样本，仅在计算指标时屏蔽缺失标签。
CSI800 沿用冻结 provider 的直接 CSI800 定义及其早期历史成分截断，
不改成重新合成的 CSI300＋CSI500 股票池。

## 评价

统一记录 IC、ICIR、RankIC、RankICIR、AR、IR。
回测使用 Qlib 0.9.7、TopkDropoutStrategy(topk=30, n_drop=30)、
close 成交价，AR/IR 为 `excess_return_without_cost`。
基准指数为 CSI300: SH000300、CSI800: SH000906。
每个数据集按五种子计算均值和总体标准差（ddof=0）。

## 本机保存的产物

正式任务至少保存命令、配置、环境与设备、seed、训练历史、停止轮数、
checkpoint、预测、逐日指标、回测、最终退出状态。
`optimizer.json` 记录实际学习率分组。
所有输出进入独立 `outputs/` 子目录；不上传结果，不覆盖旧运行目录。
失败任务可以在新 attempt/目录重新执行。

默认先在单张卡上排队；`--cpu-data-cache` 只改变缓存位置，不改变完整日截面。
不要因换机器而改标签、股票池、beta、停止规则或回测策略。

同一测试期已在开发中反复评价；新电脑重新训练仍属于 development/repeated-test，
不宣称它是未见测试期的独立验证。
