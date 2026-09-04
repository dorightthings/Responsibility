# 复现约定

## 固定协议

- 数据：`short_csi300`、`short_csi800_direct`；no-purge 边界。
- seeds：`0,1,2,3,4`。
- 训练：Adam；最多 150 epochs；首次日均 TrainMSE `<=0.95` 时停止。
- 标签：按日去除上下各 2.5% 极端值，再做横截面样本标准差 z-score。
- 学习率：主干与 RRCA body 为 `1e-5`；CRFR 与 RRCA condition 为 `1e-4`。
- 责任选择温度：从 `1.0` 线性退火至 `0.5`，前 10 epochs 完成。
- 回测：Top30/Drop30；headline 为 `excess_return_without_cost` AR/IR。

## 最小复现顺序

1. 使用 `environment.yml` 创建冻结环境；
2. 下载数据并运行 `scripts/verify_data.py`；
3. 运行 CSI300 seed0 的 smoke；
4. smoke 成功后运行两个数据集各五 seed；
5. 对完整 prediction 运行冻结 Qlib 回测；
6. 汇总 IC、ICIR、RankIC、RankICIR、AR 和 IR 的逐 seed 数值、均值及 population standard deviation。

## 完成条件

一次正式任务至少应留下：

- 完整命令、配置和环境版本；
- seed、设备和起止时间；
- 训练历史与停止轮次；
- 单一选定 checkpoint；
- validation、bridge 和 test prediction；
- 逐日 IC/RankIC；
- Top30/Drop30 回测文件；
- `result.json` 和 `DONE` 或明确的 `FAILED`；
- 失败重试使用新的 attempt/目录，不覆盖旧产物。

## 可移植性边界

- 冻结 pickle 依赖 Python、pandas 和 Qlib 的对象兼容性，优先使用给定版本。
- 不同 GPU、驱动、CUDA、BLAS 或 Qlib 进程遍历顺序可能引入浮点差异。
- 数据集专用注意力加速只对冻结形状成立；更换数据后必须退回通用实现或重新验证。
- 当前 24 GiB GPU 已验证；显存更小的机器应保持单任务并发，必要时先仅运行 CSI300 smoke。
- Qlib provider 的日历、股票字段或指数数据不同会使 AR/IR 改变，即便 prediction 完全相同。

## 证据标记

当前方法是在同一测试期被重复评价后选出的开发路线，因此相关数值必须标记为 `development/repeated-test`。如果要声称独立验证，应冻结方法后使用未参与开发的新时间窗口或新数据集。
