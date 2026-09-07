# Responsibility 开发约定

- 先确认当前主机、用户、目录和 Git 根；所有项目产物写入本仓库内部。
- 当前开发起点是 v0.4.0 的分组学习率 CRFR＋RRCA，来源提交 e7bcbb9。
- 一个 Adam、两个参数组：主干＋RRCA 适配器 1e-5；CRFR＋RRCA rho 1e-4。
  CSI300、CSI800 相同。不要自行换成统一学习率或加入四机制时间记忆。
- 默认 environment.yml 面向新主力机的 Python 3.10 / PyTorch 2.7.1+cu128；
  envs/server-legacy.yml 仅保留旧服务器环境。新显卡尚需目标机实测。
- 数据、股票池、划分、标签、beta、停止方式、seed 和评价口径以 configs/、
  docs/REPRODUCIBILITY.md 为准。科学设置变更须有用户授权并单独记录。
- 回测使用 Top30/Drop30，不计手续费的超额收益 AR/IR。
- 开发期允许重复测试和失败重试，标记 development/repeated-test；不得声称未见测试验证。
- 新实验、失败重试使用独立 outputs/ 子目录，保留旧 checkpoint、预测、日志和结果。
- 先检查已有进程和显存；新电脑默认单卡单任务，不干扰其他进程，不擅自拆分日截面。
- GitHub 只提交代码、配置、环境与说明；数据、results/、outputs/、权重、预测、
  日志和凭据均不提交。比较以新电脑实际运行结果为准。
- 不覆盖用户未提交修改，不 reset --hard、clean 或强制推送；推送指定远端和分支。
- 保留第三方代码来源和许可证；不要把服务器私钥或登录状态迁入项目。
