# Experiment summaries

这个目录用于在两台个人机器之间同步轻量实验摘要，例如配置、六项指标、
五种子汇总和简短结论。不要放入 checkpoint、prediction、sampler、Qlib 二进制
产物或完整日志。

建议每个实验使用独立子目录：

```text
results/experiments/<experiment-id>/
  config.yaml
  summary.csv
  summary.md
```

推荐用 `scripts/export_progress.py` 从本机汇总目录生成；该脚本会移除逐 seed CSV
中的本机 `run_dir`，并记录代码 commit、机器别名和配置文件。不同机器或不同实验
使用不同 `experiment-id`，不要共同覆盖一个目录。
