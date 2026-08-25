# analysis/

trace 与结果分析。

## 计划落地

- `trace_parser.py` —— 解析 M1 生成的统一 trace：route → bootstrap → prealloc → prefill → transfer → commit → decode
- `report.py` —— 生成 benchmark 报告（含拓扑、版本、配置、误差范围）
- 结果快照（JSON/CSV/png）存档于此，供 M6 汇总

## 原则

- 原始 benchmark 数据进 `analysis/raw/`，汇总报告进 `analysis/reports/`。
- 每个结论都要能回溯到具体的实验配置 + trace。
