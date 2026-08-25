# benchmarks/

TTFT / TPOT / 吞吐 / KV 传输基准。

## 计划落地

- `workloads/` —— 三类固化 workload：short-chat、long-prefill、long-decode
- `run_bench.sh` —— 一键跑 baseline，输出统一指标
- 指标：TTFT、TPOT、ITL P50/P95/P99、吞吐、KV transfer bytes/latency、bootstrap/alloc wait

## 原则

- 结果必须带硬件拓扑、版本、配置、误差范围（M6 硬性要求）。
- 每次结果进 `analysis/`，不在本目录堆临时数据。
