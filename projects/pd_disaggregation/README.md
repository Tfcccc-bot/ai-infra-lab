# Beyond the Buzz · Inference Disaggregation（P/D 分离祛魅）

> 来源：Beyond the Buzz: A Pragmatic Take on Inference Disaggregation（P/D 分离祛魅, MLSys 2026）— 对 P/D 分离部署的祛魅式评估。

## 核心技术

- **祛魅式评估**：指出 P/D 分离并非总是有益——在 prefill/decode 负载失衡、KV 传输带宽受限等场景下反而引入额外开销。
- **部署决策框架**：基于负载特征（请求长度分布、并发模式、硬件拓扑）判断何时该 P/D 分离、何时该混合部署。

## 复现内容

- 复现该文的决策逻辑与 break-even 分析流程。
- 在可控负载下复现"分离反而亏"的边界条件。

## 我们的优化

1. **结合 KsanaLLM 真实业务负载**：对 chat / 长文档 / code 三类业务分别做 P/D 分离 vs 混合部署的 break-even 分析。
2. **数据支撑选型**：为团队 P/D 架构选型提供数据依据，避免盲目跟风导致的过度工程化。

## 评测计划

- 业务负载：chat（短 prompt 高并发）/ 长文档（长 prefill）/ code（长上下文）。
- 指标：TTFT、TPOT、端到端吞吐、KV 传输开销占比、break-even 拐点。
- 状态：决策框架复现 + KsanaLLM 业务负载分析推进中 🚧

## 快速开始

```bash
# 运行 P/D 分离 break-even 分析
python projects/pd_disaggregation/breakeven.py --workload chat --kv_bw 50
```
