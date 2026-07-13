# Kitty · 2-bit KV Cache 量化

> 来源：Kitty（MLSys 2026）— 2-bit KV Cache 量化，以动态通道精度提升保护关键通道。

## 核心技术

- **Dynamic Channel-wise Precision Boost**：识别对 attention 输出贡献高的关键通道，为其分配更高有效精度（动态提升），将量化误差集中在次要通道。
- 在 **2-bit 极限压缩**下维持长上下文精度，避免传统均匀 2-bit 量化对关键维度"一刀切"带来的精度崩塌。

## 复现内容

- 复现 Kitty 的通道重要性评估 + 动态精度分配流程。
- 在长上下文（long-context）基准上对比 2-bit vs INT8 baseline 的显存占用与精度损失曲线。

## 我们的优化

1. **Kitty × BBQ 联合压缩**：将 Kitty 的动态通道保护机制与 Momenta 的 **BBQ 概率积分变换量化**结合，用于 KsanaLLM 长上下文 KV Cache 压缩（KV INT2 + 通道重要性重排）。
2. **显存-精度曲线评估**：对比 INT8 baseline，量化极致压缩下的显存节省与长文精度损失，为长文档 / 端侧场景提供压缩选型依据。

## 评测计划

- 数据集：长上下文 perplexity 基准 + 下游 long-doc QA。
- 指标：KV Cache 显存占用、长文 PPL 退化、端到端吞吐。
- 状态：复现 + KsanaLLM 适配推进中 🚧

## 快速开始

```bash
# 运行 Kitty 2-bit KV Cache 复现
python projects/kitty_kv2bit/repro.py --model Qwen/Qwen3-8B --kv_bits 2
```
