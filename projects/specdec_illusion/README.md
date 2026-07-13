# Speculative Decoding: Performance or Illusion?

> 来源：Speculative Decoding: Performance or Illusion?（MLSys 2026, Ion Stoica 组）— 对投机解码"真实加速"的理性评测。

## 核心技术

- **Throughput-Optimal 评测设定**：在固定系统吞吐约束下区分"真实加速"与"仅压低单请求延迟、却牺牲系统吞吐"的假象。
- 剖析 draft 模型的**额外算力成本**是否被接受率 / 验证 step 数真正覆盖——若 draft 开销未被接受率赚回，则只是把延迟从一处转移到另一处。

## 复现内容

- 复现该文的评测框架：在不同 batch size、上下文长度、draft budget 下实测接受率与端到端加速比。
- 量化"性能"与"假象"的边界。

## 我们的优化

1. **迁移到自研部署选型**：将评测方法用于 **EAGLE / MTP / DSpark** 三套 draft 策略的线上参数决策。
2. **数据驱动决策**：为线上投机解码参数（γ、draft 模型规模、budget）提供基于实测接受率-开销权衡的选型依据，避免"看着快、实则亏"。

## 评测计划

- 维度：batch size × 上下文长度 × draft budget。
- 指标：acceptance length、端到端加速比、系统吞吐变化（throughput-optimal 下的净收益）。
- 状态：评测框架复现 + 自研策略迁移推进中 🚧

## 快速开始

```bash
# 运行投机解码评测（覆盖 EAGLE / MTP / DSpark）
python projects/specdec_illusion/eval.py --method eagle --gamma 5 --max_len 4096
```
