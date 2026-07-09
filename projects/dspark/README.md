# DSpark: Confidence-Scheduled Speculative Decoding

> **DeepSeek × 北京大学, 2026.06** — 论文复现与优化

DSpark 是一个投机解码框架，通过**半自回归生成 + 置信度调度**在 DeepSeek-V4 生产环境中实现了相对 MTP-1 **+51%-85% 的吞吐提升**。

---

## 🎯 核心技术

### 1. 半自回归生成 (Semi-Autoregressive Generation)

```
输入: 锚点 token + (γ-1) 个 mask token
      ↓
Parallel Backbone (DFlash) — 一次性前向传播
      ↓
  h_1, ..., h_γ  (隐藏状态)
  U_1, ..., U_γ  (基础 logits)
      ↓
Markov Sequential Head — 注入 token 间依赖
      ↓
  p_k(v) ∝ exp(U_k(v) + B_k(v))
```

**关键洞察**: 并行骨干提供高初始容量, 序列头注入 token 间依赖。2 层 DSpark 在所有领域超越 5 层纯并行草稿器。

### 2. 置信度调度 (Confidence-Scheduled Verification)

```
c_k = σ(w^T · [h_k; W_1[x_{k-1}]])     # 置信度预测
a_k = Π_{i≤k} c_i                        # 前缀存活概率
     ↓
STS 校准 (Sequential Temperature Scaling)
     ↓
硬件感知前缀调度器 (贪心搜索最大化 Θ = τ · SPS(B))
```

**效果**: 对话领域通过置信度剪枝, 接受率从 45.7% 提升至 **95.7%** (+50%)。

### 3. 三损失联合训练

| 损失 | 公式 | 权重 | 作用 |
|------|------|:---:|------|
| L_ce | `-Σ w_k · log p_k^d(x_k*)` | 0.1 | 预测正确 token |
| L_tv | `Σ w_k · ∥p_k^d - p_k^t∥₁` | 0.9 | 直接最大化接受率 |
| L_conf | `BCE(c_k, c_k*)` | 1.0 | 置信度校准 |

> 位置加权: `w_k = exp(-(k-1)/γ)`, 强调块前部位置。

---

## 🚀 快速开始

```bash
# 训练 draft model
python training/train_draft.py

# 运行 demo
python demo.py --model Qwen/Qwen2.5-0.5B-Instruct --draft-len 5 --compare-baseline

# 仅测试 draft 生成
python -c "
from training.train_draft import create_trainer
trainer = create_trainer()
tokens, logits, conf = trainer.generate_draft(torch.randint(0, 1000, (2, 1)))
print(f'Draft tokens: {tokens.shape}, Confidence: {conf.shape}')
"
```

---

## 📊 性能指标

### 离线基准 (Qwen3-4B)

| 方法 | 接受长度 | vs Eagle3 | vs DFlash |
|------|:-------:|:---------:|:---------:|
| Eagle3 | 3.63 | — | — |
| DFlash | 4.17 | — | — |
| **DSpark** | **4.77** | **+30.9%** | **+16.3%** |

### 分领域接受长度

| 领域 | Eagle3 | DFlash | DSpark | 提升 (vs Eagle3) |
|------|:------:|:------:|:------:|:----------------:|
| 数学 | 4.62 | 5.13 | **5.70** | +23.4% |
| 代码 | 3.86 | 4.44 | **5.12** | +32.6% |
| 对话 | 2.40 | 2.95 | **3.49** | +45.4% |

---

## 🏗️ 我们的优化

### 1. DSpark × EAGLE-3 特征融合

将 EAGLE-3 的 Multi-Layer Feature Fusion 集成到 DSpark Parallel Backbone:

```python
# 从目标模型提取中间层特征
target_features = target_model.get_intermediate_features(
    input_ids, layers=[4, 8, 16]
)
# 融合到 draft 模型
hidden_states += feature_fusion(target_features)
```

### 2. 混合 Draft 策略

| 序列长度 | 策略 | 原因 |
|----------|------|------|
| < 512 | DSpark 半自回归 | 并行骨干效率高, 短序列接受率高 |
| ≥ 512 | EAGLE-3 自回归 | 长序列依赖建模更重要 |

### 3. 消费级 GPU 优化

- 默认 γ=5 (vs 论文的 γ=15), 适配 RTX 4090 24GB
- KV cache 使用 FP16 存储
- 批量验证融合到单个 CUDA kernel

---

## 📁 文件结构

```
dspark/
├── dspark_draft.py       # 半自回归草稿模型 (Parallel Backbone + Sequential Head)
├── confidence_head.py    # 置信度头 + STS 校准
├── scheduler.py          # 硬件感知前缀调度器
├── training/
│   ├── train_draft.py    # 训练脚本 + 混合 Draft 模型
│   └── loss.py           # 三损失联合训练
├── demo.py               # 端到端演示
└── README.md             # 本文件
```

---

## 📚 参考

- [DSpark 论文](https://arxiv.org/abs/2607.05147)
- [DeepSeek-V4 技术报告](https://arxiv.org/abs/2604.xxxxx)
- [EAGLE-3 论文](https://arxiv.org/abs/2503.01840)
- [SpecForge 框架](https://github.com/sgl-project/SpecForge)
