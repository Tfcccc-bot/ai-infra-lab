# AI Infra Lab 🔬

> **2026 前沿 AI Infra 技术本地复现与优化**
>
> 覆盖投机解码、极低比特量化、Attention Kernel 三大方向

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.2-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 项目概述

本项目对 **2025-2026 年 AI Infra 领域最新技术** 进行深入研究和工程实现，涵盖投机解码、极低比特量化与 Attention Kernel 优化三大方向，旨在推动大模型推理效率的前沿探索。

### 技术路线图

| 方向 | 技术 | 来源 | 状态 |
|------|------|------|:----:|
| **投机解码** | DSpark: Confidence-Scheduled Speculative Decoding | DeepSeek × PKU, 2026.06 | 🚧 |
| **投机解码** | EAGLE-3: On-Policy Multi-Layer Feature Fusion | NeurIPS 2025 | 📋 |
| **极低比特量化** | BBQ: Bell Box Quantization | ICLR 2026 | 🚧 |
| **极低比特量化** | QAT for Ultra-Low-Bit Reasoning LLMs | ICLR 2026 | 📋 |
| **Attention Kernel** | FlashAttention-4: Warp-Specialized Pipeline | Princeton × Together AI, 2026.03 | 🚧 |

---

## 🏗️ 项目结构

```
ai-infra-lab/
├── projects/
│   ├── dspark/              # DSpark 投机解码框架
│   ├── quant_2bit/          # 2-bit 极低比特量化
│   └── flash_attn/          # FlashAttention-4 机制复现
├── common/                  # 共享工具
├── docs/                    # 技术深度文档
└── tests/                   # 测试
```

---

## 🔥 方向一：DSpark 投机解码

### 核心技术

DSpark 是 DeepSeek 与北京大学 2026 年 6 月联合发布的投机解码框架，已在 DeepSeek-V4 生产环境验证。

- **半自回归生成**: Parallel Backbone (DFlash) + Markov Sequential Head
- **置信度调度**: Confidence Head + 序列温度校准 (STS) + 硬件感知前缀调度器
- **三损失联合训练**: L_ce + L_tv + L_conf，位置加权

### 性能指标

| 场景 | MTP-1 基线 | DSpark | 提升 |
|------|:---------:|:------:|:----:|
| V4-Flash @ 80 tok/s | 基线 | — | **+51% 吞吐** |
| 每用户生成速度 | 基线 | — | **+60-85%** |
| vs EAGLE-3 接受长度 | 3.63 | 4.77 | **+30.9%** |

### 我们的优化

1. **DSpark × EAGLE-3 融合**: 用 Multi-Layer Feature Fusion 增强 Parallel Backbone
2. **混合 Draft 策略**: 短序列半自回归，长序列自回归
3. **消费级 GPU 适配**: RTX 4090 kernel 优化

---

## 🔬 方向二：2-bit 极低比特量化

### 核心技术

ICLR 2026 是极低比特量化的爆发年，我们聚焦两个最具颠覆性的工作：

- **BBQ**: 概率积分变换将高斯分布拉平为均匀分布再量化，2-bit 比 QuEST 降 PPL 5 点
- **QAT for Reasoning**: <1B tokens 微调让 2-bit Qwen3-8B 在 MATH-500 达 80.4，超越 BitNet1.58 的 4T tokens 从头训练

### 我们的优化

1. BBQ + QAT for Reasoning 两阶段流水线融合
2. MoE 模型 2-bit 量化 (Attention W4 + FFN W2)
3. 混合精度策略自动搜索

---

## ⚡ 方向三：FlashAttention-4 机制复现

### 核心技术

FA4 针对 NVIDIA Blackwell 架构重新设计，B200 上达到 71% 硬件利用率 (1605 TFLOPs/s)。

- **Warp-Specialized Pipeline**: Load/MMA/Softmax/Correction/Epilogue 五类 warp 异步流水线
- **Hybrid Exponential**: FMA 单元三次多项式近似 2^x，释放 SFU
- **Correction Warp**: 仅 running max 变化足够大时才 rescale，减少 ~10x 操作

### 我们的优化

1. Hopper/Ada Lovelace 架构简化实现
2. GQA/MQA shared KV 访存优化
3. RMSNorm + RoPE kernel 融合

---

## 🚀 快速开始

```bash
# 克隆
git clone https://github.com/Tfcccc-bot/ai-infra-lab.git
cd ai-infra-lab

# 安装依赖
pip install -r requirements.txt

# 运行 DSpark demo
python projects/dspark/demo.py --model Qwen/Qwen2.5-0.5B-Instruct

# 运行 2-bit 量化
python projects/quant_2bit/bbq.py --model Qwen/Qwen2.5-0.5B --bits 2

# 运行 FA4 benchmark
python projects/flash_attn/benchmark.py --impl warp_specialized
```

---

## 📚 技术文档

- [DSpark 深度解析](docs/dspark_deep_dive.md)
- [2-bit 量化综述](docs/quant_2bit_survey.md)
- [FlashAttention-4 机制分析](docs/fa4_analysis.md)

---

## 📄 License

MIT License
