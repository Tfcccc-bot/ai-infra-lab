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

本项目对 **2025-2026 年 AI Infra 领域最新技术** 进行深入研究和工程实现，涵盖投机解码、极低比特量化、Attention Kernel 优化与推理系统四大方向，旨在推动大模型推理效率的前沿探索。

### 技术路线图

| 方向 | 技术 | 来源 | 状态 |
|------|------|------|:----:|
| **投机解码** | DSpark: Confidence-Scheduled Speculative Decoding | DeepSeek × PKU, 2026.06 | 🚧 |
| **投机解码** | EAGLE-3: On-Policy Multi-Layer Feature Fusion | NeurIPS 2025 | 📋 |
| **极低比特量化** | BBQ: Bell Box Quantization | ICLR 2026 | 🚧 |
| **极低比特量化** | QAT for Ultra-Low-Bit Reasoning LLMs | ICLR 2026 | 📋 |
| **投机解码** | DSpark: Confidence-Scheduled Speculative Decoding | DeepSeek × PKU, 2026.06 | 🚧 |
| **投机解码评测** | Speculative Decoding: Performance or Illusion? | MLSys 2026 (Ion Stoica 组) | 🚧 |
| **极低比特量化** | BBQ: Bell Box Quantization | ICLR 2026 | 🚧 |
| **极低比特量化** | QAT for Ultra-Low-Bit Reasoning LLMs | ICLR 2026 | 📋 |
| **KV Cache 量化** | Kitty: 2-bit KV Cache (Dynamic Channel Precision) | MLSys 2026 | 🚧 |
| **Attention Kernel** | FlashAttention-4: Warp-Specialized Pipeline | Princeton × Together AI, 2026.03 | 🚧 |
| **推理系统** | Executor Build-Ahead 快速路径 | 腾讯 KsanaLLM 实习 | ✅ |
| **推理系统** | Beyond the Buzz: P/D Disaggregation 祛魅 | MLSys 2026 | 🚧 |

---

## 🏗️ 项目结构

```
ai-infra-lab/
├── projects/
│   ├── dspark/                  # DSpark 投机解码框架
│   ├── specdec_illusion/        # 投机解码评测：Performance or Illusion?
│   ├── quant_2bit/              # 2-bit 极低比特量化
│   ├── kitty_kv2bit/            # Kitty 2-bit KV Cache 量化
│   ├── flash_attn/              # FlashAttention-4 机制复现
│   ├── ksanallm-build-ahead/    # KsanaLLM Build-Ahead 核心实现
│   └── pd_disaggregation/       # P/D 分离祛魅
├── common/                      # 共享工具
├── docs/                        # 技术深度文档
└── tests/                       # 测试
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

## 📊 方向二：投机解码评测（Performance or Illusion?）

> 来源：Speculative Decoding: Performance or Illusion?（MLSys 2026, Ion Stoica 组）

### 核心技术

- **Throughput-Optimal 评测设定**：在固定系统吞吐约束下区分"真实加速"与"仅压低单请求延迟、却牺牲系统吞吐"的假象。
- **开销-收益剖析**：剖析 draft 模型的额外算力成本是否被接受率 / 验证 step 数真正覆盖。

### 我们的优化

1. **迁移到自研部署选型**：将评测方法用于 **EAGLE / MTP / DSpark** 三套 draft 策略的线上参数决策。
2. **数据驱动决策**：为 γ、draft 模型规模、budget 提供基于实测接受率-开销权衡的选型依据，避免"看着快、实则亏"。

---

## 🔬 方向三：2-bit 极低比特量化

### 核心技术

ICLR 2026 是极低比特量化的爆发年，我们聚焦两个最具颠覆性的工作：

- **BBQ**: 概率积分变换将高斯分布拉平为均匀分布再量化，2-bit 比 QuEST 降 PPL 5 点
- **QAT for Reasoning**: <1B tokens 微调让 2-bit Qwen3-14B 在 MATH-500 达 80.4，超越 BitNet1.58 的 4T tokens 从头训练

### 我们的优化

1. BBQ + QAT for Reasoning 两阶段流水线融合
2. MoE 模型 2-bit 量化 (Attention W4 + FFN W2)
3. 混合精度策略自动搜索

---

## 💎 方向四：Kitty · 2-bit KV Cache 量化

> 来源：Kitty（MLSys 2026）— 2-bit KV Cache 量化，以动态通道精度提升保护关键通道。

### 核心技术

- **Dynamic Channel-wise Precision Boost**：识别对 attention 输出贡献高的关键通道，为其分配更高有效精度，将量化误差集中在次要通道。
- 在 **2-bit 极限压缩**下维持长上下文精度，避免均匀 2-bit 量化对关键维度"一刀切"带来的精度崩塌。

### 我们的优化

1. **Kitty × BBQ 联合压缩**：将 Kitty 的动态通道保护机制与 Momenta 的 **BBQ 概率积分变换量化**结合，用于 KsanaLLM 长上下文 KV Cache 压缩（KV INT2 + 通道重要性重排）。
2. **显存-精度曲线评估**：对比 INT8 baseline，量化极致压缩下的显存节省与长文精度损失，为长文档 / 端侧场景提供压缩选型依据。

---

## ⚡ 方向五：FlashAttention-4 机制复现

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

## 🏭 方向六：KsanaLLM Executor Build-Ahead

### 核心技术

来自腾讯 KsanaLLM 实习的核心产出。一种跨 step 的流水线深度优化机制：

- **双 Slot 乒乓**：两个 slot 各自独立 ModelInput/H2D stream/Sampler buffer，Preprocess 与 Forward 并行
- **Ring Buffer 协议**：`int64[1 + 2×max_batch_size]` 布局，ReplaceLastTokenKernel + WriteRingKernel 跨步传递采样结果
- **Placeholder 机制**：Engine 用 `kFastPathPlaceholderTokenId = -1` 占位，Executor 侧 Forward 前回填
- **FastPathController**：启动期 13 项硬条件 Gate，不做运行时判断

### 设计文档

详见 [projects/ksanallm-build-ahead/](projects/ksanallm-build-ahead/) 及内部设计文档 `build-ahead-mtp-serial.md`。

---

## 🔀 方向七：P/D 分离祛魅（Beyond the Buzz）

> 来源：Beyond the Buzz: A Pragmatic Take on Inference Disaggregation（MLSys 2026）

### 核心技术

- **祛魅式评估**：指出 P/D 分离并非总是有益——在 prefill/decode 负载失衡、KV 传输带宽受限等场景下反而引入额外开销。
- **部署决策框架**：基于负载特征（请求长度分布、并发模式、硬件拓扑）判断何时该 P/D 分离、何时该混合部署。

### 我们的优化

1. **结合 KsanaLLM 真实业务负载**：对 chat / 长文档 / code 三类业务分别做 P/D 分离 vs 混合部署的 break-even 分析。
2. **数据支撑选型**：为团队 P/D 架构选型提供数据依据，避免盲目跟风导致的过度工程化。

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

# 运行 Kitty 2-bit KV Cache 复现
python projects/kitty_kv2bit/repro.py --model Qwen/Qwen3-8B --kv_bits 2

# 运行投机解码评测（覆盖 EAGLE / MTP / DSpark）
python projects/specdec_illusion/eval.py --method eagle --gamma 5 --max_len 4096

# 运行 P/D 分离 break-even 分析
python projects/pd_disaggregation/breakeven.py --workload chat --kv_bw 50
```

---

## 📚 技术文档

- [DSpark 深度解析](docs/dspark_deep_dive.md)
- [2-bit 量化综述](docs/quant_2bit_survey.md)
- [FlashAttention-4 机制分析](docs/fa4_analysis.md)

---

## 📄 License

MIT License
