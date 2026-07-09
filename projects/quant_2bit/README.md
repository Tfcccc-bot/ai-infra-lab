# 2-bit 极低比特量化

> **ICLR 2026 前沿论文复现** — BBQ + QAT for Reasoning

---

## 🎯 核心技术

### 1. BBQ (Bell Box Quantization)

**概率积分变换 + 均匀量化**, ICLR 2026 接收。

```
传统方法: 权重空间均匀量化 → 高斯 tail 浪费量化间隔
BBQ 方法: CDF(权重) → 概率空间均匀量化 → ICDF → 去量化
          ↓
   高密度区域自动获得更小的量化间隔
```

| Bits | BBQ MSE | Uniform MSE | BBQ 提升 |
|:----:|:-------:|:-----------:|:--------:|
| 1 | — | — | **-18 PPL** vs QuEST |
| 2 | — | — | **-5 PPL** vs QuEST |

### 2. QAT for Reasoning

**<1B tokens 微调超越 BitNet 4T tokens 从头训练**, ICLR 2026。

| 方法 | 数据量 | MATH-500 | 效率 |
|------|:------:|:--------:|:----:|
| BitNet1.58 from-scratch | 4T tokens | 43.4 | 1x |
| **QAT for Reasoning** | **<1B tokens** | **80.4** | **>4000x** |

### 3. 混合精度策略

| 层类型 | 精度 | 原因 |
|--------|:----:|------|
| Attention (Q/K/V/O) | **W4A16** | 对 KV cache 质量敏感 |
| FFN (gate/up/down) | **W2A16** | 参数多, 容忍度高 |
| Embedding / LM Head | FP16 | 直接影响输出质量 |

---

## 🚀 快速开始

```python
from bbq import BBQQuantizer, BBQLinear, compute_quantization_error

# BBQ 量化
quantizer = BBQQuantizer(num_bits=2)
quantizer.calibrate(weight)
w_hat, _ = quantizer.quantize(weight)

# 评估
error = compute_quantization_error(weight, w_hat)
print(f"BBQ 2-bit: MSE={error['mse']:.6f}, Cosine={error['cosine_sim']:.4f}")
```

---

## 📁 文件结构

```
quant_2bit/
├── bbq.py                # BBQ 概率积分变换量化
├── qat_reasoning.py      # QAT for Reasoning 两阶段流水线
├── mixed_precision.py    # 混合精度策略 + MoE 量化
├── benchmarks/
│   └── bench_quant.py    # 性能基准测试
└── README.md
```

---

## 🎓 面试追问预期

1. **"BBQ 为什么比均匀量化好？"** → 传统量化在权重空间均匀分桶, 高斯分布的 tail 区域信息密度低却占用了相同的量化间隔。BBQ 在概率空间均匀量化, 等效于高密度区域使用更小的量化步长, 同时满足 ITO。

2. **"为什么 QAT 微调比 BitNet 从头训练高效？"** → FP16 预训练模型已经学到了好的表示, QAT 只需"蒸馏"到低比特空间。BitNet 从头训练浪费了大量算力在低比特表示空间的探索上。

3. **"为什么 Attention 用 W4 而 FFN 用 W2？"** → Attention 的输出直接影响每个 token 的上下文表示, 量化误差会累积到后续所有层。FFN 的误差相对局部化。而且 FFN 参数量大 (约占 60-70%), 激进量化收益更高。
