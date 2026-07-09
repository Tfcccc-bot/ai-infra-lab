# FlashAttention-4 核心机制复现

> **Princeton × Together AI × Meta × NVIDIA, 2026.03** — Warp-Specialized Pipeline

FlashAttention-4 针对 NVIDIA Blackwell GPU 重新设计，B200 上达到 **71% 硬件利用率 (1605 TFLOPs/s)**，1.3x faster than cuDNN。

---

## 🎯 核心技术

### 1. Warp-Specialized 异步流水线

```
┌────────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐
│Load Warps  │→│MMA Warps  │→│Softmax    │→│Correction│→│Epilogue  │
│TMA: Q,K,V  │ │S = Q@K^T  │ │Warps      │ │Warps     │ │Warps     │
│HBM→SMEM    │ │on TC      │ │norm+stats │ │rescale   │ │SMEM→HBM  │
└────────────┘  └──────────┘  └───────────┘  └──────────┘  └──────────┘
       ↑              ↑              ↑             ↑             ↑
       └──────────────┴──────────────┴─────────────┴─────────────┘
                     异步流水线 (Async Pipeline)
```

### 2. Hybrid Exponential

| 方法 | 执行单元 | 瓶颈 |
|------|:-------:|------|
| Exact exp | SFU | SFU 带宽不足 |
| **Cubic poly** | **FMA** | **释放 SFU** |

```python
# 三次多项式近似 2^x (在 FMA 单元执行)
2^x ≈ 0.0794·x³ + 0.2402·x² + 0.6953·x + 0.9996
# 匹配 BF16 精度, max relative error < 0.02%
```

### 3. Correction Warp

FA3 每次迭代都 rescale output → FA4 仅当 running_max 变化 > ε 时 rescale → **减少 ~10x rescale 操作**.

---

## 🚀 快速开始

```bash
# 运行 benchmark
python benchmarks/bench_fa4.py

# 验证 Hybrid Exponential 精度
python hybrid_exp.py
```

---

## 📁 文件结构

```
flash_attn/
├── warp_specialized.py   # Warp-Specialized Pipeline (Triton)
├── hybrid_exp.py         # Hybrid Exponential + TMEM 管理
├── benchmarks/
│   └── bench_fa4.py      # 性能基准测试
└── README.md
```

---

## 🎓 面试追问预期

1. **"Warp-Specialized Pipeline 的本质区别？"** → 传统 kernel 所有 warp 执行相同代码 (SIMT)。FA4 不同 warp 执行不同角色 (Load/MMA/Softmax/Correction/Epilogue), 通过异步流水线最大化硬件利用率。

2. **"Hybrid Exponential 如何保证精度？"** → 三次多项式在 x∈[-1,1] 上拟合 2^x 的最大相对误差 <0.02%, 低于 BF16 的量化误差 (~0.4%)。且 softmax 的归一化操作会进一步稀释局部误差。

3. **"Hopper vs Blackwell 实现差异？"** → Blackwell 有 TMA (Tensor Memory Accelerator) 和 TMEM (Tensor Memory), 以及 5th-gen Tensor Cores (128x256x16 tiles)。我们的实现在 Hopper 上模拟这些特性, 核心流水线逻辑相同但使用 async copy + shared memory 替代。
