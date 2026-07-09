# DSpark 深度技术解析

> 面向面试的完整技术解析, 涵盖 DSpark 的核心设计原理、实现细节和与竞品的对比。

---

## 1. 问题背景

### 投机解码的基本流程

```
Step 1: Draft    — 轻量 draft model 生成 γ 个候选 token (快)
Step 2: Verify   — 目标模型一次前向验证所有候选 (并行验证)
Step 3: Accept   — 接受匹配的 token, 拒绝不匹配的
Step 4: Repeat   — 从第一个被拒绝的位置继续
```

**关键指标**: 接受长度 (Accepted Length) = 平均每次验证被接受的 token 数。

### 为什么需要 DSpark?

| 方法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| Medusa | 并行 (多线性头) | 简单, 延迟低 | 接受率低 (~65%), 无 token 间依赖 |
| EAGLE-3 | 自回归 (feature-level) | 接受率高 (~85%) | 延迟随 γ 线性增长 |
| DFlash | 并行 (KV Injection) | 延迟 O(1) | 后缀衰减严重 |
| **DSpark** | **半自回归** | **高接受率 + O(1) 延迟** | 需要额外训练置信度头 |

---

## 2. 核心架构

### 2.1 半自回归生成

```
输入: [anchor, MASK, MASK, ..., MASK]  (1 + γ-1 个位置)
         ↓
Parallel Backbone (DFlash, 2 层) — 一次性前向传播
         ↓
  h_1, h_2, ..., h_γ     (隐藏状态, 每个位置独立计算但共享上下文)
  U_1, U_2, ..., U_γ     (基础 logits, 不含 token 间依赖)
         ↓
Sequential Head (Markov) — 注入 token 间一阶依赖
         ↓
  B_1, B_2, ..., B_γ     (序列增强 logits, 基于前一个 token)
         ↓
p_k(v) ∝ exp(U_k(v) + B_k(v))   (最终 draft 分布)
```

**为什么并行骨干 + 序列头 > 纯并行?**
- 并行骨干在位置 1 有更高的初始容量 (更深网络, 更多参数)
- 序列头以极小开销 (查表 + 向量乘法) 注入 token 间依赖
- 2 层 DSpark 在所有领域超越 5 层纯 DFlash

### 2.2 置信度调度

**核心思想**: 不是所有 draft token 都值得验证。

```
c_k = σ(w^T · [h_k; W_1[x_{k-1}]])    # 预测位置 k 的接受概率
a_k = Π_{i≤k} c_i                       # 前缀存活概率
     ↓
STS 校准 (Sequential Temperature Scaling) — 修正过度自信
     ↓
调度器: 贪心搜索最大化 Θ = τ · SPS(B)
```

**STS 为什么用温度缩放而不是 Platt Scaling?**
- Platt Scaling 是单调变换 (sigmoid), 改变概率的相对排序
- 温度缩放是保序变换 (monotonic power transform), 只修正幅度
- 保序性保证了调度器的贪心选择不受影响

### 2.3 三损失联合训练

| 损失 | 数学形式 | 权重 | 为什么重要 |
|------|---------|:---:|-----------|
| L_ce | `-Σ w_k log p_k^d(x_k*)` | 0.1 | 确保 draft 能预测正确 token |
| L_tv | `Σ w_k ||p_k^d - p_k^t||_1` | **0.9** | **直接最大化接受率** (TV 距离是接受率上界) |
| L_conf | `BCE(c_k, c_k*)` | 1.0 | 置信度估计的准确性 |

> L_tv 权重 0.9 >> L_ce 权重 0.1 是因为: 投机解码不要求 draft 和 target 完全一致, 只要求分布接近 (TV 距离小 → 接受率高)。

---

## 3. 与竞品的详细对比

### DSpark vs EAGLE-3

| 维度 | DSpark | EAGLE-3 |
|------|--------|---------|
| Draft 方式 | 半自回归 (并行骨干 + 序列头) | 全自回归 (逐 token) |
| 延迟 | O(1) (与 γ 无关) | O(γ) |
| 位置 1 接受率 | 更高 (更深网络) | 较低 (单步前向) |
| 后缀接受率 | 通过序列头缓解衰减 | 自回归天然稳定 |
| 训练策略 | Off-policy (但用 TV 损失补偿) | On-policy |
| 置信度调度 | ✅ (核心创新) | ❌ (固定长度验证) |
| 适用场景 | 高并发生产环境 | 研究/低并发场景 |

### DSpark vs MTP (DeepSeek-V4 之前的生产基线)

| 维度 | DSpark | MTP-1 |
|------|--------|-------|
| Draft 长度 | 动态 (1-7, 由调度器决定) | 固定 1 |
| 吞吐提升 | +51-85% | Baseline |
| 高并发适应 | 负载自适应缩减 | 固定 1 token |

---

## 4. 我们的优化

### 4.1 DSpark × EAGLE-3 特征融合

将 EAGLE-3 的 Multi-Layer Feature Fusion 集成到 DSpark Parallel Backbone:

```python
# 从目标模型提取 [4, 8, 16] 层的中间特征
target_features = target_model.get_intermediate_features(input_ids, layers=[4,8,16])

# 融合到 draft model 的隐藏状态
hidden_states = dspark_backbone(anchor_ids)
hidden_states += 0.1 * feature_fusion(target_features)  # 残差连接
```

### 4.2 混合 Draft 策略

| 序列长度 | 策略 | 原因 |
|----------|------|------|
| < 512 | DSpark 半自回归 | 短序列接受率高, 并行骨干效率好 |
| ≥ 512 | EAGLE-3 自回归 | 长序列依赖更重要, 自回归更稳定 |

### 4.3 消费级 GPU 适配

- γ=5 (vs 论文 γ=15): 适应 RTX 4090 24GB 显存
- KV cache FP16 存储
- 批量验证融合到单个 kernel

---

## 5. 面试高频追问

### Q1: "为什么 DSpark 的并行骨干 + 序列头比纯并行 (DFlash) 好?"

**A**: 纯并行草稿器每个位置独立预测, 无法建模 token 间依赖, 导致"多模态碰撞"——比如 "of course" 和 "no problem" 可能混合成 "of problem"。DSpark 的 Markov 序列头通过一阶转移矩阵 B(x_{k-1}, ·) 注入依赖, 以极小的计算开销 (O(r·V) vs O(γ·model_size)) 解决这个问题。

### Q2: "置信度调度器的核心优化问题是什么?"

**A**: 最大化 Θ = τ · SPS(B), 其中 τ 是期望接受 token 数, SPS(B) 是 batch size B 下的引擎吞吐量。这是一个 NP-hard 的组合优化问题, 但 DSpark 通过置信度头的存活概率 + 贪心搜索在 O(N log N) 内找到近似最优解。关键洞察: SPS 随 B 增大先升后降 (GPU 利用率 vs 延迟), 所以不是验证越多越好。

### Q3: "在消费级 GPU 上部署遇到的最大挑战?"

**A**: 三个: (1) 显存限制——论文 γ=15 需要更多 KV cache, 我们压缩到 γ=5; (2) SPS 曲线差异——需要针对 RTX 4090 重新 profiling; (3) 无 Blackwell TMA——我们使用 async copy 替代, 牺牲了部分 Load warp 的效率。

### Q4: "TV 损失为什么比 CE 损失更重要?"

**A**: 投机解码验证阶段用 rejection sampling: 从 draft 分布 p^d 采样 token x, 以概率 min(1, p^t(x)/p^d(x)) 接受。如果 p^d 和 p^t 的 TV 距离小, 接受率就高。CE 损失只优化 argmax 一致性, TV 损失直接优化分布匹配, 所以后者对接受率的提升更直接。
