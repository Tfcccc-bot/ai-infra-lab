# PD v2 on SGLang

> 基于 KsanaLLM PD v2 的架构思想，在 SGLang 上做**增量优化**的开源探索。
> 原则：基于思想重新实现，不复制内部代码；SGLang 核心改动保持小而可上游。

## 一句话目标

不做“另一套 P/D 分离”，而是把三个思想落到 SGLang 已有的成熟 PD 底座上：

1. **拓扑感知的资源放置** —— GPU ↔ NUMA ↔ NIC/HCA 亲和性进入 P/D 配对与传输资源选择。
2. **可验证的多 scheduler 状态机** —— 用 fake backend + 故障注入，把 room / KV 预分配 / staging / 传输的生命周期压测到 100% 可解释。
3. **细粒度流水重叠** —— 第二阶段再做 fast-blockwise（分块 prefill / KV 分块传输 / 解码重叠）。

## 为什么选 SGLang 而不是重写

调研基线 SGLang main `67853c5`（2026-08-25）已具备：

- P/D 物理分离 + Mooncake / NIXL / Mori / Ascend / fake 多后端；
- DP Attention、异构 P/D TP、PP、CP（部分组合有后端约束）；
- Decode KV 预分配 slot、metadata/staging/HiCache 三层门控；
- bootstrap / waiting 超时、abort、失败注入、延迟释放；
- PD + EAGLE3 投机解码 E2E。

因此本项目**不重写 connector**，而是聚焦增量优化与可验证性。

## 目录结构

```text
pd_v2/
├── README.md            # 本文档
├── TODOS.md             # M0–M7 可勾选执行清单（从这里开始）
├── configs/             # P/D/Router 与拓扑配置
├── launcher/            # 启停脚本 + 环境探测
├── topology/            # GPU-NUMA-NIC 探测、打分、绑定
├── routing/             # P/D pair 策略与调度模拟
├── benchmarks/          # TTFT/TPOT/吞吐/KV 传输基准
├── tests/               # fake backend 状态机与故障测试
├── analysis/            # trace 与结果分析
└── patches/             # 指向 SGLang fork commit，不复制大段源码
```

核心改动放在 SGLang fork 独立分支（`Tfcccc-bot/sglang: pd-v2-topology`）；本仓库只留实验、配置、基准、测试与分析。

## 快速开始

先做 M0 环境探测：

```bash
bash launcher/env_probe.sh
```

它会输出软件版本、GPU/PCIe/NUMA/NIC/HCA/RDMA 拓扑，并在 `topology/` 下生成 `topology-<hostname>.json`。

完整路线见 [TODOS.md](./TODOS.md)，设计细节见 [研究规划文档](../../../outputs/pd_v2_sglang_implementation_plan.md)（仓库外，或随发布同步）。

## 里程碑总览

| 里程碑 | 主题 | 产出 |
|---|---|---|
| M0 | 范围冻结与基线固定 | 拓扑 JSON、启动配置、固定基线 commit |
| M1 | SGLang 原生 PD 基线 | 一键运行 + 三类 workload trace |
| M2 | 多 scheduler 状态机测试 | fake backend 故障矩阵 1000 次稳定 |
| M3 | Topology-aware PD（第一主线） | per-GPU IB 映射 + affinity + pair 评分 |
| M4 | Stage pool 与背压 | 统一资源租约 + 水位/泄漏指标 |
| M5 | Fast-blockwise（第二主线） | 传输流水化 PoC，输出一致 |
| M6 | 兼容性 + 性能矩阵 | 公开 benchmark 报告 |
| M7 | 开源交付 | 小提交 + 上游 PR + 合规检查 |
