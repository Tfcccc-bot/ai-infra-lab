# PD v2 → SGLang Fork 实施计划（FORK_PLAN）

> 本文档是「把 KsanaLLM PD v2 思想落到 SGLang fork」的**落地计划**。
> 原则：**文档先行、边界清晰、TDD 驱动、任务可独立验收、不大规模重构核心代码**。
> 配套文档：`DEVLOG.md`（逐模块技术记录）、`WORKLOG.md`（工作日志）、`TODOS.md`（勾选清单）。

---

## 0. 文档目的与纪律

1. **文档先行**：在动 fork 代码前，先把边界、单测预期、任务拆解写清楚并 review。
2. **TDD 驱动**：每个任务**先写/扩单测 → 定义预期结果 → 跑红 → 实现 → 跑绿 → 记 DEVLOG**。
3. **不大规模重构**：只在 SGLang 已有 PD 底座上加**隔离的小 hook**，不重写 connector、不搬内部实现。
4. **任务可判断完成**：每个任务都有明确「验收标准」，满足即可勾掉，不依赖后续任务。

---

## 1. 背景与已就绪资产

纯 Python 原型（`topology / routing / tests 状态机 / benchmarks / analysis / patches`）已按 TDD 完成，**48 单测全绿**，作为 fork 落地的 **spec / 参考实现**：

| 原型模块 | 已验证能力 | 在 fork 中的落点 |
|---|---|---|
| M1 topology | GPU↔NUMA↔NIC 图、per-GPU IB 映射、affinity、评分 | `--disaggregation-ib-device` 自动映射、worker 绑定 |
| M2 状态机 | 8 状态生命周期 + 4 不变量（1000 次随机稳定） | 多 scheduler room 生命周期集成测试（fake backend） |
| M3 routing | TopologyAwarePolicy / NaivePolicy + 离散事件模拟 | Router 的 (P,D) 配对策略 |
| M4 benchmarks | TTFT/TPOT/ITL P50–P99/吞吐/KV 指标 | 真机指标采集 |
| M5 analysis | 8 阶段 trace 解析 + compare 报告 | 真机 trace 解析与 A/B 对比 |
| M6 patches | manifest 校验（fork_repo/branch/commit/intent…） | 每笔 fork 提交补一条 entry |

Fork 信息（已确认）：`Tfcccc-bot/sglang`，基线 commit `67853c5`，分支 `pd-v2-topology`，目标参数 `--disaggregation-ib-device`。

### 1.1 现实校验（2026-08-26 迁移启动时复核）

- 实际 fork 的 disaggregation 子系统比本文档初拟时更成熟：
  - `disaggregation_ib_device` 已原生支持 per-GPU JSON 映射（`{"0":"mlx5_0","1":"mlx5_1"}`）与 JSON 文件路径；`_validate_ib_devices` 已对 `/sys/class/infiniband` 校验。**T3 的 `auto` 是在此之上的可加 hook，不是另起炉灶。**
  - SGLang PD 以 **bootstrap 连接模型**为主（decode 经 `bootstrap_addr` 连 prefill），**没有计划假设的单一「Router (P,D) 配对策略」调用点**。因此 **T5 需重新定位 hook 点**：`score_pair` 更现实的落点是「decode server 选择 prefill server / 多 prefill 间选源」或作为可插拔 scheduler 策略，而非替换一个不存在的 Router 配对主流程。
  - `srt/utils/numa_utils.py` 已有 NUMA 绑定基建（`SGLANG_SET_CPU_AFFINITY` 等）。**T4 的增量是「按 worker 角色（transfer/bootstrap/staging）做拓扑感知的 NUMA 本地绑定」，不是重写绑定机制。**
- 结论：T1–T3 按原计划落地；T4/T5 复用已移植的纯函数，但 hook 点需在真机定位、并入验证阶段。

---

## 2. 边界（明确做什么 / 不做什么）

### 2.1 仓库边界

- **本仓库**（`ai-infra-lab/projects/pd_v2`）：实验脚本、配置、基准、单测、分析、patches 指针。**不存大段 fork 源码**。
- **SGLang fork**（`Tfcccc-bot/sglang: pd-v2-topology`）：核心改动，保持**小提交、可上游**。

### 2.2 代码边界（第一批，关键）

| 动作 | 范围 | 不动 |
|---|---|---|
| per-GPU IB 映射 | disaggregation 启动参数解析 + 拓扑查询替换静态全局 IB 列表 | Mooncake/NIXL connector 内部实现 |
| worker 绑定 | transfer/bootstrap/staging worker 启动处加 NUMA 本地 CPU affinity | worker 内部逻辑 |
| Router 配对 | 新增 `TopologyAwarePolicy` 类，主流程留 `NaivePolicy` 对照 | Router 调度主流程 |
| trace/指标 | 启动脚本采集 + 阶段对齐 M5 的 8 阶段 | SGLang 指标内部实现 |

### 2.3 明确不做（Do-not，第一批）

- 不复制 KsanaLLM 内部实现或单测（基于思想重新实现）。
- 不重写 SGLang 已有 Mooncake/NIXL connector。
- 不同时改 Router + 传输层 + scheduler + speculative decoding。
- 无 baseline 数据前，**不宣称** NUMA / blockwise 一定提升性能。
- fast-blockwise / stage pool（第二阶段）**不在本计划边界内**，仅预留接口与文档。

---

## 3. 单测实现计划（TDD 纪律）

### 3.1 分层

- **纯函数单测**（离线可跑）：直接复用 M1–M6 的 `tests/test_*.py`，fork 改动先在本仓库用注入数据验证。
- **集成单测**（需 fork）：标记 `@pytest.mark.sglang_integration` + `skip_without_sglang` fixture 兜底；fork 不可用时 skip 而非 fail。

### 3.2 复用与新增

- 每个 fork 任务先在 `tests/` 下**写一条对应单测**（定义预期结果），再改 fork 代码。
- 复用 M2 的 4 条不变量、M3 的「auto 永不被 dominate」、M1 的「不满足同 NUMA 显式降级」作为**验收断言**。

### 3.3 必守不变量清单

1. （M2）状态只走白名单转换；资源只释放一次；失败不污染其它 room；无孤儿 slot。
2. （M3）拓扑感知策略每次决策都不被可用备选 dominate；平均路径成本 ≤ 朴素策略。
3. （M1）per-GPU IB 无同 NUMA 时 `downgraded=True` 且 `reason` 记录，绝不静默跨 NUMA。
4. （M6）每条 fork 提交都在 `patches/manifest.json` 登记且 `validate` 通过。

---

## 4. 任务拆解（可独立验收）

> 每个任务自带：目标 / 边界 / 单测计划（预期结果）/ 验收标准（可判断完成）/ 落点 / 关联模块。

### T1 — Fork 接入与基线锁定
- **目标**：本地拿到 fork 并钉死基线。
- **边界**：`git clone` 自己账户下的 fork → `git remote add upstream https://github.com/Tfcccc-bot/sglang.git` → `git fetch upstream 67853c5` → `git checkout -b pd-v2-topology 67853c5`。不动 `main`。
- **单测计划**：无代码单测；用 `python -c "import sglang"` 验证可导入。
- **验收**：分支 = `pd-v2-topology`，HEAD = `67853c5`，`import sglang` 成功。可判断完成。

### T2 — 真机拓扑探测接入 M1 topology
- **目标**：真机 `env_probe` 产出的 JSON 能被 M1 正确解析。
- **边界**：`launcher/env_probe.sh` 真机产出 `topology-<host>.json`；M1 `probe.py` 的 `ib_numa_map` 注入点用真机数据；不改 M1 纯函数。
- **单测计划**：扩 `tests/test_topology_probe.py`，用真机 sample JSON 断言：GPU↔NUMA↔NIC 还原、`per_gpu_ib_device` 在真机满足同 NUMA 时 `downgraded=False`。
- **验收**：真机 JSON 通过 M1 单测；解析无异常。可判断完成。

### T3 — per-GPU IB 设备自动映射落 fork
- **目标**：用拓扑图自动映射 per-GPU IB，替代静态全局列表。
- **边界**：改 SGLang disaggregation 参数解析，挂 `--disaggregation-ib-device=auto`；用 M1 `topology_graph` 查询。**不碰 connector 内部**。
- **单测计划**：`tests/test_ib_device_mapping.py` 断言：给定 topology，P/D 实例拿到各自 GPU 本地 IB；缺同 NUMA 时 `downgraded=True` 且 `reason` 记录。
- **验收**：单测绿；真机 1P1D 用 `auto` 启动成功。可判断完成。

### T4 — worker CPU affinity 落 fork
- **目标**：transfer/bootstrap/staging worker 绑 NUMA 本地 CPU。
- **边界**：用 M1 `bind.cpu_affinity`；仅在 worker 启动处加绑定调用。**不改 worker 内部逻辑**。
- **单测计划**：`tests/test_affinity_bind.py` 断言：affinity mask == GPU 本地 NUMA CPUs；非法角色抛 `ValueError`。
- **验收**：单测绿；真机 `numactl --show` 验证绑定生效。可判断完成。

### T5 — Router (P,D) 配对策略落 fork
- **目标**：Router 选对用拓扑感知评分。
- **边界**：新增 `TopologyAwarePolicy`（复用 M3 `score_pair`），主流程保留 `NaivePolicy` 对照。**不改 Router 调度主流程**。
- **单测计划**：`tests/test_routing_policy.py` 断言：同 NUMA 候选优先；`rank_pairs` 升序；无候选 `ValueError`；模拟 300 请求平均成本 < 朴素。
- **验收**：单测绿；可切 Naive/TopologyAware 对照。可判断完成。

### T6 — 多 scheduler room 状态机集成
- **目标**：M2 不变量复用到 fork（fake backend）。
- **边界**：在 fork 用 `fake` backend 复用 M2 状态机场景；**不搬内部测试代码**。
- **单测计划**：`@skip_without_sglang` 集成用例跑 M2 的 11 类场景 + 1000 次随机调度；失败注入后服务仍可服务新请求。
- **验收**：集成用例绿；故障注入后无 KV 泄漏、无孤儿 slot。可判断完成。

### T7 — 统一指标 + trace 接入 fork
- **目标**：真机产出可被 M4/M5 解析的 trace。
- **边界**：启动脚本采集；trace 阶段对齐 M5 的 8 阶段（route/bootstrap/prealloc/prefill/transfer/commit/decode/e2e）。**不改 SGLang 指标内部**。
- **单测计划**：用 M4/M5 单测解析真机 trace JSON，断言 TTFT/ITL/KV 派生时长正确。
- **验收**：真机 trace 通过 M4/M5 解析；三类 workload 可复现。可判断完成。

### T8 — patches manifest 回填
- **目标**：每笔 fork 提交登记到 manifest。
- **边界**：仅维护 `patches/manifest.json`；commit 字段从 `TBD` 填实。
- **单测计划**：`test_patches.py` 校验每条必填 + 不重 + status 合法。
- **验收**：`validate(load_manifest())` 通过；每条对应真实 commit。可判断完成。

### T9 — A/B 对比报告
- **目标**：产出 local-NUMA / cross-NUMA / auto 三组的指标 delta。
- **边界**：用 M5 `compare` 产出报告；**不改策略代码**。
- **单测计划**：用 M5 `compare` 单测断言候选延迟全降时 `better=True`。
- **验收**：生成可读报告；auto 不被 dominate。可判断完成。

---

## 5. 文档先行与重构约束

- 第一批所有核心改动 = **加隔离 hook**，不动 SGLang 主干逻辑；任何主干改动先提 PR 讨论。
- 每个任务开始**先补单测与本文档对应小节**，再写实现；实现完补 `DEVLOG.md` 一节。
- `patches/manifest.json` 每提交一条，保证可回退、可 review。

## 6. 回退与 tag 约定

- 纯 Python 原型回退点：`pdv2-m0` … `pdv2-m6`（待首次 `git tag`）。
- fork 每完成一个 T 任务打一个轻量 tag（如 `pdv2-t3`），便于二分回退。
- 第二阶段（fast-blockwise / stage pool）单独开文档，不在本计划边界内。
