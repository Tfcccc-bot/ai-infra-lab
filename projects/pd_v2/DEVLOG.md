# PD v2 → SGLang 迁移开发记录（DEVLOG）

> 用途：按模块记录实现了什么、单测预期 vs 实际、关键效果与回退点，方便 review 与回退。
> 约定：每完成一个模块追加一节；回退点用 `git tag pdv2-m<N>` 标记。
> TDD 纪律：**先写单测（定义预期结果）→ 跑红 → 实现 → 跑绿 → 记本节 → 再下一模块**。

## 关联文档

- `FORK_PLAN.md`：**边界 + TDD 单测计划 + T1–T9 任务拆解**（fork 落地的总计划，文档先行）。
- `WORKLOG.md`：chronological 工作日志（决策 / 进展 / 阻塞）。
- `TODOS.md`：T1–T9 可勾选执行清单（从 FORK_PLAN 派生）。
- 本报告 M0–M6 的纯 Python 模块即 fork 落地的 **spec / 参考实现**，T2–T9 复用其单测与不变量。

---

## M0 — TDD 脚手架（已完成 2026-08-26）

- 新增文件：`pyproject.toml`、`tests/conftest.py`
- 关键设计：
  - pytest `pythonpath=["."]`，`topology`/`routing` 等可直接 import。
  - `tests/conftest.py` 探测 `sglang` 是否可导入；不可用则集成测试自动 skip，不阻塞纯 Python 模块。
  - marker `sglang_integration`：标记需连 fork 的端到端用例。
- 预期结果：单测可离线运行；缺 fork 时相关用例 skip 而非 fail。
- 实际结果：✅ `pytest` 可运行（19 用例全绿，含 M1）。
- 回退点：`pdv2-m0`（待首次 git tag）。

---

## M1 — topology：GPU↔NUMA↔NIC 拓扑与亲和性（已完成 2026-08-26）

- 新增文件：
  - `topology/topology_graph.py`（节点定义 + NUMA/IB 映射/距离）
  - `topology/probe.py`（解析 `env_probe.sh` 原始 JSON）
  - `topology/score.py`（P/D pair 可插拔评分纯函数）
  - `topology/bind.py`（transfer/bootstrap/staging worker CPU affinity）
  - `topology/__init__.py`
- 单测（预期结果 → 实际）：`tests/test_topology_*.py` 共 19 例
  - 节点还原：`from_dict` 还原 4 GPU/2 NUMA/2 NIC ✅
  - GPU↔NUMA：`gpu_numa` 正确，跨 NUMA 距离=21 ✅
  - **M3 第一主线 `per_gpu_ib_device`**：优先同 NUMA IB；无同 NUMA 时显式 `downgraded=True` 而非静默跨 NUMA ✅
  - 评分：`score_pair` 同 NUMA pair 总分 < 跨 NUMA pair；降级路径 `downgraded` 且 `reason` 说明跨 NUMA；队列负载↑、kv_bytes↑ 均使总分↑；分量可加 ✅
  - 绑定：`cpu_affinity` 返回 GPU 所在 NUMA 本地 CPU；无 CPU 列表时显式降级；非法角色抛 `ValueError` ✅
- 关键效果（设计落地）：
  - 拓扑 JSON 成为唯一事实来源，评分/绑定都从图查询，消灭了「静态全局 IB 列表」的脆弱写法。
  - 所有「无法满足同 NUMA」的情况都显式降级并留 `reason`，为 M6 A/B metric 打点提供钩子。
- 已知缺口（留给 M2/M6）：真实环境 GPU→IB 的 NUMA 亲和靠 PCIe 关联，目前 `probe` 用可注入的 `ib_numa_map` 兜底。
- 回退点：`pdv2-m1`（待 git tag）。

---

## M2 — tests：多 scheduler room 生命周期状态机（已完成 2026-08-26）

- 新增文件：`tests/pd_state_machine.py`（参考实现 / fake backend）、`tests/test_room_lifecycle.py`（11 例）
- 状态（8+终态）：`CREATED → ALLOCATED → P_LOADING → P_RUNNING → KV_TRANSFERRING → D_DECODING → FINISHED`，外加 `ABORTED`/`FAILED` 终态。
- 4 条不变量：①状态一致（只走白名单转换）②资源只释放一次（release_events 计数）③失败不污染其它 room④无孤儿 slot（occupied 只含非终态 room）。
- 预期结果 → 实际：
  - 完整 happy path 到 FINISHED 并释放 slot ✅
  - P 先到 / D 先到：单到达停留 CREATED，双到达才进 ALLOCATED ✅
  - 重复 room → `DuplicateRoom`；slot 冲突 → `SlotConflict` ✅
  - 任意阶段 ABORT→ABORTED、FAIL→FAILED 均释放 slot ✅
  - 失败不污染：A FAIL 后 B 独立跑完 FINISHED ✅
  - 迟到到达 / 终态后再发事件 → `InvalidTransition`（不变量①）✅
  - **1000 次随机调度（多线程）不变量恒成立** ✅
- 关键效果：这是 PD v2 的「安全网」验收基准。真机端到端（sglang fork 就绪后）必须复用同一组不变量。
- 缺口/前置：`sglang` fork 尚未 clone 成功（SSH 公钥未授权，`Permission denied (publickey)`），端到端集成用例（`@skip_without_sglang`）暂缓；fork 就绪后补 M2 集成用例。
- 回退点：`pdv2-m2`（待 git tag）。

---

## M3 — routing：P/D 选择 + 离散事件模拟（已完成 2026-08-26）

- 新增文件：`routing/{__init__,pair_score,simulate}.py`、`tests/test_routing.py`（5 例）
- 内容：
  - `RoutingPolicy` 可插拔抽象；`TopologyAwarePolicy`（枚举候选 (P,D) 用 M1 的 `score_pair` 选最低分）；`NaivePolicy`（忽略拓扑、取首个可用，作对照基线）。
  - `Simulator` 离散事件模拟 PD 集群：泊松式到达 → policy 选对 → 按 KV 量模拟传输耗时 → 释放；记录每决策的「所选分 / 最优备选分」。
- 预期结果 → 实际：
  - 存在同 NUMA (P,D) 时，所选对 `numa_crossing==0` ✅
  - `rank_pairs` 升序；无候选 `select` 抛 `ValueError` ✅
  - 拓扑感知平均路径成本 < 朴素策略（300 请求，seed=7）✅
  - 拓扑感知每次决策都不被可用备选 dominate（1000 请求）✅
  - 1000 次模拟稳定无异常、不变量成立 ✅
- 关键效果：证明「拓扑感知路由」相对朴素路由确实更优，且 auto 永不选明显更差路径——满足 M3「先模拟后上真机」原则。
- 回退点：`pdv2-m3`（待 git tag）。

## M4 — benchmarks：统一指标计算（已完成 2026-08-26）

- 新增文件：`benchmarks/{__init__,metrics}.py`、`benchmarks/workloads/{short,long_prefill,long_decode}.json`、`tests/test_benchmarks.py`（4 例）
- 内容：
  - `RequestRecord` / `BenchmarkReport` / `aggregate` / `percentile`（自实现分位数，无 numpy 依赖）。
  - 指标：TTFT、TPOT（≈ITL 均值）、ITL P50/P99、吞吐（总 token/总时长）、KV bytes + KV 传输延迟。
  - 三类 workload 固化：short（压 TTFT/吞吐）、long_prefill（压 prefill + KV 传输）、long_decode（压 decode 吞吐 + TPOT/ITL）。
- 预期结果 → 实际：
  - percentile 线性插值正确（含边界/空序列）✅
  - TTFT = first_token - prefill_start；ITL = 相邻 decode 间隔；TPOT = ITL 均值 ✅
  - 吞吐 = 总 token / 总时长；KV 指标聚合正确；分位数存在 ✅
  - 空记录安全（n=0、吞吐=0）✅
- 关键效果：指标定义与 M6 A/B 对齐，三类 workload 让每次优化都能回到具体配置复现。
- 回退点：`pdv2-m4`（待 git tag）。

## M5 — analysis：统一 trace 解析 + 对比报告（已完成 2026-08-26）

- 新增文件：`analysis/{__init__,trace_parser,report}.py`、`tests/test_analysis.py`（4 例）
- 内容：
  - 统一 8 阶段 trace（prefill / kv_send / kv_recv / decode_first_token ...），`parse_events` 按 request_id 聚合，`stage_durations` 派生 prefill / kv_transfer / ttft / decode / e2e。
  - `RunSummary` 绑定 `config_ref`（拓扑/版本/commit），`compare` 产出 delta 与 `better` 判定，供 M6 A/B 回溯到具体配置。
- 预期结果 → 实际：
  - 统一 trace 按 request_id 聚合 ✅
  - 派生时长正确（prefill=end-start；kv_transfer=send_end-send_start；ttft=first_token-prefill_start）✅
  - summarize 绑定 config_ref；compare delta 正确、候选延迟全降时 better=True ✅
- 关键效果：每次优化都能从 trace 还原到「哪次配置改动带来哪一档指标变化」。
- 回退点：`pdv2-m5`（待 git tag）。

## M6 — patches：上游改动指针文档（已完成 2026-08-26）

- 新增文件：`patches/manifest.py`、`patches/manifest.json`（模板 1 条待填 commit）、`tests/test_patches.py`（4 例）
- 内容：每条上游改动登记 `PatchEntry`（fork_repo / branch / commit / intent / status / metric_result / related_module）；`load_manifest` 解析并校验必填字段，`validate` 检重与 status 合法性。
- 预期结果 → 实际：
  - 合法 manifest 解析为 PatchEntry 列表 ✅
  - 缺必填字段（id/fork_repo/branch/commit/intent）抛 ValueError ✅
  - validate 检重 id / 非法 status / 顶层须为列表 ✅
- 关键效果：每个改动可追溯、可回退；fork 就绪后每笔提交都补一条 entry（commit 从 TBD 填实）。
- 回退点：`pdv2-m6`（待 git tag）。

## 模块顺序总览

| 阶 | 模块 | 状态 |
|---|---|---|
| 0 | TDD 脚手架 | ✅ |
| 1 | topology | ✅ |
| 2 | tests 状态机 | ✅ |
| 3 | routing（pair_score + 模拟） | ✅ |
| 4 | benchmarks | ✅ |
| 5 | analysis | ✅ |
| 6 | patches 指针文档 | ✅ |
| 7 | DEVLOG 维护 | ✅ |

---

## F1 — fork 迁移启动：topology 包 + auto IB 映射（2026-08-26）

- 移植文件（fork 内 `python/sglang/srt/disaggregation/topology/`）：
  - `topology_graph.py`（GPU↔NUMA↔NIC 图、`per_gpu_ib_device` 拓扑感知映射、显式降级）
  - `probe.py`（`parse_probe_json` 解析 env_probe JSON + `probe_local_topology` 本机 sysfs 探测，依赖全可注入）
  - `score.py`（`score_pair` 可插拔 P/D 评分纯函数）
  - `bind.py`（`cpu_affinity` transfer/bootstrap/staging 角色 NUMA 本地 CPU 绑定）
  - `auto_ib.py`（`build_auto_ib_mapping` 纯函数 + `resolve_auto_ib_devices` 薄包装，`auto` 哨兵）
  - `__init__.py`（统一导出）
- 改动文件：`python/sglang/srt/server_args.py` 的 `_validate_ib_devices`：新增 `auto` 分支 → `_resolve_auto_ib_mapping`，委托 topology 包。其余路径（None/共享列表/per-GPU JSON/文件路径）行为不变。
- 单测：`test/pd_v2/test_topology.py`（独立导入 topology 包，绕开 sglang 顶层 import）12 例全绿，覆盖：
  - 同 NUMA → 各自本地 IB；缺同 NUMA → `downgraded=True` + `reason`；空设备抛 `ValueError`。
  - `resolve_auto_ib_devices` 非 `auto` 返回 None；`auto` 产出 per-GPU JSON（mock sysfs）。
  - `probe_local_topology` 注入式构造；`parse_probe_json` 解析 env_probe 形状。
  - `cpu_affinity` 命中本地 NUMA CPUs / 降级 / 非法角色 `ValueError`。
  - `score_pair` 同 NUMA pair 零跨 NUMA 惩罚；跨 NUMA pair 惩罚 + `downgraded`。
- 关键效果（设计落地）：拓扑图成为 fork 内 IB 映射/亲和/评分的唯一事实来源；「静态全局 IB 列表」被替换为拓扑感知映射；所有跨 NUMA 兜底都显式降级、可观测。
- 已知缺口（留真机）：`probe_local_topology` 默认走 sysfs 探测 GPU NUMA（/sys/class/drm）+ IB NUMA（/sys/class/infiniband/<dev>/device/numa_node），缺数据时可经 `SGLANG_TOPOLOGY_PROBE_JSON` 注入 env_probe JSON；T4/T5 在 sglang worker/router 的实际挂钩点需在真机定位。
- 回退点：`pdv2-t3`（待首个 fork commit 后打 tag）。
