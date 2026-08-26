# PD v2 on SGLang — 执行清单（T1–T9）

> 本清单是 `FORK_PLAN.md` 任务拆解的落地勾选版，每个任务可独立验收。
> 纪律：**先补单测与 FORK_PLAN 对应小节 → 实现 → 补 DEVLOG.md 一节 → 勾掉**。
> 纯 Python 原型 M0–M6 已完成（48 单测全绿），作为 fork 落地的 spec / 参考实现。
> 边界、单测计划、不变量见 `FORK_PLAN.md`；决策与进展见 `WORKLOG.md`。

## 准备（前置，阻塞于 GitHub 公钥）

- [ ] 公钥 `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFbxxz8dixLmX8twZ6wzIo1Zuvg0D7GRJmCQVSqJRtq3 fangtang` 加到 GitHub，`ssh -T git@github.com` 通过
- [ ] clone 自己账户下的 fork + `git remote add upstream https://github.com/Tfcccc-bot/sglang.git` + `git fetch upstream 67853c5` + `git checkout -b pd-v2-topology 67853c5`

## T1 — Fork 接入与基线锁定

- [x] 分支 `pd-v2-topology`，HEAD = `67853c5`（已创建，基线锁定）
- [ ] `python -c "import sglang"` 成功

**验收**：分支/commit 锁定，sglang 可导入。可判断完成。

## T2 — 真机拓扑探测接入 M1 topology

- [ ] `launcher/env_probe.sh` 真机产出 `topology-<host>.json`
- [ ] 扩 `tests/test_topology_probe.py`：真机 JSON 还原 GPU↔NUMA↔NIC；`per_gpu_ib_device` 满足同 NUMA 时 `downgraded=False`
- [ ] 真机 JSON 通过 M1 单测

**验收**：解析无异常。可判断完成。

## T3 — per-GPU IB 设备自动映射落 fork

- [x] 改 disaggregation 参数解析，挂 `--disaggregation-ib-device=auto`，用 M1 `topology_graph` 查询（替换静态全局 IB 列表）
- [ ] `tests/test_ib_device_mapping.py`：P/D 命中各自 GPU 本地 IB；缺同 NUMA 时 `downgraded=True` + `reason`
- [ ] 真机 1P1D 用 `auto` 启动成功

**验收**：单测绿 + 真机启动成功。可判断完成。

## T4 — worker CPU affinity 落 fork

- [ ] 用 M1 `bind.cpu_affinity` 在 transfer/bootstrap/staging worker 启动处加 NUMA 本地 CPU 绑定（不改 worker 内部）
- [ ] `tests/test_affinity_bind.py`：affinity mask == GPU 本地 NUMA CPUs；非法角色 `ValueError`
- [ ] 真机 `numactl --show` 验证绑定生效

**验收**：单测绿 + 绑定生效。可判断完成。

## T5 — Router (P,D) 配对策略落 fork

- [ ] 新增 `TopologyAwarePolicy`（复用 M3 `score_pair`），主流程保留 `NaivePolicy` 对照（不改 Router 调度主流程）
- [ ] `tests/test_routing_policy.py`：同 NUMA 候选优先；`rank_pairs` 升序；无候选 `ValueError`；300 请求平均成本 < 朴素
- [ ] 可切 Naive / TopologyAware 对照

**验收**：单测绿 + 可对照。可判断完成。

## T6 — 多 scheduler room 状态机集成

- [ ] fork 用 `fake` backend 复用 M2 状态机场景（不搬内部测试代码）
- [ ] `@skip_without_sglang` 集成用例跑 M2 的 11 类场景 + 1000 次随机调度
- [ ] 失败注入后服务仍可服务新请求、无 KV 泄漏、无孤儿 slot

**验收**：集成绿 + 不变量成立。可判断完成。

## T7 — 统一指标 + trace 接入 fork

- [ ] 启动脚本采集；trace 阶段对齐 M5 的 8 阶段（route/bootstrap/prealloc/prefill/transfer/commit/decode/e2e）
- [ ] 用 M4/M5 单测解析真机 trace JSON，断言 TTFT/ITL/KV 派生时长正确
- [ ] 三类 workload（short/long_prefill/long_decode）可复现

**验收**：真机 trace 通过 M4/M5 解析。可判断完成。

## T8 — patches manifest 回填

- [ ] 每笔 fork 提交补 `patches/manifest.json` entry，commit 从 `TBD` 填实
- [ ] `test_patches.py` 校验必填 + 不重 + status 合法

**验收**：`validate(load_manifest())` 通过，每条对应真实 commit。可判断完成。

## T9 — A/B 对比报告

- [ ] 用 M5 `compare` 产出 local-NUMA / cross-NUMA / auto 三组指标 delta
- [ ] 生成可读报告；auto 不被可用备选 dominate

**验收**：报告产出，`better` 判定正确。可判断完成。

## 明确不做（与 FORK_PLAN §2.3 一致）

- [ ] 不复制 KsanaLLM 内部实现 / 单测
- [ ] 不重写 SGLang 已有 Mooncake / NIXL connector
- [ ] 第一批不同时改 Router + 传输层 + scheduler + speculative decoding
- [ ] 无 baseline 数据前不宣称 NUMA / blockwise 一定提升性能
- [ ] fast-blockwise / stage pool 留作第二阶段，不在本清单边界内

## 进度快照（2026-08-26 迁移启动）

- **T1 分支**：✅ 已创建 `pd-v2-topology` @ `67853c5`。
- **T2 foundation**：✅ M1 topology 包已移植进 fork（`sglang.srt.disaggregation.topology`）；真机 `env_probe` JSON 接入待 T2 真机步骤。
- **T3 auto IB**：✅ `server_args` 已挂 `auto`；离线单测 12 例全绿；真机 1P1D 启动待验证。
- **T4 / T5 逻辑**：✅ `cpu_affinity` / `score_pair` 已随 topology 包移植并附单测；**接入 sglang worker/router 的具体 hook 点待真机定位**（fork 实际 PD 为 bootstrap 模型，无计划假设的单一 Router 配对点）。
- **待验证阶段**：真实 `import sglang` + 真机 env_probe + 真机 1P1D 启动（T2/T3 真机部分、T4/T5 集成、T6/T7/T9）。

## 回退与 tag

- [ ] 纯 Python 原型回退点先打 `pdv2-m0` … `pdv2-m6`
- [ ] 每个 T 任务完成打轻量 tag（如 `pdv2-t3`），便于二分回退

## 进度快照（2026-08-26 决策核心 + 整套管理落地）

> 本批次纠偏：先前 T5 的「静态 `TopologyAwarePolicy`」与 KsanaLLM 真实实现不符；
> 真实核心是 **LinUCB 上下文赌博机 + 生产者-消费者 + 延迟/即时奖励 + 快照驱动 peer 管理**。
> 据此在 `ai-infra-lab` prototype 重建 `bandit/`+`management/`，全程 TDD，73 例单测全绿
> （commit `c4ef9b1`，分支 `pd-v2-docs`，未推送）。

- **A 数学原语 / B 奖励函数 / C 双臂选择器 / D 决策流**：✅ `bandit/`（37 单测）。`LinUCBRouter` 平局偏 REMOTE、在线 Sherman-Morrison 更新、UCB 探索-利用收敛验证通过。
- **E PeerSelector / F1 AliveStore+Keepalive / F2 发现+广播+配对 / G 拓扑接入**：✅ `management/`（36 单测）。快照驱动路由表、Redis 存活键 epoch 协议、S3/S1 发现回退、reverse-route 配对、复用 `topology/` 作 peer 亲和度排名。
- **T5 重写说明**：T5 的实际实现 = `bandit/router.py`(LinUCB) + `management/peer_selector.py`，已 **supersede** 原计划的静态 `TopologyAwarePolicy`；`NaivePolicy` 对照概念保留。
- **H 落点已定位（未实现）**：
  - sglang fork PD = bootstrap 模型、decode 实例不跑 prefill → 须 **co-locate 同节点 prefill 实例**，使 `decide_route` 选 local/remote prefill 实例。
  - 落点① `scheduler.py::_add_request_to_queue`（~2811 行 `disagg_prefill_bootstrap_queue.add`）挂 `decide_route`；
  - 落点② `decode.py::_resolve_prefill_dp_rank`（661 行）挂 `PeerSelector.select()`；
  - 落点③ `mooncake/conn.py` 连接层 + `pd_disaggregation_hook.py`(148) 挂 Snapshot/Alive 生命周期。
  - 注入接口：`LinUCBRouter` + `TopologyAwarePeerSelector` + `AliveStore(Redis)` + `SnapshotReceiver`(decode) / `SnapshotBroadcaster` + `ReverseRouteTable`(prefill)。
- **下一步（H，待用户授权）**：把 `bandit/`+`management/` 移植进 sglang 为 `sglang/srt/disaggregation/pdv2/`，按落点焊接；真机验证（本机 `import sglang` 失败，无法离线单测）。
