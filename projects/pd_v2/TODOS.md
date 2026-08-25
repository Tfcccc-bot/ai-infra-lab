# PD v2 on SGLang — 执行清单

> 这是把规划文档落成可勾选的执行清单。按顺序推进，M0/M1 是硬前提。
> 依赖关系：`M0 → M1 → {M2 安全网, M3 第一主线} → M4 → M5 → M6 → M7`。

## 现在就能动手的第一刀

- [ ] `bash launcher/env_probe.sh`，确认能拿到拓扑 JSON
- [ ] Fork `sgl-project/sglang`，固定基线 commit `67853c5`，建 `pd-v2-topology` 分支
- [ ] 确定首个模型/硬件：建议 Qwen3-8B、2 GPU 单机（有 RDMA 再扩 1P1D 跨机）

---

## M0：范围冻结与基线固定

- [ ] Fork SGLang，固定基线 commit `67853c5`，创建 `pd-v2-topology` 分支
- [ ] 在 README 明确目标顺序：topology-aware → state-machine hardening → fast-blockwise
- [ ] 确认首个模型与硬件（Qwen3-8B / 2 GPU 单机起步）
- [ ] 写环境探测脚本：GPU、PCIe、NUMA、NIC/HCA、RDMA、Mooncake 版本
- [ ] 固化软件版本清单（cuda / torch / sglang / mooncake / rdma-core）

**验收**：能输出可复现的软件版本、拓扑 JSON、启动配置。

## M1：SGLang 原生 PD 基线

- [ ] 启动 1P1D + Router，先用 Mooncake；无 RDMA 先跑 fake / NIXL 本地路径
- [ ] 固化三类 workload：short-chat、long-prefill、long-decode
- [ ] 采集 TTFT、TPOT、ITL P50/P95/P99、吞吐、KV transfer bytes/latency、bootstrap/alloc wait
- [ ] 跑 correctness：确定性输出、并发、abort、首 token EOS、结构化输出
- [ ] 生成统一 trace：route → bootstrap → prealloc → prefill → transfer → commit → decode

**验收**：baseline 一键运行；连续三次偏差可解释；请求无 KV 泄漏。

## M2：多 scheduler 状态机测试（安全网）

- [ ] 基于 SGLang `fake` backend 自行实现测试场景（不搬内部测试代码）
- [ ] 建立 room 生命周期：Created → Bootstrapping → DestPublished → Transferring → Committed / Failed / Aborted
- [ ] 覆盖：P 先到、D 先到、bootstrap 超时、传输超时、某 TP rank 失败、重复 room、abort 与迟到 ACK、decode retraction
- [ ] 校验不变量：各 rank 状态一致；资源只释放一次；失败不污染复用 room；无孤儿 prealloc slot
- [ ] 加并发压力 + 随机故障注入

**验收**：状态机用例 1000 次随机调度稳定；失败后服务继续可用。

## M3：Topology-aware PD v2（第一主线）

- [ ] 探测 GPU ↔ NUMA ↔ NIC/HCA 距离，生成 topology graph
- [ ] 实现 per-GPU `--disaggregation-ib-device` 自动映射（替代静态全局列表）
- [ ] 为 Mooncake transfer worker / bootstrap thread / staging worker 加 CPU affinity 策略
- [ ] Router/P-D pair 评分加入：队列负载、KV 字节、GPU-NIC 距离、NUMA 跨越成本、链路带宽
- [ ] 无法同 NUMA 时显式降级并打 metric（不静默跨 NUMA）
- [ ] 对比 local-NUMA / cross-NUMA / auto 三组实验

**验收**：auto 不选明显更差路径；P95 KV transfer latency 与 CPU 开销有可复现改善。

## M4：Stage pool 与背压

- [ ] 定义统一 `TransferResourceLease`：room、KV pages、metadata slot、staging alloc、generation、owner
- [ ] 把“地址已发布但未入 TransferQueue”的窗口纳入资源计数
- [ ] 按 decode 可用 KV、extra slots、transfer queue 水位动态控制 prefill admission
- [ ] 优先级区分：bootstrap / metadata(aux) / KV payload / abort(cleanup)
- [ ] 加水位与泄漏指标：allocated / published / transferring / committed / aborted

**验收**：高并发下不出现 decode 预分配死锁；队列水位有界；超时/abort 后资源归零。

## M5：Fast-blockwise（第二主线）

- [ ] 定义 block 级协议：chunk id、token range、KV page range、last-chunk、generation
- [ ] 做仅传输流水化 PoC：prefill chunk N+1 与 chunk N 的 KV transfer 重叠
- [ ] Decode 在依赖块与 metadata 完整后才 commit，禁止读半成品 KV
- [ ] 与 chunked prefill、staging、异构 TP 对齐
- [ ] 评估与 EAGLE3/NEXTN 叠加；首版不同时改 blockwise 与 speculative 状态机

**验收**：长 prompt 下 KV 传输被 prefill 计算部分隐藏；输出与非 blockwise 一致；失败按 room 回收全部 chunk。

## M6：兼容性与性能矩阵

- [ ] 后端：Mooncake / NIXL（fake 仅逻辑测试）
- [ ] 拓扑：同机同 NUMA、同机跨 NUMA、跨机 RDMA
- [ ] 并行：同 TP、P-TP>D-TP、P-TP<D-TP、DP Attention；PP/CP 作扩展
- [ ] 功能：Radix Cache、HiCache、KV offload、EAGLE3；不兼容组合跳过并说明
- [ ] workload：ISL/OSL/并发二维 sweep，画出分离收益与退化边界
- [ ] 回归：accuracy、abort、node failure、timeout、资源泄漏、长稳压

**验收**：公开 benchmark 报告，含硬件拓扑、版本、配置、误差范围。

## M7：开源交付

- [ ] `projects/pd_v2/` 提供设计、启动脚本、基准、测试、结果
- [ ] SGLang fork 小提交拆分：topology discovery / affinity / routing policy / metrics / tests
- [ ] 可复用部分提交上游 PR；实验性策略留在本仓库
- [ ] 合规检查：不含内部文件路径、提交哈希、专有类实现、原始代码片段

## 明确不做

- 不重写 SGLang 已有 Mooncake/NIXL connector
- 不复制 KsanaLLM 内部实现或单测
- 第一阶段不同时改 Router、传输层、scheduler、speculative decoding
- 无基线数据前，不宣称 NUMA 或 blockwise 一定提升性能
