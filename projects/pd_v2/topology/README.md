# topology/

GPU ↔ NUMA ↔ NIC/HCA 拓扑探测、打分、绑定（M3 核心）。

## 现状与缺口

SGLang 侧已有 `numa_node` 参数和 `numa_utils`，但 **IB 设备 ↔ NUMA 的自动按索引映射未定义**——这正是 M3 要补的“per-GPU `--disaggregation-ib-device` 自动映射”。

## 计划落地

- `probe.py` —— 解析 `env_probe.sh` 的 JSON，构建 topology graph（GPU 节点 + NUMA 节点 + NIC/HCA 节点 + 边权 = 距离/带宽）
- `score.py` —— P/D pair 评分：队列负载、KV 字节、GPU-NIC 距离、NUMA 跨越成本、链路带宽
- `bind.py` —— transfer worker / bootstrap thread / staging worker 的 CPU affinity 策略

## 原则

- 无法满足同 NUMA 时**显式降级并打 metric**，不静默跨 NUMA。
- 拓扑 JSON 是唯一事实来源，路由与配置都从它生成。
