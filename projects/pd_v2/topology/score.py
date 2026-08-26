"""P/D pair 评分（M3 第一主线）。

评分是可插拔的纯函数：输入拓扑图 + 一对 GPU + 负载/KV 信息，输出带分量拆解的总分。
约定：**分数越低越好**。每个分量都可独立解释，便于 M6 的 A/B metric 打点。

分量：
- numa_crossing : 跨 NUMA 惩罚（P/D 各自 IB 跨 NUMA + P/D 互跨 NUMA）
- gpu_nic_distance : GPU 到所选 IB 的归一化距离之和
- queue_load    : P/D 两侧队列负载之和
- kv_bottleneck : kv_bytes / 链路带宽（越大越糟）
"""

from __future__ import annotations

from dataclasses import dataclass

from .topology_graph import TopologyGraph, IbAssignment

DEFAULT_WEIGHTS = {
    "cross_numa": 100.0,      # 单个 IB 跨 NUMA 惩罚
    "cross_numa_pair": 60.0,  # P/D 互跨 NUMA 惩罚
    "gpu_nic": 5.0,           # GPU↔IB 距离权重
    "queue": 20.0,            # 队列负载权重
    "kv": 1e-6,              # KV 字节 / 带宽 权重
}


@dataclass
class ScoreComponents:
    numa_crossing: float
    gpu_nic_distance: float
    queue_load: float
    kv_bottleneck: float


@dataclass
class PairScore:
    p_gpu: int
    d_gpu: int
    total: float
    components: ScoreComponents
    downgraded: bool
    reason: str = ""


def score_pair(
    graph: TopologyGraph,
    p_gpu: int,
    d_gpu: int,
    *,
    kv_bytes: float = 0.0,
    p_queue_load: float = 0.0,
    d_queue_load: float = 0.0,
    ib_devices: list[str] | None = None,
    weights: dict | None = None,
) -> PairScore:
    w = weights or DEFAULT_WEIGHTS
    ib_devices = ib_devices or []

    p_ib: IbAssignment = graph.per_gpu_ib_device(p_gpu, ib_devices)
    d_ib: IbAssignment = graph.per_gpu_ib_device(d_gpu, ib_devices)
    downgraded = p_ib.downgraded or d_ib.downgraded

    numa_crossing = 0.0
    if p_ib.numa_node != graph.gpu_numa(p_gpu):
        numa_crossing += w["cross_numa"]
    if d_ib.numa_node != graph.gpu_numa(d_gpu):
        numa_crossing += w["cross_numa"]
    if graph.gpu_numa(p_gpu) != graph.gpu_numa(d_gpu):
        numa_crossing += w["cross_numa_pair"]

    gpu_nic_distance = w["gpu_nic"] * (
        graph.gpu_nic_distance(p_gpu, p_ib.device)
        + graph.gpu_nic_distance(d_gpu, d_ib.device)
    )
    queue_load = w["queue"] * (p_queue_load + d_queue_load)

    bw_p = graph.nic_bandwidth(p_ib.device)
    bw_d = graph.nic_bandwidth(d_ib.device)
    bw = min(bw_p, bw_d)
    kv_bottleneck = w["kv"] * (kv_bytes / bw) if bw > 0 else w["kv"] * kv_bytes

    comp = ScoreComponents(
        numa_crossing=numa_crossing,
        gpu_nic_distance=gpu_nic_distance,
        queue_load=queue_load,
        kv_bottleneck=kv_bottleneck,
    )
    total = sum(comp.__dict__.values())
    reason = " ".join(r.reason for r in (p_ib, d_ib) if r.reason)
    return PairScore(p_gpu, d_gpu, total, comp, downgraded, reason.strip())
