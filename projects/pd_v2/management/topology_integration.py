"""模块 G：拓扑/IB 接入 —— 复用 T3 auto IB 拓扑作为 PeerSelector 的亲和性排名输入。

对齐：
  - topology/score.py::score_pair 与 topology/topology_graph.py::TopologyGraph
    （T3 移植前的原型，纯 stdlib，本模块直接复用，不另写）。
  - sglang 移植版 topology/auto_ib.py::resolve_auto_ib_devices 产出的
    peer→IB 设备映射，作为本模块的 `peer_ib` 输入（运行时注入，离线可测）。

职责边界：
  - 本模块只把「物理亲和性」转成 peer 排名偏置，**不替代** E 的 PeerSelector 路由表；
    TopologyAwarePeerSelector 包裹 PeerSelector，仅在选择时按亲和度排序/分层。
  - 真正的 IB 设备解析（T3 auto）发生在 sglang fork，结果以 peer_ib 注入此处。
"""

from typing import Dict, List, Optional, Tuple

from topology.topology_graph import TopologyGraph

from .peer_selector import PeerSelector


def peer_topology_affinity(graph: TopologyGraph, local_gpu: int, ib_device: str) -> float:
    """单 peer 的物理亲和性 ∈ [0,1]，越高越好。

    - 同 NUMA 且未降级 → 1.0
    - 跨 NUMA 降级   → 按 GPU↔NIC 距离衰减（≥1 → ≤1）
    - IB 设备不在拓扑图 → 0.0（无法评估，最差）
    """
    if ib_device not in graph.nics:
        return 0.0
    asg = graph.per_gpu_ib_device(local_gpu, [ib_device])
    if asg.downgraded:
        dist = graph.gpu_nic_distance(local_gpu, ib_device)  # 同 NUMA=1.0，跨 NUMA>1
        return max(0.1, 1.0 / dist)
    return 1.0


def build_peer_ib_map(auto_ib_mapping: Dict[str, str]) -> Dict[str, str]:
    """T3 auto IB 解析结果 -> {peer_key: ib_device}。

    sglang 移植版 resolve_auto_ib_devices 产出 {"<gpu/peer>": "mlx5_x"}；
    此处原样透传，调用方负责把 key 对齐到 PeerSelector 的 server_name。
    """
    return dict(auto_ib_mapping)


class TopologyAwarePeerSelector:
    """在 E 的 PeerSelector 之上叠加拓扑亲和度排名（G 核心）。

    - select() 优先选亲和度最高的健康 peer；同亲和度组内 round-robin（维持负载均衡）。
    - 缺拓扑信息（peer 无 IB 映射）时不偏置，退化为原 round-robin。
    """

    def __init__(
        self,
        base: PeerSelector,
        graph: TopologyGraph,
        local_gpu: int,
        peer_ib: Dict[str, str],
    ):
        self._base = base
        self._graph = graph
        self._local_gpu = local_gpu
        self._peer_ib = peer_ib
        self._tier_cursor = 0

    def _affinity_for(self, segment_id: int) -> float:
        snap = self._base.get_snapshot(segment_id)
        if snap is None:
            return 0.0
        ib = self._peer_ib.get(snap.server_name)
        if ib is None:
            return 1.0  # 缺拓扑信息时不偏置
        return peer_topology_affinity(self._graph, self._local_gpu, ib)

    def ranked_healthy(self) -> List[Tuple[int, float]]:
        healthy = self._base.healthy_peers()
        return sorted(
            ((sid, self._affinity_for(sid)) for sid in healthy),
            key=lambda x: (-x[1], x[0]),
        )

    def select(self) -> Optional[int]:
        ranked = self.ranked_healthy()
        if not ranked:
            return None
        # 取最高亲和度层，组内 round-robin（优先同 NUMA，跨 NUMA 降级）
        best_aff = ranked[0][1]
        tier = [sid for sid, a in ranked if a == best_aff]
        chosen = tier[self._tier_cursor % len(tier)]
        self._tier_cursor += 1
        return chosen
