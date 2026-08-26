"""模块 G 单测：拓扑/IB 接入，复用 topology 包对齐亲和性排名。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from management.peer_selector import PeerSelector
from management.topology_integration import (
    build_peer_ib_map,
    peer_topology_affinity,
    TopologyAwarePeerSelector,
)
from topology.topology_graph import TopologyGraph


def _graph():
    return TopologyGraph.from_dict({
        "gpus": [{"index": 0, "name": "g0", "memory_total_mb": 0, "pci_bus_id": "x", "numa_node": 0}],
        "numa_nodes": [
            {"node_id": 0, "cpus": [0], "distances": {0: 10, 1: 21}},
            {"node_id": 1, "cpus": [1], "distances": {0: 21, 1: 10}},
        ],
        "nics": [
            {"name": "mlx5_0", "numa_node": 0, "link_layer": "IB", "bandwidth_gbps": 200},
            {"name": "mlx5_1", "numa_node": 1, "link_layer": "IB", "bandwidth_gbps": 200},
        ],
    })


def test_affinity_same_numa_is_one():
    g = _graph()
    assert peer_topology_affinity(g, 0, "mlx5_0") == 1.0


def test_affinity_cross_numa_downgraded_penalized():
    g = _graph()
    a = peer_topology_affinity(g, 0, "mlx5_1")
    assert 0.0 < a < 1.0  # 跨 NUMA 降级，按距离衰减


def test_affinity_unknown_ib_zero():
    g = _graph()
    assert peer_topology_affinity(g, 0, "mlx5_x") == 0.0


def test_build_peer_ib_map_passthrough():
    m = build_peer_ib_map({"0": "mlx5_0", "1": "mlx5_1"})
    assert m == {"0": "mlx5_0", "1": "mlx5_1"}


def _selector_with_peers():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    ps.register_peer(20, "p1")
    ps.update_snapshot(10, _health("p0", True))
    ps.update_snapshot(20, _health("p1", True))
    return ps


def _health(server_name, healthy):
    from management.types import PeerHealth
    h = PeerHealth()
    h.server_name = server_name
    h.healthy = healthy
    return h


def test_aware_selector_prefers_same_numa_peer():
    ps = _selector_with_peers()
    g = _graph()
    # p0 走同 NUMA 的 mlx5_0；p1 走跨 NUMA 的 mlx5_1
    aware = TopologyAwarePeerSelector(ps, g, local_gpu=0, peer_ib={"p0": "mlx5_0", "p1": "mlx5_1"})
    for _ in range(5):
        assert aware.select() == 10  # 始终优先同 NUMA 的 p0


def test_aware_selector_round_robin_within_same_tier():
    ps = _selector_with_peers()
    g = _graph()
    # 两 peer 都用同 NUMA 设备 -> 同亲和度 -> round-robin
    aware = TopologyAwarePeerSelector(ps, g, local_gpu=0, peer_ib={"p0": "mlx5_0", "p1": "mlx5_0"})
    picks = [aware.select() for _ in range(4)]
    assert picks == [10, 20, 10, 20]


def test_aware_selector_no_bias_when_ib_missing():
    ps = _selector_with_peers()
    g = _graph()
    # 无 IB 映射 -> 不偏置 -> round-robin（与 E 行为一致）
    aware = TopologyAwarePeerSelector(ps, g, local_gpu=0, peer_ib={})
    picks = [aware.select() for _ in range(4)]
    assert picks == [10, 20, 10, 20]


def test_aware_selector_returns_none_when_empty():
    ps = PeerSelector()
    aware = TopologyAwarePeerSelector(ps, _graph(), local_gpu=0, peer_ib={})
    assert aware.select() is None


def test_ranked_healthy_orders_by_affinity_desc():
    ps = _selector_with_peers()
    g = _graph()
    aware = TopologyAwarePeerSelector(ps, g, local_gpu=0, peer_ib={"p0": "mlx5_0", "p1": "mlx5_1"})
    ranked = aware.ranked_healthy()
    assert ranked[0] == (10, 1.0)        # p0 同 NUMA 最高
    assert ranked[1][0] == 20            # p1 跨 NUMA 次之
    assert ranked[1][1] < 1.0
