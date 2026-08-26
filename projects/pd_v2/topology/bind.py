"""CPU affinity 策略（M3 资源绑定）。

为 transfer worker / bootstrap thread / staging worker 选定 NUMA 本地的 CPU 集合。
无法满足同 NUMA 时显式降级并打 reason（不静默跨 NUMA）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .topology_graph import TopologyGraph

ROLE_TRANSFER_WORKER = "transfer_worker"
ROLE_BOOTSTRAP = "bootstrap_thread"
ROLE_STAGING = "staging_worker"

_VALID_ROLES = {ROLE_TRANSFER_WORKER, ROLE_BOOTSTRAP, ROLE_STAGING}


@dataclass
class AffinityResult:
    role: str
    cpus: list[int]
    numa_node: int
    downgraded: bool
    reason: str = ""


def cpu_affinity(
    role: str,
    graph: TopologyGraph,
    gpu_index: int,
) -> AffinityResult:
    if role not in _VALID_ROLES:
        raise ValueError(f"未知角色 {role!r}，应为 {sorted(_VALID_ROLES)}")
    numa_node = graph.gpu_numa(gpu_index)
    node = graph.numa_nodes.get(numa_node)
    if node is not None and node.cpus:
        return AffinityResult(role, list(node.cpus), numa_node, False)
    # 降级：该 NUMA 无可用 CPU 列表，退而用节点 0
    fallback = graph.numa_nodes.get(0)
    cpus = list(fallback.cpus) if fallback and fallback.cpus else []
    return AffinityResult(
        role, cpus, 0, True,
        reason=f"{role} 在 NUMA{numa_node} 无可用 CPU 列表，降级绑定到 NUMA0",
    )
