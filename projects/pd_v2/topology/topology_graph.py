"""结构化的 GPU/NUMA/NIC 拓扑图。

设计要点（对应 M3「拓扑感知资源放置」）：
- 拓扑 JSON 是唯一事实来源，本类只描述「节点 + 边权」。
- GPU → NUMA、GPU → 最近 IB 设备、NUMA 间距离，全部从这里查，不写死全局列表。
- 任何「无法满足同 NUMA」的情况都要显式标记 downgraded，绝不静默跨 NUMA。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuNode:
    index: int
    name: str
    memory_total_mb: int
    pci_bus_id: str
    numa_node: int


@dataclass
class NumaNode:
    node_id: int
    cpus: list[int] = field(default_factory=list)
    # node_id -> 相对距离（10 表示同节点，21 表示跨节点一跳，越大越远）
    distances: dict[int, int] = field(default_factory=dict)


@dataclass
class NicNode:
    name: str
    numa_node: int
    link_layer: str = ""        # "IB" / "ETH" / ""
    bandwidth_gbps: float = 0.0


@dataclass
class IbAssignment:
    device: str
    numa_node: int
    downgraded: bool
    reason: str = ""


class TopologyGraph:
    """GPU ↔ NUMA ↔ NIC 的归一化拓扑图。"""

    def __init__(
        self,
        gpus: list[GpuNode],
        numa_nodes: list[NumaNode],
        nics: list[NicNode],
    ) -> None:
        self.gpus = {g.index: g for g in gpus}
        self.numa_nodes = {n.node_id: n for n in numa_nodes}
        self.nics = {n.name: n for n in nics}

    # ---------------- 构造 ----------------
    @classmethod
    def from_dict(cls, d: dict) -> "TopologyGraph":
        """从归一化 dict 构造（tests / probe 都产出这种格式）。

        d = {
          "gpus": [{"index","name","memory_total_mb","pci_bus_id","numa_node"}, ...],
          "numa_nodes": [{"node_id","cpus":[int...],"distances":{int:int}}, ...],
          "nics": [{"name","numa_node","link_layer","bandwidth_gbps"}, ...],
        }
        """
        gpus = [GpuNode(**g) for g in d.get("gpus", [])]
        numas = [
            NumaNode(
                node_id=n["node_id"],
                cpus=list(n.get("cpus", [])),
                distances={int(k): int(v) for k, v in n.get("distances", {}).items()},
            )
            for n in d.get("numa_nodes", [])
        ]
        nics = [NicNode(**n) for n in d.get("nics", [])]
        return cls(gpus, numas, nics)

    def to_dict(self) -> dict:
        return {
            "gpus": [vars(g) for g in self.gpus.values()],
            "numa_nodes": [vars(n) for n in self.numa_nodes.values()],
            "nics": [vars(n) for n in self.nics.values()],
        }

    # ---------------- NUMA 查询 ----------------
    def gpu_numa(self, gpu_index: int) -> int:
        if gpu_index not in self.gpus:
            raise KeyError(f"unknown gpu index {gpu_index}")
        return self.gpus[gpu_index].numa_node

    def are_same_numa(self, a: int, b: int) -> bool:
        return self.gpu_numa(a) == self.gpu_numa(b)

    def numa_distance(self, a: int, b: int) -> int:
        """a,b 为 NUMA node id；取已知距离，未知回退为跨节点惩罚。"""
        if a == b:
            return 10
        return self.numa_nodes.get(a, NumaNode(a)).distances.get(b, 21)

    def _nic_numa(self, device: str) -> int:
        nic = self.nics.get(device)
        return nic.numa_node if nic is not None else 0

    def ib_devices_for_numa(self, numa_node: int) -> list[str]:
        return [name for name, nic in self.nics.items() if nic.numa_node == numa_node]

    # ---------------- M3: per-GPU IB 自动映射 ----------------
    def per_gpu_ib_device(self, gpu_index: int, ib_devices: list[str]) -> IbAssignment:
        """替代静态全局列表：按 GPU 的 NUMA 节点自动选 IB 设备。

        规则：
        - 优先选与该 GPU 同 NUMA 的 IB 设备，多个时按 gpu_index 轮询。
        - 同 NUMA 无 IB 时，显式降级（downgraded=True），绝不静默跨 NUMA。
        """
        if not ib_devices:
            raise ValueError("ib_devices 为空，无法为 GPU 分配 IB 设备")
        gpu_numa = self.gpu_numa(gpu_index)
        same_numa = [d for d in ib_devices if self._nic_numa(d) == gpu_numa]
        if same_numa:
            dev = same_numa[gpu_index % len(same_numa)]
            return IbAssignment(device=dev, numa_node=gpu_numa, downgraded=False)
        # 降级：跨 NUMA 取第一个（按索引轮询），记录原因
        dev = ib_devices[gpu_index % len(ib_devices)]
        return IbAssignment(
            device=dev,
            numa_node=self._nic_numa(dev),
            downgraded=True,
            reason=f"GPU{gpu_index} 在 NUMA{gpu_numa} 无同 NUMA IB，降级跨 NUMA 使用 {dev} (NUMA{self._nic_numa(dev)})",
        )

    # ---------------- GPU↔NIC 距离 ----------------
    def gpu_nic_distance(self, gpu_index: int, device: str) -> float:
        """同 NUMA=1.0，跨 NUMA 用相对距离归一化（/10）。"""
        g_numa = self.gpu_numa(gpu_index)
        n_numa = self._nic_numa(device)
        if g_numa == n_numa:
            return 1.0
        return self.numa_distance(g_numa, n_numa) / 10.0

    def nic_bandwidth(self, device: str) -> float:
        nic = self.nics.get(device)
        return nic.bandwidth_gbps if nic is not None else 0.0
