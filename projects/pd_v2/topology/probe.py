"""解析 env_probe.sh 生成的原始 topology JSON 为归一化 TopologyGraph。

env_probe.sh 输出的是「人读」的聚合 JSON：GPU/NUMA/NIC/HCA 都是原始命令行文本的
字符串数组。这里做 best-effort 解析，产出结构化图供上层使用。

真实环境里 GPU→IB 的 NUMA 亲和需要靠 PCIe 拓扑关联（nvidia-smi topo -m 的 NUMA 列 +
lspci 的 NIC NUMA），本模块把这部分做成可注入的 ib_numa_map，便于测试与缺数据兜底。
"""

from __future__ import annotations

import json
from pathlib import Path

from .topology_graph import GpuNode, NumaNode, NicNode, TopologyGraph


def _expand_cpu_list(spec: str) -> list[int]:
    """展开 '0-7,16-23' -> [0..7,16..23]。"""
    cpus: list[int] = []
    spec = spec.strip()
    if not spec:
        return cpus
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _parse_gpus(gpu_lines: list[str]) -> list[GpuNode]:
    gpus: list[GpuNode] = []
    for line in gpu_lines:
        # nvidia-smi csv,noheader: index,name,memory.total,pci.bus_id,pci.domain
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        name = parts[1]
        mem = int("".join(filter(str.isdigit, parts[2])) or 0)
        bus_id = parts[3]
        gpus.append(GpuNode(index=idx, name=name, memory_total_mb=mem,
                            pci_bus_id=bus_id, numa_node=0))
    return gpus


def _parse_numa(numa_lines: list[str]) -> list[NumaNode]:
    nodes: dict[int, NumaNode] = {}
    distances_block = False
    dist_header: list[int] = []
    for line in numa_lines:
        line = line.strip()
        if line.startswith("node") and "cpus:" in line:
            # "node 0 cpus: 0-63"
            head, _, rest = line.partition("cpus:")
            nid = int(head.replace("node", "").strip())
            nodes[nid] = NumaNode(node_id=nid, cpus=_expand_cpu_list(rest))
        elif line.startswith("node distances:"):
            distances_block = True
            dist_header = []
        elif distances_block:
            if line.startswith("node"):
                # "node   0   1" -> 表头
                dist_header = [int(x) for x in line.split()[1:]]
            elif dist_header and line and line[0].isdigit():
                # "  0:  10  21"
                tokens = line.split()
                src = int(tokens[0].rstrip(":"))
                if src in nodes:
                    nodes[src].distances = {h: int(v) for h, v in zip(dist_header, tokens[1:])}
    return list(nodes.values())


def _parse_hca(hca_lines: list[str]) -> list[str]:
    return [ln.split(":", 1)[1].strip() for ln in hca_lines if ln.strip().startswith("hca_id")]


def parse_probe_json(
    data: dict | str | Path,
    ib_numa_map: dict[str, int] | None = None,
) -> TopologyGraph:
    """把 env_probe 原始 JSON 解析为 TopologyGraph。

    ib_numa_map: 可选的 {ib_device: numa_node}，用于补全真实环境里 HCA 输出不含的
                NUMA 归属。缺省时 IB 默认挂在 NUMA 0。
    """
    if isinstance(data, (str, Path)):
        with open(data, "r") as f:
            data = json.load(f)

    gpus = _parse_gpus(data.get("gpu", []))
    numa_nodes = _parse_numa(data.get("numa", []))

    ib_numa_map = ib_numa_map or {}
    hca_ids = _parse_hca(data.get("hca", []))
    # 没有真实 HCA 时，退而从 nic 列表推断（ip link 里的 ib* 接口）
    if not hca_ids:
        hca_ids = [n.split()[0] for n in data.get("nic", []) if n.strip().startswith("ib")]
    nics = [
        NicNode(name=n, numa_node=ib_numa_map.get(n, 0), link_layer="IB")
        for n in hca_ids
    ]

    return TopologyGraph(gpus, numa_nodes, nics)
