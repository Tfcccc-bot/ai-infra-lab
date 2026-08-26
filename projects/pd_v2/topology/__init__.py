"""PD v2 topology 包：GPU ↔ NUMA ↔ NIC/HCA 拓扑图与亲和性原语。

唯一事实来源是 env_probe.sh 生成的 topology JSON（见 launcher/env_probe.sh）。
本包把它解析成结构化拓扑图，供 routing/ 评分与 launcher/ 绑定复用。
"""

from .topology_graph import (
    GpuNode,
    NumaNode,
    NicNode,
    TopologyGraph,
    IbAssignment,
)
from .probe import parse_probe_json
from .score import score_pair, ScoreComponents, PairScore, DEFAULT_WEIGHTS
from .bind import cpu_affinity, AffinityResult, ROLE_TRANSFER_WORKER, ROLE_BOOTSTRAP, ROLE_STAGING

__all__ = [
    "GpuNode", "NumaNode", "NicNode", "TopologyGraph", "IbAssignment",
    "parse_probe_json",
    "score_pair", "ScoreComponents", "PairScore", "DEFAULT_WEIGHTS",
    "cpu_affinity", "AffinityResult",
    "ROLE_TRANSFER_WORKER", "ROLE_BOOTSTRAP", "ROLE_STAGING",
]
