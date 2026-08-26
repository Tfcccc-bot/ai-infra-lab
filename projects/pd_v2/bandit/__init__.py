"""PD v2 决策核心：LinUCB 上下文赌博机（对齐 KsanaLLM csrc/pd_v2/decode）。

模块划分（纯函数、离线可测）：
- linucb_math : 4 维线性代数原语（Ainv/b、Sherman-Morrison、UCB）
- reward      : 奖励函数（LocalReward / TtftCredit / RemoteReward）
- router      : LinUCBRouter 双臂选择器（BuildFeatures / SelectArm / RecordReward）

两臂：ROUTE_REMOTE（转发到 P 节点，默认路径） vs ROUTE_LOCAL（留 D 节点本地跑）。
"""
from .linucb_math import (
    DIM,
    scaled_identity,
    dot,
    matvec,
    quad_form,
    sherman_morrison_update,
    ucb_score,
)
from .reward import local_reward, ttft_credit, remote_reward
from .router import (
    LinUCBRouter,
    LinUCBConfig,
    LocalArmContext,
    ROUTE_LOCAL,
    ROUTE_REMOTE,
)

__all__ = [
    "DIM",
    "scaled_identity",
    "dot",
    "matvec",
    "quad_form",
    "sherman_morrison_update",
    "ucb_score",
    "local_reward",
    "ttft_credit",
    "remote_reward",
    "LinUCBRouter",
    "LinUCBConfig",
    "LocalArmContext",
    "ROUTE_LOCAL",
    "ROUTE_REMOTE",
]
