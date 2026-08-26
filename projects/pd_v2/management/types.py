"""对齐 KsanaLLM csrc/pd_v2/public/pd_v2_types.h 的核心数据结构。

仅包含离线可测、与传输层无关的数据类型（PeerHealth / Snapshot 包装）。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PeerHealth:
    """对齐 pd_v2_types.h::PeerHealth（MAP 编码，新字段可缺省为 0）。

    PeerSelector 以此为唯一输入做 Select()；无预留计数 / in-flight 簿记。
    """

    segment_id: int = 0
    server_name: str = ""
    free_kv_blocks: int = 0
    last_snapshot_ms: int = 0
    healthy: bool = False
    # 快照驱动路由的负载特征（LinUCB 等使用）。缺省即旧版兼容的零值。
    future_free_kv_blocks: int = 0   # 已释放但未合并的块（swapout 在途）
    waiting_reqs: int = 0            # 排队（prefilling）请求数
    running_reqs: int = 0            # 运行（decoding）请求数
    pending_prefill_tokens: int = 0  # waiting 请求规划序列长度之和
    step_token_num: int = 0          # 每调度步处理 token 的 EWMA
    step_latency_us: int = 0         # 每步墙钟时间的 EWMA（微秒）


@dataclass
class StageAssignment:
    """对齐 pd_v2 的 HostStagePool 分区（per-peer 控制面暂存池）。"""

    pool_base_va: int = 0
    base_slot: int = 0
    n_slots: int = 0
    slot_size: int = 0


@dataclass
class PrefillSnapshot:
    """对齐 pd_v2 的 PrefillSnapshot：snapshot 跨接收方相同，stage 逐 peer 不同。"""

    snapshot: PeerHealth = field(default_factory=PeerHealth)
    stage: StageAssignment = field(default_factory=StageAssignment)


@dataclass
class IncomingPrefillRequest:
    """对齐 pd_v2_types.h::IncomingPrefillRequest。

    decode_engine_inference_addr 同时作为并发 req_id 的发送方区分键与回程路由目标。
    """

    decode_engine_inference_addr: str = ""
    req_id: int = 0
    input_tokens: List[int] = field(default_factory=list)
    sampling_config_wire: object = None
    decode_targets_wire: object = None
    num_decode_targets: int = 0
    decode_attn_dp_group_id: int = 0
