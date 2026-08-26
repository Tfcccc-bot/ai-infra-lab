"""模块 F2：快照发现/广播 + 入站 reverse-route 配对。

对齐 KsanaLLM csrc/pd_v2/mooncake/mooncake_engine_connector_discovery.cpp：

  - Prefill 侧 `BroadcastSnapshot`：
      * 构建 snapshot（server_name=本节点地址，字段来自快照提供者）
      * 选 recipients：若 alive_store 已连（S3 发现），KEYS `pd_v2/decode_alive/<cluster>/*`，
        逐个解析 epoch，跳过自己与无效键；否则回退 `incoming_routes_`（S1 自发现）。
      * epoch 缓存 diff：first_seen / restarted / disappeared
      * 逐个 push（snapshot 共享，stage 逐 peer 不同）
  - Decode 侧 `OnPrefillSnapshot`：
      * 解析本地 segment_id：先按 server_name（epoch 旋转时重新 open_segment），
        否则按 segment_id；两者皆空则丢弃。
      * 注册/刷新 PeerSelector 快照；缓存 stage 分区。
  - 入站 reverse-route 配对：IncomingPrefillRequest 以 decode_engine_inference_addr
    作为并发 req_id 的发送方区分键与回程路由目标。

传输层（mooncake/Redis/网络）以可注入函数/存储模拟，离线可测。
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from .alive_store import alive_key_prefix, parse_epoch_from_json
from .peer_selector import PeerSelector
from .types import IncomingPrefillRequest, PeerHealth, PrefillSnapshot, StageAssignment


class SnapshotReceiver:
    """Decode 侧 OnPrefillSnapshot（discovery.cpp §3.1）。"""

    def __init__(
        self,
        peer_selector: PeerSelector,
        open_segment_fn: Optional[Callable[[str], Optional[int]]] = None,
        stage_pool: Optional[object] = None,
    ):
        self._ps = peer_selector
        self._open_segment = open_segment_fn
        self._stage_pool = stage_pool

    def on_snapshot(self, snap: PrefillSnapshot) -> Optional[int]:
        name = snap.snapshot.server_name
        local_seg_id = 0
        if name:
            cached = self._ps.segment_id_by_name(name)
            if cached is not None:
                local_seg_id = cached
            else:
                local_seg_id = self._open_segment(name) if self._open_segment else 0
                if not local_seg_id:
                    return None  # openSegment 失败 -> 丢弃
                self._ps.register_peer(local_seg_id, name)
        elif snap.snapshot.segment_id != 0:
            local_seg_id = snap.snapshot.segment_id
        else:
            return None  # 两者皆空 -> 丢弃

        self._ps.update_snapshot(local_seg_id, snap.snapshot)

        if name and snap.stage.n_slots > 0 and self._stage_pool is not None:
            self._stage_pool.register_peer(name, snap.stage)
        return local_seg_id


@dataclass
class EpochDiff:
    first_seen: List[str] = field(default_factory=list)
    restarted: List[str] = field(default_factory=list)
    disappeared: List[str] = field(default_factory=list)


class SnapshotBroadcaster:
    """Prefill 侧 BroadcastSnapshot + 发现（discovery.cpp §3.2/§3.3）。"""

    def __init__(
        self,
        inference_addr: str,
        cluster: str,
        snapshot_provider: Callable[[], PeerHealth],
        alive_store,
        incoming_routes: Optional[List[str]] = None,
        stage_pool: Optional[object] = None,
    ):
        self._addr = inference_addr
        self._cluster = cluster
        self._provider = snapshot_provider
        self._alive = alive_store
        self._incoming = list(incoming_routes or [])
        self._stage_pool = stage_pool
        self._prev_addrs: List[str] = []
        self._prev_epochs: Dict[str, int] = {}

    def build_snapshot(self) -> PrefillSnapshot:
        snap = PrefillSnapshot()
        snap.snapshot = self._provider()
        snap.snapshot.server_name = self._addr
        snap.snapshot.segment_id = 0
        snap.snapshot.healthy = True
        snap.snapshot.last_snapshot_ms = int(time.monotonic() * 1000)
        return snap

    def select_decode_addrs(self) -> (List[str], EpochDiff):
        if self._alive is not None and self._alive.is_connected():
            prefix = alive_key_prefix(self._cluster)
            current: List[str] = []
            epochs: Dict[str, int] = {}
            for key in self._alive.keys(prefix):
                addr = key[len(prefix):]
                if addr == self._addr:
                    continue  # 跳过自己
                raw = self._alive.get(key)
                epoch = parse_epoch_from_json(raw)
                if epoch == 0:
                    continue  # 无效键
                current.append(addr)
                epochs[addr] = epoch
            diff = self._compute_diff(current, epochs)
        else:
            # 回退 S1：自发现（incoming_routes）
            current = sorted(set(self._incoming))
            diff = self._compute_diff(current, {})
        self._prev_addrs = current
        self._prev_epochs = epochs if self._alive and self._alive.is_connected() else {}
        return current, diff

    def _compute_diff(self, current: List[str], epochs: Dict[str, int]) -> EpochDiff:
        cur_set = set(current)
        diff = EpochDiff()
        diff.disappeared = [a for a in self._prev_addrs if a not in cur_set]
        for a in current:
            if a not in self._prev_addrs:
                diff.first_seen.append(a)
            elif epochs.get(a, 0) < self._prev_epochs.get(a, 0):
                diff.restarted.append(a)
        return diff

    def broadcast(self, send_fn: Callable[[str, PrefillSnapshot], None]) -> (List[str], EpochDiff):
        snap = self.build_snapshot()
        addrs, diff = self.select_decode_addrs()
        for addr in addrs:
            stage = StageAssignment()
            if self._stage_pool is not None:
                stage = self._stage_pool.register_peer(addr, StageAssignment())
            per_peer = PrefillSnapshot(snapshot=snap.snapshot, stage=stage)
            send_fn(addr, per_peer)
        return addrs, diff


class ReverseRouteTable:
    """入站 reverse-route 配对：以 decode_engine_inference_addr 作为并发请求区分键。"""

    def __init__(self):
        self._by_addr: Dict[str, Dict[int, IncomingPrefillRequest]] = {}

    def register(self, req: IncomingPrefillRequest) -> None:
        self._by_addr.setdefault(req.decode_engine_inference_addr, {})[req.req_id] = req

    def resolve_target(self, decode_addr: str, req_id: int) -> Optional[str]:
        # 回程路由目标即发送方地址（prefill 完成后据此回送 decode）。
        entry = self._by_addr.get(decode_addr)
        if entry and req_id in entry:
            return decode_addr
        return None

    def pop(self, decode_addr: str, req_id: int) -> Optional[IncomingPrefillRequest]:
        entry = self._by_addr.get(decode_addr)
        if entry and req_id in entry:
            return entry.pop(req_id)
        return None

    def pending_count(self, decode_addr: str) -> int:
        return len(self._by_addr.get(decode_addr, {}))
