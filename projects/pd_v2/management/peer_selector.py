"""模块 E：PeerSelector —— 对齐 KsanaLLM csrc/pd_v2/decode/peer_selector.{h,cpp}。

Decode 引擎本地的 peer 路由表。Snapshot 驱动：PeerHealth 经 Ack/Complete/周期
Snapshot 推入，Select() 仅从快照表选 peer_id（== segment_id）。

设计要点（§4.3）：
  - 无 per-request 簿记。Select() 不记录 (req_id, peer_id)；无预留计数。
  - 快照是唯一真相来源；选中的 peer 视图可能略旧，恢复靠对下一个快照重试。
  - 策略：kRoundRobin 按 segment_id 升序循环；kLinUCB 远程 peer 仍走 round-robin
    （LinUCB 只决定「是否留本地」，详见 bandit/）。
"""

import threading
from typing import Dict, List, Optional

from .types import PeerHealth


class SelectorPolicy:
    ROUND_ROBIN = "round_robin"
    LINUCB = "linucb"


class PeerSelector:
    def __init__(self, policy: str = SelectorPolicy.ROUND_ROBIN):
        self.policy_ = policy
        self._mu = threading.RLock()
        self._peers: Dict[int, PeerHealth] = {}
        self._rr_cursor = 0  # 单线程测试下用普通 int 即可；加锁保护

    # ---- 注册 / 注销 ----

    def register_peer(self, segment_id: int, server_name: str) -> None:
        with self._mu:
            it = self._peers.get(segment_id)
            if it is None:
                h = PeerHealth()
                h.segment_id = segment_id
                h.server_name = server_name
                h.healthy = False  # 首个快照到达前不可选
                self._peers[segment_id] = h
            else:
                it.server_name = server_name

    def unregister_peer(self, segment_id: int) -> None:
        with self._mu:
            self._peers.pop(segment_id, None)

    # ---- 快照刷新 ----

    def update_snapshot(self, segment_id: int, health: PeerHealth) -> None:
        with self._mu:
            it = self._peers.get(segment_id)
            if it is None:
                # 回退：若 server_name 命中已注册 peer（epoch 变更 -> 新 seg_id），
                # 在原键下就地更新，再换到新 segment_id 键。
                if health.server_name:
                    for sid, h in list(self._peers.items()):
                        if h.server_name == health.server_name:
                            del self._peers[sid]
                            break
                h = PeerHealth()
                h.__dict__.update(health.__dict__)
                h.segment_id = segment_id
                self._peers[segment_id] = h
                return
            # 周期推送可能只带 segment_id；保留已有 server_name。
            preserved = it.server_name
            it.__dict__.update(health.__dict__)
            it.segment_id = segment_id  # 线端 segment_id 冲突时以之为准
            if not it.server_name:
                it.server_name = preserved

    # ---- 查询 ----

    def segment_id_by_name(self, server_name: str) -> Optional[int]:
        with self._mu:
            for sid, h in self._peers.items():
                if h.server_name == server_name:
                    return sid
            return None

    def get_snapshot(self, segment_id: int) -> Optional[PeerHealth]:
        with self._mu:
            it = self._peers.get(segment_id)
            return None if it is None else PeerHealth(**it.__dict__)

    def healthy_peer_count(self) -> int:
        with self._mu:
            return sum(1 for h in self._peers.values() if h.healthy)

    def registered_peer_count(self) -> int:
        with self._mu:
            return len(self._peers)

    def healthy_peers(self) -> List[int]:
        with self._mu:
            return sorted(sid for sid, h in self._peers.items() if h.healthy)

    # ---- 选择 ----

    def select(self) -> Optional[int]:
        with self._mu:
            if not self._peers:
                return None
            healthy = [h for h in self._peers.values() if h.healthy]
            if not healthy:
                return None
            # 按 segment_id 升序稳定排序；游标回绕。
            healthy.sort(key=lambda h: h.segment_id)
            idx = self._rr_cursor % len(healthy)
            self._rr_cursor += 1
            return healthy[idx].segment_id
