"""模块 F2 单测：快照发现/广播 + 入站配对，对齐 discovery.cpp。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from management.alive_store import FakeAliveStore, alive_key
from management.discovery import (
    ReverseRouteTable,
    SnapshotBroadcaster,
    SnapshotReceiver,
)
from management.peer_selector import PeerSelector
from management.types import IncomingPrefillRequest, PeerHealth, PrefillSnapshot, StageAssignment


def _health(**kw):
    h = PeerHealth()
    for k, v in kw.items():
        setattr(h, k, v)
    return h


# ---------------- Decode 侧 OnPrefillSnapshot ----------------

def test_receiver_resolves_by_server_name_and_registers():
    ps = PeerSelector()
    rcv = SnapshotReceiver(ps, open_segment_fn=lambda name: 42)
    snap = PrefillSnapshot(snapshot=_health(server_name="p0", healthy=True, free_kv_blocks=7))
    seg = rcv.on_snapshot(snap)
    assert seg == 42
    assert ps.segment_id_by_name("p0") == 42
    assert ps.get_snapshot(42).free_kv_blocks == 7


def test_receiver_reuses_cached_segment_on_epoch_rotation():
    # 第一次解析到 seg 42；后续同名新 seg_id 应复用 42（不新增条目）
    ps = PeerSelector()
    rcv = SnapshotReceiver(ps, open_segment_fn=lambda name: 42)
    rcv.on_snapshot(PrefillSnapshot(snapshot=_health(server_name="p0", segment_id=1, healthy=True)))
    rcv.on_snapshot(PrefillSnapshot(snapshot=_health(server_name="p0", segment_id=99, healthy=True)))
    assert ps.registered_peer_count() == 1
    assert ps.segment_id_by_name("p0") == 42


def test_receiver_drops_when_open_segment_fails():
    ps = PeerSelector()
    rcv = SnapshotReceiver(ps, open_segment_fn=lambda name: 0)  # 失败
    seg = rcv.on_snapshot(PrefillSnapshot(snapshot=_health(server_name="p0", healthy=True)))
    assert seg is None
    assert ps.registered_peer_count() == 0


def test_receiver_drops_when_both_empty():
    ps = PeerSelector()
    rcv = SnapshotReceiver(ps)
    seg = rcv.on_snapshot(PrefillSnapshot(snapshot=_health(server_name="", segment_id=0)))
    assert seg is None
    assert ps.registered_peer_count() == 0


def test_receiver_caches_stage_assignment():
    ps = PeerSelector()
    stage = {}
    class FakeStagePool:
        def register_peer(self, name, st):
            stage[name] = st
            return st
    rcv = SnapshotReceiver(ps, open_segment_fn=lambda n: 42, stage_pool=FakeStagePool())
    st = StageAssignment(n_slots=4, slot_size=1024)
    rcv.on_snapshot(PrefillSnapshot(snapshot=_health(server_name="p0", healthy=True), stage=st))
    assert stage.get("p0") == st


# ---------------- Prefill 侧 BroadcastSnapshot ----------------

class _Provider:
    def __init__(self, **kw):
        self.kw = kw
    def __call__(self):
        return _health(**self.kw)


def test_broadcaster_selects_via_alive_store_skips_self_and_invalid():
    store = FakeAliveStore(connected=True)
    # 两个存活 decode + 一个自己 + 一个无效 epoch
    store.set(alive_key("c1", "d1:1"), '{"epoch":100}', 5.0)
    store.set(alive_key("c1", "d2:1"), '{"epoch":100}', 5.0)
    store.set(alive_key("c1", "self:1"), '{"epoch":100}', 5.0)
    store.set(alive_key("c1", "bad:1"), '{"epoch":0}', 5.0)  # 无效
    b = SnapshotBroadcaster("self:1", "c1", _Provider(free_kv_blocks=9), store)
    addrs, diff = b.select_decode_addrs()
    assert sorted(addrs) == ["d1:1", "d2:1"]
    assert sorted(diff.first_seen) == ["d1:1", "d2:1"]
    assert diff.disappeared == []


def test_broadcaster_fallback_to_incoming_routes():
    store = FakeAliveStore(connected=False)  # 未连 -> S1 自发现
    b = SnapshotBroadcaster("p:1", "c1", _Provider(), store,
                            incoming_routes=["d1:1", "d1:1", "d2:1"])
    addrs, diff = b.select_decode_addrs()
    assert addrs == ["d1:1", "d2:1"]  # 去重排序


def test_broadcaster_epoch_diff_restarted_and_disappeared():
    store = FakeAliveStore(connected=True)
    clk = {"t": 0.0}
    def now():
        return clk["t"]
    # 初始：d1, d2 存活
    store._now = now
    store.set(alive_key("c1", "d1:1"), '{"epoch":100}', 5.0)
    store.set(alive_key("c1", "d2:1"), '{"epoch":100}', 5.0)
    b = SnapshotBroadcaster("self:1", "c1", _Provider(), store)
    addrs, diff = b.select_decode_addrs()
    assert sorted(addrs) == ["d1:1", "d2:1"]
    assert sorted(diff.first_seen) == ["d1:1", "d2:1"]

    # 第二轮：d2 消失（键过期），d1 epoch 推进（仍存活），新增 d3
    clk["t"] = 6.0  # d1/d2 过期
    store.set(alive_key("c1", "d1:1"), '{"epoch":200}', 5.0)  # epoch 推进
    store.set(alive_key("c1", "d3:1"), '{"epoch":200}', 5.0)
    addrs, diff = b.select_decode_addrs()
    assert sorted(addrs) == ["d1:1", "d3:1"]
    assert diff.disappeared == ["d2:1"]
    assert diff.first_seen == ["d3:1"]
    assert diff.restarted == []  # d1 epoch 推进但 >= 上次？这里 200>=100 不算重启
    # 再让 d1 epoch 回退模拟重启
    clk["t"] = 12.0
    store.set(alive_key("c1", "d1:1"), '{"epoch":50}', 5.0)
    store.set(alive_key("c1", "d3:1"), '{"epoch":200}', 5.0)
    addrs, diff = b.select_decode_addrs()
    assert diff.restarted == ["d1:1"]


def test_broadcaster_builds_snapshot_with_self_name():
    store = FakeAliveStore(connected=False)
    b = SnapshotBroadcaster("p:1", "c1", _Provider(free_kv_blocks=11), store)
    snap = b.build_snapshot()
    assert snap.snapshot.server_name == "p:1"
    assert snap.snapshot.free_kv_blocks == 11
    assert snap.snapshot.healthy is True


def test_broadcaster_sends_per_peer():
    store = FakeAliveStore(connected=False)
    b = SnapshotBroadcaster("p:1", "c1", _Provider(), store,
                            incoming_routes=["d1:1", "d2:1"])
    sent = []
    def send(addr, snap):
        sent.append((addr, snap))
    addrs, _ = b.broadcast(send)
    assert sorted(addrs) == ["d1:1", "d2:1"]
    assert len(sent) == 2
    for addr, snap in sent:
        assert snap.snapshot.server_name == "p:1"


# ---------------- 入站 reverse-route 配对 ----------------

def test_reverse_route_register_resolve_pop():
    t = ReverseRouteTable()
    r1 = IncomingPrefillRequest(decode_engine_inference_addr="d1:1", req_id=1)
    r2 = IncomingPrefillRequest(decode_engine_inference_addr="d1:1", req_id=2)
    r3 = IncomingPrefillRequest(decode_engine_inference_addr="d2:2", req_id=1)
    t.register(r1); t.register(r2); t.register(r3)
    assert t.pending_count("d1:1") == 2
    assert t.resolve_target("d1:1", 1) == "d1:1"
    assert t.pop("d1:1", 1) is r1
    assert t.pending_count("d1:1") == 1
    assert t.resolve_target("d1:1", 9) is None  # 不存在
    assert t.pop("d2:2", 1) is r3
