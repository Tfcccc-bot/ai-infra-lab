"""模块 E 单测：PeerSelector，对齐 peer_selector.cpp 行为。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from management.peer_selector import PeerSelector, SelectorPolicy
from management.types import PeerHealth


def _health(segment_id, server_name, healthy=True, **kw):
    h = PeerHealth()
    h.segment_id = segment_id
    h.server_name = server_name
    h.healthy = healthy
    for k, v in kw.items():
        setattr(h, k, v)
    return h


def test_register_idempotent_healthy_false_until_snapshot():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    assert ps.registered_peer_count() == 1
    assert ps.healthy_peer_count() == 0  # 首个快照前不可选
    # 重复注册只更新 server_name，不产生多余条目
    ps.register_peer(10, "p0-renamed")
    assert ps.registered_peer_count() == 1
    assert ps.segment_id_by_name("p0-renamed") == 10


def test_unregister():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    ps.unregister_peer(10)
    assert ps.registered_peer_count() == 0
    assert ps.segment_id_by_name("p0") is None


def test_update_snapshot_marks_healthy_and_preserves_name():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    h = _health(10, "", healthy=True, free_kv_blocks=5)  # 周期推送常不带 server_name
    ps.update_snapshot(10, h)
    snap = ps.get_snapshot(10)
    assert snap.healthy is True
    assert snap.free_kv_blocks == 5
    assert snap.server_name == "p0"  # 保留已注册名


def test_update_snapshot_by_segment_rotation_via_server_name():
    # epoch 变更导致 seg_id 旋转：新 seg_id + 旧 server_name 应就地更新
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    new_seg = 99
    h = _health(new_seg, "p0", healthy=True)
    ps.update_snapshot(new_seg, h)
    assert ps.registered_peer_count() == 1
    assert ps.segment_id_by_name("p0") == new_seg
    assert ps.get_snapshot(new_seg).healthy is True


def test_healthy_peers_sorted_ascending():
    ps = PeerSelector()
    ps.register_peer(30, "p2")
    ps.register_peer(10, "p0")
    ps.register_peer(20, "p1")
    ps.update_snapshot(10, _health(10, "p0", healthy=True))
    ps.update_snapshot(20, _health(20, "p1", healthy=True))
    ps.update_snapshot(30, _health(30, "p2", healthy=False))  # 不健康
    assert ps.healthy_peers() == [10, 20]


def test_select_round_robin_over_healthy():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    ps.register_peer(20, "p1")
    ps.update_snapshot(10, _health(10, "p0", healthy=True))
    ps.update_snapshot(20, _health(20, "p1", healthy=True))
    picks = [ps.select() for _ in range(4)]
    assert picks == [10, 20, 10, 20]  # 升序 + 游标回绕


def test_select_returns_none_when_no_healthy():
    ps = PeerSelector()
    assert ps.select() is None  # 空表
    ps.register_peer(10, "p0")  # 未更新快照 -> 不健康
    assert ps.select() is None


def test_select_skips_unhealthy_and_wraps():
    ps = PeerSelector()
    ps.register_peer(10, "p0")
    ps.register_peer(20, "p1")
    ps.register_peer(30, "p2")
    ps.update_snapshot(10, _health(10, "p0", healthy=True))
    ps.update_snapshot(20, _health(20, "p1", healthy=False))
    ps.update_snapshot(30, _health(30, "p2", healthy=True))
    picks = [ps.select() for _ in range(4)]
    assert picks == [10, 30, 10, 30]  # 跳过不健康，仅 10/30 参与
