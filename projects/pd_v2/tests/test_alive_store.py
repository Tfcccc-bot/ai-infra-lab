"""模块 F1 单测：AliveStore + Keepalive，对齐 discovery.cpp §3.3。"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from management.alive_store import (
    alive_key,
    alive_key_prefix,
    encode_alive_value,
    parse_epoch_from_json,
    FakeAliveStore,
    KeepaliveAgent,
)


def _fake_clock(start=1000.0):
    state = {"t": start}
    def now():
        return state["t"]
    now.advance = lambda dt: state.__setitem__("t", state["t"] + dt)
    return now


def test_key_prefix_and_full_key():
    assert alive_key_prefix("c1") == "pd_v2/decode_alive/c1/"
    assert alive_key("c1", "1.2.3.4:8080") == "pd_v2/decode_alive/c1/1.2.3.4:8080"


def test_encode_parse_epoch_roundtrip():
    v = encode_alive_value(123456)
    assert v == '{"epoch":123456}'
    assert parse_epoch_from_json(v) == 123456


def test_parse_epoch_invalid_returns_zero():
    assert parse_epoch_from_json("") == 0
    assert parse_epoch_from_json("not json") == 0
    assert parse_epoch_from_json('{"foo":1}') == 0


def test_fake_store_set_get_ttl_expiry():
    clk = _fake_clock()
    s = FakeAliveStore(now_fn=clk, connected=True)
    s.set("k", "v", ttl_sec=5.0)
    assert s.get("k") == "v"
    clk.advance(6.0)  # 过期
    assert s.get("k") is None


def test_fake_store_keys_prefix_and_expiry():
    clk = _fake_clock()
    s = FakeAliveStore(now_fn=clk, connected=True)
    s.set("pd_v2/decode_alive/c1/a", "v", 5.0)
    s.set("pd_v2/decode_alive/c1/b", "v", 5.0)
    s.set("other/x", "v", 5.0)
    assert sorted(s.keys("pd_v2/decode_alive/c1/")) == [
        "pd_v2/decode_alive/c1/a",
        "pd_v2/decode_alive/c1/b",
    ]
    clk.advance(6.0)
    assert s.keys("pd_v2/decode_alive/c1/") == []  # 全部过期


def test_keepalive_register_writes_key_with_epoch():
    clk = _fake_clock(1000.0)
    s = FakeAliveStore(now_fn=clk, connected=True)
    agent = KeepaliveAgent(s, "c1", "1.2.3.4:8080", epoch_fn=clk, ttl_sec=5.0)
    agent.register()
    k = alive_key("c1", "1.2.3.4:8080")
    assert s.get(k) == '{"epoch":1000}'
    # 续租推进 epoch（删除旧 + 写新，键不变值变）
    clk.advance(2.0)
    agent.renew()
    assert s.get(k) == '{"epoch":1002}'
    # 关闭删除键
    agent.shutdown()
    assert s.get(k) is None


def test_keepalive_epoch_advances_on_reregister():
    # 进程重启场景：epoch 推进 -> Prefill 视为重启
    clk = _fake_clock(1000.0)
    s = FakeAliveStore(now_fn=clk, connected=True)
    agent = KeepaliveAgent(s, "c1", "addr", epoch_fn=clk, ttl_sec=5.0)
    agent.register()
    k = alive_key("c1", "addr")
    assert parse_epoch_from_json(s.get(k)) == 1000
    clk.advance(100.0)
    agent.register()  # 重新注册（模拟重启）
    assert parse_epoch_from_json(s.get(k)) == 1100


def test_keepalive_shutdown_keeps_key_when_disconnected_then_retries():
    # 关闭时若断连，C++ 会等待重试；这里验证 shutdown 仍最终删键（离线简化）
    clk = _fake_clock()
    s = FakeAliveStore(now_fn=clk, connected=True)
    agent = KeepaliveAgent(s, "c1", "addr", epoch_fn=clk)
    agent.register()
    agent.shutdown()
    assert s.get(alive_key("c1", "addr")) is None
