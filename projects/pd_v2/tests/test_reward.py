"""模块 B 单测：奖励函数（对齐 KsanaLLM linucb_router.cpp）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit.reward import local_reward, ttft_credit, remote_reward
from bandit.router import LinUCBConfig, LocalArmContext


def _cfg(**kw):
    c = LinUCBConfig()
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_local_reward_idle_is_one():
    ctx = LocalArmContext(step_latency_us=0, step_latency_us_inst=0)
    assert local_reward(ctx, LinUCBConfig()) == 1.0


def test_local_reward_at_target_is_zero():
    # worst(=target*safety) => (target - worst)/target = 0
    cfg = LinUCBConfig(tpot_target_us=50000.0, tpot_safety_factor=0.85)
    worst = 50000.0 * 0.85
    ctx = LocalArmContext(step_latency_us=int(worst), step_latency_us_inst=int(worst))
    assert local_reward(ctx, cfg) == 0.0


def test_local_reward_above_target_clamped():
    cfg = LinUCBConfig(tpot_target_us=50000.0, tpot_safety_factor=0.85)
    ctx = LocalArmContext(step_latency_us=10_000_000, step_latency_us_inst=10_000_000)
    assert local_reward(ctx, cfg) == 0.0


def test_local_reward_conservative_takes_worse():
    # inst 尖峰应使奖励低于仅看 ewma 的情况
    cfg = LinUCBConfig(tpot_target_us=50000.0, tpot_safety_factor=0.85)
    ctx_inst = LocalArmContext(step_latency_us=0, step_latency_us_inst=30_000)
    r_inst = local_reward(ctx_inst, cfg)
    ctx_ewma = LocalArmContext(step_latency_us=30_000, step_latency_us_inst=0)
    r_ewma = local_reward(ctx_ewma, cfg)
    assert r_inst <= r_ewma  # 瞬时尖峰更保守


def test_ttft_credit_below_threshold_zero():
    cfg = LinUCBConfig(ttft_threshold_us=1_500_000)
    assert ttft_credit(1_000_000, cfg) == 0.0


def test_ttft_credit_double_threshold_is_one():
    cfg = LinUCBConfig(ttft_threshold_us=1_500_000)
    assert ttft_credit(3_000_000, cfg) == 1.0


def test_remote_reward_zero_latency_is_one():
    cfg = LinUCBConfig(ttft_threshold_us=1_500_000)
    assert remote_reward(0.0, cfg) == 1.0


def test_remote_reward_at_threshold_zero():
    cfg = LinUCBConfig(ttft_threshold_us=1_500_000)
    assert remote_reward(1_500_000, cfg) == 0.0


def test_remote_reward_above_threshold_clamped():
    cfg = LinUCBConfig(ttft_threshold_us=1_500_000)
    assert remote_reward(5_000_000, cfg) == 0.0
