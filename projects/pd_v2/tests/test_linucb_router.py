"""模块 C 单测：LinUCBRouter 双臂选择器（对齐 KsanaLLM linucb_router.{h,cpp}）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit.router import (
    LinUCBRouter,
    LinUCBConfig,
    LocalArmContext,
    ROUTE_LOCAL,
    ROUTE_REMOTE,
)


def test_select_tie_favors_remote_initial():
    # 初始两臂对称（Ainv=I,b=0），平局偏向 REMOTE
    r = LinUCBRouter()
    ctx = LocalArmContext()  # 全 0 => idle, 高余量
    arm, x = r.select_arm(ctx)
    assert arm == ROUTE_REMOTE
    assert len(x) == 4 and x[0] == 1.0


def test_features_idle_node():
    r = LinUCBRouter()
    ctx = LocalArmContext(running_reqs=0, remaining_prefill_tokens=0)
    x = r.build_features(ctx)
    assert x[0] == 1.0
    assert x[1] == 1.0  # idle
    assert x[2] == 1.0  # tpot_headroom（无延迟）
    assert x[3] == 1.0  # remaining_small


def test_features_busy_node():
    r = LinUCBRouter()
    ctx = LocalArmContext(running_reqs=100, remaining_prefill_tokens=1_000_000)
    x = r.build_features(ctx)
    assert x[1] == 0.0  # 满负载 => idle=0
    assert x[3] == 0.0  # 剩余很大 => remaining_small=0


def test_record_reward_updates_scores():
    r = LinUCBRouter()
    x = [1.0, 0.0, 0.0, 0.0]
    r.record_reward(ROUTE_LOCAL, x, 1.0)
    local_s = r.score_for_test(ROUTE_LOCAL, x)
    remote_s = r.score_for_test(ROUTE_REMOTE, x)
    assert local_s > remote_s  # 在 bias 维度奖励 LOCAL 后应胜出


def test_local_learns_when_rewarded():
    # 闭环（对齐真实调度）：select -> 路由 -> 记录实际 reward。
    # LOCAL 环境优秀(reward=1)、REMOTE 差(reward=0)，UCB 应学到优先 LOCAL。
    # 注意不能"硬奖励 LOCAL"：UCB 探索项会让从未更新的 REMOTE 保持最高 bonus 而永远胜出。
    r = LinUCBRouter()
    ctx = LocalArmContext()  # idle 节点
    for _ in range(300):
        arm, x = r.select_arm(ctx)
        reward = 1.0 if arm == ROUTE_LOCAL else 0.0
        r.record_reward(arm, x, reward)
    arm, _ = r.select_arm(ctx)
    assert arm == ROUTE_LOCAL


def test_remote_learns_when_rewarded():
    # 闭环：REMOTE 环境优秀、LOCAL 差，UCB 应学到优先 REMOTE。
    r = LinUCBRouter()
    ctx = LocalArmContext(running_reqs=50, remaining_prefill_tokens=100_000)
    for _ in range(300):
        arm, x = r.select_arm(ctx)
        reward = 1.0 if arm == ROUTE_REMOTE else 0.0
        r.record_reward(arm, x, reward)
    arm, _ = r.select_arm(ctx)
    assert arm == ROUTE_REMOTE


def test_reward_clamped_to_unit():
    r = LinUCBRouter()
    x = [1.0, 0.0, 0.0, 0.0]
    r.record_reward(ROUTE_LOCAL, x, 5.0)  # 超出 [0,1]
    # 不应抛错；b 累加被 clamp 到 1.0 的奖励
    local_s = r.score_for_test(ROUTE_LOCAL, x)
    assert local_s > r.score_for_test(ROUTE_REMOTE, x)
