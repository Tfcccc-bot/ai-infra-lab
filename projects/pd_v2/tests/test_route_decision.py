"""模块 D 单测：D 节点生产者-消费者决策流（对齐 KsanaLLM PdV2DecodeHook::Process）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandit.route_decision import (
    decide_route,
    PrefillRouter,
    RouteRequestContext,
    ROUTE_BYPASS_LOCAL,
    ROUTE_LOCAL,
    ROUTE_REMOTE,
    ROUTE_FAIL,
    POLICY_NAIVE,
    POLICY_LINUCB,
)
from bandit.router import LinUCBRouter, LocalArmContext, ROUTE_LOCAL as ARM_LOCAL, ROUTE_REMOTE as ARM_REMOTE


def _ctx(**kw):
    base = dict(req_id=1, prefill_token_count=1000, selector_policy=POLICY_LINUCB)
    base.update(kw)
    return RouteRequestContext(**base)


def _ctx_to_arm_ctx(ctx):
    return LocalArmContext(
        running_reqs=ctx.running_reqs,
        step_latency_us=ctx.step_latency_us,
        step_latency_us_inst=ctx.step_latency_us_inst,
        remaining_prefill_tokens=max(0, ctx.prefill_token_count - max(ctx.prefix_cache_len, ctx.fetcher_prefix_len)),
    )


def test_bypass_warmup():
    ctx = _ctx(req_id=-1, is_warmup=True)
    assert decide_route(ctx)[0] == ROUTE_BYPASS_LOCAL


def test_bypass_empty_prefill():
    ctx = _ctx(prefill_token_count=0)
    assert decide_route(ctx)[0] == ROUTE_BYPASS_LOCAL


def test_bypass_global_cache_full_hit():
    ctx = _ctx(infer_stage_is_decode=True)
    assert decide_route(ctx)[0] == ROUTE_BYPASS_LOCAL


def test_bypass_no_residual():
    ctx = _ctx(prefill_token_count=100, prefix_cache_len=100)
    assert decide_route(ctx)[0] == ROUTE_BYPASS_LOCAL


def test_static_threshold_bypass_only_for_naive():
    ctx = _ctx(selector_policy=POLICY_NAIVE, local_complete_threshold=200, prefix_cache_len=900)
    # remaining = 1000 - 900 = 100 <= 200 => bypass
    assert decide_route(ctx)[0] == ROUTE_BYPASS_LOCAL
    # kLinUCB 下同样情况由 bandit 决定（remaining_small 特征），不静态 bypass
    ctx_lin = _ctx(selector_policy=POLICY_LINUCB, local_complete_threshold=200, prefix_cache_len=900)
    assert decide_route(ctx_lin)[0] in (ROUTE_LOCAL, ROUTE_REMOTE)


def test_linucb_idle_node_defaults_remote():
    r = LinUCBRouter()
    ctx = _ctx()
    decision, arm_ctx, arm = decide_route(ctx, r)
    assert decision == ROUTE_REMOTE
    assert arm == ARM_REMOTE
    assert arm_ctx is not None and arm_ctx.remaining_prefill_tokens == 1000


def test_naive_forwards_when_peer_present():
    ctx = _ctx(selector_policy=POLICY_NAIVE, has_healthy_peer=True)
    decision, _a, arm = decide_route(ctx)
    assert decision == ROUTE_REMOTE and arm == ARM_REMOTE


def test_naive_fails_without_peer():
    ctx = _ctx(selector_policy=POLICY_NAIVE, has_healthy_peer=False)
    decision, _a, arm = decide_route(ctx)
    assert decision == ROUTE_FAIL and arm is None


def test_producer_consumer_submits_on_remote():
    submitted = []
    r = LinUCBRouter()
    router = PrefillRouter(r, lambda rid, off: submitted.append((rid, off)), policy=POLICY_LINUCB)
    ctx = _ctx(prefix_cache_len=100, fetcher_prefix_len=50)
    decision = router.route(ctx)
    # 初始对称 => REMOTE（平局偏 REMOTE），投递一次，prefix_offset = max(100,50)=100
    assert decision == ROUTE_REMOTE
    assert submitted == [(1, 100)]


def test_local_does_not_submit():
    submitted = []
    r = LinUCBRouter()
    router = PrefillRouter(r, lambda rid, off: submitted.append((rid, off)), policy=POLICY_LINUCB)
    ctx = _ctx()
    # 闭环让 bandit 学到 LOCAL 优先
    for _ in range(300):
        arm_ctx = _ctx_to_arm_ctx(ctx)
        arm, x = r.select_arm(arm_ctx)
        r.record_reward(arm, x, 1.0 if arm == ARM_LOCAL else 0.0)
    decision = router.route(ctx)
    assert decision == ROUTE_LOCAL
    assert submitted == []  # LOCAL 不投递到 P


def test_remote_reward_settled_on_complete():
    # 验证生产者-消费者闭环：REMOTE 完成时即时结算，奖励写入 REMOTE 臂的 bandit 状态。
    # 注意：单次奖励不会让 UCB 分数上升（探索 bonus 同期收缩），故断言内部 b 被修改。
    r = LinUCBRouter()
    router = PrefillRouter(r, lambda rid, off: None, policy=POLICY_LINUCB)
    ctx = _ctx()
    arm_ctx = _ctx_to_arm_ctx(ctx)
    router.pending[1] = {"arm_ctx": arm_ctx, "arm": ARM_REMOTE}
    before_b = list(r.remote.b)
    router.on_remote_complete(1, elapsed_us=100_000)  # 快完成 -> 高奖励
    after_b = list(r.remote.b)
    assert after_b != before_b  # REMOTE 奖励已写入 bandit


def test_local_reward_settled_on_complete():
    # 验证延迟结算：LOCAL 完成时结算，奖励写入 LOCAL 臂的 bandit 状态。
    r = LinUCBRouter()
    router = PrefillRouter(r, lambda rid, off: None, policy=POLICY_LINUCB)
    ctx = _ctx()
    arm_ctx = _ctx_to_arm_ctx(ctx)
    router.pending[1] = {"arm_ctx": arm_ctx, "arm": ARM_LOCAL}
    before_b = list(r.local.b)
    router.on_local_complete(1, LocalArmContext(running_reqs=0, step_latency_us=0, step_latency_us_inst=0))
    after_b = list(r.local.b)
    assert after_b != before_b  # LOCAL 奖励已写入 bandit


def test_remote_wins_when_faster_than_local():
    # 端到端（生产者-消费者闭环）：P 很快（REMOTE 高奖励）、本地重载（LOCAL 0 奖励），
    # 多轮后 bandit 应稳定选 REMOTE。
    r = LinUCBRouter()
    router = PrefillRouter(r, lambda rid, off: None, policy=POLICY_LINUCB)
    ctx = _ctx()
    for _ in range(500):
        decision = router.route(ctx)
        if decision == ROUTE_REMOTE:
            router.on_remote_complete(ctx.req_id, elapsed_us=100_000)
        else:  # LOCAL：重载 => 低奖励(0)
            router.on_local_complete(ctx.req_id, LocalArmContext(100, 10_000_000, 10_000_000))
    assert router.route(_ctx()) == ROUTE_REMOTE
