"""模块 D：D 节点生产者-消费者决策流（对齐 KsanaLLM PdV2DecodeHook::Process）。

决策流（decide_route，纯函数、离线可测）：
  bypass_local : warmup / 空 prompt / global-cache 全命中 / 无残差 / 静态小请求 -> 本地跑
  local        : kLinUCB 选 kLocal（延迟奖励，请求完成时结算）
  remote       : kLinUCB 选 kRemotePeer，或 NAIVE 策略下有健康 peer -> 投递给 P（生产者-消费者）
  fail         : 无健康 peer（NAIVE 下）

投递与结算（PrefillRouter，状态类）：
  - route()         : 决策为 remote 时调用 submit_fn(req_id, prefix_offset)（生产者投递到 P）
  - on_remote_complete(req_id, elapsed_us) : REMOTE 即时奖励（round-trip 越快越高）
  - on_local_complete(req_id, local_ctx)   : LOCAL 延迟奖励（用实际本地负载结算）
  二者按 req_id 关联 in_flight 特征，分别喂给 LinUCBRouter.record_reward。
"""
from __future__ import annotations

from dataclasses import dataclass

from .router import LinUCBRouter, LocalArmContext, ROUTE_LOCAL as ARM_LOCAL, ROUTE_REMOTE as ARM_REMOTE
from . import reward as _reward

# 决策结果
ROUTE_BYPASS_LOCAL = "bypass_local"
ROUTE_LOCAL = "local"
ROUTE_REMOTE = "remote"
ROUTE_FAIL = "fail"

# 选择器策略
POLICY_NAIVE = "naive"
POLICY_LINUCB = "linucb"


@dataclass
class RouteRequestContext:
    """对齐 PdV2DecodeHook::Process 判定所需的请求 + 节点状态。"""
    req_id: int
    prefill_token_count: int
    prefix_cache_len: int = 0
    fetcher_prefix_len: int = 0
    infer_stage_is_decode: bool = False
    is_warmup: bool = False
    has_healthy_peer: bool = True
    selector_policy: str = POLICY_LINUCB
    local_complete_threshold: int = 0  # >0 且非 kLinUCB 时启用静态软 bypass
    # 以下仅 kLinUCB 使用，来自 ReadLocalLoad / per-step EWMA
    running_reqs: int = 0
    step_latency_us: int = 0
    step_latency_us_inst: int = 0


def decide_route(ctx: RouteRequestContext, router: LinUCBRouter | None = None):
    """返回 (decision, arm_ctx, arm)。

    arm_ctx 供后续奖励结算重建特征；arm 为 ARM_LOCAL / ARM_REMOTE（bypass/fail 为 None）。
    """
    # 1) warmup bypass（req_id < 0）
    if ctx.is_warmup or ctx.req_id < 0:
        return ROUTE_BYPASS_LOCAL, None, None
    # 2) 空 prefilling_tokens -> 本地
    if ctx.prefill_token_count <= 0:
        return ROUTE_BYPASS_LOCAL, None, None
    # 3) global cache 全命中（infer_stage == kDecode）
    if ctx.infer_stage_is_decode:
        return ROUTE_BYPASS_LOCAL, None, None

    prefix_len = max(ctx.prefix_cache_len, ctx.fetcher_prefix_len)
    remaining = ctx.prefill_token_count - prefix_len
    # 4) 无残差工作 -> 本地
    if remaining <= 0:
        return ROUTE_BYPASS_LOCAL, None, None

    use_linucb = (ctx.selector_policy == POLICY_LINUCB)
    # 5) 静态软 bypass（仅非 kLinUCB）
    if (not use_linucb) and ctx.local_complete_threshold > 0 and remaining <= ctx.local_complete_threshold:
        return ROUTE_BYPASS_LOCAL, None, None

    # 6) LOCAL vs REMOTE 路由
    arm_ctx = LocalArmContext(
        running_reqs=ctx.running_reqs,
        step_latency_us=ctx.step_latency_us,
        step_latency_us_inst=ctx.step_latency_us_inst,
        remaining_prefill_tokens=max(0, remaining),
    )
    if use_linucb:
        if router is None:
            router = LinUCBRouter()
        arm, _features = router.select_arm(arm_ctx)
    else:
        # NAIVE: 有 peer 就转发，否则 fail（RouteRequest == CanAccept）
        if not ctx.has_healthy_peer:
            return ROUTE_FAIL, arm_ctx, None
        arm = ARM_REMOTE

    if arm == ARM_LOCAL:
        return ROUTE_LOCAL, arm_ctx, arm
    return ROUTE_REMOTE, arm_ctx, arm


class PrefillRouter:
    """D 节点生产者-消费者投递 + 延迟/即时奖励结算（对齐 connector RouteRequest）。"""

    def __init__(self, router: LinUCBRouter | None, submit_fn, policy: str = POLICY_LINUCB):
        self.router = router
        self.submit_fn = submit_fn          # 生产者投递：submit_fn(req_id, prefix_offset) -> 触发 P 端 prefill
        self.policy = policy
        self.pending = {}                   # req_id -> arm_ctx（用于延迟/即时奖励结算）

    def route(self, ctx: RouteRequestContext) -> str:
        decision, arm_ctx, arm = decide_route(
            ctx, self.router if self.policy == POLICY_LINUCB else None
        )
        # 记录 in-flight：生产者-消费者投递（REMOTE）与本地延迟结算（LOCAL）都按 req_id 关联。
        # 仅 LOCAL/REMOTE 决策入账；bypass/fail 无待结算项。
        if arm_ctx is not None and arm is not None:
            self.pending[ctx.req_id] = {"arm_ctx": arm_ctx, "arm": arm}
        if decision == ROUTE_REMOTE:
            prefix_offset = max(ctx.prefix_cache_len, ctx.fetcher_prefix_len)
            self.submit_fn(ctx.req_id, prefix_offset)  # 生产者-消费者：投递到 P
        return decision

    def on_remote_complete(self, req_id: int, elapsed_us: float) -> None:
        """REMOTE 即时奖励：P 的 round-trip 完成时结算（按臂校验）。"""
        entry = self.pending.pop(req_id, None)
        if entry is None or entry["arm"] != ARM_REMOTE or self.router is None:
            return
        r = _reward.remote_reward(elapsed_us, self.router.config)
        self.router.record_reward(ARM_REMOTE, self.router.build_features(entry["arm_ctx"]), r)

    def on_local_complete(self, req_id: int, local_reward_ctx: LocalArmContext) -> None:
        """LOCAL 延迟奖励：D 本地解码完成时结算（按臂校验，用实际本地负载）。"""
        entry = self.pending.pop(req_id, None)
        if entry is None or entry["arm"] != ARM_LOCAL or self.router is None:
            return
        r = _reward.local_reward(local_reward_ctx, self.router.config)
        self.router.record_reward(ARM_LOCAL, self.router.build_features(entry["arm_ctx"]), r)
