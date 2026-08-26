"""LinUCB 双臂选择器（对齐 KsanaLLM csrc/pd_v2/decode/linucb_router.{h,cpp}）。

两臂：ROUTE_REMOTE（转发到 P 节点，默认路径） vs ROUTE_LOCAL（留 D 节点本地跑）。
- BuildFeatures : 由 LocalArmContext 构造共享 4 维特征（bias, idle, tpot_headroom, remaining_small）
- SelectArm    : 两臂各算 UCB 分数，取高分臂；平局偏向 REMOTE
- RecordReward : Sherman-Morrison 更新 Ainv + b += r*x（在线学习）

线程安全：router_mu_ 保护两臂的临界区（与 C++ 一致）。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from .linucb_math import DIM, scaled_identity, ucb_score, sherman_morrison_update
from . import reward as _reward

ROUTE_REMOTE = 0  # kRemotePeer
ROUTE_LOCAL = 1   # kLocal


@dataclass
class LocalArmContext:
    """Decode 节点在路由某请求时的实时负载（对齐 KsanaLLM LocalArmContext）。"""
    running_reqs: int = 0               # 解码中请求数；低 = 空闲，在等 Prefill
    step_latency_us: int = 0            # 每步解码延迟 EWMA（本地 TPOT 代理）
    step_latency_us_inst: int = 0       # 最近一次（非 EWMA）每步延迟
    remaining_prefill_tokens: int = 0   # 本地前缀缓存命中后剩余的待 prefill token 数


@dataclass
class LinUCBConfig:
    """可调参数（对齐 KsanaLLM LinUCBConfig，默认值一致）。"""
    alpha: float = 1.0                  # 探索权重 alpha * sqrt(x^T Ainv x)
    tpot_target_us: float = 50000.0     # TPOT 预算（us）
    tpot_safety_factor: float = 0.85    # LOCAL 门控瞄准 tpot_target * this，预留余量
    ttft_threshold_us: float = 1500000.0  # TTFT 预算（us）
    running_ref: float = 16.0           # running_reqs 视为 "全忙" 的参考值
    remaining_ref_tokens: float = 4096.0  # remaining_prefill 视为 "太大不宜留本地" 的参考值


class _Arm:
    def __init__(self):
        self.ainv = scaled_identity(1.0)  # A0 = I => Ainv0 = I
        self.b = [0.0] * DIM


class LinUCBRouter:
    def __init__(self, config: LinUCBConfig | None = None):
        self.config = config or LinUCBConfig()
        self._mu = threading.Lock()
        self.local = _Arm()
        self.remote = _Arm()

    def _arm(self, arm: int) -> _Arm:
        return self.local if arm == ROUTE_LOCAL else self.remote

    def build_features(self, ctx: LocalArmContext) -> list[float]:
        x = [0.0] * DIM
        x[0] = 1.0  # bias
        running_ref = max(1.0, self.config.running_ref)
        x[1] = 1.0 - min(ctx.running_reqs / running_ref, 1.0)  # idle
        x[2] = _reward.local_reward(ctx, self.config)          # tpot_headroom
        remaining_ref = max(1.0, self.config.remaining_ref_tokens)
        x[3] = 1.0 - min(ctx.remaining_prefill_tokens / remaining_ref, 1.0)  # remaining_small
        return x

    def _score_locked(self, arm: _Arm, x: list[float]) -> float:
        return ucb_score(arm.ainv, arm.b, x, self.config.alpha)

    def select_arm(self, ctx: LocalArmContext):
        """返回 (arm, features)。平局偏向 REMOTE。features 供后续 RecordReward 使用。"""
        x = self.build_features(ctx)
        with self._mu:
            local_score = self._score_locked(self.local, x)
            remote_score = self._score_locked(self.remote, x)
        arm = ROUTE_LOCAL if local_score > remote_score else ROUTE_REMOTE
        return arm, x

    def record_reward(self, arm: int, features: list[float], reward: float) -> None:
        """对 arm 应用 [0,1] 奖励：Sherman-Morrison 更新 Ainv + b += r*x。"""
        r = max(0.0, min(1.0, reward))
        a = self._arm(arm)
        with self._mu:
            sherman_morrison_update(a.ainv, features)
            for i in range(DIM):
                a.b[i] += r * features[i]

    def score_for_test(self, arm: int, x: list[float]) -> float:
        with self._mu:
            return self._score_locked(self._arm(arm), x)
