"""奖励函数（对齐 KsanaLLM csrc/pd_v2/decode/linucb_router.cpp）。

三路奖励，值域均为 [0, 1]：
- local_reward   : LOCAL 臂的 TPOT 余量（节点越闲越高），取瞬时与 EWMA 中较差值（保守）
- ttft_credit    : LOCAL 臂的 TTFT 信用（远端 RTT 超阈值后线性升到 1.0）
- remote_reward  : REMOTE 臂奖励（Submit→Complete 越快越高，超 TTFT 预算即 0）

注意 LOCAL 臂的延迟奖励 = TtftCredit(select 时 peer RTT) * LocalReward(settle 时负载)，
是一个 AND 门控：仅当 REMOTE 会慢且我们仍有 TPOT 余量时才值得留本地。
"""
from __future__ import annotations


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def local_reward(ctx, cfg) -> float:
    """TPOT 余量：低于硬预算（带安全系数）越高越好。"""
    tpot_target = max(1.0, cfg.tpot_target_us * cfg.tpot_safety_factor)
    inst = float(ctx.step_latency_us_inst)
    ewma = float(ctx.step_latency_us)
    worst = max(inst, ewma)  # 保守：取较差者，负载尖峰立刻关门
    return _clamp01((tpot_target - worst) / tpot_target)


def ttft_credit(peer_ttft_us: float, cfg) -> float:
    """LOCAL 臂 TTFT 信用：peer RTT 超阈值后线性升到 1.0（2x 阈值时满）。"""
    threshold = max(1.0, cfg.ttft_threshold_us)
    return _clamp01((peer_ttft_us - threshold) / threshold)


def remote_reward(elapsed_us: float, cfg) -> float:
    """REMOTE 臂奖励：round-trip 越快越高，超过 TTFT 预算即 0。"""
    ref = max(1.0, cfg.ttft_threshold_us)
    return _clamp01(1.0 - elapsed_us / ref)
