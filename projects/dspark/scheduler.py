"""
DSpark 硬件感知前缀调度器 (Hardware-Aware Prefix Scheduler)

核心优化问题:
  最大化: Θ = τ · SPS(B)
  其中:
    B = Σ_r(1 + ℓ_r): 验证批次大小 (token 数)
    τ = Σ_r(1 + Σ_j a_{r,j}): 期望接受 token 数
    SPS(B): 批次大小 B 下的引擎吞吐量 (步/秒)

算法:
1. 计算存活概率 a_{r,j} = Π_{i≤j} c_{r,i}
2. 全局排序: 按 a_{r,j} 降序排列所有候选 token
3. 贪心搜索: 逐个接纳 token, 通过 SPS 查找表评估吞吐量
4. 提前停止: 当 Θ ≤ Θ_best 时立即终止

参考: DSpark Section 4.3: Hardware-Aware Prefix Scheduler
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import heapq


@dataclass
class ScheduleDecision:
    """单个请求的调度决策."""
    request_id: int
    accept_count: int       # 预期接受的 token 数 (ℓ_r)
    survival_probs: torch.Tensor  # 前缀存活概率
    tokens_to_verify: int   # 实际验证的 token 数


@dataclass
class BatchSchedule:
    """批调度结果."""
    decisions: List[ScheduleDecision]
    total_tokens: int       # 总验证 token 数 B
    expected_accepted: int  # 期望接受 token 数 τ
    throughput_estimate: float  # 估计吞吐量 Θ


class HardwareAwareScheduler:
    """
    硬件感知前缀调度器.

    核心功能:
    1. 基于置信度预测决定每个请求验证多少 token
    2. 使用 SPS 查找表建模硬件行为
    3. 贪心搜索最大化全局吞吐量
    """

    def __init__(
        self,
        max_tokens_per_step: int = 8192,  # 单步最大 token 数
        draft_len: int = 5,               # 草稿块长度 γ
        # SPS 查找表: (batch_size_tokens) -> steps_per_second
        sps_lookup: Optional[Dict[int, float]] = None,
    ):
        self.max_tokens_per_step = max_tokens_per_step
        self.draft_len = draft_len

        # 默认 SPS 查找表 (模拟数据, 实际需 profiling)
        # SPS 随 batch size 增加先升后降 (由于 GPU 利用率 vs 延迟权衡)
        self.sps_lookup = sps_lookup or {
            64:   120.0,
            128:  180.0,
            256:  240.0,
            512:  280.0,
            1024: 300.0,
            2048: 290.0,
            4096: 260.0,
            8192: 220.0,
        }

    def schedule(
        self,
        requests: List[Tuple[int, torch.Tensor]],  # [(req_id, survival_probs)]
    ) -> BatchSchedule:
        """
        执行批调度.

        Args:
            requests: 每个元素是 (request_id, survival_probs)
                      survival_probs: [draft_len] 前缀存活概率

        Returns:
            BatchSchedule 包含调度决策和性能估计
        """
        # Step 1: 计算每个请求的前缀存活概率
        candidates: List[Tuple[float, int, int]] = []  # (survival_prob, req_id, position)

        for req_id, probs in requests:
            for pos in range(len(probs)):
                # 位置 pos 的 token 的存活概率 = Π_{i≤pos} c_i
                survival = probs[:pos+1].prod().item()
                candidates.append((-survival, req_id, pos))  # 负值用于最大堆

        # Step 2: 按存活概率降序排列
        heapq.heapify(candidates)

        # Step 3: 贪心搜索
        accepted = {req_id: 0 for req_id, _ in requests}
        total_tokens = 0
        best_theta = 0.0
        best_schedule = None

        current_schedule: List[ScheduleDecision] = []
        req_accepted: Dict[int, int] = {req_id: 0 for req_id, _ in requests}
        req_tokens: Dict[int, int] = {req_id: 1 for req_id, _ in requests}  # +1 for anchor

        # 临时存储: 每个请求已接受的 token 位置
        req_positions: Dict[int, set] = {req_id: set() for req_id, _ in requests}

        while candidates and total_tokens < self.max_tokens_per_step:
            neg_survival, req_id, pos = heapq.heappop(candidates)

            # 检查是否已经被接受
            if pos in req_positions[req_id]:
                continue

            # 检查前缀完整性: 位置 pos 被接受的前提是 pos-1 也被接受
            if pos > 0 and (pos - 1) not in req_positions[req_id]:
                continue  # 前缀不完整, 跳过 (这个 token 还不能被接受)

            # 接纳这个 token
            req_positions[req_id].add(pos)
            req_accepted[req_id] = max(req_accepted[req_id], pos + 1)
            req_tokens[req_id] = 1 + req_accepted[req_id]
            total_tokens = sum(req_tokens.values())

            # 评估当前吞吐量
            theta = self._evaluate_throughput(req_accepted, req_tokens)

            if theta > best_theta:
                best_theta = theta
                best_schedule = {
                    req_id: (req_accepted[req_id], req_tokens[req_id])
                    for req_id in req_accepted
                }

        # 构建输出
        if best_schedule is None:
            # Fallback: 每个请求只验证 1 个 token (MTP-1 行为)
            best_schedule = {
                req_id: (1, 2) for req_id, _ in requests
            }
            best_theta = self._evaluate_throughput(
                {r: 1 for r, _ in requests},
                {r: 2 for r, _ in requests}
            )

        decisions = []
        for req_id, surv_probs in requests:
            accepted_count = best_schedule[req_id][0]
            decisions.append(ScheduleDecision(
                request_id=req_id,
                accept_count=accepted_count,
                survival_probs=surv_probs,
                tokens_to_verify=best_schedule[req_id][1],
            ))

        return BatchSchedule(
            decisions=decisions,
            total_tokens=sum(d.tokens_to_verify for d in decisions),
            expected_accepted=sum(d.accept_count for d in decisions),
            throughput_estimate=best_theta,
        )

    def _evaluate_throughput(
        self,
        accepted: Dict[int, int],
        tokens: Dict[int, int],
    ) -> float:
        """
        评估调度方案的吞吐量.

        Θ = τ · SPS(B)
        τ = Σ(1 + accepted_count)  — 期望接受 token 数
        B = Σ tokens_per_request   — 验证批次大小
        """
        total_B = sum(tokens.values())
        total_tau = sum(1 + a for a in accepted.values())

        # 查找 SPS (使用最近邻插值)
        sps = self._lookup_sps(total_B)

        return total_tau * sps

    def _lookup_sps(self, batch_tokens: int) -> float:
        """SPS 查找表 (最近邻插值)."""
        keys = sorted(self.sps_lookup.keys())
        for k in keys:
            if batch_tokens <= k:
                return self.sps_lookup[k]
        return self.sps_lookup[keys[-1]]

    def get_verification_budget(self, decision: ScheduleDecision) -> int:
        """
        获取单个请求的验证预算 (用于 CUDA Graph 等需要固定 batch size 的场景).

        生产环境适配:
        - CUDA Graph 和 ZOS 要求已知批次大小
        - 使用异步调度: 两步前的置信度预测确定截断长度 K
        - 因果性保证: 两步前预测形成因果屏障
        """
        return decision.tokens_to_verify - 1  # -1 for anchor token


# ============================================================
# Production Scheduler (异步调度 + 因果屏障)
# ============================================================

class AsyncProductionScheduler(HardwareAwareScheduler):
    """
    异步生产调度器.

    解决 CUDA Graph / ZOS 需要固定 batch size 的问题:
    1. 使用两步前的置信度预测确定截断长度 K
    2. 两步前预测形成因果屏障, 隔离当前 token 信息
    3. 异步执行: 调度器在 GPU 执行验证的同时准备下一批
    """

    def __init__(self, *args, lookahead: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = lookahead  # 因果屏障步数
        self._pending_decisions: Dict[int, ScheduleDecision] = {}

    def schedule_async(
        self,
        requests: List[Tuple[int, torch.Tensor]],
        current_step: int,
    ) -> BatchSchedule:
        """
        异步调度: 使用 lookahead 步前的置信度.

        Args:
            requests: 当前请求列表
            current_step: 当前步数
        """
        # 使用 lookahead 步前的置信度 (因果屏障)
        # 在实际系统中, 这需要维护置信度的历史记录
        adjusted_requests = []
        for req_id, probs in requests:
            # 截断到因果屏障允许的范围
            adjusted_probs = probs[:self.draft_len - self.lookahead]
            adjusted_requests.append((req_id, adjusted_probs))

        schedule = self.schedule(adjusted_requests)

        # 缓存决策, 供后续步骤使用
        for decision in schedule.decisions:
            self._pending_decisions[decision.request_id] = decision

        return schedule


# ============================================================
# SPS Profiler
# ============================================================

class SPSProfiler:
    """
    SPS (Steps Per Second) Profiler.

    在实际部署前, 需要对目标硬件进行 profiling,
    建立精确的 SPS 查找表.
    """

    def __init__(self, engine):
        self.engine = engine
        self.profile_data: Dict[int, float] = {}

    def profile(self, batch_sizes: List[int], num_warmup: int = 5, num_steps: int = 50):
        """对不同 batch size 进行 profiling."""
        for batch_size in batch_sizes:
            # 构造测试请求
            # ... (实际实现取决于推理引擎)
            pass

    def build_lookup(self) -> Dict[int, float]:
        """构建 SPS 查找表."""
        return self.profile_data

    def fit_sps_curve(self) -> callable:
        """
        拟合 SPS 曲线, 用于连续插值.

        SPS(B) 通常可以拟合为:
        SPS(B) = a / (1 + b * B^c)
        或使用分段线性插值.
        """
        import numpy as np

        sizes = np.array(sorted(self.profile_data.keys()))
        sps_values = np.array([self.profile_data[s] for s in sizes])

        # 简单分段线性插值
        def interpolate(batch_tokens):
            return np.interp(batch_tokens, sizes, sps_values)

        return interpolate
