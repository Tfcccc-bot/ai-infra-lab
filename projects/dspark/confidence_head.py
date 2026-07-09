"""
DSpark 置信度头 (Confidence Head) + 序列温度校准 (STS)

Confidence Head 预测每个位置的前缀存活概率, 用于调度器做验证预算分配.
STS (Sequential Temperature Scaling) 校准置信度估计的系统性偏差.

参考: DSpark Section 4: Confidence-Scheduled Verification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class ConfidenceHead(nn.Module):
    """
    置信度头: 预测每个位置 k 的 token 通过验证的条件概率.

    c_k = σ(w^T · [h_k; W_1[x_{k-1}]])

    其中:
    - h_k: Parallel Backbone 在位置 k 的隐藏状态
    - W_1[x_{k-1}]: 前一个 token 的低秩嵌入
    - c_k ∈ (0, 1): 在之前所有 token 被接受的前提下, 位置 k 通过验证的概率

    关键: 置信度头是 DSpark 区分于其他投机解码方案的核心组件.
    它使得调度器能够动态决定验证多少 token, 而不是固定长度.
    """

    def __init__(
        self,
        hidden_size: int = 896,
        token_rank: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_rank = token_rank

        # Token embedding (与 SequentialHead 共享)
        self.token_embed = nn.Embedding(151936, token_rank)

        # 置信度预测器
        input_dim = hidden_size + token_rank
        self.confidence_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_size // 2, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for module in self.confidence_predictor:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [batch, draft_len, hidden_size]
        prev_token_ids: torch.Tensor,   # [batch] 锚点 token
    ) -> torch.Tensor:
        """
        计算每个位置的置信度 c_k.

        Returns:
            confidences: [batch, draft_len] 每个位置的置信度 (0-1)
        """
        batch_size, draft_len, _ = hidden_states.shape
        device = hidden_states.device

        confidences = torch.zeros(batch_size, draft_len, device=device)

        current_prev = prev_token_ids

        for k in range(draft_len):
            h_k = hidden_states[:, k, :]
            token_emb = self.token_embed(current_prev)
            combined = torch.cat([h_k, token_emb], dim=-1)

            logit = self.confidence_predictor(combined).squeeze(-1)
            confidences[:, k] = torch.sigmoid(logit)

            # 下一个位置的 "前一个 token" (用 argmax 近似)
            current_prev = torch.zeros_like(current_prev)  # placeholder

        return confidences

    def compute_survival_probabilities(
        self, confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        计算前缀存活概率: a_{k} = Π_{i≤k} c_i

        这是调度器的核心输入 — 表示位置 k 的 token 被接受的概率.
        """
        return torch.cumprod(confidences, dim=-1)


# ============================================================
# Sequential Temperature Scaling (STS)
# ============================================================

class SequentialTemperatureScaling:
    """
    序列温度校准 (STS).

    问题: 神经网络置信度估计通常过于自信 (ECE 3%-8%).
    解决方案: 对累积乘积 Π_{i≤k} c_i 进行逐位置温度缩放.

    算法:
    1. 使用验证集, 从左到右逐位置校准
    2. 对每个位置 k, 独立执行 1D 网格搜索找到最优温度 T_k
    3. 目标: 最小化 ECE (Expected Calibration Error)
    4. 最终将平均 ECE 降至约 1%

    温度缩放是保序变换: 修正概率幅度而不破坏相对排序.
    """

    def __init__(self, draft_len: int = 5, n_bins: int = 10):
        self.draft_len = draft_len
        self.n_bins = n_bins
        self.temperatures = nn.Parameter(torch.ones(draft_len))

    def calibrate(
        self,
        raw_confidences: torch.Tensor,    # [N, draft_len]
        acceptance_labels: torch.Tensor,   # [N, draft_len] 0/1 真实接受标签
        n_search: int = 100,
    ) -> None:
        """
        逐位置校准温度参数.

        Args:
            raw_confidences: 未校准的置信度
            acceptance_labels: 真实接受标签 (1=接受, 0=拒绝)
            n_search: 每个位置的网格搜索点数
        """
        for k in range(self.draft_len):
            best_temp = 1.0
            best_ece = float("inf")

            # 1D 网格搜索 (温度范围 0.1 ~ 10.0)
            for temp in torch.logspace(-1, 1, n_search):
                calib = raw_confidences[:, :k+1].prod(dim=1) ** (1.0 / temp.item())
                ece = self._compute_ece(calib, acceptance_labels[:, k].float())
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp.item()

            self.temperatures.data[k] = best_temp

    def apply(self, raw_confidences: torch.Tensor) -> torch.Tensor:
        """
        应用校准温度.

        校准后置信度: c'_k = c_k^{1/T_k}
        校准后存活概率: a'_k = Π_{i≤k} c_i^{1/T_i}
        """
        calibrated = raw_confidences ** (1.0 / self.temperatures.unsqueeze(0))
        return calibrated

    def _compute_ece(
        self,
        confidences: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """计算 Expected Calibration Error."""
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n = len(confidences)

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_size = in_bin.sum().item()
            if bin_size > 0:
                bin_conf = confidences[in_bin].mean().item()
                bin_acc = labels[in_bin].float().mean().item()
                ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return ece


# ============================================================
# 监督信号生成
# ============================================================

def compute_soft_acceptance_labels(
    draft_logits: torch.Tensor,     # [batch, draft_len, vocab] 草稿模型 logits
    target_logits: torch.Tensor,    # [batch, draft_len, vocab] 目标模型 logits
) -> torch.Tensor:
    """
    生成置信度头的软接受标签.

    c_k* = 1 - 1/2 · ||p_k^d - p_k^t||_1

    其中 p_k^d 和 p_k^t 分别是草稿模型和目标模型在位置 k 的分布.
    总变分距离 (L1 distance / 2) 是严格的接受率上界.
    """
    draft_probs = F.softmax(draft_logits, dim=-1)
    target_probs = F.softmax(target_logits, dim=-1)

    # Total Variation distance = 1/2 * ||p - q||_1
    tv_distance = 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)

    return 1.0 - tv_distance  # [batch, draft_len]
