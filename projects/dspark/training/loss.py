"""
DSpark 训练损失函数.

三个损失联合训练:
  L = α_ce · L_ce + α_tv · L_tv + α_conf · L_conf

1. L_ce (交叉熵损失): 训练草稿模型预测正确 token
2. L_tv (分布匹配损失): 最小化草稿与目标分布的总变分距离, 直接最大化接受率
3. L_conf (置信度损失): 训练置信度头预测软接受标签

所有损失使用位置加权 w_k = exp(-(k-1)/γ), 强调块前部位置.

参考: DSpark Section 3.2: Training Objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class DSparkLoss(nn.Module):
    """
    DSpark 联合训练损失.

    配置:
    - α_ce = 0.1 (交叉熵权重较小, 因为分布匹配更重要)
    - α_tv = 0.9 (分布匹配是主要优化目标)
    - α_conf = 1.0 (置信度预测与分布匹配同等重要)
    """

    def __init__(
        self,
        draft_len: int = 5,
        alpha_ce: float = 0.1,
        alpha_tv: float = 0.9,
        alpha_conf: float = 1.0,
        gamma_decay: float = 1.0,  # γ for position weighting
        reduction: str = "mean",
    ):
        super().__init__()
        self.draft_len = draft_len
        self.alpha_ce = alpha_ce
        self.alpha_tv = alpha_tv
        self.alpha_conf = alpha_conf
        self.gamma_decay = gamma_decay

        # 预计算位置权重
        positions = torch.arange(1, draft_len + 1).float()
        self.register_buffer(
            "position_weights",
            torch.exp(-(positions - 1) / gamma_decay)
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        draft_logits: torch.Tensor,         # [batch, draft_len, vocab]
        target_logits: torch.Tensor,        # [batch, draft_len, vocab]
        target_token_ids: torch.Tensor,     # [batch, draft_len] ground truth token IDs
        confidence_logits: torch.Tensor,    # [batch, draft_len] raw confidence logits
        soft_acceptance_labels: torch.Tensor, # [batch, draft_len] c_k* from compute_soft_acceptance_labels
    ) -> dict:
        """
        计算联合训练损失.

        Returns:
            dict with keys: total, ce, tv, conf (用于 logging)
        """
        batch_size = draft_logits.shape[0]
        device = draft_logits.device
        weights = self.position_weights.to(device)  # [draft_len]

        # ============================================================
        # 1. 交叉熵损失 L_ce
        # ============================================================
        # L_ce = -Σ_k w_k · log p_k^d(x_k*)
        draft_logits_flat = draft_logits.reshape(-1, draft_logits.size(-1))
        target_ids_flat = target_token_ids.reshape(-1)
        ce_per_token = self.ce_loss(draft_logits_flat, target_ids_flat)
        ce_per_token = ce_per_token.reshape(batch_size, self.draft_len)

        # 位置加权
        L_ce = (ce_per_token * weights.unsqueeze(0)).sum(dim=-1).mean()

        # ============================================================
        # 2. 分布匹配损失 L_tv (Total Variation)
        # ============================================================
        # L_tv = Σ_k w_k · ||p_k^d - p_k^t||_1
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)

        tv_per_position = (draft_probs - target_probs).abs().sum(dim=-1)  # [batch, draft_len]

        L_tv = (tv_per_position * weights.unsqueeze(0)).sum(dim=-1).mean()

        # ============================================================
        # 3. 置信度损失 L_conf
        # ============================================================
        # L_conf = Σ_k w_k · BCE(c_k, c_k*)
        conf_per_position = self.bce_loss(
            confidence_logits.reshape(-1),
            soft_acceptance_labels.reshape(-1)
        ).reshape(batch_size, self.draft_len)

        L_conf = (conf_per_position * weights.unsqueeze(0)).sum(dim=-1).mean()

        # ============================================================
        # 总损失
        # ============================================================
        L_total = self.alpha_ce * L_ce + self.alpha_tv * L_tv + self.alpha_conf * L_conf

        return {
            "total": L_total,
            "ce": L_ce,
            "tv": L_tv,
            "conf": L_conf,
        }


# ============================================================
# Position-weighted loss helper
# ============================================================

def position_weighted_loss(
    per_position_loss: torch.Tensor,  # [batch, draft_len]
    draft_len: int,
    gamma_decay: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    对逐位置损失应用指数衰减权重.

    w_k = exp(-(k-1)/γ)

    权重衰减强调块前部位置, 因为:
    1. 位置 1 的 token 接受率最高, 对整体加速贡献最大
    2. 后缀位置即使接受, 贡献也递减
    """
    positions = torch.arange(1, draft_len + 1, device=per_position_loss.device).float()
    weights = torch.exp(-(positions - 1) / gamma_decay)

    weighted = per_position_loss * weights.unsqueeze(0)

    if reduction == "mean":
        return weighted.sum(dim=-1).mean()
    elif reduction == "sum":
        return weighted.sum()
    elif reduction == "none":
        return weighted
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ============================================================
# Acceptance rate computation (for monitoring)
# ============================================================

def compute_acceptance_rate(
    draft_token_ids: torch.Tensor,     # [batch, draft_len]
    target_token_ids: torch.Tensor,    # [batch, draft_len]
) -> torch.Tensor:
    """
    计算投机解码的接受率.

    接受率 = 匹配的 token 数 / 总 draft token 数
    """
    matches = (draft_token_ids == target_token_ids).float()
    return matches.mean()


def compute_accepted_length(
    draft_token_ids: torch.Tensor,     # [batch, draft_len]
    target_token_ids: torch.Tensor,    # [batch, draft_len]
) -> torch.Tensor:
    """
    计算期望接受长度.

    接受长度 = 第一个不匹配位置之前的连续匹配数
    """
    matches = (draft_token_ids == target_token_ids)
    # 找到第一个 False 的位置
    accepted = matches.long().cumsum(dim=-1)
    # 第一个不匹配后, 所有位置都重置为 0
    first_mismatch = (~matches).float().cumsum(dim=-1)
    mask = (first_mismatch == 0).float()
    return (accepted * mask).sum(dim=-1).float().mean()
