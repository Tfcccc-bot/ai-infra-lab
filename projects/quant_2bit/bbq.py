"""
BBQ (Bell Box Quantization) — 概率积分变换极低比特量化

ICLR 2026 论文复现.

核心思想:
  通过概率积分变换将权重的高斯分布"拉平"为均匀分布,
  然后进行均匀量化, 同时满足 ITO (Information-Theoretic Optimal)
  和 compute-efficient.

算法流程:
  1. 估计权重分布的 CDF (累积分布函数)
  2. 对每个权重 w, 计算 u = CDF(w) ∈ (0, 1)
  3. 将 u 均匀量化为 2^bits 个区间
  4. 去量化: w_hat = CDF^{-1}(u_hat)

为什么有效:
  - 高斯分布的两端 (tail) 信息密度低, 但传统均匀量化分配了相同的量化间隔
  - BBQ 在概率空间均匀量化, 等效于在高密度区域使用更小的量化间隔
  - 2-bit 比 QuEST 降 PPL 5 点, 1-bit 降 18 点

参考: BBQ: Bell Box Quantization, ICLR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============================================================
# BBQ 核心量化器
# ============================================================

class BBQQuantizer:
    """
    Bell Box 量化器.

    实现概率积分变换 + 均匀量化的完整流程.
    """

    def __init__(
        self,
        num_bits: int = 2,
        symmetric: bool = True,
        per_channel: bool = True,
        percentile: float = 0.9999,  # 截断百分位 (避免 tail outliers)
    ):
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.percentile = percentile

        # 量化参数
        self.num_levels = 2 ** num_bits
        self.qmin = 0 if not symmetric else -(2 ** (num_bits - 1))
        self.qmax = 2 ** num_bits - 1 if not symmetric else 2 ** (num_bits - 1) - 1

        # 缓存 CDF 参数 (在校准阶段填充)
        self.cdf_params: Dict[str, torch.Tensor] = {}

    def calibrate(self, weight: torch.Tensor, name: str = "default"):
        """
        校准: 估计权重的高斯 CDF 参数.

        假设权重近似服从 N(μ, σ²), 估计 μ 和 σ.
        """
        if self.per_channel and weight.dim() > 1:
            # Per-channel: 每个输出通道独立估计
            w_flat = weight.reshape(weight.shape[0], -1)
            mu = w_flat.mean(dim=-1)
            sigma = w_flat.std(dim=-1) + 1e-8

            # 截断到百分位
            if self.percentile < 1.0:
                k = math.sqrt(2) * torch.erfinv(
                    2 * torch.tensor(self.percentile, device=weight.device) - 1
                )
                bound = mu + k * sigma if self.symmetric else k * sigma
            else:
                bound = w_flat.abs().max(dim=-1).values

            self.cdf_params[name] = {
                "mu": mu,
                "sigma": sigma,
                "bound": bound,
                "per_channel": True,
                "shape": weight.shape,
            }
        else:
            # Per-tensor
            w_flat = weight.flatten()
            mu = w_flat.mean()
            sigma = w_flat.std() + 1e-8

            if self.percentile < 1.0:
                k = math.sqrt(2) * torch.erfinv(
                    2 * torch.tensor(self.percentile, device=weight.device) - 1
                )
                bound = mu + k * sigma if self.symmetric else k * sigma
            else:
                bound = w_flat.abs().max()

            self.cdf_params[name] = {
                "mu": mu,
                "sigma": sigma,
                "bound": bound,
                "per_channel": False,
                "shape": weight.shape,
            }

    def _gaussian_cdf(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """高斯 CDF: Φ((x - μ) / σ)."""
        return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * math.sqrt(2))))

    def _gaussian_icdf(self, u: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """高斯 ICDF: μ + σ · √2 · erf^{-1}(2u - 1)."""
        # 裁剪 u 以避免数值问题
        u = u.clamp(1e-7, 1.0 - 1e-7)
        return mu + sigma * math.sqrt(2) * torch.erfinv(2 * u - 1)

    def quantize(self, weight: torch.Tensor, name: str = "default") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量化权重.

        Returns:
            quantized: 量化后的权重 (dequantized, 用于前向传播)
            scale: 缩放因子 (用于反量化)
        """
        params = self.cdf_params.get(name)
        if params is None:
            # 自动校准
            self.calibrate(weight, name)
            params = self.cdf_params[name]

        mu = params["mu"]
        sigma = params["sigma"]
        per_channel = params["per_channel"]

        # Step 1: 截断到 [-bound, bound]
        if self.symmetric:
            bound = params["bound"]
            if per_channel:
                bound = bound.view(-1, *([1] * (weight.dim() - 1)))
                mu = mu.view(-1, *([1] * (weight.dim() - 1)))
                sigma = sigma.view(-1, *([1] * (weight.dim() - 1)))
            weight_clipped = weight.clamp(-bound, bound)
        else:
            weight_clipped = weight

        # Step 2: 概率积分变换 u = CDF(w)
        u = self._gaussian_cdf(weight_clipped, mu, sigma)

        # Step 3: 在概率空间均匀量化
        if self.symmetric:
            # 对称量化: u ∈ (0,1) → q ∈ [qmin, qmax]
            q = torch.round(u * (self.qmax - self.qmin) + self.qmin)
            q = q.clamp(self.qmin, self.qmax)
            u_hat = (q - self.qmin) / (self.qmax - self.qmin)
        else:
            q = torch.round(u * (self.num_levels - 1))
            q = q.clamp(0, self.num_levels - 1)
            u_hat = q / (self.num_levels - 1)

        # Step 4: 逆变换 w_hat = ICDF(u_hat)
        w_hat = self._gaussian_icdf(u_hat, mu, sigma)

        return w_hat, None  # BBQ 不需要显式的 scale

    def get_quantized_integer(self, weight: torch.Tensor, name: str = "default") -> torch.Tensor:
        """获取整数表示 (用于存储)."""
        params = self.cdf_params.get(name)
        if params is None:
            self.calibrate(weight, name)
            params = self.cdf_params[name]

        mu = params["mu"]
        sigma = params["sigma"]
        per_channel = params["per_channel"]

        if per_channel:
            mu = mu.view(-1, *([1] * (weight.dim() - 1)))
            sigma = sigma.view(-1, *([1] * (weight.dim() - 1)))

        u = self._gaussian_cdf(weight, mu, sigma)

        if self.symmetric:
            q = torch.round(u * (self.qmax - self.qmin) + self.qmin)
            q = q.clamp(self.qmin, self.qmax)
        else:
            q = torch.round(u * (self.num_levels - 1))
            q = q.clamp(0, self.num_levels - 1)

        return q.to(torch.int8)


# ============================================================
# BBQ 量化模块 (可直接替换 nn.Linear)
# ============================================================

class BBQLinear(nn.Module):
    """
    BBQ 量化的线性层.

    使用概率积分变换将权重从 FP32/FP16 量化到 2-bit.
    前向传播: y = dequantize(quantize(W)) @ x + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 2,
        symmetric: bool = True,
        per_channel: bool = True,
        percentile: float = 0.9999,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits

        # 原始权重 (FP32, 训练用)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # BBQ 量化器
        self.quantizer = BBQQuantizer(
            num_bits=num_bits,
            symmetric=symmetric,
            per_channel=per_channel,
            percentile=percentile,
        )

        self.reset_parameters()
        self._calibrated = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def calibrate(self):
        """校准量化参数."""
        self.quantizer.calibrate(self.weight.data, "weight")
        self._calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._calibrated:
            self.calibrate()

        # 量化 + 反量化
        w_hat, _ = self.quantizer.quantize(self.weight, "weight")

        return F.linear(x, w_hat, self.bias)

    def get_storage_size(self) -> int:
        """计算量化后的存储大小 (bits)."""
        # 2-bit 权重 + CDF 参数 (mu, sigma, bound per channel)
        weight_bits = self.out_features * self.in_features * self.num_bits
        # CDF 参数: 3 * out_features * 32 bits (FP32)
        cdf_bits = 3 * self.out_features * 32
        return weight_bits + cdf_bits

    def compression_ratio(self) -> float:
        """压缩比 (vs FP32)."""
        original_bits = self.out_features * self.in_features * 32
        return original_bits / self.get_storage_size()


# ============================================================
# BBQ + SmoothQuant 融合: 激活平滑 + 权重 BBQ 量化
# ============================================================

class BBQWithSmoothQuant(nn.Module):
    """
    BBQ 量化 + SmoothQuant 激活平滑.

    结合你的简历经验:
    1. SmoothQuant: 通过激活-权重平滑迁移降低量化误差
    2. BBQ: 概率积分变换进一步压缩到 2-bit
    """

    def __init__(
        self,
        linear: nn.Linear,
        num_bits: int = 2,
        smooth_alpha: float = 0.5,
    ):
        super().__init__()
        self.linear = BBQLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            num_bits=num_bits,
        )

        # 复制原始权重
        self.linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            self.linear.bias.data.copy_(linear.bias.data)

        # SmoothQuant: 计算平滑因子
        self.smooth_alpha = smooth_alpha
        self.smooth_scale = None  # 在校准阶段计算

    def calibrate_smooth(self, calibration_data: torch.Tensor):
        """
        SmoothQuant 校准: 计算激活-权重平滑因子.

        scale = max(|X|)^α / max(|W|)^(1-α)
        """
        with torch.no_grad():
            # 激活范围
            x_max = calibration_data.abs().max(dim=0).values

            # 权重范围
            w_max = self.linear.weight.data.abs().max(dim=0).values

            # SmoothQuant 平滑因子
            self.smooth_scale = (
                x_max.pow(self.smooth_alpha) /
                w_max.pow(1 - self.smooth_alpha)
            )
            self.smooth_scale = self.smooth_scale.clamp(min=1e-8)

            # 应用平滑到权重
            self.linear.weight.data = self.linear.weight.data / self.smooth_scale.unsqueeze(0)

            # 重新校准 BBQ
            self.linear.calibrate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_scale is not None:
            x = x * self.smooth_scale
        return self.linear(x)


# ============================================================
# 工具函数
# ============================================================

def bbq_quantize_model(
    model: nn.Module,
    num_bits: int = 2,
    skip_layers: Optional[list] = None,
) -> nn.Module:
    """
    将模型的所有 Linear 层替换为 BBQLinear.

    Args:
        model: 原始模型
        num_bits: 量化比特数
        skip_layers: 跳过的层名列表 (如 lm_head, embedding)
    """
    skip_layers = skip_layers or ["lm_head", "embed_tokens", "embed"]

    def _replace_module(parent, name, child):
        if isinstance(child, nn.Linear):
            # 检查是否在跳过列表中
            if any(skip in name for skip in skip_layers):
                return

            bbq_linear = BBQLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                num_bits=num_bits,
            )
            bbq_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                bbq_linear.bias.data.copy_(child.bias.data)
            setattr(parent, name, bbq_linear)

    for name, module in model.named_children():
        _replace_module(model, name, module)

    return model


def compute_quantization_error(
    original_weight: torch.Tensor,
    quantized_weight: torch.Tensor,
) -> Dict[str, float]:
    """
    计算量化误差指标.
    """
    diff = original_weight - quantized_weight

    return {
        "mse": (diff ** 2).mean().item(),
        "mae": diff.abs().mean().item(),
        "max_error": diff.abs().max().item(),
        "cosine_sim": F.cosine_similarity(
            original_weight.flatten().unsqueeze(0),
            quantized_weight.flatten().unsqueeze(0),
        ).item(),
        "snr_db": 10 * math.log10(
            (original_weight ** 2).mean().item() / (diff ** 2).mean().item()
        ),
    }
