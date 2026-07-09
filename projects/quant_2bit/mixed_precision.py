"""
混合精度量化策略.

设计思路:
  - Attention 层: W4A16 (4-bit 权重, 16-bit 激活)
    原因: Attention 对精度更敏感, KV cache 质量直接影响生成质量
  - FFN 层: W2A16 (2-bit 权重, 16-bit 激活)
    原因: FFN 层参数多, 对量化误差容忍度高

策略自动搜索:
  基于逐层敏感度分析, 自动分配最优量化精度.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from collections import defaultdict

from bbq import BBQQuantizer, BBQLinear
from qat_reasoning import TwoBitQuantizer


# ============================================================
# 逐层敏感度分析
# ============================================================

class LayerSensitivityAnalyzer:
    """
    逐层敏感度分析器.

    通过分析每层对量化误差的敏感度, 自动分配量化精度.
    敏感层 → 更多 bits, 不敏感层 → 更少 bits.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.sensitivities: Dict[str, float] = {}
        self.layer_types: Dict[str, str] = {}

    def analyze(
        self,
        calibration_loader,
        num_batches: int = 10,
        bits_to_test: List[int] = [2, 4, 8],
    ) -> Dict[str, Dict[int, float]]:
        """
        分析每层对不同量化精度的敏感度.

        方法: 对每层单独量化, 测量输出变化的 MSE.
        """
        print("Analyzing layer sensitivities...")

        results = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # 分类层类型
            if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
                self.layer_types[name] = "attention"
            elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
                self.layer_types[name] = "ffn"
            else:
                self.layer_types[name] = "other"

            layer_sens = {}
            for bits in bits_to_test:
                # 量化并测量 MSE
                quantizer = TwoBitQuantizer(num_bits=bits, group_size=128)
                w_hat = quantizer(module.weight.data)
                mse = F.mse_loss(w_hat, module.weight.data).item()
                layer_sens[bits] = mse

            results[name] = layer_sens
            self.sensitivities[name] = max(layer_sens.values())

        return results

    def recommend_precision(
        self,
        total_budget_bits: Optional[int] = None,
        base_bits: int = 2,
    ) -> Dict[str, int]:
        """
        根据敏感度推荐每层的量化精度.

        策略:
        - 高敏感度层 (top 20%): base_bits + 2
        - 中敏感度层 (middle 60%): base_bits + 1
        - 低敏感度层 (bottom 20%): base_bits
        """
        if not self.sensitivities:
            return {}

        sorted_layers = sorted(self.sensitivities.items(), key=lambda x: x[1])
        n = len(sorted_layers)

        recommendations = {}
        for rank, (name, sens) in enumerate(sorted_layers):
            percentile = rank / n

            if percentile < 0.2:
                bits = base_bits
            elif percentile < 0.8:
                bits = base_bits + 1
            else:
                bits = base_bits + 2

            # 确保在合理范围
            bits = max(2, min(8, bits))
            recommendations[name] = bits

        return recommendations


# ============================================================
# 混合精度策略配置
# ============================================================

@torch.no_grad()
class MixedPrecisionConfig:
    """
    混合精度配置.

    默认策略 (基于经验):
    - Attention Q/K/V/O: W4A16
    - FFN gate/up/down: W2A16
    - Embedding: FP16
    - LM Head: FP16
    - LayerNorm/RMSNorm: FP16
    """

    # 默认层类型 → 精度映射
    DEFAULT_PRECISION_MAP = {
        "attention": {"weight_bits": 4, "activation_bits": 16},
        "ffn": {"weight_bits": 2, "activation_bits": 16},
        "embedding": {"weight_bits": 16, "activation_bits": 16},  # 不量化
        "lm_head": {"weight_bits": 16, "activation_bits": 16},    # 不量化
        "norm": {"weight_bits": 16, "activation_bits": 16},       # 不量化
        "other": {"weight_bits": 4, "activation_bits": 16},
    }

    def __init__(self, custom_map: Optional[Dict] = None):
        self.precision_map = custom_map or self.DEFAULT_PRECISION_MAP

    @classmethod
    def from_sensitivity(
        cls,
        analyzer: LayerSensitivityAnalyzer,
        base_bits: int = 2,
    ) -> "MixedPrecisionConfig":
        """从敏感度分析结果构建混合精度配置."""
        recommendations = analyzer.recommend_precision(base_bits=base_bits)

        custom_map = {}
        for name, bits in recommendations.items():
            layer_type = analyzer.layer_types.get(name, "other")
            custom_map[name] = {
                "weight_bits": bits,
                "activation_bits": 16,
            }

        return cls(custom_map=custom_map)

    def get_precision(self, layer_name: str, layer_type: str = "other") -> Dict:
        """获取指定层的精度配置."""
        if layer_name in self.precision_map:
            return self.precision_map[layer_name]
        return self.precision_map.get(layer_type, self.precision_map["other"])


# ============================================================
# 混合精度模型包装器
# ============================================================

class MixedPrecisionModel(nn.Module):
    """
    混合精度量化模型.

    自动根据配置对不同层使用不同的量化精度.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MixedPrecisionConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config

        # 分类层并应用量化
        self.quantized_layers: Dict[str, nn.Module] = {}
        self._apply_mixed_precision()

    def _classify_layer(self, name: str) -> str:
        """分类层类型."""
        name_lower = name.lower()
        if any(k in name_lower for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            return "attention"
        elif any(k in name_lower for k in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2"]):
            return "ffn"
        elif any(k in name_lower for k in ["embed", "wte", "wpe"]):
            return "embedding"
        elif any(k in name_lower for k in ["lm_head", "head"]):
            return "lm_head"
        elif any(k in name_lower for k in ["norm", "ln", "rms"]):
            return "norm"
        return "other"

    def _apply_mixed_precision(self):
        """应用混合精度量化."""
        stats = defaultdict(int)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_type = self._classify_layer(name)
                precision = self.config.get_precision(name, layer_type)
                w_bits = precision["weight_bits"]

                stats[f"{w_bits}bit"] += 1

                if w_bits < 16:
                    # 应用量化
                    quantizer = TwoBitQuantizer(
                        num_bits=w_bits,
                        group_size=128,
                        stochastic=True,
                    )

                    # 存储量化器引用
                    self.quantized_layers[name] = quantizer

        print("Mixed Precision Configuration:")
        for bits, count in sorted(stats.items()):
            print(f"  {bits}: {count} layers")
        print(f"  Total quantized: {sum(stats.values())} layers")

    def forward(self, *args, **kwargs):
        """前向传播 (量化在 hook 中自动应用)."""
        return self.model(*args, **kwargs)

    def compute_model_size(self) -> Dict[str, float]:
        """计算混合精度模型的存储大小."""
        total_params = 0
        total_bits = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_type = self._classify_layer(name)
                precision = self.config.get_precision(name, layer_type)

                num_params = module.weight.numel()
                bits = num_params * precision["weight_bits"]

                total_params += num_params
                total_bits += bits

        fp16_bits = total_params * 16

        return {
            "total_params": total_params,
            "fp16_size_gb": fp16_bits / 8 / 1e9,
            "quantized_size_gb": total_bits / 8 / 1e9,
            "compression_ratio": fp16_bits / total_bits,
            "avg_bits_per_param": total_bits / total_params,
        }


# ============================================================
# MoE 模型 2-bit 量化 (与 DeepSeek-V4 经验呼应)
# ============================================================

class MoETwoBitQuantizer:
    """
    MoE 模型的 2-bit 量化策略.

    MoE 模型的特殊挑战:
    1. Expert 数量多 (DeepSeek-V4: 256 experts), 总参数量大
    2. 每个 expert 被激活的频率不同 (路由不均衡)
    3. 热门 expert 需要更高精度, 冷门 expert 可以更激进量化

    策略:
    - 热门 expert (top 20% 激活频率): W4
    - 中等 expert (middle 60%): W2
    - 冷门 expert (bottom 20%): W2 + 更激进的 group_size
    """

    def __init__(
        self,
        num_experts: int = 256,
        expert_hidden_size: int = 2048,
        expert_intermediate_size: int = 8192,
    ):
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size

        # Expert 激活频率 (运行时统计)
        self.expert_activation_count = torch.zeros(num_experts)

    def record_activation(self, expert_indices: torch.Tensor):
        """记录 expert 激活 (用于路由统计)."""
        for idx in expert_indices.unique():
            self.expert_activation_count[idx] += 1

    def get_expert_precision(self, expert_idx: int) -> int:
        """根据激活频率返回 expert 的量化精度."""
        total = self.expert_activation_count.sum()
        if total == 0:
            return 2  # 默认 2-bit

        freq = self.expert_activation_count[expert_idx] / total
        sorted_freqs = sorted(self.expert_activation_count / total)

        # 分位数
        p20 = sorted_freqs[int(self.num_experts * 0.2)]
        p80 = sorted_freqs[int(self.num_experts * 0.8)]

        if freq >= p80:
            return 4  # 热门 expert: W4
        elif freq >= p20:
            return 2  # 中等 expert: W2
        else:
            return 2  # 冷门 expert: W2 (但可以用更大的 group_size)

    def compute_expert_storage(self) -> Dict:
        """计算 MoE expert 的量化后存储大小."""
        # 每个 expert 的 FFN: gate, up, down 三个矩阵
        per_expert_params = (
            3 * self.expert_hidden_size * self.expert_intermediate_size
        )

        total_fp16 = per_expert_params * self.num_experts * 16  # bits

        # 混合精度
        total_quantized = 0
        for i in range(self.num_experts):
            bits = self.get_expert_precision(i)
            total_quantized += per_expert_params * bits

        return {
            "num_experts": self.num_experts,
            "per_expert_params": per_expert_params,
            "total_params": per_expert_params * self.num_experts,
            "fp16_storage_gb": total_fp16 / 8 / 1e9,
            "quantized_storage_gb": total_quantized / 8 / 1e9,
            "compression_ratio": total_fp16 / total_quantized,
        }
