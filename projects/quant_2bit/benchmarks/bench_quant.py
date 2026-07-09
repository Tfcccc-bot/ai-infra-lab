"""
2-bit 量化基准测试.

评测维度:
1. BBQ vs 传统均匀量化 (不同 bits 下的 MSE)
2. QAT for Reasoning 收敛曲线
3. 混合精度策略的存储节省
4. MoE 模型量化效率
"""

import torch
import torch.nn as nn
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbq import BBQQuantizer, BBQLinear, compute_quantization_error
from qat_reasoning import TwoBitQuantizer, QATForReasoning
from mixed_precision import (
    LayerSensitivityAnalyzer,
    MixedPrecisionConfig,
    MixedPrecisionModel,
    MoETwoBitQuantizer,
)


def benchmark_bbq_vs_uniform():
    """
    对比 BBQ 和传统均匀量化在不同 bits 下的 MSE.
    """
    print("=" * 60)
    print("Benchmark: BBQ vs Uniform Quantization")
    print("=" * 60)

    # 模拟高斯分布权重
    torch.manual_seed(42)
    weight = torch.randn(1024, 1024) * 0.5  # 高斯分布

    bits_list = [1, 2, 3, 4, 8]
    results = []

    for bits in bits_list:
        # BBQ
        bbq = BBQQuantizer(num_bits=bits, symmetric=True, per_channel=True)
        bbq.calibrate(weight)
        w_bbq, _ = bbq.quantize(weight)
        bbq_error = compute_quantization_error(weight, w_bbq)

        # 均匀量化
        uniform = TwoBitQuantizer(num_bits=bits, group_size=128)
        w_uniform = uniform(weight)
        uniform_error = compute_quantization_error(weight, w_uniform)

        results.append({
            "bits": bits,
            "bbq_mse": bbq_error["mse"],
            "uniform_mse": uniform_error["mse"],
            "bbq_cosine": bbq_error["cosine_sim"],
            "uniform_cosine": uniform_error["cosine_sim"],
        })

        improvement = (uniform_error["mse"] - bbq_error["mse"]) / uniform_error["mse"] * 100
        print(f"  {bits}-bit: BBQ MSE={bbq_error['mse']:.6f}, "
              f"Uniform MSE={uniform_error['mse']:.6f}, "
              f"BBQ improvement: {improvement:.1f}%")

    print()
    return results


def benchmark_two_bit_quality():
    """
    评测 2-bit 量化的质量 (模拟推理任务).
    """
    print("\n" + "=" * 60)
    print("Benchmark: 2-bit Quality Simulation")
    print("=" * 60)

    # 创建一个简单的 MLP 模拟 transformer FFN 层
    class MockFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(896, 4864)
            self.up = nn.Linear(896, 4864)
            self.down = nn.Linear(4864, 896)

        def forward(self, x):
            return self.down(nn.functional.silu(self.gate(x)) * self.up(x))

    model = MockFFN()

    # 生成测试数据
    torch.manual_seed(42)
    test_input = torch.randn(32, 128, 896)

    # FP32 baseline
    with torch.no_grad():
        fp32_output = model(test_input)

    # 2-bit 量化
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            bbq = BBQQuantizer(num_bits=2, symmetric=True, per_channel=True)
            bbq.calibrate(module.weight.data)
            w_hat, _ = bbq.quantize(module.weight.data)

            # 计算误差
            error = compute_quantization_error(module.weight.data, w_hat)
            print(f"  {name}: MSE={error['mse']:.6f}, "
                  f"Cosine={error['cosine_sim']:.4f}, "
                  f"SNR={error['snr_db']:.1f}dB")

            # 替换权重
            module.weight.data = w_hat

    # 量化后输出
    with torch.no_grad():
        quantized_output = model(test_input)

    # 输出相似度
    output_cosine = nn.functional.cosine_similarity(
        fp32_output.flatten().unsqueeze(0),
        quantized_output.flatten().unsqueeze(0),
    ).item()

    output_mse = nn.functional.mse_loss(fp32_output, quantized_output).item()

    print(f"\n  Output quality:")
    print(f"    Cosine similarity: {output_cosine:.4f}")
    print(f"    MSE: {output_mse:.6f}")

    return output_cosine, output_mse


def benchmark_moe_storage():
    """
    MoE 模型量化存储分析.
    """
    print("\n" + "=" * 60)
    print("Benchmark: MoE Model Storage Analysis")
    print("=" * 60)

    # DeepSeek-V4 风格配置
    configs = [
        {"name": "DeepSeek-V4-Flash", "num_experts": 128, "hidden": 2048, "ffn": 8192},
        {"name": "DeepSeek-V4-Pro", "num_experts": 256, "hidden": 4096, "ffn": 16384},
        {"name": "Qwen3-MoE-15B", "num_experts": 64, "hidden": 2048, "ffn": 6144},
    ]

    for cfg in configs:
        moe = MoETwoBitQuantizer(
            num_experts=cfg["num_experts"],
            expert_hidden_size=cfg["hidden"],
            expert_intermediate_size=cfg["ffn"],
        )

        # 模拟激活分布 (Zipf 分布: 少数 expert 被大量激活)
        torch.manual_seed(42)
        activations = torch.distributions.Categorical(
            logits=torch.log(1.0 / torch.arange(1, cfg["num_experts"] + 1).float())
        ).sample((10000,))

        moe.record_activation(activations)

        storage = moe.compute_expert_storage()

        print(f"\n  {cfg['name']}:")
        print(f"    Experts: {storage['num_experts']}, "
              f"Per-expert params: {storage['per_expert_params']:,}")
        print(f"    FP16 storage: {storage['fp16_storage_gb']:.2f} GB")
        print(f"    2-bit storage: {storage['quantized_storage_gb']:.2f} GB")
        print(f"    Compression: {storage['compression_ratio']:.1f}x")

        # 精度分布统计
        precision_dist = {4: 0, 2: 0}
        for i in range(cfg["num_experts"]):
            bits = moe.get_expert_precision(i)
            precision_dist[bits] = precision_dist.get(bits, 0) + 1

        print(f"    Precision distribution: "
              f"W4={precision_dist[4]}, W2={precision_dist[2]}")


def benchmark_mixed_precision_strategy():
    """
    混合精度策略分析: Attention W4 + FFN W2.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Mixed Precision Strategy")
    print("=" * 60)

    # Qwen2.5-0.5B 的层分布
    model_layers = {
        "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],  # 24 layers × 4 = 96
        "ffn": ["gate_proj", "up_proj", "down_proj"],            # 24 layers × 3 = 72
        "embedding": ["embed_tokens"],
        "lm_head": ["lm_head"],
    }

    # 每层的参数大小
    hidden = 896
    intermediate = 4864
    vocab = 151936

    param_sizes = {
        "q_proj": hidden * hidden,
        "k_proj": hidden * hidden // 7,  # GQA: 2 kv heads vs 14 q heads
        "v_proj": hidden * hidden // 7,
        "o_proj": hidden * hidden,
        "gate_proj": hidden * intermediate,
        "up_proj": hidden * intermediate,
        "down_proj": intermediate * hidden,
        "embed_tokens": vocab * hidden,
        "lm_head": vocab * hidden,
    }

    # 计算不同策略的存储
    strategies = [
        {"name": "FP16 (baseline)", "attention_bits": 16, "ffn_bits": 16},
        {"name": "W4A16 (uniform)", "attention_bits": 4, "ffn_bits": 4},
        {"name": "W2A16 (uniform)", "attention_bits": 2, "ffn_bits": 2},
        {"name": "Mixed: Attn W4 + FFN W2", "attention_bits": 4, "ffn_bits": 2},
    ]

    num_layers = 24

    for strat in strategies:
        total_bits = 0
        total_params = 0

        # Attention layers
        for name in model_layers["attention"]:
            bits = strat["attention_bits"]
            params = param_sizes[name]
            total_bits += params * bits * num_layers
            total_params += params * num_layers

        # FFN layers
        for name in model_layers["ffn"]:
            bits = strat["ffn_bits"]
            params = param_sizes[name]
            total_bits += params * bits * num_layers
            total_params += params * num_layers

        # Embedding + LM Head (always FP16)
        for name in ["embed_tokens", "lm_head"]:
            params = param_sizes[name]
            total_bits += params * 16
            total_params += params

        size_gb = total_bits / 8 / 1e9
        avg_bits = total_bits / total_params
        compression = (total_params * 16) / total_bits

        print(f"\n  {strat['name']}:")
        print(f"    Total params: {total_params:,}")
        print(f"    Storage: {size_gb:.2f} GB")
        print(f"    Avg bits/param: {avg_bits:.2f}")
        print(f"    Compression: {compression:.1f}x")


if __name__ == "__main__":
    benchmark_bbq_vs_uniform()
    benchmark_two_bit_quality()
    benchmark_moe_storage()
    benchmark_mixed_precision_strategy()

    print("\n" + "=" * 60)
    print("All 2-bit quantization benchmarks completed! ✓")
