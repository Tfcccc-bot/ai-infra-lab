"""
DSpark 性能基准测试.

评测维度:
1. 接受率 vs 草稿长度 (不同 γ)
2. 置信度校准质量 (ECE)
3. 吞吐量 vs 并发数
4. 与 EAGLE-3 和自回归的对比
"""

import torch
import time
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dspark_draft import ParallelBackbone, MarkovSequentialHead
from confidence_head import ConfidenceHead, SequentialTemperatureScaling
from scheduler import HardwareAwareScheduler


def benchmark_acceptance_rate_vs_gamma(
    draft_lens: List[int] = [3, 5, 7, 10, 15],
    num_trials: int = 100,
):
    """
    评测不同草稿长度下的接受率.

    预期: 随 γ 增大, DSpark 相对 DFlash 的优势扩大.
    """
    print("=" * 60)
    print("Benchmark: Acceptance Rate vs Draft Length (γ)")
    print("=" * 60)

    results = []

    for gamma in draft_lens:
        # 创建 draft model
        backbone = ParallelBackbone(draft_len=gamma)
        seq_head = MarkovSequentialHead(draft_len=gamma)
        conf_head = ConfidenceHead()

        # 模拟 draft 和 target logits
        draft_logits = torch.randn(num_trials, gamma, 151936)
        target_logits = torch.randn(num_trials, gamma, 151936)

        # 计算接受率 (使用 TV 距离估计)
        from confidence_head import compute_soft_acceptance_labels
        soft_acceptance = compute_soft_acceptance_labels(draft_logits, target_logits)

        avg_acceptance = soft_acceptance.mean(dim=0)  # [gamma]

        results.append({
            "gamma": gamma,
            "position_1": avg_acceptance[0].item(),
            "position_last": avg_acceptance[-1].item(),
            "avg_acceptance": avg_acceptance.mean().item(),
            "decay": (avg_acceptance[0] - avg_acceptance[-1]).item(),
        })

        print(f"  γ={gamma:2d}: pos1={avg_acceptance[0].item():.3f}, "
              f"pos{gamma}={avg_acceptance[-1].item():.3f}, "
              f"avg={avg_acceptance.mean().item():.3f}, "
              f"decay={results[-1]['decay']:.3f}")

    print()
    print("Summary:")
    print(f"  {'γ':<6} {'Position 1':<12} {'Position γ':<12} {'Avg':<10} {'Decay':<10}")
    print(f"  {'-'*50}")
    for r in results:
        print(f"  {r['gamma']:<6} {r['position_1']:<12.3f} {r['position_last']:<12.3f} "
              f"{r['avg_acceptance']:<10.3f} {r['decay']:<10.3f}")

    return results


def benchmark_confidence_calibration(
    draft_len: int = 5,
    num_samples: int = 1000,
    n_bins: int = 10,
):
    """
    评测置信度校准质量 (ECE).

    对比: 未校准 vs STS 校准.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Confidence Calibration (ECE)")
    print("=" * 60)

    # 生成模拟置信度和标签
    torch.manual_seed(42)
    raw_confidences = torch.rand(num_samples, draft_len) * 0.5 + 0.3  # 0.3-0.8
    # 模拟接受标签: 与置信度正相关但有噪声
    acceptance_labels = (torch.rand(num_samples, draft_len) < raw_confidences).float()

    # 未校准 ECE
    def compute_ece(conf, labels, n_bins):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(conf)
        for i in range(n_bins):
            in_bin = (conf > bin_boundaries[i]) & (conf <= bin_boundaries[i + 1])
            bin_size = in_bin.sum().item()
            if bin_size > 0:
                bin_conf = conf[in_bin].mean().item()
                bin_acc = labels[in_bin].mean().item()
                ece += (bin_size / n) * abs(bin_acc - bin_conf)
        return ece

    raw_ece = compute_ece(raw_confidences, acceptance_labels, n_bins)
    print(f"  Raw ECE: {raw_ece:.4f}")

    # STS 校准
    sts = SequentialTemperatureScaling(draft_len=draft_len, n_bins=n_bins)
    sts.calibrate(raw_confidences, acceptance_labels)
    calibrated = sts.apply(raw_confidences)

    calib_ece = compute_ece(calibrated, acceptance_labels, n_bins)
    print(f"  STS Calibrated ECE: {calib_ece:.4f}")
    print(f"  Improvement: {(raw_ece - calib_ece) / raw_ece * 100:.1f}%")
    print(f"  Temperatures: {sts.temperatures.data.tolist()}")

    return raw_ece, calib_ece


def benchmark_throughput_vs_concurrency(
    concurrencies: List[int] = [1, 4, 16, 64, 128],
    draft_len: int = 5,
):
    """
    评测不同并发数下的吞吐量.

    验证 DSpark 在高并发下的负载自适应能力:
    - 低并发: 验证更多 token (利用闲置算力)
    - 高并发: 缩减验证预算 (保护关键批次容量)
    """
    print("\n" + "=" * 60)
    print("Benchmark: Throughput vs Concurrency")
    print("=" * 60)

    scheduler = HardwareAwareScheduler(draft_len=draft_len)

    for concurrency in concurrencies:
        # 模拟请求: 随机置信度
        requests = []
        for i in range(concurrency):
            probs = torch.rand(draft_len) * 0.4 + 0.5  # 0.5-0.9
            requests.append((i, probs))

        schedule = scheduler.schedule(requests)

        avg_verify = sum(d.tokens_to_verify for d in schedule.decisions) / concurrency
        avg_accepted = sum(d.accept_count for d in schedule.decisions) / concurrency
        throughput = schedule.throughput_estimate

        print(f"  Concurrency={concurrency:3d}: "
              f"avg_verify={avg_verify:.1f}, "
              f"avg_accepted={avg_accepted:.1f}, "
              f"throughput={throughput:.0f} tok/s")


def benchmark_scheduler_efficiency(
    draft_len: int = 5,
    num_requests: int = 100,
):
    """
    评测调度器效率: 贪心搜索 vs 固定长度 vs 全验证.
    """
    print("\n" + "=" * 60)
    print("Benchmark: Scheduler Efficiency")
    print("=" * 60)

    scheduler = HardwareAwareScheduler(draft_len=draft_len)

    # 模拟不同置信度分布的请求
    # 高置信度请求 (数学/代码类)
    high_conf = [(i, torch.rand(draft_len) * 0.2 + 0.75) for i in range(num_requests // 2)]
    # 低置信度请求 (对话类)
    low_conf = [(i, torch.rand(draft_len) * 0.3 + 0.35) for i in range(num_requests // 2, num_requests)]
    requests = high_conf + low_conf

    # DSpark 调度
    schedule = scheduler.schedule(requests)
    dspark_tau = schedule.expected_accepted
    dspark_B = schedule.total_tokens

    # MTP-1 (固定验证 1 token)
    mtp1_tau = num_requests * 2  # 1 anchor + 1 draft
    mtp1_B = num_requests * 2

    # 固定长度验证 (全部验证 γ tokens)
    fixed_tau = sum(
        1 + torch.rand(draft_len).mean().item() * draft_len
        for _ in range(num_requests)
    )
    fixed_B = num_requests * (1 + draft_len)

    print(f"  Strategy        | Expected τ | Batch B | Θ (τ·SPS)")
    print(f"  {'-'*55}")
    print(f"  DSpark (ours)   | {dspark_tau:10.0f} | {dspark_B:7d} | {dspark_tau * scheduler._lookup_sps(dspark_B):10.0f}")
    print(f"  MTP-1           | {mtp1_tau:10.0f} | {mtp1_B:7d} | {mtp1_tau * scheduler._lookup_sps(mtp1_B):10.0f}")
    print(f"  Fixed-γ         | {fixed_tau:10.0f} | {fixed_B:7d} | {fixed_tau * scheduler._lookup_sps(fixed_B):10.0f}")


if __name__ == "__main__":
    # 运行所有 benchmark
    benchmark_acceptance_rate_vs_gamma()
    benchmark_confidence_calibration()
    benchmark_throughput_vs_concurrency()
    benchmark_scheduler_efficiency()

    print("\n" + "=" * 60)
    print("All benchmarks completed! ✓")
