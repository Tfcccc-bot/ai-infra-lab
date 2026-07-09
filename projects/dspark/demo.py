"""
DSpark 端到端演示脚本.

演示 DSpark 投机解码的完整流程:
1. 加载目标模型 + DSpark draft 模型
2. 半自回归生成 draft tokens
3. 置信度调度决定验证预算
4. 目标模型验证 + 接受/拒绝
5. 性能对比: 自回归 vs DSpark vs EAGLE-3

用法:
    python demo.py --model Qwen/Qwen2.5-0.5B-Instruct --draft-len 5
"""

import torch
import argparse
import time
from typing import Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dspark_draft import ParallelBackbone, MarkovSequentialHead, KVCache
from confidence_head import ConfidenceHead, SequentialTemperatureScaling, compute_soft_acceptance_labels
from scheduler import HardwareAwareScheduler, ScheduleDecision
from training.train_draft import DSparkTrainer, create_trainer


# ============================================================
# 验证器: 目标模型验证 draft tokens
# ============================================================

class TargetModelVerifier:
    """
    目标模型验证器.

    使用目标模型验证 draft tokens, 接受匹配的 token, 拒绝不匹配的.
    这是投机解码的核心: 验证保证输出与自回归生成完全一致.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None  # 延迟加载

    def load_model(self):
        """加载目标模型."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading target model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,         # [batch, seq_len]
        draft_tokens: torch.Tensor,      # [batch, draft_len]
        max_verify: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证 draft tokens.

        Returns:
            accepted_tokens: [batch, accepted_len] 被接受的 token
            acceptance_mask: [batch, draft_len] 接受/拒绝掩码
        """
        if max_verify is None:
            max_verify = draft_tokens.shape[1]

        batch_size = draft_tokens.shape[0]
        draft_len = draft_tokens.shape[1]

        # 目标模型前向传播
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        outputs = self.model(full_input)
        logits = outputs.logits  # [batch, seq+draft, vocab]

        # 获取目标模型预测 (从 input_ids 最后一个位置开始)
        target_logits = logits[:, input_ids.shape[1]-1:-1, :]  # [batch, draft_len, vocab]
        target_tokens = target_logits.argmax(dim=-1)  # [batch, draft_len]

        # 逐位置比较
        acceptance_mask = (draft_tokens == target_tokens)

        # 找到第一个不匹配的位置
        first_mismatch = (~acceptance_mask).float().cumsum(dim=-1)
        valid_mask = (first_mismatch == 0).float()

        accepted_len = valid_mask.sum(dim=-1).long()  # [batch]

        return target_tokens, acceptance_mask, accepted_len


# ============================================================
# 自回归生成 (baseline)
# ============================================================

@torch.no_grad()
def autoregressive_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> Tuple[str, float]:
    """
    标准自回归生成 (baseline).

    Returns:
        generated_text: 生成的文本
        elapsed_time: 耗时 (秒)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    start = time.perf_counter()

    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    elapsed = time.perf_counter() - start

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return generated_text, elapsed


# ============================================================
# DSpark 投机解码生成
# ============================================================

@torch.no_grad()
def dspark_generate(
    verifier: TargetModelVerifier,
    draft_model,
    scheduler: HardwareAwareScheduler,
    prompt: str,
    max_new_tokens: int = 100,
    draft_len: int = 5,
) -> Tuple[str, float, dict]:
    """
    DSpark 投机解码生成.

    流程:
    1. 用目标模型处理 prompt, 获取锚点 token 的 KV cache
    2. 循环:
       a. DSpark draft model 生成 draft_len 个候选 token
       b. Confidence Head 预测每个位置的存活概率
       c. Scheduler 决定验证预算
       d. 目标模型验证, 接受匹配的 token
    3. 直到达到 max_new_tokens

    Returns:
        generated_text: 生成的文本
        elapsed_time: 耗时 (秒)
        stats: 统计信息
    """
    inputs = verifier.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    generated_ids = input_ids.clone()
    total_accepted = 0
    total_drafted = 0
    total_steps = 0

    start = time.perf_counter()

    while total_accepted < max_new_tokens:
        # Step 1: Draft
        anchor = generated_ids[:, -1:]  # 最后一个 token 作为锚点
        draft_tokens, draft_logits, confidences = draft_model.generate_draft(anchor)

        # Step 2: Schedule (使用存活概率)
        survival_probs = torch.cumprod(confidences, dim=-1)

        # 简化调度: 根据平均置信度决定验证预算
        avg_confidence = confidences.mean().item()
        if avg_confidence > 0.9:
            verify_budget = draft_len
        elif avg_confidence > 0.7:
            verify_budget = max(1, draft_len - 1)
        elif avg_confidence > 0.5:
            verify_budget = max(1, draft_len - 2)
        else:
            verify_budget = 1

        # Step 3: Verify
        target_tokens, acceptance_mask, accepted_len = verifier.verify(
            generated_ids, draft_tokens[:, :verify_budget]
        )

        # Step 4: Accept
        accepted_count = accepted_len[0].item()
        if accepted_count > 0:
            accepted_tokens = target_tokens[:, :accepted_count]
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=1)
        else:
            # 全部拒绝, 接受目标模型的第一个 token
            generated_ids = torch.cat([generated_ids, target_tokens[:, :1]], dim=1)
            accepted_count = 1

        total_accepted += accepted_count
        total_drafted += verify_budget
        total_steps += 1

    elapsed = time.perf_counter() - start

    generated_text = verifier.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    stats = {
        "total_accepted": total_accepted,
        "total_drafted": total_drafted,
        "acceptance_rate": total_accepted / total_drafted if total_drafted > 0 else 0,
        "total_steps": total_steps,
        "speedup_vs_ar": max_new_tokens / total_steps,  # 理想加速比
        "time": elapsed,
        "tokens_per_second": total_accepted / elapsed,
    }

    return generated_text, elapsed, stats


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="DSpark Speculative Decoding Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Target model name")
    parser.add_argument("--draft-len", type=int, default=5,
                        help="Draft block length (γ)")
    parser.add_argument("--seq-head", type=str, default="markov",
                        choices=["markov", "rnn"],
                        help="Sequential head type")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--prompt", type=str,
                        default="Explain the key idea behind speculative decoding in large language models:",
                        help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to draft model checkpoint")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare with autoregressive baseline")

    args = parser.parse_args()

    print("=" * 60)
    print("DSpark: Confidence-Scheduled Speculative Decoding Demo")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Draft length (γ): {args.draft_len}")
    print(f"  Sequential head: {args.seq_head}")
    print(f"  Max tokens: {args.max_tokens}")
    print()

    # 初始化
    print("Initializing components...")
    verifier = TargetModelVerifier(args.model)

    # 创建 draft model
    trainer = create_trainer(
        target_model_name=args.model,
        draft_len=args.draft_len,
        seq_head_type=args.seq_head,
    )
    draft_model = trainer

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        draft_model.load_checkpoint(args.checkpoint)

    scheduler = HardwareAwareScheduler(draft_len=args.draft_len)

    print("  Ready!")
    print()

    # 加载目标模型 (仅当需要 baseline 对比时)
    if args.compare_baseline:
        verifier.load_model()

    # DSpark 生成
    print(f"Prompt: {args.prompt[:100]}...")
    print("-" * 60)

    if args.compare_baseline and verifier.model is not None:
        # Baseline: 自回归
        print("Running autoregressive baseline...")
        ar_text, ar_time = autoregressive_generate(
            verifier.model, verifier.tokenizer,
            args.prompt, args.max_tokens
        )
        print(f"  AR time: {ar_time:.2f}s ({args.max_tokens / ar_time:.1f} tok/s)")
        print()

    # DSpark
    print("Running DSpark speculative decoding...")
    dspark_text, dspark_time, dspark_stats = dspark_generate(
        verifier, draft_model, scheduler,
        args.prompt, args.max_tokens, args.draft_len
    )

    print(f"  DSpark time: {dspark_time:.2f}s ({dspark_stats['tokens_per_second']:.1f} tok/s)")
    print(f"  Acceptance rate: {dspark_stats['acceptance_rate']:.2%}")
    print(f"  Total steps: {dspark_stats['total_steps']}")
    print(f"  Speedup vs AR (ideal): {dspark_stats['speedup_vs_ar']:.2f}x")
    print()

    if args.compare_baseline:
        actual_speedup = ar_time / dspark_time
        print(f"  Actual speedup: {actual_speedup:.2f}x")
        print()

    # 输出生成文本
    print("Generated text:")
    print("-" * 60)
    print(dspark_text[-500:])  # 最后 500 字符
    print("-" * 60)

    print("\nDSpark demo completed! ✓")


if __name__ == "__main__":
    main()
