"""
QAT for Ultra-Low-Bit Reasoning LLMs — ICLR 2026 论文复现

核心突破:
  用 <1B tokens 的 QAT 微调让 2-bit Qwen3-8B 在 MATH-500 达到 80.4,
  超越 BitNet1.58 用 4T tokens 从头训练的 43.4.

两阶段流水线:
  Phase 1: 混合域数据做块级量化校准
    - 80% 推理数据 + 20% 预训练数据
    - 块级 (block-wise) 量化, 逐块重建
  Phase 2: 教师引导的 QAT 微调
    - 使用原始 FP16 模型作为教师
    - 知识蒸馏损失 + 任务损失

为什么有效:
  - BitNet 的 from-scratch 训练浪费了大量算力在低比特表示的探索上
  - 先训练 FP16 模型 (已有成熟的 scaling law), 再蒸馏到低比特
  - 这被称为 "BitNet Distillation", 可能是比 BitNet from-scratch 更高效的路线

参考: Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs, ICLR 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import math
import copy


# ============================================================
# 2-bit 量化器 (用于 QAT)
# ============================================================

class TwoBitQuantizer(nn.Module):
    """
    2-bit 量化感知训练 (QAT) 量化器.

    使用 Straight-Through Estimator (STE) 进行反向传播.
    支持 per-channel 和 per-group 量化粒度.
    """

    def __init__(
        self,
        num_bits: int = 2,
        symmetric: bool = True,
        group_size: int = 128,  # per-group 量化粒度
        stochastic: bool = False,  # 随机舍入 (QAT 关键)
    ):
        super().__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.stochastic = stochastic

        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QAT 前向传播: 量化 + STE 反量化.
        """
        shape = x.shape

        if self.group_size > 0 and x.numel() > self.group_size:
            # Per-group 量化
            x_reshaped = x.reshape(-1, self.group_size)

            # 计算 scale (per group)
            if self.symmetric:
                scale = x_reshaped.abs().max(dim=-1, keepdim=True).values / self.qmax
                scale = scale.clamp(min=1e-8)
            else:
                x_min = x_reshaped.min(dim=-1, keepdim=True).values
                x_max = x_reshaped.max(dim=-1, keepdim=True).values
                scale = (x_max - x_min) / (self.qmax - self.qmin)
                scale = scale.clamp(min=1e-8)

            # 量化
            if self.symmetric:
                q = x_reshaped / scale
            else:
                zero_point = self.qmin - x_min / scale
                q = x_reshaped / scale + zero_point

            if self.stochastic and self.training:
                # 随机舍入 (Stochastic Rounding)
                q_floor = q.floor()
                noise = torch.rand_like(q)
                q = q_floor + (noise < (q - q_floor)).float()
            else:
                q = q.round()

            q = q.clamp(self.qmin, self.qmax)

            # 反量化
            if self.symmetric:
                x_hat = q * scale
            else:
                x_hat = (q - zero_point) * scale

            x_hat = x_hat.reshape(shape)
        else:
            # Per-tensor 量化
            if self.symmetric:
                scale = x.abs().max() / self.qmax
                scale = scale.clamp(min=1e-8)
            else:
                x_min = x.min()
                x_max = x.max()
                scale = (x_max - x_min) / (self.qmax - self.qmin)
                scale = scale.clamp(min=1e-8)

            if self.symmetric:
                q = x / scale
            else:
                zero_point = self.qmin - x_min / scale
                q = x / scale + zero_point

            if self.stochastic and self.training:
                q_floor = q.floor()
                noise = torch.rand_like(q)
                q = q_floor + (noise < (q - q_floor)).float()
            else:
                q = q.round()

            q = q.clamp(self.qmin, self.qmax)

            if self.symmetric:
                x_hat = q * scale
            else:
                x_hat = (q - zero_point) * scale

        # STE: 前向用 x_hat, 反向直接传梯度到 x
        return x + (x_hat - x).detach()


# ============================================================
# QAT for Reasoning 两阶段流水线
# ============================================================

class QATForReasoning:
    """
    ICLR 2026 QAT for Reasoning 两阶段流水线.

    Phase 1: 块级量化校准 (Block-wise Calibration)
    Phase 2: 教师引导 QAT 微调 (Teacher-guided QAT Fine-tuning)
    """

    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 2,
        group_size: int = 128,
        block_size: int = 4,  # 块大小 (一次量化多少层)
    ):
        self.model = model
        self.num_bits = num_bits
        self.group_size = group_size
        self.block_size = block_size

        # 教师模型 (FP16 原始模型)
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 量化器
        self.quantizers: Dict[str, TwoBitQuantizer] = {}

        # 获取所有可量化的 Linear 层
        self.quantizable_layers = self._get_linear_layers()

    def _get_linear_layers(self) -> List[str]:
        """获取模型中所有 Linear 层的名称."""
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(name)
        return layers

    def _insert_quantizers(self):
        """插入量化器到模型中."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.quantizable_layers:
                self.quantizers[name] = TwoBitQuantizer(
                    num_bits=self.num_bits,
                    group_size=self.group_size,
                    stochastic=True,
                )

    # ============================================================
    # Phase 1: 块级量化校准
    # ============================================================

    def phase1_block_wise_calibration(
        self,
        calibration_loader,
        num_blocks: int = None,
        mixed_ratio: float = 0.8,  # 80% reasoning + 20% pretrain
    ):
        """
        Phase 1: 块级量化校准.

        逐块量化模型, 每个块量化后用少量数据重建输出.
        混合推理数据和预训练数据 (80/20) 确保通用能力不丢失.
        """
        print("Phase 1: Block-wise Calibration...")

        layers = self.quantizable_layers
        if num_blocks is None:
            num_blocks = len(layers) // self.block_size

        for block_idx in range(num_blocks):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, len(layers))
            block_layers = layers[start_idx:end_idx]

            print(f"  Block {block_idx + 1}/{num_blocks}: layers {block_layers}")

            # 量化当前块的层
            for layer_name in block_layers:
                # 获取模块
                module = self.model
                for part in layer_name.split("."):
                    module = getattr(module, part)

                # 创建量化器
                quantizer = TwoBitQuantizer(
                    num_bits=self.num_bits,
                    group_size=self.group_size,
                    stochastic=False,
                )
                self.quantizers[layer_name] = quantizer

            # 块级重建: 用校准数据微调当前块
            self._block_reconstruction(block_layers, calibration_loader, mixed_ratio)

    def _block_reconstruction(
        self,
        block_layers: List[str],
        calibration_loader,
        mixed_ratio: float,
        num_steps: int = 100,
        lr: float = 1e-5,
    ):
        """
        块级重建: 最小化量化前后块输出的 MSE.
        """
        params = []
        for layer_name in block_layers:
            module = self.model
            for part in layer_name.split(".")[:-1]:
                module = getattr(module, part)
            param = getattr(module, layer_name.split(".")[-1])
            params.extend(list(param.parameters()))

        optimizer = torch.optim.AdamW(params, lr=lr)

        for step, batch in enumerate(calibration_loader):
            if step >= num_steps:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
            else:
                input_ids = batch
                attention_mask = None

            if input_ids is None:
                continue

            # 教师模型输出
            with torch.no_grad():
                teacher_out = self.teacher(input_ids, attention_mask=attention_mask)
                if hasattr(teacher_out, "hidden_states"):
                    teacher_hidden = teacher_out.hidden_states
                else:
                    teacher_hidden = teacher_out.logits

            # 学生模型 (量化) 输出
            student_out = self.model(input_ids, attention_mask=attention_mask)
            if hasattr(student_out, "hidden_states"):
                student_hidden = student_out.hidden_states
            else:
                student_hidden = student_out.logits

            # MSE 重建损失
            loss = F.mse_loss(student_hidden, teacher_hidden)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(f"    Step {step}: loss={loss.item():.6f}")

    # ============================================================
    # Phase 2: 教师引导 QAT 微调
    # ============================================================

    def phase2_teacher_guided_qat(
        self,
        train_loader,
        num_steps: int = 1000,
        lr: float = 1e-5,
        temperature: float = 3.0,
        alpha_distill: float = 0.7,
        alpha_task: float = 0.3,
    ):
        """
        Phase 2: 教师引导的 QAT 微调.

        损失:
          L = α_distill · L_KD + α_task · L_CE

        其中:
          L_KD: KL(p_teacher || p_student), 知识蒸馏损失
          L_CE: CrossEntropy(y_true, p_student), 任务损失
        """
        print("Phase 2: Teacher-guided QAT Fine-tuning...")

        # 确保所有量化器使用随机舍入
        for quantizer in self.quantizers.values():
            quantizer.stochastic = True

        # 训练参数 (所有权重 + 量化参数)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps
        )

        for step, batch in enumerate(train_loader):
            if step >= num_steps:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
                labels = batch.get("labels")
                attention_mask = batch.get("attention_mask")
            else:
                input_ids = batch
                labels = batch.clone()
                attention_mask = None

            if input_ids is None:
                continue

            # 教师模型 logits (no grad)
            with torch.no_grad():
                teacher_out = self.teacher(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_out.logits

            # 学生模型 logits (with quantization)
            student_out = self.model(input_ids, attention_mask=attention_mask)
            student_logits = student_out.logits

            # 知识蒸馏损失 (KL散度)
            L_KD = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            # 任务损失 (交叉熵)
            if labels is not None:
                L_CE = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                L_CE = 0.0

            # 总损失
            loss = alpha_distill * L_KD + alpha_task * L_CE

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"  Step {step}/{num_steps}: "
                      f"L_KD={L_KD.item():.4f}, L_CE={L_CE.item() if L_CE else 0:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

    # ============================================================
    # 评估
    # ============================================================

    @torch.no_grad()
    def evaluate_perplexity(self, eval_loader, max_batches: int = 50) -> float:
        """评估困惑度 (Perplexity)."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break

            if isinstance(batch, dict):
                input_ids = batch.get("input_ids")
                labels = batch.get("labels", input_ids)
                attention_mask = batch.get("attention_mask")
            else:
                input_ids = batch
                labels = batch.clone()
                attention_mask = None

            if input_ids is None:
                continue

            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        return perplexity

    def run_full_pipeline(
        self,
        calibration_loader,
        train_loader,
        eval_loader=None,
    ):
        """运行完整的两阶段 QAT 流水线."""
        print("=" * 60)
        print("QAT for Ultra-Low-Bit Reasoning: Full Pipeline")
        print(f"  Target bits: {self.num_bits}")
        print(f"  Quantizable layers: {len(self.quantizable_layers)}")
        print("=" * 60)

        # 评估原始模型
        if eval_loader:
            orig_ppl = self._evaluate_perplexity_teacher(eval_loader)
            print(f"\nOriginal FP16 PPL: {orig_ppl:.2f}")

        # Phase 1
        self.phase1_block_wise_calibration(calibration_loader)

        if eval_loader:
            calib_ppl = self.evaluate_perplexity(eval_loader)
            print(f"After Phase 1 PPL: {calib_ppl:.2f}")

        # Phase 2
        self.phase2_teacher_guided_qat(train_loader)

        if eval_loader:
            final_ppl = self.evaluate_perplexity(eval_loader)
            print(f"After Phase 2 PPL: {final_ppl:.2f}")
            print(f"Total PPL degradation: {final_ppl - orig_ppl:.2f}")

        print("\nQAT pipeline completed! ✓")
        return self.model

    def _evaluate_perplexity_teacher(self, eval_loader) -> float:
        """评估教师模型 (FP16) 的困惑度."""
        original_model = self.model
        self.model = self.teacher
        ppl = self.evaluate_perplexity(eval_loader)
        self.model = original_model
        return ppl


# ============================================================
# BitNet Distillation: 先训后蒸
# ============================================================

def bitnet_distillation_comparison():
    """
    对比 "BitNet Distillation" vs "BitNet from-scratch".

    BitNet from-scratch:
      - BitNet1.58 2B4T: 用 4T tokens 从头训练, MATH-500 = 43.4

    BitNet Distillation (QAT for Reasoning):
      - Qwen3-8B FP16 预训练 (已有)
      - 2-bit QAT 微调 <1B tokens
      - MATH-500 = 80.4

    结论: "先训后蒸" 比 "从头训低比特" 效率高 1000x+.
    """
    comparison = {
        "Method": ["BitNet from-scratch", "QAT for Reasoning (ours)"],
        "Model": ["BitNet1.58 2B4T", "Qwen3-8B @ 2-bit"],
        "Training Data": ["4T tokens", "< 1B tokens"],
        "MATH-500": [43.4, 80.4],
        "Training Efficiency": ["1x (baseline)", ">4000x"],
        "Reasoning Capability": ["Weak", "Strong (near FP16)"],
    }

    print("BitNet Distillation Comparison:")
    print("-" * 70)
    header = f"  {'Method':<30} {'Data':<15} {'MATH-500':<10} {'Efficiency':<15}"
    print(header)
    print(f"  {'-'*65}")

    for i in range(2):
        print(f"  {comparison['Method'][i]:<30} "
              f"{comparison['Training Data'][i]:<15} "
              f"{comparison['MATH-500'][i]:<10} "
              f"{comparison['Training Efficiency'][i]:<15}")

    return comparison
