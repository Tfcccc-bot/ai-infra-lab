"""
DSpark 草稿模型训练脚本.

训练流程:
1. 使用目标模型生成训练数据 (KV cache + logits)
2. 联合训练 Parallel Backbone + Sequential Head + Confidence Head
3. 三损失联合优化: L_ce + L_tv + L_conf

参考: DSpark Section 3.2 & Appendix B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import math
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dspark_draft import (
    ParallelBackbone,
    MarkovSequentialHead,
    RNNSequentialHead,
    DraftTransformerLayer,
)
from confidence_head import (
    ConfidenceHead,
    SequentialTemperatureScaling,
    compute_soft_acceptance_labels,
)
from training.loss import DSparkLoss


@dataclass
class DSparkTrainConfig:
    """DSpark 训练配置."""

    # Model
    target_model_name: str = "Qwen/Qwen2.5-0.5B"
    hidden_size: int = 896
    num_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 4864
    vocab_size: int = 151936
    target_num_layers: int = 24

    # Draft model
    draft_num_layers: int = 2
    draft_len: int = 5           # γ: 草稿块长度
    sequential_head_type: str = "markov"  # "markov" or "rnn"
    markov_rank: int = 256       # Markov 头低秩维度
    rnn_state_size: int = 512    # RNN 头状态维度

    # Training
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # Loss weights
    alpha_ce: float = 0.1
    alpha_tv: float = 0.9
    alpha_conf: float = 1.0

    # Data
    max_seq_len: int = 2048
    dataset_name: str = "openwebtext"  # or custom

    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    output_dir: str = "./dspark_checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True


class DSparkTrainer:
    """
    DSpark 训练器.

    管理完整的训练流程: 数据准备 → 模型初始化 → 训练循环 → 验证.
    """

    def __init__(self, config: DSparkTrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        # 初始化模型组件
        self.parallel_backbone = ParallelBackbone(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size,
            num_layers=config.draft_num_layers,
            vocab_size=config.vocab_size,
            draft_len=config.draft_len,
        ).to(self.device)

        # Sequential Head
        if config.sequential_head_type == "markov":
            self.sequential_head = MarkovSequentialHead(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                rank=config.markov_rank,
                draft_len=config.draft_len,
            ).to(self.device)
        else:
            self.sequential_head = RNNSequentialHead(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                state_size=config.rnn_state_size,
                rank=config.markov_rank,
                draft_len=config.draft_len,
            ).to(self.device)

        # Confidence Head
        self.confidence_head = ConfidenceHead(
            hidden_size=config.hidden_size,
            token_rank=config.markov_rank,
        ).to(self.device)

        # Loss
        self.criterion = DSparkLoss(
            draft_len=config.draft_len,
            alpha_ce=config.alpha_ce,
            alpha_tv=config.alpha_tv,
            alpha_conf=config.alpha_conf,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self._get_trainable_params(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # STS calibrator
        self.sts = SequentialTemperatureScaling(draft_len=config.draft_len)

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        self.global_step = 0

    def _get_trainable_params(self):
        """获取所有可训练参数."""
        params = []
        for model in [self.parallel_backbone, self.sequential_head, self.confidence_head]:
            params.extend(model.parameters())
        return params

    def _build_scheduler(self):
        """构建学习率调度器 (cosine with warmup)."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(
        self,
        anchor_ids: torch.Tensor,          # [batch, 1]
        target_token_ids: torch.Tensor,     # [batch, draft_len] ground truth
        target_logits: torch.Tensor,        # [batch, draft_len, vocab] from target model
    ) -> Dict[str, float]:
        """
        单步训练.

        流程:
        1. Parallel Backbone 生成 hidden states + base logits
        2. Sequential Head 生成序列增强 logits
        3. 组合得到最终 draft logits
        4. Confidence Head 预测置信度
        5. 计算联合损失并反向传播
        """
        self.parallel_backbone.train()
        self.sequential_head.train()
        self.confidence_head.train()

        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # Step 1: Parallel Backbone
            hidden_states, base_logits = self.parallel_backbone(anchor_ids, None)

            # Step 2: Sequential Head
            prev_token_ids = anchor_ids.squeeze(-1)
            sequential_logits = self.sequential_head(hidden_states, prev_token_ids)

            # Step 3: Combined draft logits
            draft_logits = base_logits + sequential_logits  # [batch, draft_len, vocab]

            # Step 4: Confidence prediction
            confidence_logits = self.confidence_head(hidden_states, prev_token_ids)

            # Step 5: Soft acceptance labels
            soft_labels = compute_soft_acceptance_labels(draft_logits, target_logits)

            # Step 6: Compute loss
            losses = self.criterion(
                draft_logits=draft_logits,
                target_logits=target_logits,
                target_token_ids=target_token_ids,
                confidence_logits=confidence_logits,
                soft_acceptance_labels=soft_labels,
            )

        # Backward
        if self.scaler:
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._get_trainable_params(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                self._get_trainable_params(), self.config.max_grad_norm
            )
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def generate_draft(
        self,
        anchor_ids: torch.Tensor,    # [batch, 1]
        target_kv_cache=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成 draft tokens (推理模式).

        Returns:
            draft_tokens: [batch, draft_len] draft token IDs
            draft_logits: [batch, draft_len, vocab]
            confidences: [batch, draft_len]
        """
        self.parallel_backbone.eval()
        self.sequential_head.eval()
        self.confidence_head.eval()

        hidden_states, base_logits = self.parallel_backbone(anchor_ids, target_kv_cache)

        prev_token_ids = anchor_ids.squeeze(-1)
        sequential_logits = self.sequential_head(hidden_states, prev_token_ids)

        draft_logits = base_logits + sequential_logits
        draft_tokens = draft_logits.argmax(dim=-1)

        confidences = torch.sigmoid(
            self.confidence_head(hidden_states, prev_token_ids)
        )

        # 应用 STS 校准
        confidences = self.sts.apply(confidences)

        return draft_tokens, draft_logits, confidences

    def calibrate_confidence(
        self,
        val_loader: DataLoader,
    ):
        """
        STS 校准: 在验证集上校准置信度头的温度参数.
        """
        self.parallel_backbone.eval()
        self.sequential_head.eval()
        self.confidence_head.eval()

        all_confidences = []
        all_labels = []

        for batch in val_loader:
            anchor_ids = batch["anchor_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            hidden_states, base_logits = self.parallel_backbone(anchor_ids, None)
            prev_token_ids = anchor_ids.squeeze(-1)
            sequential_logits = self.sequential_head(hidden_states, prev_token_ids)

            draft_logits = base_logits + sequential_logits
            draft_tokens = draft_logits.argmax(dim=-1)

            raw_conf = self.confidence_head(hidden_states, prev_token_ids)
            raw_conf = torch.sigmoid(raw_conf)

            # 真实接受标签
            accepted = (draft_tokens == target_ids).float()

            all_confidences.append(raw_conf)
            all_labels.append(accepted)

        all_confidences = torch.cat(all_confidences, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.sts.calibrate(all_confidences, all_labels)

        return self.sts.temperatures.data

    def save_checkpoint(self, path: str):
        """保存训练检查点."""
        checkpoint = {
            "global_step": self.global_step,
            "parallel_backbone": self.parallel_backbone.state_dict(),
            "sequential_head": self.sequential_head.state_dict(),
            "confidence_head": self.confidence_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "sts_temperatures": self.sts.temperatures.data,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载训练检查点."""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint["global_step"]
        self.parallel_backbone.load_state_dict(checkpoint["parallel_backbone"])
        self.sequential_head.load_state_dict(checkpoint["sequential_head"])
        self.confidence_head.load_state_dict(checkpoint["confidence_head"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.sts.temperatures.data = checkpoint["sts_temperatures"]


# ============================================================
# 混合 Draft 策略: DSpark + EAGLE-3 Feature Fusion
# ============================================================

class HybridDraftModel(nn.Module):
    """
    混合 Draft 模型: DSpark 半自回归 + EAGLE-3 Multi-Layer Feature Fusion.

    我们的优化:
    1. 用目标模型中间层特征 (Multi-Layer Feature Fusion) 增强 DSpark Parallel Backbone
    2. 短序列 (<512 tokens) 使用 DSpark 半自回归
    3. 长序列 (≥512 tokens) 切换 EAGLE-3 风格自回归 draft
    """

    def __init__(
        self,
        dspark_backbone: ParallelBackbone,
        dspark_seq_head: MarkovSequentialHead,
        target_feature_layers: List[int] = None,  # 目标模型中间层
        seq_len_threshold: int = 512,
    ):
        super().__init__()
        self.dspark_backbone = dspark_backbone
        self.dspark_seq_head = dspark_seq_head
        self.target_feature_layers = target_feature_layers or [4, 8, 16]
        self.seq_len_threshold = seq_len_threshold

        # Multi-Layer Feature Fusion (EAGLE-3 风格)
        self.feature_fusion = nn.ModuleDict({
            str(layer): nn.Sequential(
                nn.Linear(dspark_backbone.hidden_size, dspark_backbone.hidden_size),
                nn.SiLU(),
                nn.Linear(dspark_backbone.hidden_size, dspark_backbone.hidden_size),
            )
            for layer in self.target_feature_layers
        })

    def forward(
        self,
        anchor_ids: torch.Tensor,
        target_features: Optional[Dict[int, torch.Tensor]] = None,  # 目标模型中间层特征
        seq_len: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合 draft 生成.

        Args:
            anchor_ids: 锚点 token
            target_features: {layer_idx: hidden_states} 目标模型中间层特征
            seq_len: 当前序列长度

        Returns:
            draft_tokens, draft_logits
        """
        if seq_len < self.seq_len_threshold:
            # 短序列: DSpark 半自回归
            return self._dspark_draft(anchor_ids, target_features)
        else:
            # 长序列: EAGLE-3 风格自回归
            return self._eagle_draft(anchor_ids, target_features)

    def _dspark_draft(self, anchor_ids, target_features):
        """DSpark 半自回归 draft (融合目标模型特征)."""
        hidden_states, base_logits = self.dspark_backbone(anchor_ids, None)

        # 如果提供了目标模型特征, 进行融合
        if target_features is not None:
            fused_features = []
            for layer_key, features in target_features.items():
                if layer_key in self.feature_fusion:
                    fused = self.feature_fusion[str(layer_key)](features)
                    fused_features.append(fused)
            if fused_features:
                feature_boost = sum(fused_features) / len(fused_features)
                # 将融合特征加到 hidden states 中
                hidden_states = hidden_states + 0.1 * feature_boost.unsqueeze(1)

        sequential_logits = self.dspark_seq_head(
            hidden_states, anchor_ids.squeeze(-1)
        )
        draft_logits = base_logits + sequential_logits
        draft_tokens = draft_logits.argmax(dim=-1)

        return draft_tokens, draft_logits

    def _eagle_draft(self, anchor_ids, target_features):
        """
        EAGLE-3 风格自回归 draft.

        关键差异:
        1. 逐 token 自回归生成 (而非并行)
        2. 每个 token 的预测基于之前生成的 token
        3. 使用 Tree Attention 支持多路径验证
        """
        # 简化实现: 使用 DSpark backbone 但改为自回归模式
        batch_size = anchor_ids.shape[0]
        draft_len = self.dspark_backbone.draft_len
        device = anchor_ids.device

        all_logits = []
        current_ids = anchor_ids

        for _ in range(draft_len):
            # 单步前向 (模拟自回归)
            hidden, logits = self.dspark_backbone(
                current_ids[:, -1:], None
            )
            # 只取最后一个位置的 logits
            next_logits = logits[:, -1:, :]  # [batch, 1, vocab]
            next_token = next_logits.argmax(dim=-1)

            all_logits.append(next_logits)
            current_ids = torch.cat([current_ids, next_token], dim=1)

        draft_logits = torch.cat(all_logits, dim=1)  # [batch, draft_len, vocab]
        draft_tokens = draft_logits.argmax(dim=-1)

        return draft_tokens, draft_logits


# ============================================================
# 便捷训练入口
# ============================================================

def create_trainer(
    target_model_name: str = "Qwen/Qwen2.5-0.5B",
    draft_len: int = 5,
    seq_head_type: str = "markov",
    output_dir: str = "./dspark_checkpoints",
    **kwargs,
) -> DSparkTrainer:
    """
    创建 DSpark 训练器的便捷函数.

    Args:
        target_model_name: HuggingFace 目标模型名称
        draft_len: 草稿块长度 γ
        seq_head_type: 序列头类型 ("markov" or "rnn")
        output_dir: 输出目录
    """
    # 根据模型名称推断配置
    model_configs = {
        "Qwen/Qwen2.5-0.5B": {
            "hidden_size": 896,
            "num_heads": 14,
            "num_kv_heads": 2,
            "head_dim": 64,
            "intermediate_size": 4864,
            "vocab_size": 151936,
            "target_num_layers": 24,
        },
        "Qwen/Qwen2.5-1.5B": {
            "hidden_size": 1536,
            "num_heads": 12,
            "num_kv_heads": 2,
            "head_dim": 128,
            "intermediate_size": 8960,
            "vocab_size": 151936,
            "target_num_layers": 28,
        },
    }

    if target_model_name in model_configs:
        cfg = model_configs[target_model_name]
    else:
        # 默认 Qwen2.5-0.5B 配置
        cfg = model_configs["Qwen/Qwen2.5-0.5B"]

    config = DSparkTrainConfig(
        target_model_name=target_model_name,
        draft_len=draft_len,
        sequential_head_type=seq_head_type,
        output_dir=output_dir,
        **cfg,
        **kwargs,
    )

    return DSparkTrainer(config)


if __name__ == "__main__":
    # 快速验证
    print("Creating DSpark trainer...")
    trainer = create_trainer()
    print(f"  Parallel Backbone: {sum(p.numel() for p in trainer.parallel_backbone.parameters()):,} params")
    print(f"  Sequential Head: {sum(p.numel() for p in trainer.sequential_head.parameters()):,} params")
    print(f"  Confidence Head: {sum(p.numel() for p in trainer.confidence_head.parameters()):,} params")

    # 测试前向传播
    dummy_anchor = torch.randint(0, 1000, (2, 1))
    dummy_target_ids = torch.randint(0, 1000, (2, 5))
    dummy_target_logits = torch.randn(2, 5, 151936)

    print("\nRunning train step...")
    losses = trainer.train_step(dummy_anchor, dummy_target_ids, dummy_target_logits)
    print(f"  Losses: {losses}")

    print("\nDSpark trainer created successfully! ✓")
