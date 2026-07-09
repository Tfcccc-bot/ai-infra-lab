"""
共享的 CUDA/Triton kernel 工具函数。
用于 FlashAttention-4 风格 warp-specialized pipeline 和量化 kernel。
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS Normalization (used in LLaMA/Qwen models)."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input (RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to Q and K tensors.
    Fused in attention kernel for efficiency — extracted here for reference.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def scaled_dot_product_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference implementation of scaled dot-product attention.
    Used as correctness baseline for optimized kernels.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))

    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        L, S = query.size(-2), key.size(-2)
        causal_mask = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=S - L + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    return torch.matmul(attn_weights, value)


# ============================================================
# FlashAttention-style tiling utilities
# ============================================================

def compute_tile_size(
    head_dim: int,
    num_heads: int,
    smem_bytes: int = 48 * 1024,  # 48KB default shared memory
    dtype_size: int = 2,  # BF16
) -> Tuple[int, int]:
    """
    Compute optimal tile sizes for FlashAttention-style kernel.
    Returns (Br, Bc) — tile sizes for Q and KV tiles.
    """
    # We need: Br * d * 2 (Q), Bc * d * 2 (K, V), Br * Bc * 2 (S)
    # Br * Bc * 2 + (Br + 2*Bc) * d * 2 <= smem_bytes
    Br = 64  # Start with reasonable default
    Bc = 64
    while True:
        smem_used = Br * Bc * dtype_size + (Br + 2 * Bc) * head_dim * dtype_size
        if smem_used <= smem_bytes:
            Bc += 16
        else:
            Bc -= 16
            break
    return Br, Bc


# ============================================================
# Benchmark utilities
# ============================================================

class KernelBenchmark:
    """Simple micro-benchmark for CUDA/Triton kernels."""

    def __init__(self, warmup: int = 10, repeat: int = 100):
        self.warmup = warmup
        self.repeat = repeat

    def run(self, fn, *args, **kwargs):
        """Benchmark a function. Returns mean and std in milliseconds."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(self.repeat):
                fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()

            elapsed = start.elapsed_time(end) / self.repeat
        else:
            import time

            start = time.perf_counter()
            for _ in range(self.repeat):
                fn(*args, **kwargs)
            elapsed = (time.perf_counter() - start) / self.repeat * 1000

        return elapsed
