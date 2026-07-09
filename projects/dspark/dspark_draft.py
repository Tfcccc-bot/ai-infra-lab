"""
DSpark 半自回归草稿模型 (Semi-Autoregressive Draft Model)

实现 DSpark 的核心组件:
1. Parallel Backbone (基于 DFlash 架构) — 一次性生成所有候选 draft token
2. Markov Sequential Head — 注入 token 间一阶依赖关系
3. KV Injection — 从目标模型提取上下文特征

参考: DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
       https://arxiv.org/abs/2607.05147
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class KVCache:
    """轻量级 KV Cache，模拟推理引擎中的缓存机制."""

    def __init__(self, max_batch_size: int, max_seq_len: int, num_layers: int,
                 num_kv_heads: int, head_dim: int, dtype=torch.float32):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # [layers, batch, heads, seq, dim]
        shape = (num_layers, max_batch_size, num_kv_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype)
        self.v_cache = torch.zeros(shape, dtype=dtype)
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor,
               batch_indices: torch.Tensor, positions: torch.Tensor):
        """更新指定层的 KV cache."""
        for i, (b, p) in enumerate(zip(batch_indices, positions)):
            self.k_cache[layer_idx, b, :, p] = k[i]
            self.v_cache[layer_idx, b, :, p] = v[i]

    def get(self, layer_idx: int, batch_idx: int, start: int, end: int):
        """获取指定范围的 KV cache."""
        return (self.k_cache[layer_idx, batch_idx, :, start:end],
                self.v_cache[layer_idx, batch_idx, :, start:end])


class RMSNorm(nn.Module):
    """RMS Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class ParallelBackbone(nn.Module):
    """
    DSpark 并行骨干网络 (Parallel Backbone).

    基于 DFlash 架构，通过 KV Injection 从目标模型提取上下文特征，
    一次性生成所有 γ 个候选 draft token 的隐藏状态和基础 logits。

    关键设计:
    - 输入: 锚点 token + (γ-1) 个 mask token
    - KV Injection: 将锚点 token 的 KV 对注入到每个 mask 位置的 attention 中
    - 输出: h_1, ..., h_γ (隐藏状态) 和 U_1, ..., U_γ (基础 logits)
    - 仅需一次前向传播, T_draft 几乎与块大小无关
    """

    def __init__(
        self,
        hidden_size: int = 896,      # Qwen2.5-0.5B hidden size
        num_heads: int = 14,
        num_kv_heads: int = 2,       # GQA
        head_dim: int = 64,
        intermediate_size: int = 4864,
        num_layers: int = 2,          # Draft model 层数 (远小于目标模型)
        vocab_size: int = 151936,
        max_seq_len: int = 2048,
        draft_len: int = 5,           # γ: 草稿块长度
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.draft_len = draft_len

        # Token embedding (共享目标模型的 embedding)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # 位置编码 (learnable, 因为我们用 mask token 填充)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        # Transformer layers (轻量级, 只有 2 层)
        self.layers = nn.ModuleList([
            DraftTransformerLayer(
                hidden_size, num_heads, num_kv_heads, head_dim,
                intermediate_size, dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(hidden_size)

        # 输出投影: 隐藏状态 → logits
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 与目标模型 embedding 权重绑定
        self.lm_head.weight = self.embed_tokens.weight

        # 初始化
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        anchor_ids: torch.Tensor,       # [batch, 1] 锚点 token
        target_kv_cache: KVCache,        # 目标模型的 KV cache (用于 KV Injection)
        target_num_layers: int = 24,     # 目标模型层数
        kv_inject_layers: Optional[List[int]] = None,  # KV Injection 的目标层
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播: 生成 draft_len 个候选 token 的隐藏状态和 logits.

        Returns:
            hidden_states: [batch, draft_len, hidden_size] 隐藏状态
            base_logits: [batch, draft_len, vocab_size] 基础 logits U_k
        """
        batch_size = anchor_ids.shape[0]
        device = anchor_ids.device

        if kv_inject_layers is None:
            # 默认从目标模型中间层注入
            kv_inject_layers = [target_num_layers // 2]

        # Step 1: 构造输入序列 [anchor, mask_1, mask_2, ..., mask_{γ-1}]
        # 使用 PAD token 或特殊的 mask token
        mask_token_id = 0  # 使用 pad token 作为 mask
        mask_tokens = torch.full(
            (batch_size, self.draft_len - 1), mask_token_id,
            dtype=torch.long, device=device
        )
        input_ids = torch.cat([anchor_ids, mask_tokens], dim=1)  # [batch, γ]

        # Step 2: Embedding + Position encoding
        positions = torch.arange(
            anchor_ids.shape[1], anchor_ids.shape[1] + self.draft_len,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

        hidden = self.embed_tokens(input_ids) + self.position_embedding(positions)

        # Step 3: KV Injection — 从目标模型提取锚点 token 的 KV 特征
        # 这是 DSpark 的核心创新: draft model 直接复用目标模型的中间特征
        injected_kv = self._inject_kv(
            anchor_ids, target_kv_cache, kv_inject_layers, target_num_layers
        )

        # Step 4: 通过 draft transformer layers
        for layer_idx, layer in enumerate(self.layers):
            # 如果这一层有注入的 KV, 将其拼接到 attention 的 KV 中
            layer_kv = injected_kv.get(layer_idx, None)
            hidden = layer(hidden, injected_kv=layer_kv)

        hidden = self.norm(hidden)  # [batch, γ, hidden_size]

        # Step 5: 输出 logits
        base_logits = self.lm_head(hidden)  # [batch, γ, vocab_size]

        return hidden, base_logits

    def _inject_kv(
        self,
        anchor_ids: torch.Tensor,
        target_kv_cache: KVCache,
        inject_layers: List[int],
        target_num_layers: int,
    ) -> dict:
        """
        KV Injection: 将目标模型指定层的 KV cache 注入 draft model.

        在 DSpark 实现中, 这一步通过以下方式完成:
        1. 使用目标模型对锚点 token 做一次前向传播
        2. 缓存中间层的 K, V
        3. 将这些 K, V 作为 draft model attention 的额外 context

        这里我们提供一个模拟实现, 实际部署时需要与推理引擎集成。
        """
        # 模拟: 为每个 draft layer 分配一组注入层
        injected = {}
        for i, layer in enumerate(self.layers):
            target_layer = inject_layers[i % len(inject_layers)]
            injected[i] = target_layer
        return injected

    def compute_draft_logits(
        self,
        hidden_states: torch.Tensor,     # [batch, draft_len, hidden_size]
        base_logits: torch.Tensor,        # [batch, draft_len, vocab_size]
        sequential_logits: torch.Tensor,  # [batch, draft_len, vocab_size] from SequentialHead
    ) -> torch.Tensor:
        """
        组合并行骨干和序列头的 logits 得到最终 draft 分布:

        p_k(v | x_0, x_{<k}) ∝ exp(U_k(v) + B_k(x_0, x_{<k}, v))

        其中 U_k 是并行骨干的输出, B_k 是序列头的输出.
        """
        combined = base_logits + sequential_logits
        return F.log_softmax(combined, dim=-1)


class DraftTransformerLayer(nn.Module):
    """Draft model 中的单个 Transformer 层 (轻量级)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.input_norm = RMSNorm(hidden_size)
        self.post_attn_norm = RMSNorm(hidden_size)

        # QKV 投影 (使用 GQA)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # FFN
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        injected_kv: Optional[int] = None,
    ) -> torch.Tensor:
        # Self-attention with causal mask
        residual = hidden
        hidden = self.input_norm(hidden)

        batch, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # GQA: repeat KV heads
        if self.num_heads > self.num_kv_heads:
            n_repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_repeat, dim=2)
            v = v.repeat_interleave(n_repeat, dim=2)

        # Scaled dot-product attention (causal)
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        hidden = residual + self.dropout(attn_output)

        # FFN (SwiGLU)
        residual = hidden
        hidden = self.post_attn_norm(hidden)
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        hidden = self.down_proj(F.silu(gate) * up)
        hidden = residual + self.dropout(hidden)

        return hidden


# ============================================================
# Markov Sequential Head
# ============================================================

class MarkovSequentialHead(nn.Module):
    """
    Markov 序列头 — DSpark 默认方案.

    通过一阶马尔可夫转移矩阵注入 token 间依赖:
    B(x_{k-1}, ·) = W_1[x_{k-1}] · W_2 ∈ R^V

    其中 W_1 ∈ R^{V×r}, W_2 ∈ R^{r×V}, 默认 r=256.
    低秩分解大幅减少存储和计算开销.

    关键洞察:
    - 极小的存储开销: r * V * 2 = 256 * 151936 * 2 ≈ 78M 参数 (可接受)
    - 极小的计算开销: 每次只需查表 + 向量乘法
    - 却能有效缓解纯并行草稿的"多模态碰撞"问题
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        rank: int = 256,
        draft_len: int = 5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.draft_len = draft_len

        # W_1 ∈ R^{V×r}: 将前一个 token 映射到低秩空间
        self.W1 = nn.Embedding(vocab_size, rank)

        # W_2 ∈ R^{r×V}: 从低秩空间映射到词汇表
        self.W2 = nn.Linear(rank, vocab_size, bias=False)

        # 隐藏状态到 logits 的投影 (与 Parallel Backbone 输出融合)
        self.hidden_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        # 初始化
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.normal_(self.W2.weight, std=0.02)
        nn.init.normal_(self.hidden_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, draft_len, hidden_size] 来自 Parallel Backbone
        prev_token_ids: torch.Tensor,  # [batch] 锚点 token (位置 0)
    ) -> torch.Tensor:
        """
        计算序列增强 logits B_k.

        B(x_{k-1}, ·) = W_1[x_{k-1}] · W_2  + hidden_proj(h_k)

        Args:
            hidden_states: Parallel Backbone 输出的隐藏状态
            prev_token_ids: 锚点 token ID (k=0)

        Returns:
            sequential_logits: [batch, draft_len, vocab_size]
        """
        batch_size, draft_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        sequential_logits = torch.zeros(batch_size, draft_len, self.vocab_size, device=device)

        # 位置 0 的 "前一个 token" 是锚点 token
        current_prev = prev_token_ids

        for k in range(draft_len):
            # 马尔可夫转移: B_k = W_1[current_prev] · W_2
            w1_out = self.W1(current_prev)  # [batch, rank]
            transition_logits = self.W2(w1_out)  # [batch, vocab_size]

            # 加上隐藏状态投影
            hidden_logits = self.hidden_proj(hidden_states[:, k, :])  # [batch, vocab_size]

            sequential_logits[:, k, :] = transition_logits + hidden_logits

            # 下一个位置的 "前一个 token" 使用 argmax draft (训练时可用 teacher forcing)
            # 在实际推理中, 这里会用 greedy/采样 的结果
            draft_token = (hidden_logits + transition_logits).argmax(dim=-1)
            current_prev = draft_token

        return sequential_logits


# ============================================================
# RNN Sequential Head (扩展方案)
# ============================================================

class RNNSequentialHead(nn.Module):
    """
    RNN 序列头 — DSpark 扩展方案.

    维护循环状态 s_k, 可访问块内完整前缀历史:
    z_k = [s_{k-1}; W_1[x_{k-1}]; h_k]
    s_k = σ(W_g·z_k) ⊙ s_{k-1} + (1-σ(W_g·z_k)) ⊙ tanh(W_c·z_k)

    相比 Markov 头, RNN 头能捕获更长的 token 间依赖,
    但实现复杂度更高, 性能提升边际递减.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        state_size: int = 512,
        rank: int = 256,
        draft_len: int = 5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.rank = rank
        self.draft_len = draft_len

        # W_1: token → 低秩空间
        self.W1 = nn.Embedding(vocab_size, rank)

        # 门控网络输入: [state, token_emb, hidden_state]
        input_size = state_size + rank + hidden_size
        self.W_gate = nn.Linear(input_size, state_size, bias=False)
        self.W_candidate = nn.Linear(input_size, state_size, bias=False)

        # 输出投影
        self.output_proj = nn.Linear(state_size, vocab_size, bias=False)
        self.hidden_proj = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_gate, self.W_candidate, self.output_proj, self.hidden_proj]:
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, draft_len, _ = hidden_states.shape
        device = hidden_states.device

        # 初始状态: 零向量
        state = torch.zeros(batch_size, self.state_size, device=device)

        current_prev = prev_token_ids
        sequential_logits = torch.zeros(batch_size, draft_len, self.vocab_size, device=device)

        for k in range(draft_len):
            # Token embedding
            token_emb = self.W1(current_prev)  # [batch, rank]
            h_k = hidden_states[:, k, :]  # [batch, hidden_size]

            # RNN cell
            z = torch.cat([state, token_emb, h_k], dim=-1)
            gate = torch.sigmoid(self.W_gate(z))
            candidate = torch.tanh(self.W_candidate(z))
            state = gate * state + (1 - gate) * candidate

            # 输出 logits
            state_logits = self.output_proj(state)
            hidden_logits = self.hidden_proj(h_k)
            sequential_logits[:, k, :] = state_logits + hidden_logits

            # Greedy next token
            draft_token = (state_logits + hidden_logits).argmax(dim=-1)
            current_prev = draft_token

        return sequential_logits
