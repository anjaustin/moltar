from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from executorch.examples.models.llama.feed_forward import FeedForward
from executorch.examples.models.llama.llama_transformer import TransformerBlock
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.norm import RMSNorm
from executorch.examples.models.llama.rope import Rope


class ShortConvExplicit(nn.Module):
    """
    ShortConv that takes conv_state as an explicit input and returns updated state.
    This avoids mutating internal buffers, which breaks backend partitioning/verification.
    """

    def __init__(self, dim: int, L_cache: int = 3, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.L_cache = L_cache
        self.bias = bias

        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=L_cache,
            padding=0,  # we handle padding manually
            groups=dim,
            bias=bias,
        )

        # better performance in ExecuTorch with separate projections
        self.B_proj = nn.Linear(dim, dim, bias=bias)
        self.C_proj = nn.Linear(dim, dim, bias=bias)
        self.x_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [1, S, D]
            conv_state: [1, D, L_cache-1]
        Returns:
            y: [1, S, D]
            new_conv_state: [1, D, L_cache-1]
        """
        batch_size, seqlen, dim = x.size()
        torch._check(batch_size == 1)
        torch._check(dim == self.dim)
        torch._check(conv_state.dim() == 3)
        torch._check(conv_state.size(0) == 1)
        torch._check(conv_state.size(1) == self.dim)
        torch._check(conv_state.size(2) == (self.L_cache - 1))

        B = self.B_proj(x).transpose(-1, -2)  # [1, D, S]
        C = self.C_proj(x).transpose(-1, -2)  # [1, D, S]
        x_proj = self.x_proj(x).transpose(-1, -2)  # [1, D, S]

        Bx = B * x_proj  # [1, D, S]

        # Manual padding using explicit state
        Bx_padded = torch.cat([conv_state, Bx], dim=-1)  # [1, D, S+L_cache-1]

        new_conv_state = Bx_padded[..., -(self.L_cache - 1) :]  # [1, D, L_cache-1]

        conv_out = self.conv(Bx_padded)[..., : x_proj.size(-1)]  # [1, D, S]
        y = C * conv_out  # [1, D, S]

        y = y.transpose(-1, -2).contiguous()  # [1, S, D]
        y = self.out_proj(y)
        return y, new_conv_state


class ShortConvBlockExplicit(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.L_cache = 3  # matches upstream
        self.conv = ShortConvExplicit(dim, self.L_cache, bias=False)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        # unify with TransformerBlock naming
        self.attention_norm = RMSNorm(dim, norm_eps)

    def forward(
        self, x: torch.Tensor, conv_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, new_state = self.conv(self.attention_norm(x), conv_state)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_state


class AttentionMHAExplicitState(nn.Module):
    """
    MHA attention that takes KV-cache as explicit inputs and returns updated caches.

    We intentionally do NOT use KVCache.register_buffer() or in-place cache updates.
    """

    def __init__(self, args: ModelArgs, layer_id: int, rope: Rope):
        super().__init__()
        self.use_kv_cache = True
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_heads if args.n_kv_heads is None else args.n_kv_heads
        torch._check(self.n_heads % self.n_kv_heads == 0)

        model_parallel_size = 1
        self.n_local_heads = self.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.head_dim
        self.max_context_len = args.max_context_len
        self.dim = args.dim
        self.attention_qkv_bias = args.attention_qkv_bias
        self.use_qk_norm = args.use_qk_norm
        self.qk_norm_before_rope = args.qk_norm_before_rope

        if self.use_qk_norm:
            self.q_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)
            self.k_norm_fn = RMSNorm(self.head_dim, eps=args.norm_eps)

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=self.attention_qkv_bias)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=self.attention_qkv_bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.layer_id = layer_id
        self.rope = rope

        causal_mask = torch.tril(
            torch.ones(
                self.max_context_len,
                self.max_context_len,
                dtype=torch.bool,
                device="cpu",
            )
        )
        self.register_buffer("mask", causal_mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [1, S, D]
            input_pos: [S] (positions to write into cache)
            k_cache/v_cache: [1, H_kv, C, Hd] where C=max_context_len
        Returns:
            out: [1, S, D]
            k_cache_out/v_cache_out: updated caches (same shape)
        """
        bsz, seqlen, _ = x.shape
        torch._check(bsz == 1)
        torch._check(k_cache.size(0) == 1)
        torch._check(v_cache.size(0) == 1)
        torch._check(k_cache.size(1) == self.n_local_kv_heads)
        torch._check(v_cache.size(1) == self.n_local_kv_heads)
        torch._check(k_cache.size(2) == self.max_context_len)
        torch._check(v_cache.size(2) == self.max_context_len)
        torch._check(k_cache.size(3) == self.head_dim)
        torch._check(v_cache.size(3) == self.head_dim)

        # QKV projections
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        q, k = self.rope.forward(q, k, freqs_cos, freqs_sin)

        q = q.transpose(1, 2)  # [1, H, S, Hd]
        k = k.transpose(1, 2)  # [1, H_kv, S, Hd]
        v = v.transpose(1, 2)  # [1, H_kv, S, Hd]

        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm_fn(q)
            k = self.k_norm_fn(k)

        # Build causal mask rows for these positions.
        # mask is always 2D; indexing yields [S, C], which broadcasts in SDPA.
        attn_mask = self.mask[input_pos]

        # Out-of-place cache update to avoid mutation nodes in export.
        # index_copy uses input_pos as write indices along sequence dimension.
        k_cache_out = k_cache.index_copy(2, input_pos, k)
        v_cache_out = v_cache.index_copy(2, input_pos, v)

        # Expand out keys/values for GQA
        k_full = k_cache_out.repeat_interleave(self.n_rep, dim=1)
        v_full = v_cache_out.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        out = self.wo(y)
        return out, k_cache_out, v_cache_out


class TransformerBlockExplicitState(nn.Module):
    """
    Like TransformerBlock, but with explicit KV-cache in/out (no internal mutation).
    """

    def __init__(self, args: ModelArgs, attention: AttentionMHAExplicitState):
        super().__init__()
        self.dim = args.dim
        self.attention = attention
        assert args.hidden_dim is not None
        self.feed_forward = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        input_pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k_out, v_out = self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin, input_pos, k_cache, v_cache
        )
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, k_out, v_out


@dataclass(frozen=True)
class Lfm2ExplicitStateLayout:
    conv_layer_ids: List[int]
    attn_layer_ids: List[int]


class Lfm2ExplicitStateModel(nn.Module):
    """
    LFM2 (hybrid conv + attention) with explicit state:
      inputs:  tokens, input_pos, conv_state, k_cache, v_cache
      outputs: logits, conv_state_out, k_cache_out, v_cache_out
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        if not params.layer_types:
            raise ValueError("LFM2 requires params.layer_types to be set (hybrid conv/attention).")
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.max_context_len = params.max_context_len

        self.rope = Rope(params)
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        layers = nn.ModuleList()
        conv_ids: List[int] = []
        attn_ids: List[int] = []
        for layer_id in range(params.n_layers):
            if params.layer_types[layer_id] == "conv":
                conv_ids.append(layer_id)
                layers.append(
                    ShortConvBlockExplicit(
                        dim=params.dim,
                        hidden_dim=params.hidden_dim,  # type: ignore[arg-type]
                        norm_eps=params.norm_eps,
                    )
                )
            else:
                attn_ids.append(layer_id)
                attn = AttentionMHAExplicitState(params, layer_id, self.rope)
                layers.append(TransformerBlockExplicitState(params, attn))
        self.layers = layers
        self.layout = Lfm2ExplicitStateLayout(conv_layer_ids=conv_ids, attn_layer_ids=attn_ids)

        # LFM2 assumes batch=1 currently (matches upstream ShortConv)
        self.conv_L_cache_minus_1 = 2  # L_cache=3

    def forward(
        self,
        tokens: torch.LongTensor,          # [1, S]
        input_pos: torch.LongTensor,       # [S]
        conv_state: torch.Tensor,          # [Nconv, D, 2]
        k_cache: torch.Tensor,             # [Nattn, H_kv, C, Hd]
        v_cache: torch.Tensor,             # [Nattn, H_kv, C, Hd]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.tok_embeddings(tokens)  # [1, S, D]
        seqlen = h.shape[1]
        freqs_cos, freqs_sin = self.rope.get_freqs(input_pos, seqlen)

        conv_i = 0
        attn_i = 0
        conv_state_out = conv_state
        k_cache_out = k_cache
        v_cache_out = v_cache

        for layer_id, layer in enumerate(self.layers):
            if self.params.layer_types[layer_id] == "conv":
                # conv_state slice: [D, 2] -> [1, D, 2]
                cs = conv_state_out[conv_i].unsqueeze(0)
                h, cs_out = layer(h, cs)  # type: ignore[misc]
                conv_state_out = conv_state_out.index_copy(
                    0, torch.tensor([conv_i], dtype=torch.long), cs_out.squeeze(0).unsqueeze(0)
                )
                conv_i += 1
            else:
                kc = k_cache_out[attn_i].unsqueeze(0)
                vc = v_cache_out[attn_i].unsqueeze(0)
                h, kc_out, vc_out = layer(  # type: ignore[misc]
                    h, freqs_cos, freqs_sin, input_pos, kc, vc
                )
                k_cache_out = k_cache_out.index_copy(
                    0, torch.tensor([attn_i], dtype=torch.long), kc_out
                )
                v_cache_out = v_cache_out.index_copy(
                    0, torch.tensor([attn_i], dtype=torch.long), vc_out
                )
                attn_i += 1

        # LFM2 typically uses last token logits for decode; keep full logits for simplicity.
        h = self.norm(h)
        logits = self.output(h)  # [1, S, vocab]
        return logits, conv_state_out, k_cache_out, v_cache_out

