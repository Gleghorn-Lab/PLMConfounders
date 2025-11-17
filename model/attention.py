import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from functools import partial

from .utils import LinearLayer, ParameterLayer, LayerNorm
from .rotary import RotaryEmbedding


class PAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int,
            dropout: float = 0.2,
            spectral_norm: bool = False,
    ):
        super(PAttention, self).__init__()
        self.n_tokens = n_tokens
        self.Wq = LinearLayer(hidden_size, hidden_size, spectral_norm=spectral_norm)
        self.Pk = ParameterLayer((1, n_tokens, hidden_size))
        self.Pv = ParameterLayer((1, n_tokens, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, :].expand(b, L, self.n_tokens).bool()
            else:
                raise ValueError(f"Invalid attention mask shape: {attention_mask.shape}")
        q = self.Wq(x) # (b, L, d)
        out = F.scaled_dot_product_attention(q, self.Pk, self.Pv, attn_mask=attention_mask, is_causal=False) # (b, L, d)
        return self.dropout(out)


class MultiHeadPAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            n_tokens: int,
            dropout: float = 0.2,
            rotary: bool = True,
            causal: bool = False,
            spectral_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout, spectral_norm=spectral_norm)
        self.Wk = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout, spectral_norm=spectral_norm)
        self.Wv = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout, spectral_norm=spectral_norm)
        self.out_proj = LinearLayer((hidden_size // n_heads) * n_heads, hidden_size, spectral_norm=spectral_norm)
        self.q_ln = LayerNorm(hidden_size)
        self.k_ln = LayerNorm(hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        b, L, _ = x.shape
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
            else:
                raise ValueError(f"Invalid attention mask shape: {attention_mask.shape}")
        
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True, causal: bool = False, spectral_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            LayerNorm(hidden_size), LinearLayer(hidden_size, hidden_size * 3, spectral_norm=spectral_norm)
        )
        self.out_proj = LinearLayer((hidden_size // n_heads) * n_heads, hidden_size, spectral_norm=spectral_norm)
        self.q_ln = LayerNorm(hidden_size, bias=False)
        self.k_ln = LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        b, L, _ = x.shape
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
        qkv = self.layernorm_qkv(x) # (bs, seq_len, hidden_size * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (bs, seq_len, hidden_size)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class AttentionPooler(nn.Module):
    """
    Cross-attention mechanism for pooling (b, L, d) -> (b, n_tokens, d)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int = 1,
            n_heads: int = 16,
            spectral_norm: bool = False,
    ):
        super(AttentionPooler, self).__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.d_head = hidden_size // n_heads
        self.Q = ParameterLayer((1, n_tokens, hidden_size))
        self.Wq = LinearLayer(hidden_size, hidden_size, spectral_norm=spectral_norm)
        self.Wv = LinearLayer(hidden_size, hidden_size, spectral_norm=spectral_norm)
        self.Wk = LinearLayer(hidden_size, hidden_size, spectral_norm=spectral_norm)
        self.Wo = LinearLayer((hidden_size // n_heads) * n_heads, hidden_size, spectral_norm=spectral_norm)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, d = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, self.n_tokens, L).bool()
        q = self.Wq(self.Q).expand(b, -1, -1)  # (b, n_tokens, d)
        v = self.Wv(x)  # (b, L, d)
        k = self.Wk(x)  # (b, L, d)
        q, k, v = map(self.reshaper, (q, k, v))  # (b, n_heads, n_tokens, d_head) (b, n_heads, L, d_head)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        ) # (b, n_heads, n_tokens, d_head)
        attn = rearrange(attn, "b h s d -> b s (h d)")  # (b, n_tokens, n_heads * d_head)
        return self.Wo(attn)  # (b, n_tokens, d)
    