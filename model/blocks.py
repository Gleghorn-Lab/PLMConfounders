import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import PAttention, MultiHeadPAttention, MultiHeadAttention
from .utils import LayerNorm, LinearLayer, correction_fn_256


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ffn(hidden_size: int, expansion_ratio: float, dropout: float = 0.1, spectral_norm: bool = False):
    return nn.Sequential(
        LayerNorm(hidden_size),
        LinearLayer(
            hidden_size, correction_fn_256(expansion_ratio, hidden_size) * 2, spectral_norm=spectral_norm
        ),
        SwiGLU(),
        nn.Dropout(dropout),
        LinearLayer(correction_fn_256(expansion_ratio, hidden_size), hidden_size, spectral_norm=spectral_norm),
    )


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            rotary=config.rotary,
            causal=False,
            spectral_norm=config.spectral_norm
        )
        self.ffn = swiglu_ffn(config.hidden_size, config.expansion_ratio, config.dropout, config.spectral_norm)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, attention_mask) + x
        x = self.ffn(x) + x
        return x


class PTransformerBlock(nn.Module):
    def __init__(self, config):
        super(PTransformerBlock, self).__init__()
        self.ln1 = LayerNorm(config.hidden_size)
        self.attention = MultiHeadPAttention(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            n_tokens=config.n_tokens,
            dropout=config.dropout,
            rotary=config.rotary,
            causal=False,
            spectral_norm=config.spectral_norm
        )
        self.ln2 = LayerNorm(config.hidden_size)
        self.mlp = PAttention(
            hidden_size=config.hidden_size,
            n_tokens=correction_fn_256(config.expansion_ratio, config.hidden_size),
            dropout=config.dropout,
            spectral_norm=config.spectral_norm
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(self.ln1(x), attention_mask) + x
        x = self.mlp(self.ln2(x)) + x
        return x
