import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SpectralNormWrapper
from functools import partial
from typing import Tuple


def LinearLayer(input_size: int, output_size: int, spectral_norm: bool = False, bias: bool = False):
    layer = nn.Linear(input_size, output_size, bias=bias)
    nn.init.xavier_normal_(layer.weight)
    if bias:
        nn.init.zeros_(layer.bias)
    if spectral_norm:
        return SpectralNormWrapper(layer)
    return layer


def ParameterLayer(size: Tuple[int, ...], spectral_norm: bool = False, **kwargs):
    layer = nn.Parameter(torch.randn(size, **kwargs))
    nn.init.xavier_normal_(layer)
    if spectral_norm:
        return SpectralNormWrapper(layer)
    return layer


LayerNorm = partial(nn.LayerNorm, bias=False)


def correction_fn_256(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)
