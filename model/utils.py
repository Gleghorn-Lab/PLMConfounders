import torch
import torch.nn as nn
import torch.nn.functional as F
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


Linear = partial(nn.Linear, bias=False)
LayerNorm = partial(nn.LayerNorm, bias=False)


def correction_fn_256(expansion_ratio: float, hidden_size: int) -> int:
    return int(((expansion_ratio * hidden_size) + 255) // 256 * 256)


def mean_pooling_ab(              # (b, b, l1, l2) -> (b, b)
    y: torch.Tensor,              # pairwise scores
    a_mask: torch.Tensor,         # (b, l1) 0/1
    b_mask: torch.Tensor          # (b, l2) 0/1
) -> torch.Tensor:
    """
    Mean-pool `y` over all (l1,l2) cells where *both*
    sequence-specific masks are 1.

    Returns
    -------
    torch.Tensor
        Shape (b, b) - the mean of valid cells for every pair (i, j).
    """
    # Broadcast the 1/0 masks to (b, b, l1, l2)
    pair_mask = (a_mask[:, None, :, None] *    # (b,1,l1,1)
                 b_mask[None, :, None, :])     # (1,b,1,l2)
    pair_mask = pair_mask.to(y.dtype)
    # Apply mask
    masked_y   = y * pair_mask
    # Sum masked values and count how many were kept
    sums = masked_y.sum(dim=(-1, -2))              # (b, b)
    counts = pair_mask.sum(dim=(-1, -2)).clamp(min=1) # (b, b) – avoid /0
    return sums / counts


def max_pooling_ab(               # (b, b, l1, l2) -> (b, b)
    y: torch.Tensor,              # pairwise scores
    a_mask: torch.Tensor,         # (b, l1) 0/1
    b_mask: torch.Tensor          # (b, l2) 0/1
) -> torch.Tensor:
    """
    Max-pool `y` over all (l1,l2) cells where *both*
    sequence-specific masks are 1.

    Returns
    -------
    torch.Tensor
        Shape (b, b) - the max of valid cells for every pair (i, j).
    """
    # Broadcast the 1/0 masks to (b, b, l1, l2)
    pair_mask = (a_mask[:, None, :, None] *    # (b,1,l1,1)
                 b_mask[None, :, None, :])     # (1,b,1,l2)
    # Set masked-out values to a very low value (so they don't affect max)
    masked_y = y.masked_fill(pair_mask == 0, float('-inf'))
    # Max over last two dims
    max_vals, _ = masked_y.max(dim=-1)
    max_vals, _ = max_vals.max(dim=-1)
    return max_vals
