"""Reference depthwise conv implementation using torch.nn.functional.conv2d."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ref_dwconv(x_nhwc: torch.Tensor, w_rsc: torch.Tensor, pad: int) -> torch.Tensor:
    """Compute depthwise conv2d on an NHWC tensor using PyTorch.

    Args:
        x_nhwc: [N, H, W, C] input tensor.
        w_rsc:  [R, S, C] filter tensor (depthwise, one RxS kernel per channel).
        pad:    symmetric spatial padding.

    Returns:
        [N, P, Q, C] output tensor with the same dtype as x_nhwc.
    """
    if x_nhwc.dim() != 4 or w_rsc.dim() != 3:
        raise ValueError("x must be [N,H,W,C] and w must be [R,S,C]")
    N, H, W, C = x_nhwc.shape
    R, S, Cw = w_rsc.shape
    if C != Cw:
        raise ValueError("channel mismatch between x and w")
    x = x_nhwc.permute(0, 3, 1, 2).contiguous()      # [N, C, H, W]
    w = w_rsc.permute(2, 0, 1).unsqueeze(1).contiguous()  # [C, 1, R, S]
    y = F.conv2d(x, w, bias=None, stride=1, padding=pad, groups=C)
    return y.permute(0, 2, 3, 1).contiguous()
