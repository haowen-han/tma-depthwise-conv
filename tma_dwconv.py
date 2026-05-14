"""Python wrapper around the C++/CUDA extension + reference implementation."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import _tma_dwconv_C as _C  # type: ignore

__all__ = ["dwconv_v0", "dwconv_ws", "dwconv_ws_probe", "ref_dwconv"]


# ------------------------------ reference ----------------------------------

def ref_dwconv(x_nhwc: torch.Tensor, w_rsc: torch.Tensor, pad: int) -> torch.Tensor:
    """Depthwise conv2d reference via torch.nn.functional.conv2d.

    Args:
        x_nhwc: [N, H, W, C] input tensor.
        w_rsc:  [R, S, C] filter tensor (depthwise, one RxS kernel per channel).
        pad:    symmetric spatial padding.

    Returns:
        [N, P, Q, C] output tensor with the same dtype as x_nhwc.
    """
    if x_nhwc.dim() != 4 or w_rsc.dim() != 3:
        raise ValueError("x must be [N,H,W,C] and w must be [R,S,C]")
    _, _, _, C = x_nhwc.shape
    _, _, Cw = w_rsc.shape
    if C != Cw:
        raise ValueError("channel mismatch between x and w")
    x = x_nhwc.permute(0, 3, 1, 2).contiguous()           # [N, C, H, W]
    w = w_rsc.permute(2, 0, 1).unsqueeze(1).contiguous()  # [C, 1, R, S]
    y = F.conv2d(x, w, bias=None, stride=1, padding=pad, groups=C)
    return y.permute(0, 2, 3, 1).contiguous()


# ------------------------------ validation ---------------------------------

def _check(x: torch.Tensor, w: torch.Tensor, pad: int) -> None:
    if not (x.is_cuda and w.is_cuda):
        raise ValueError("x and w must be CUDA tensors")
    if x.dtype != torch.float16 or w.dtype != torch.float16:
        raise ValueError("x and w must be fp16")
    if x.dim() != 4:
        raise ValueError(f"x must be [N,H,W,C], got shape {tuple(x.shape)}")
    if w.dim() != 3:
        raise ValueError(f"w must be [R,S,C], got shape {tuple(w.shape)}")
    if x.shape[3] != w.shape[2]:
        raise ValueError("x and w must agree on C")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("x and w must be contiguous")
    if pad < 0:
        raise ValueError("pad must be non-negative")


# ------------------------------ public API ---------------------------------

def dwconv_v0(x: torch.Tensor, w: torch.Tensor, pad: int = 1) -> torch.Tensor:
    _check(x, w, pad)
    return _C.dwconv_v0(x, w, int(pad))


def dwconv_ws(x: torch.Tensor, w: torch.Tensor,
              pad: int = 1, n_stage: int = 2) -> torch.Tensor:
    """TMA + warp-specialized depthwise conv2d.

    Args:
        x: [N, H, W, C] fp16, contiguous, C % 64 == 0.
        w: [R, S, C]    fp16, contiguous.
        pad: zero-padding on both H and W.
        n_stage: SMEM ring-buffer depth (pipeline stages). Supported: 2, 3, 4.
    """
    _check(x, w, pad)
    if x.shape[3] % 64 != 0:
        raise ValueError("dwconv_ws requires C % 64 == 0")
    if n_stage not in (2, 3, 4):
        raise ValueError(f"n_stage must be one of {{2, 3, 4}}, got {n_stage}")
    return _C.dwconv_ws(x, w, int(pad), int(n_stage))


def dwconv_ws_probe(x: torch.Tensor, w: torch.Tensor,
                    pad: int = 1, n_stage: int = 2):
    """Same as ``dwconv_ws`` but instruments producer/consumer stalls.

    Returns:
        (y, stall): ``y`` is the output tensor; ``stall`` is an int64 CUDA
        tensor of shape ``[num_blocks, 4]`` whose columns are, per CTA:

          * col 0 — producer stall cycles (waiting on ``empty_bar``)
          * col 1 — consumer stall cycles, summed across consumer warps
                    (waiting on ``full_bar``)
          * col 2 — producer total cycles (warp 0 lane 0 lifetime)
          * col 3 — consumer total cycles, summed across consumer warps
                    (each warp's own lane-0 lifetime; producer typically
                    retires earlier so use this — not col 2 — when computing
                    consumer-side fractions)

        Block linear index = ``(z*gy + y)*gx + x`` with
        ``gx=ceil(P/8)``, ``gy=C/64``, ``gz=N``.

        Compare ``mean(col 0)`` vs ``mean(col 1)/CONSUMER_WARPS``:
          * col0 >> col1  -> consumers are slower; producer is starved.
          * col1 >> col0  -> producer (TMA) is slower; consumers are starved.
    """
    _check(x, w, pad)
    if x.shape[3] % 64 != 0:
        raise ValueError("dwconv_ws_probe requires C % 64 == 0")
    if n_stage not in (2, 3, 4):
        raise ValueError(f"n_stage must be one of {{2, 3, 4}}, got {n_stage}")
    return _C.dwconv_ws_probe(x, w, int(pad), int(n_stage))
