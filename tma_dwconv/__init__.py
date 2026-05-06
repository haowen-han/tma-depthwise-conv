"""Python wrapper around the C++/CUDA extension.

All shape / dtype validation happens here so the C++ side can stay thin.
"""
from __future__ import annotations

import torch

from . import _C  # type: ignore
from .reference import ref_dwconv

__all__ = ["dwconv_v0", "dwconv_v1", "dwconv_v2", "ref_dwconv"]


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


def dwconv_v0(x: torch.Tensor, w: torch.Tensor, pad: int = 1) -> torch.Tensor:
    _check(x, w, pad)
    return _C.dwconv_v0(x, w, int(pad))


def dwconv_v1(x: torch.Tensor, w: torch.Tensor, pad: int = 1) -> torch.Tensor:
    _check(x, w, pad)
    if x.shape[3] % 64 != 0:
        raise ValueError("v1 requires C % 64 == 0")
    return _C.dwconv_v1(x, w, int(pad))


def dwconv_v2(x: torch.Tensor, w: torch.Tensor, pad: int = 1) -> torch.Tensor:
    _check(x, w, pad)
    if x.shape[3] % 64 != 0:
        raise ValueError("v2 requires C % 64 == 0")
    return _C.dwconv_v2(x, w, int(pad))
