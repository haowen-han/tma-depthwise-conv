"""Benchmark v0 / ws(n_stage=2) / ws(n_stage=4) against cuDNN.

Usage:
    python -m tests.bench
"""
from __future__ import annotations

import os
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tma_dwconv import dwconv_v0, dwconv_ws


def benchmark(fn: Callable, *args, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


SHAPES: List[Tuple[int, int, int, int]] = [
    (8,  56, 224, 128),
    (16, 112, 224, 128),
    (32, 224, 336, 256),
    (64, 224, 336, 256),
    (72, 224, 336, 256),
    (80, 224, 336, 320),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    pad = 1
    rows = []
    for (N, H, W, C) in SHAPES:
        x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
        w = torch.randn(3, 3, C, device="cuda", dtype=torch.float16) * 0.1

        # Pre-permute to NCHW layout once; cuDNN benchmark should NOT pay the
        # NHWC<->NCHW conversion cost on every iteration.
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        w_oihw = w.permute(2, 0, 1).unsqueeze(1).contiguous()
        cudnn_fn = lambda: F.conv2d(x_nchw, w_oihw, bias=None,
                                    stride=1, padding=pad, groups=C)

        row = {"N": N, "H": H, "W": W, "C": C}
        row["cudnn_ms"] = benchmark(cudnn_fn)
        row["v0_ms"]    = benchmark(dwconv_v0, x, w, pad)
        row["ws2_ms"]   = benchmark(dwconv_ws, x, w, pad, 2)
        row["ws4_ms"]   = benchmark(dwconv_ws, x, w, pad, 4)
        print(row)
        rows.append(row)

    plot(rows)


def plot(rows: List[dict]) -> None:
    xs = [r["N"] * r["H"] * r["W"] * r["C"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    for key, label, color in [
        ("cudnn_ms", "cuDNN",        "tab:blue"),
        ("v0_ms",    "v0",           "tab:orange"),
        ("ws2_ms",   "ws n_stage=2", "tab:green"),
        ("ws4_ms",   "ws n_stage=4", "tab:red"),
    ]:
        ax.plot(xs, [r[key] for r in rows], marker="o", color=color, label=label)

    ax.set_xlabel("N * H * W * C")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Depthwise Conv2D latency vs. shape product")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "bench.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()
