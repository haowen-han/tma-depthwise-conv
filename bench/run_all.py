"""Benchmark v0 / v1 / v2 against cuDNN (torch.nn.functional.conv2d).

Usage:
    python -m bench.run_all
"""
from __future__ import annotations

import csv
import itertools
import os
import time
from typing import Callable, List, Tuple

import torch

from tma_dwconv import dwconv_v0, dwconv_v1, dwconv_v2, ref_dwconv


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
    (1,  56, 56,  64),
    (8,  56, 56, 128),
    (8, 112, 112, 128),
    (8,  56, 56, 256),
    (8,  28, 28, 512),
    (16, 56, 56, 128),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    pad = 1
    results = []
    for (N, H, W, C) in SHAPES:
        x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
        w = torch.randn(3, 3, C, device="cuda", dtype=torch.float16) * 0.1

        # cuDNN reference (via ref_dwconv)
        row = {"N": N, "H": H, "W": W, "C": C}
        row["cudnn_ms"] = benchmark(ref_dwconv, x, w, pad)
        row["v0_ms"]    = benchmark(dwconv_v0, x, w, pad)
        row["v1_ms"]    = benchmark(dwconv_v1, x, w, pad)
        row["v2_ms"]    = benchmark(dwconv_v2, x, w, pad)
        row["v1_over_v0"] = row["v0_ms"] / row["v1_ms"]
        row["v2_over_v1"] = row["v1_ms"] / row["v2_ms"]
        row["v2_vs_cudnn"] = row["cudnn_ms"] / row["v2_ms"]
        print(row)
        results.append(row)

    out = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
