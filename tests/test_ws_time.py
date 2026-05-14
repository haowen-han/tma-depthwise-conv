"""Producer / consumer wait-cycle probe for the warp-specialized kernel.

Uses the ``dwconv_ws_probe`` entry point which instruments the in-kernel
``cute::wait_barrier`` calls with ``clock64()``.

Per CTA, the kernel writes 4 cycle counters:

  * col 0 — producer stall: cycles warp 0 spent waiting on ``empty_bar``
            (i.e. waiting for some consumer warp to release a stage).
  * col 1 — consumer stall: cycles spent waiting on ``full_bar``,
            **summed across all 4 consumer warps** (each contributes via
            its lane 0).
  * col 2 — producer total cycles (warp 0 lane 0 lifetime).
  * col 3 — consumer total cycles, **summed across consumer warps** (each
            warp contributes its own lane-0 lifetime). Producer typically
            retires earlier than consumers, so use this — not col 2 — as
            the denominator for consumer-side fractions.

Reading the table:
  * ``prod_stall`` >> ``cons_stall_per_warp``  -> consumers are the bottleneck
    (TMA is fast, compute is slow; producer keeps waiting for empty buffers).
  * ``cons_stall_per_warp`` >> ``prod_stall``  -> producer is the bottleneck
    (compute is fast, TMA is slow; consumers keep waiting for full buffers).

Run as either pytest (will print summary) or directly:
    pytest -s tests/test_ws_time.py
    python tests/test_ws_time.py
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("tma_dwconv")
from tma_dwconv import dwconv_ws_probe, ref_dwconv


# Number of consumer warps per CTA (must match WARPS_PER_CTA - 1 in the .cu).
CONSUMER_WARPS = 4

# (N, H, W, C, pad) — a small subset of the full correctness suite, focused
# on shapes large enough to amortize start-up and produce stable stalls.
CASES = [
    (8,  56,  56, 256, 1),
    (16, 112, 224, 128, 1),
    (32, 224, 336, 256, 1),
    (64, 224, 336, 256, 1),
]

WS_STAGES = [2, 3, 4]

# How many warm-up + measured launches per (case, stage). We accumulate stall
# counters across the measured launches to smooth out per-block jitter.
N_WARMUP = 2
N_MEASURE = 5


def _make_inputs(N, H, W, C, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
    w = torch.randn(3, 3, C, device="cuda", dtype=torch.float16) * 0.1
    return x, w


def _probe_once(x, w, pad, n_stage):
    """Single probed launch -> stall tensor [num_blocks, 3] (int64, on CUDA)."""
    _, stall = dwconv_ws_probe(x, w, pad=pad, n_stage=n_stage)
    return stall


def _summarize(stall_cpu: torch.Tensor) -> dict:
    """Reduce a [num_blocks, 4] int64 CPU tensor into summary stats."""
    prod = stall_cpu[:, 0].double()
    cons_total_stall = stall_cpu[:, 1].double()
    cons_per_warp = cons_total_stall / CONSUMER_WARPS  # comparable to prod
    prod_total = stall_cpu[:, 2].double()
    cons_total_life = stall_cpu[:, 3].double() / CONSUMER_WARPS  # per-warp
    return {
        "num_blocks": stall_cpu.shape[0],
        "prod_mean": prod.mean().item(),
        "prod_p50":  prod.median().item(),
        "prod_p95":  torch.quantile(prod, 0.95).item(),
        "cons_mean": cons_per_warp.mean().item(),
        "cons_p50":  cons_per_warp.median().item(),
        "cons_p95":  torch.quantile(cons_per_warp, 0.95).item(),
        "prod_total_mean": prod_total.mean().item(),
        "cons_total_mean": cons_total_life.mean().item(),
        # Each side's stall as a fraction of its OWN lifetime — apples to apples.
        "prod_frac": (prod / prod_total.clamp(min=1)).mean().item(),
        "cons_frac": (cons_per_warp / cons_total_life.clamp(min=1)).mean().item(),
    }


def _format_row(case, n_stage, s):
    N, H, W, C, _ = case
    bottleneck = "consumer" if s["prod_mean"] > s["cons_mean"] else "producer"
    ratio = (s["prod_mean"] / s["cons_mean"]) if s["cons_mean"] > 0 else float("inf")
    return (
        f"  N={N:>3} H={H:>3} W={W:>3} C={C:>4}  stage={n_stage}  "
        f"blks={s['num_blocks']:>5}  "
        f"prod μ={s['prod_mean']:>10.0f}  p95={s['prod_p95']:>10.0f}  "
        f"cons/warp μ={s['cons_mean']:>10.0f}  p95={s['cons_p95']:>10.0f}  "
        f"prod_tot μ={s['prod_total_mean']:>11.0f}  "
        f"cons_tot/warp μ={s['cons_total_mean']:>11.0f}  "
        f"prod/prod_tot={s['prod_frac']*100:>5.1f}%  "
        f"cons/cons_tot={s['cons_frac']*100:>5.1f}%  "
        f"-> {bottleneck} bottleneck (prod/cons={ratio:.2f})"
    )


def _run_case(case, n_stage):
    N, H, W, C, pad = case
    x, w = _make_inputs(N, H, W, C)

    # Optional sanity check on the very first run, against torch reference.
    # Cheap relative to the cost of building x.
    y, _ = dwconv_ws_probe(x, w, pad=pad, n_stage=n_stage)
    y_ref = ref_dwconv(x, w, pad=pad)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)

    # Warmup.
    for _ in range(N_WARMUP):
        _probe_once(x, w, pad, n_stage)
    torch.cuda.synchronize()

    # Accumulate stalls across N_MEASURE launches.
    accum = None
    for _ in range(N_MEASURE):
        stall = _probe_once(x, w, pad, n_stage)
        accum = stall if accum is None else accum + stall
    torch.cuda.synchronize()

    return _summarize(accum.cpu())


@pytest.mark.parametrize("n_stage", WS_STAGES)
@pytest.mark.parametrize("N,H,W,C,pad", CASES)
def test_ws_stall_probe(N, H, W, C, pad, n_stage, capsys):
    """Run the probe and print a per-(case, stage) summary line.

    This test does not assert numerical bottleneck thresholds; it only
    asserts that the probe ran and produced non-zero ``total`` cycles.
    Use ``pytest -s`` to see the printed table.
    """
    case = (N, H, W, C, pad)
    summary = _run_case(case, n_stage)
    line = _format_row(case, n_stage, summary)
    # Print via capsys.disabled so it shows up even without -s.
    with capsys.disabled():
        print()
        print(line)
    assert summary["prod_total_mean"] > 0 and summary["cons_total_mean"] > 0


def main():
    print("=" * 80)
    print("Warp-specialized depthwise conv: producer/consumer stall probe")
    print(f"  CONSUMER_WARPS = {CONSUMER_WARPS} "
          f"(consumer stall column is summed across warps; "
          f"divide by {CONSUMER_WARPS} for per-warp comparison with producer)")
    print(f"  warmup = {N_WARMUP}, measured launches accumulated = {N_MEASURE}")
    print("=" * 80)
    for case in CASES:
        for n_stage in WS_STAGES:
            summary = _run_case(case, n_stage)
            print(_format_row(case, n_stage, summary))
        print()


if __name__ == "__main__":
    main()
