import pytest
import torch

pytest.importorskip("tma_dwconv")
from tma_dwconv import dwconv_v0, dwconv_ws, ref_dwconv


CASES = [
    # (N, H, W, C, pad)
    (1,  8,  8, 64, 1),
    (2, 16, 16, 64, 1),
    (4, 32, 32, 128, 1),
    (8, 56, 56, 256, 1),
    (8,  56, 224, 128, 1),
    (16, 112, 224, 128, 1),
    (32, 224, 336, 256, 1),
    (64, 224, 336, 256, 1),
    (72, 224, 336, 256, 1),
    (80, 224, 336, 320, 1),    
]

WS_STAGES = [2, 3, 4]


def _make_inputs(N, H, W, C, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
    w = torch.randn(3, 3, C, device="cuda", dtype=torch.float16) * 0.1
    return x, w


@pytest.mark.parametrize("N,H,W,C,pad", CASES)
def test_v0_matches_reference(N, H, W, C, pad):
    x, w = _make_inputs(N, H, W, C)
    y_ref = ref_dwconv(x, w, pad=pad)
    y     = dwconv_v0(x, w, pad=pad)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n_stage", WS_STAGES)
@pytest.mark.parametrize("N,H,W,C,pad", CASES)
def test_ws_matches_reference(N, H, W, C, pad, n_stage):
    x, w = _make_inputs(N, H, W, C)
    y_ref = ref_dwconv(x, w, pad=pad)
    y     = dwconv_ws(x, w, pad=pad, n_stage=n_stage)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
