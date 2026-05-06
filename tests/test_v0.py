import pytest
import torch

pytest.importorskip("tma_dwconv")
from tma_dwconv import dwconv_v0, ref_dwconv


CASES = [
    # (N, H, W, C, pad)
    (1,  8,  8, 64, 1),
    (2, 16, 16, 64, 1),
    (4, 32, 32, 128, 1),
    (8, 56, 56, 256, 1),
]


@pytest.mark.parametrize("N,H,W,C,pad", CASES)
def test_v0_matches_reference(N, H, W, C, pad):
    torch.manual_seed(0)
    x = torch.randn(N, H, W, C, device="cuda", dtype=torch.float16)
    w = torch.randn(3, 3, C, device="cuda", dtype=torch.float16) * 0.1
    y_ref = ref_dwconv(x, w, pad=pad)
    y     = dwconv_v0(x, w, pad=pad)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
