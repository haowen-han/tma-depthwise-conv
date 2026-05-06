#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tma_dwconv {

// ---------------- Fixed kernel / tile configuration ----------------
// Keep these 'constexpr' so both host launcher and device kernels agree.
// BlockC must be 64  -> fp16 => 128B per row along C (128B swizzle friendly).
// BlockP=8, BlockQ=32 -> input-tile (w/ halo 2) = 10 x 34 x 64.
constexpr int kR        = 3;
constexpr int kS        = 3;
constexpr int kBlockP   = 8;
constexpr int kBlockQ   = 32;
constexpr int kBlockC   = 64;
constexpr int kHaloP    = kBlockP + (kR - 1);   // 10
constexpr int kHaloQ    = kBlockQ + (kS - 1);   // 34
constexpr int kVec      = 8;                    // 128-bit fp16 vector

// 128B alignment for TMA / SMEM
constexpr int kSmemAlign = 128;

// Per-stage input-tile size in bytes (fp16): 10 * 34 * 64 * 2 = 43520.
constexpr int kInputTileElems = kHaloP * kHaloQ * kBlockC;     // 21760
constexpr int kInputTileBytes = kInputTileElems * sizeof(__half);

// Filter tile: R * S * BlockC fp16 = 3*3*64*2 = 1152 bytes.
constexpr int kFilterTileElems = kR * kS * kBlockC;
constexpr int kFilterTileBytes = kFilterTileElems * sizeof(__half);

// ---------------- Convolution parameters passed to kernels ----------------
struct ConvParams {
    int N, H, W, C;
    int P, Q;
    int pad_h, pad_w;
    // stride/dilation fixed to 1 in the first version, kept here to make it
    // easy to extend later.
    int stride_h, stride_w;
    int dilation_h, dilation_w;
};

} // namespace tma_dwconv
