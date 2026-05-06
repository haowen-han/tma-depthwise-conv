// Shared template for v1 / v2 warp-specialized, multi-stage TMA depthwise-
// conv kernels.
//
// Both v1 (N_STAGE=2, "double buffer") and v2 (N_STAGE=4, "multi-stage")
// instantiate the same kernel; only N_STAGE differs.
//
// Architecture
//   grid  = (P_tiles, C_blocks, N)
//   block = WARPS_PER_CTA * 32 threads
//     warp 0                      -> producer  (issues TMA loads)
//     warps 1 .. WARPS_PER_CTA-1  -> consumers (depthwise FMA)
//
// One CTA iterates along Q, loading one (kHaloP x kHaloQ x kBlockC) input
// tile per step into an N_STAGE ring of smem buffers via TMA. A pair of
// mbarriers (full/empty) per stage synchronizes the producer/consumer.
//
// The filter F[R, S, C] slice for the CTA's c_block is loaded once into smem.

#pragma once

#include "common.cuh"

#include <cuda_fp16.h>
#include <cstdint>
#include <cuda.h>         // CUtensorMap, cuTensorMapEncodeTiled

namespace tma_dwconv {

// ---------------------------- PTX helpers ----------------------------------
// These wrap the subset of mbarrier / TMA PTX we need.  Each helper takes a
// *shared-memory* pointer (already cvta'd to .shared::cta).

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    uint32_t addr = smem_ptr_u32(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
    uint32_t addr = smem_ptr_u32(bar);
    asm volatile("{ .reg .b64 t; mbarrier.arrive.shared::cta.b64 t, [%0]; }\n" :: "r"(addr));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t tx_count) {
    uint32_t addr = smem_ptr_u32(bar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(addr), "r"(tx_count));
}

__device__ __forceinline__ void mbar_wait_parity(uint64_t* bar, uint32_t phase) {
    uint32_t addr = smem_ptr_u32(bar);
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;");
}

// 4D TMA load (global -> smem) with mbarrier::complete_tx.
__device__ __forceinline__ void tma_load_4d(
    void*             smem_dst,
    const CUtensorMap* tmap,
    int32_t c0, int32_t c1, int32_t c2, int32_t c3,
    uint64_t*         bar)
{
    uint32_t smem_addr = smem_ptr_u32(smem_dst);
    uint32_t bar_addr  = smem_ptr_u32(bar);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
        :
        : "r"(smem_addr),
          "l"(reinterpret_cast<uint64_t>(tmap)),
          "r"(bar_addr),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
        : "memory");
}

// ---------------------------- shared-mem layout ----------------------------
template <int N_STAGE>
struct SmemLayout {
    alignas(128) __half   x_tiles[N_STAGE][kInputTileElems];
    alignas(128) __half   f_tile[kFilterTileElems];
    alignas(8)   uint64_t full_bar[N_STAGE];
    alignas(8)   uint64_t empty_bar[N_STAGE];
};

// ----------------------------- device kernel -------------------------------
template <int N_STAGE, int WARPS_PER_CTA>
__global__ void dwconv_tma_ws_kernel(
    const __grid_constant__ CUtensorMap tmap_x,
    const __half* __restrict__ F,
    __half*       __restrict__ Y,
    ConvParams    params)
{
    static_assert(WARPS_PER_CTA >= 2, "need 1 producer + >=1 consumer warp");
    constexpr int CONSUMER_WARPS   = WARPS_PER_CTA - 1;
    constexpr int CONSUMER_THREADS = CONSUMER_WARPS * 32;

    extern __shared__ __align__(128) char smem_raw[];
    auto& smem = *reinterpret_cast<SmemLayout<N_STAGE>*>(smem_raw);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int n_idx   = blockIdx.z;
    const int c_block = blockIdx.y;
    const int p_block = blockIdx.x;

    const int c_base = c_block * kBlockC;
    const int p_base = p_block * kBlockP;

    const int num_q_tiles = (params.Q + kBlockQ - 1) / kBlockQ;

    // ---- Barrier initialization (single elected thread) ----
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < N_STAGE; ++i) {
            // full_bar: 1 arrival (TMA completion signals via
            //           mbarrier.complete_tx; we call expect_tx each round).
            mbar_init(&smem.full_bar[i], 1);
            // empty_bar: every consumer thread arrives once.
            mbar_init(&smem.empty_bar[i], CONSUMER_THREADS);
        }
        fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ---- Load filter slice (producer lane 0) ----
    if (warp_id == 0 && lane_id == 0) {
        // F[R, S, C], c_base selects a contiguous 64-channel strip.
        // Small (1152 B) so a vectorized scalar copy is fine.
        const float4* g = reinterpret_cast<const float4*>(&F[c_base]);
        float4*       s = reinterpret_cast<float4*>(smem.f_tile);
        const int f_stride_vec = params.C / kVec;
        #pragma unroll
        for (int rs = 0; rs < kR * kS; ++rs) {
            #pragma unroll
            for (int v = 0; v < kBlockC / kVec; ++v) {
                s[rs * (kBlockC / kVec) + v] = g[rs * f_stride_vec + v];
            }
        }
    }
    __syncthreads();  // all consumer threads must see filter before compute

    // ---- Warp-specialized main loop ----
    if (warp_id == 0) {
        // Producer warp: lane 0 issues TMA loads, other lanes are idle.
        if (lane_id == 0) {
            for (int tile = 0; tile < num_q_tiles; ++tile) {
                const int s     = tile % N_STAGE;
                const int round = tile / N_STAGE;
                // Wait for consumer to finish previous use of this stage.
                if (round > 0) {
                    const uint32_t empty_phase = (round - 1) & 1u;
                    mbar_wait_parity(&smem.empty_bar[s], empty_phase);
                }
                // Prepare barrier for TMA and issue load.
                mbar_expect_tx(&smem.full_bar[s], kInputTileBytes);
                // 4D NHWC TMA coords (innermost first: C, W, H, N).
                const int q_base = tile * kBlockQ;
                const int32_t c0 = c_base;                           // C
                const int32_t c1 = q_base - params.pad_w;            // W start
                const int32_t c2 = p_base - params.pad_h;            // H start
                const int32_t c3 = n_idx;                            // N
                tma_load_4d(&smem.x_tiles[s][0], &tmap_x,
                            c0, c1, c2, c3, &smem.full_bar[s]);
            }
        }
    } else {
        // Consumer warps
        const int consumer_tid = threadIdx.x - 32;  // 0..CONSUMER_THREADS-1

        // Each consumer thread owns a set of (p, q, c-vec) outputs inside
        // the tile.  Linearize: total vec-slots = BlockP * BlockQ * BlockC/Vec
        // = 8 * 32 * 8 = 2048.  With 128/96/... consumer threads, each one
        // strides through the work list.
        constexpr int kOutVecsPerTile =
            kBlockP * kBlockQ * (kBlockC / kVec);  // 2048

        // For each q_tile, consume and compute
        for (int tile = 0; tile < num_q_tiles; ++tile) {
            const int s     = tile % N_STAGE;
            const int round = tile / N_STAGE;
            const uint32_t full_phase = round & 1u;
            mbar_wait_parity(&smem.full_bar[s], full_phase);

            const int q_base = tile * kBlockQ;

            // Compute outputs for this tile.
            for (int idx = consumer_tid; idx < kOutVecsPerTile;
                 idx += CONSUMER_THREADS) {
                // idx = ((p * kBlockQ) + q) * (kBlockC/kVec) + cv
                const int cv = idx % (kBlockC / kVec);
                const int tmp = idx / (kBlockC / kVec);
                const int q_local = tmp % kBlockQ;
                const int p_local = tmp / kBlockQ;

                const int c_local = cv * kVec;  // 0..BlockC step VEC

                const int p_glb = p_base + p_local;
                const int q_glb = q_base + q_local;
                if (p_glb >= params.P || q_glb >= params.Q) continue;

                float acc[kVec];
                #pragma unroll
                for (int v = 0; v < kVec; ++v) acc[v] = 0.f;

                #pragma unroll
                for (int r = 0; r < kR; ++r) {
                    #pragma unroll
                    for (int ss = 0; ss < kS; ++ss) {
                        // smem.x_tiles layout: [kHaloP, kHaloQ, kBlockC]
                        const int ph = p_local + r;
                        const int qh = q_local + ss;
                        const __half* xp =
                            &smem.x_tiles[s][(ph * kHaloQ + qh) * kBlockC + c_local];
                        const __half* wp =
                            &smem.f_tile[(r * kS + ss) * kBlockC + c_local];
                        float4 xv = *reinterpret_cast<const float4*>(xp);
                        float4 wv = *reinterpret_cast<const float4*>(wp);
                        __half xh[kVec]; __half wh[kVec];
                        *reinterpret_cast<float4*>(xh) = xv;
                        *reinterpret_cast<float4*>(wh) = wv;
                        #pragma unroll
                        for (int v = 0; v < kVec; ++v) {
                            acc[v] += __half2float(xh[v]) * __half2float(wh[v]);
                        }
                    }
                }

                // Store fp16
                __half yh[kVec];
                #pragma unroll
                for (int v = 0; v < kVec; ++v) yh[v] = __float2half(acc[v]);
                __half* yp =
                    &Y[((n_idx * params.P + p_glb) * params.Q + q_glb) *
                         params.C + c_base + c_local];
                *reinterpret_cast<float4*>(yp) =
                    *reinterpret_cast<float4*>(yh);
            }

            // Signal this stage's buffer is free.
            mbar_arrive(&smem.empty_bar[s]);
        }
    }
}

// ------------------ Host helper : build TMA descriptor for X ----------------
inline CUresult build_tmap_x_f16(
    CUtensorMap& tmap,
    const void*  gmem_ptr,
    const ConvParams& p)
{
    // X is [N, H, W, C] fp16 row-major, innermost dim = C.
    // TMA descriptor sizes are listed innermost-first.
    cuuint64_t size[4]    = { (cuuint64_t)p.C, (cuuint64_t)p.W,
                              (cuuint64_t)p.H, (cuuint64_t)p.N };
    cuuint64_t stride[3]  = {
        (cuuint64_t)p.C * sizeof(__half),                              // stride of W
        (cuuint64_t)p.C * p.W * sizeof(__half),                        // stride of H
        (cuuint64_t)p.C * p.W * p.H * sizeof(__half),                  // stride of N
    };
    cuuint32_t box[4]     = { (cuuint32_t)kBlockC, (cuuint32_t)kHaloQ,
                              (cuuint32_t)kHaloP, 1u };
    cuuint32_t elem_stride[4] = { 1u, 1u, 1u, 1u };

    return cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /*tensorRank=*/4,
        const_cast<void*>(gmem_ptr),
        size,
        stride,
        box,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        // First version uses SWIZZLE_NONE so the smem tile has a linear
        // (kHaloP, kHaloQ, kBlockC) layout that the compute loop can index
        // directly.  Swizzle can be revisited as a perf micro-optimization
        // (see plan.md Step 5).
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

} // namespace tma_dwconv
