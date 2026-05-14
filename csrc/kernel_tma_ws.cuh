// Shared template for v1 / v2 warp-specialized, multi-stage TMA depthwise-
// conv kernels, using CuTe instead of raw PTX.
//
// Both v1 (N_STAGE=2, "double buffer") and v2 (N_STAGE=4, "multi-stage")
// instantiate the same kernel; only N_STAGE differs.
//
// Architecture
//   grid  = (P_tiles, C_blocks, N)
//   block = WARPS_PER_CTA * 32 threads
//     warp 0                      -> producer  (issues TMA loads via cute::copy)
//     warps 1 .. WARPS_PER_CTA-1  -> consumers (depthwise FMA)
//
// The TMA descriptor is constructed on the host via cute::make_tma_atom
// against a 4D NHWC view of X (mode order in CuTe: (C, W, H, N),
// innermost = C).  The producer warp issues loads through the CuTe wrapper
// cute::SM90_TMA_LOAD_4D::copy, passing (C, W, H, N) coords directly — no
// coord-tensor / tma_partition machinery on the device side.  Barriers use
// cute::{initialize,wait,arrive}_barrier and cute::set_barrier_transaction_bytes.

#pragma once

#include "common.cuh"

#include <cuda_fp16.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>  // cute::{initialize,wait,arrive}_barrier
#include <cutlass/half.h>                // cutlass::half_t (TMA descriptor dtype)

namespace tma_dwconv {

using namespace cute;

// ---------------- CuTe layouts (shared by host/device) ---------------------

// SMEM tile layout for one stage: (BlockC, HaloQ, HaloP), innermost = C.
// Compact linear, no swizzle -> matches the index arithmetic in the
// consumer compute loop (offset = (p*kHaloQ + q)*kBlockC + c).
//
// NOTE on mode order: CuTe / TMA convention is fastest-first (mode 0 = the
// contiguous, stride-1 dim).  This is the OPPOSITE of the C++/NumPy "NHWC"
// reading order.  make_tma_atom asserts `stride<0>(gmem) == 1`, so we must
// write it as (C, W, H, N) even though the tensor is stored as NHWC.
CUTE_HOST_DEVICE constexpr auto make_smem_x_layout() {
    return make_layout(
        make_shape (Int<kBlockC>{}, Int<kHaloQ>{}, Int<kHaloP>{}),
        make_stride(Int<1>{},       Int<kBlockC>{},
                    Int<kBlockC * kHaloQ>{}));
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
//
// Stall-probe mode (kStallProbe == true):
//   stall_buf layout = [num_blocks][3] of unsigned long long:
//     [block][0] = producer  cycles spent inside wait_barrier(empty_bar)
//     [block][1] = consumer  cycles spent inside wait_barrier(full_bar),
//                  summed across all consumer warps (each warp contributes
//                  via lane 0)
//     [block][2] = total kernel cycles measured by warp 0 lane 0
//                  (clock64 end - clock64 start)
//   When kStallProbe == false, stall_buf is ignored and the probe code is
//   compiled out (`if constexpr`).
template <class TmaAtomX, int N_STAGE, int WARPS_PER_CTA,
          bool kStallProbe = false>
__global__ void dwconv_tma_ws_kernel(
    CUTE_GRID_CONSTANT TmaAtomX const tma_atom_x,
    const __half* __restrict__ F,
    __half*       __restrict__ Y,
    ConvParams    params,
    unsigned long long* stall_buf = nullptr)
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

    // ---- Stall-probe accumulators (compiled out when kStallProbe=false) ----
    uint64_t prod_stall_cycles = 0;
    uint64_t cons_stall_cycles = 0;
    uint64_t kernel_start_cyc  = 0;  // producer warp 0 lane 0 only
    uint64_t cons_start_cyc    = 0;  // each consumer warp lane 0
    if constexpr (kStallProbe) {
        if (threadIdx.x == 0) kernel_start_cyc = clock64();
        // Consumer warps record their own start (lane 0 per warp), so we can
        // compute their actual lifetime — producer typically retires earlier.
        if (warp_id != 0 && lane_id == 0) cons_start_cyc = clock64();
    }

    // ---- Barrier initialization (single elected thread) ----
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < N_STAGE; ++i) {
            cute::initialize_barrier(smem.full_bar[i],  /*arrive_count=*/1);
            cute::initialize_barrier(smem.empty_bar[i],
                                     /*arrive_count=*/CONSUMER_THREADS);
        }
    }
    __syncthreads();

    // ---- Load filter slice (producer lane 0) ----
    if (warp_id == 0 && lane_id == 0) {
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
        if (lane_id == 0) {
            // Skip the coord-tensor / tma_partition machinery: call the CuTe
            // PTX wrapper directly.  Coords are in TMA-descriptor order
            // (C, W, H, N), matching the layout passed to make_tma_atom_x.
            // Negative W/H coords at image borders are handled by the TMA OOB
            // policy (zero fill).
            cute::TmaDescriptor const* tma_desc =
                tma_atom_x.get_tma_descriptor();
            constexpr uint64_t kCacheHint =
                static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);

            const int32_t c_crd = c_base;
            const int32_t h_crd = p_base - int(params.pad_h);
            const int32_t n_crd = n_idx;

            for (int tile = 0; tile < num_q_tiles; ++tile) {
                const int s_idx = tile % N_STAGE;
                const int round = tile / N_STAGE;

                if (round > 0) {
                    const uint32_t empty_phase = (round - 1) & 1u;
                    uint64_t t0 = 0;
                    if constexpr (kStallProbe) t0 = clock64();
                    cute::wait_barrier(smem.empty_bar[s_idx], empty_phase);
                    if constexpr (kStallProbe) {
                        prod_stall_cycles += clock64() - t0;
                    }
                }

                cute::set_barrier_transaction_bytes(
                    smem.full_bar[s_idx], kInputTileBytes);

                const int32_t w_crd = tile * kBlockQ - int(params.pad_w);
                cute::SM90_TMA_LOAD_4D::copy(
                    tma_desc,
                    &smem.full_bar[s_idx],
                    kCacheHint,
                    &smem.x_tiles[s_idx][0],
                    c_crd, w_crd, h_crd, n_crd);
            }
        }
    } else {
        // ---- Consumer warps: depthwise FMA over smem tile ----
        const int consumer_tid = threadIdx.x - 32;  // 0..CONSUMER_THREADS-1
        constexpr int kOutVecsPerTile =
            kBlockP * kBlockQ * (kBlockC / kVec);  // 2048

        for (int tile = 0; tile < num_q_tiles; ++tile) {
            const int s_idx = tile % N_STAGE;
            const int round = tile / N_STAGE;
            const uint32_t full_phase = round & 1u;
            uint64_t t0 = 0;
            if constexpr (kStallProbe) {
                if (lane_id == 0) t0 = clock64();
            }
            cute::wait_barrier(smem.full_bar[s_idx], full_phase);
            if constexpr (kStallProbe) {
                if (lane_id == 0) cons_stall_cycles += clock64() - t0;
            }

            const int q_base = tile * kBlockQ;

            for (int idx = consumer_tid; idx < kOutVecsPerTile;
                 idx += CONSUMER_THREADS) {
                const int cv      = idx % (kBlockC / kVec);
                const int tmp     = idx / (kBlockC / kVec);
                const int q_local = tmp % kBlockQ;
                const int p_local = tmp / kBlockQ;
                const int c_local = cv * kVec;

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
                        const int ph = p_local + r;
                        const int qh = q_local + ss;
                        const __half* xp =
                            &smem.x_tiles[s_idx][(ph * kHaloQ + qh) * kBlockC + c_local];
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
            cute::arrive_barrier(smem.empty_bar[s_idx]);
        }
    }

    // ---- Stall-probe writeback ----
    if constexpr (kStallProbe) {
        const int block_lin =
            (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        // Producer warp lane 0: producer stall + total kernel cycles.
        if (warp_id == 0 && lane_id == 0) {
            const uint64_t kernel_end = clock64();
            atomicAdd(&stall_buf[block_lin * 4 + 0],
                      static_cast<unsigned long long>(prod_stall_cycles));
            atomicAdd(&stall_buf[block_lin * 4 + 2],
                      static_cast<unsigned long long>(
                          kernel_end - kernel_start_cyc));
        }
        // Each consumer warp's lane 0 contributes to the consumer-stall slot
        // and to the consumer-total slot (its own lifetime).
        if (warp_id != 0 && lane_id == 0) {
            const uint64_t cons_end = clock64();
            atomicAdd(&stall_buf[block_lin * 4 + 1],
                      static_cast<unsigned long long>(cons_stall_cycles));
            atomicAdd(&stall_buf[block_lin * 4 + 3],
                      static_cast<unsigned long long>(
                          cons_end - cons_start_cyc));
        }
    }
}

// -------------------- Host helper : build TMA atom for X -------------------
// Builds cute::Copy_Atom<SM90_TMA_LOAD, cutlass::half_t> over a 4D view of X.
//
// CuTe mode order is fastest-first (opposite of C++/NumPy): the contiguous
// dim must sit at mode 0.  For an NHWC tensor that means (C, W, H, N) here,
// with stride (1, C, C*W, C*W*H).  make_tma_atom asserts stride<0> == 1 at
// runtime; violating this silently produces wrong TMA descriptors.
//
// SM90_TMA_LOAD is the CuTe op name for cp.async.bulk.tensor and is the right
// choice on both Hopper (SM90) and Blackwell (SM100) for single-CTA TMA loads;
// SM100 only introduces new ops for 2SM multicast, which we do not use here.
// Element type must be cutlass::half_t (not __half) so CuTe's TMA descriptor
// machinery dispatches FP16 correctly.
inline auto make_tma_atom_x(const void* x_ptr, ConvParams const& p) {
    auto gX = make_tensor(
        make_gmem_ptr(reinterpret_cast<cutlass::half_t const*>(x_ptr)),
        make_layout(
            make_shape (int(p.C), int(p.W), int(p.H), int(p.N)),
            make_stride(Int<1>{},
                        int64_t(p.C),
                        int64_t(p.C) * int64_t(p.W),
                        int64_t(p.C) * int64_t(p.W) * int64_t(p.H))));

    // CTA tiler over (C, W, H, N); N tile = 1 -> one image per TMA.
    auto cta_tiler = make_shape(Int<kBlockC>{}, Int<kHaloQ>{},
                                Int<kHaloP>{}, Int<1>{});

    return make_tma_atom(SM90_TMA_LOAD{}, gX, make_smem_x_layout(), cta_tiler);
}

} // namespace tma_dwconv
