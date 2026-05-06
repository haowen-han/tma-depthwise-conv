// v0: naive depthwise conv, no shared memory, no TMA, but float4 vectorized.
// Produces the same result as v1 / v2 and acts as a performance floor.

#include "common.cuh"
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace tma_dwconv {

using Vec = float4;                // 128 bit = 8 x fp16

template <int R, int S, int VEC>
__global__ void dwconv_v0_kernel(
    const __half* __restrict__ X,  // [N, H, W, C]
    const __half* __restrict__ F,  // [R, S, C]
    __half*       __restrict__ Y,  // [N, P, Q, C]
    int N, int H, int W, int C,
    int P, int Q,
    int pad_h, int pad_w,
    int stride_h, int stride_w)
{
    // 1D launch. Each thread owns one (n, p, q, c_vec) output.
    // Linearization: tid -> (n, p, q, cvec) with cvec innermost.
    const int C_vec = C / VEC;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * P * Q * C_vec;
    if (tid >= total) return;

    int t = tid;
    const int cvec = t % C_vec;          t /= C_vec;
    const int q    = t % Q;              t /= Q;
    const int p    = t % P;              t /= P;
    const int n    = t;
    const int c0   = cvec * VEC;

    float acc[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.f;

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int h_in = p * stride_h + r - pad_h;
        #pragma unroll
        for (int s = 0; s < S; ++s) {
            const int w_in = q * stride_w + s - pad_w;

            __half x_v[VEC];
            __half w_v[VEC];

            *reinterpret_cast<Vec*>(w_v) =
                *reinterpret_cast<const Vec*>(&F[(r * S + s) * C + c0]);

            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                *reinterpret_cast<Vec*>(x_v) =
                    *reinterpret_cast<const Vec*>(
                        &X[((n * H + h_in) * W + w_in) * C + c0]);
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; ++v) x_v[v] = __float2half(0.f);
            }

            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                acc[v] += __half2float(x_v[v]) * __half2float(w_v[v]);
            }
        }
    }

    __half y_v[VEC];
    #pragma unroll
    for (int v = 0; v < VEC; ++v) y_v[v] = __float2half(acc[v]);
    *reinterpret_cast<Vec*>(&Y[((n * P + p) * Q + q) * C + c0]) =
        *reinterpret_cast<Vec*>(y_v);
}

// Host launcher -----------------------------------------------------------
torch::Tensor dwconv_v0(torch::Tensor x, torch::Tensor w, int64_t pad) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must live on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be fp16");
    TORCH_CHECK(x.dim() == 4, "x must be [N,H,W,C]");
    TORCH_CHECK(w.dim() == 3, "w must be [R,S,C]");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x/w must be contiguous");

    const int N = x.size(0);
    const int H = x.size(1);
    const int W = x.size(2);
    const int C = x.size(3);
    const int R = w.size(0);
    const int S = w.size(1);
    TORCH_CHECK(R == kR && S == kS, "Only 3x3 kernel supported in v0");
    TORCH_CHECK(C == w.size(2), "channel mismatch between x and w");
    TORCH_CHECK(C % kVec == 0, "C must be a multiple of VEC=8 (fp16 float4)");

    const int pad_h = pad, pad_w = pad;
    const int stride_h = 1, stride_w = 1;
    const int P = (H + 2 * pad_h - R) / stride_h + 1;
    const int Q = (W + 2 * pad_w - S) / stride_w + 1;

    auto y = torch::empty({N, P, Q, C}, x.options());

    const int threads_per_block = 128;
    const int total = N * P * Q * (C / kVec);
    const int num_blocks = (total + threads_per_block - 1) / threads_per_block;
    dim3 block(threads_per_block);
    dim3 grid(num_blocks);

    auto stream = at::cuda::getCurrentCUDAStream();
    dwconv_v0_kernel<kR, kS, kVec><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, H, W, C, P, Q,
        pad_h, pad_w, stride_h, stride_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

} // namespace tma_dwconv
