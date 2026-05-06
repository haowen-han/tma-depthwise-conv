// Unified TMA + warp-specialized depthwise-conv launcher.
//
// N_STAGE (pipeline depth of the SMEM ring buffer) is a compile-time
// template parameter of the kernel; we dispatch the Python-provided value to
// a small set of supported instantiations.

#include "kernel_tma_ws.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace tma_dwconv {

namespace {

template <int N_STAGE>
void launch_impl(torch::Tensor const& x,
                 torch::Tensor const& w,
                 torch::Tensor& y,
                 ConvParams const& p)
{
    auto tma_atom_x = make_tma_atom_x(x.data_ptr(), p);

    constexpr int WARPS_PER_CTA = 5;   // 1 producer + 4 consumer
    const int threads_per_cta = WARPS_PER_CTA * 32;

    dim3 grid((p.P + kBlockP - 1) / kBlockP,
              p.C / kBlockC,
              p.N);
    dim3 block(threads_per_cta);

    size_t smem_bytes = sizeof(SmemLayout<N_STAGE>);
    auto* kptr = &dwconv_tma_ws_kernel<decltype(tma_atom_x),
                                       N_STAGE, WARPS_PER_CTA>;
    TORCH_CHECK(cudaFuncSetAttribute(
        kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)) == cudaSuccess, "smem set failed");

    auto stream = at::cuda::getCurrentCUDAStream();
    kptr<<<grid, block, smem_bytes, stream>>>(
        tma_atom_x,
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

torch::Tensor dwconv_ws(torch::Tensor x, torch::Tensor w,
                        int64_t pad, int64_t n_stage) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "tensors must live on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be fp16");
    TORCH_CHECK(x.dim() == 4, "x must be [N,H,W,C]");
    TORCH_CHECK(w.dim() == 3, "w must be [R,S,C]");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x/w must be contiguous");

    ConvParams p{};
    p.N = x.size(0); p.H = x.size(1); p.W = x.size(2); p.C = x.size(3);
    TORCH_CHECK(w.size(0) == kR && w.size(1) == kS, "only 3x3 kernel supported");
    TORCH_CHECK(w.size(2) == p.C, "channel mismatch");
    TORCH_CHECK(p.C % kBlockC == 0, "C must be a multiple of BlockC=64");

    p.pad_h = pad; p.pad_w = pad;
    p.stride_h = 1; p.stride_w = 1;
    p.P = (p.H + 2 * p.pad_h - kR) / p.stride_h + 1;
    p.Q = (p.W + 2 * p.pad_w - kS) / p.stride_w + 1;

    auto y = torch::empty({p.N, p.P, p.Q, p.C}, x.options());

    switch (n_stage) {
        case 2: launch_impl<2>(x, w, y, p); break;
        case 3: launch_impl<3>(x, w, y, p); break;
        case 4: launch_impl<4>(x, w, y, p); break;
        default:
            TORCH_CHECK(false, "n_stage must be one of {2, 3, 4}, got ", n_stage);
    }
    return y;
}

} // namespace tma_dwconv
