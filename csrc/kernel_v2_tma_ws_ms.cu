// v2: TMA + warp-specialized + multi-stage (N_STAGE = 4).
//
// Same kernel body as v1, just a deeper pipeline.

#include "kernel_tma_ws.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace tma_dwconv {

torch::Tensor dwconv_v2(torch::Tensor x, torch::Tensor w, int64_t pad) {
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
    p.dilation_h = 1; p.dilation_w = 1;
    p.P = (p.H + 2 * p.pad_h - kR) / p.stride_h + 1;
    p.Q = (p.W + 2 * p.pad_w - kS) / p.stride_w + 1;

    auto y = torch::empty({p.N, p.P, p.Q, p.C}, x.options());

    CUtensorMap tmap_x{};
    CUresult res = build_tmap_x_f16(tmap_x, x.data_ptr(), p);
    TORCH_CHECK(res == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed, res=", int(res));

    constexpr int N_STAGE       = 4;
    constexpr int WARPS_PER_CTA = 5;   // 1 producer + 4 consumer = 5 warps
    const int threads_per_cta = WARPS_PER_CTA * 32;

    dim3 grid((p.P + kBlockP - 1) / kBlockP,
              p.C / kBlockC,
              p.N);
    dim3 block(threads_per_cta);

    size_t smem_bytes = sizeof(SmemLayout<N_STAGE>);
    auto* kptr = &dwconv_tma_ws_kernel<N_STAGE, WARPS_PER_CTA>;
    TORCH_CHECK(cudaFuncSetAttribute(
        kptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)) == cudaSuccess, "smem set failed");

    auto stream = at::cuda::getCurrentCUDAStream();
    kptr<<<grid, block, smem_bytes, stream>>>(
        tmap_x,
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

} // namespace tma_dwconv
