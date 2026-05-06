// PyBind entry points.
#include <torch/extension.h>

namespace tma_dwconv {
torch::Tensor dwconv_v0(torch::Tensor x, torch::Tensor w, int64_t pad);
torch::Tensor dwconv_v1(torch::Tensor x, torch::Tensor w, int64_t pad);
torch::Tensor dwconv_v2(torch::Tensor x, torch::Tensor w, int64_t pad);
}  // namespace tma_dwconv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dwconv_v0", &tma_dwconv::dwconv_v0,
          "Depthwise conv2d NHWC, naive float4 baseline (v0).",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("pad"));
    m.def("dwconv_v1", &tma_dwconv::dwconv_v1,
          "Depthwise conv2d NHWC, TMA + warp-specialized + double-buffer (v1).",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("pad"));
    m.def("dwconv_v2", &tma_dwconv::dwconv_v2,
          "Depthwise conv2d NHWC, TMA + warp-specialized + multi-stage (v2).",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("pad"));
}
