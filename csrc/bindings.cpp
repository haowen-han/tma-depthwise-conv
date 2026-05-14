// PyBind entry points.
#include <torch/extension.h>
#include <tuple>

namespace tma_dwconv {
torch::Tensor dwconv_v0(torch::Tensor x, torch::Tensor w, int64_t pad);
torch::Tensor dwconv_ws(torch::Tensor x, torch::Tensor w,
                        int64_t pad, int64_t n_stage);
std::tuple<torch::Tensor, torch::Tensor> dwconv_ws_probe(
    torch::Tensor x, torch::Tensor w, int64_t pad, int64_t n_stage);
}  // namespace tma_dwconv

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dwconv_v0", &tma_dwconv::dwconv_v0,
          "Depthwise conv2d NHWC, naive float4 baseline (v0).",
          pybind11::arg("x"), pybind11::arg("w"), pybind11::arg("pad"));
    m.def("dwconv_ws", &tma_dwconv::dwconv_ws,
          "Depthwise conv2d NHWC, TMA + warp-specialized, configurable pipeline depth.",
          pybind11::arg("x"), pybind11::arg("w"),
          pybind11::arg("pad"), pybind11::arg("n_stage"));
    m.def("dwconv_ws_probe", &tma_dwconv::dwconv_ws_probe,
          "Same as dwconv_ws but also returns a per-CTA stall-cycle tensor "
          "of shape [num_blocks, 3] (cols: producer_stall, consumer_stall, total).",
          pybind11::arg("x"), pybind11::arg("w"),
          pybind11::arg("pad"), pybind11::arg("n_stage"));
}
