# TMA Depthwise Conv2D (NHWC) on Blackwell B200

This project implements a **NHWC Depthwise Conv2D forward** operator for NVIDIA
Blackwell B200 (sm_100) in order to study **TMA (Tensor Memory Accelerator)**,
`mbarrier`, and **warp-specialized multi-stage pipelines**.

C++/CUDA only does the kernel work. All correctness checks and benchmarks are
driven from Python using PyTorch CUDA Extensions.

## Versions

| id | description |
| -- | ----------- |
| v0 | Naive kernel, no smem, no TMA; `float4`-vectorized global loads/stores. |
| v1 | TMA + warp specialization + **double buffer** (N_STAGE=2). |
| v2 | TMA + warp specialization + **multi-stage pipeline** (N_STAGE=4). |

Fixed config (first version):
- fp16 input, fp32 accumulator, fp16 output
- `R = S = 3`, `stride = 1`, `padding = 1` (same), `dilation = 1`
- `BlockP = 8`, `BlockQ = 32`, `BlockC = 64`

## Layout

```
tma-depthwise-conv/
├── setup.py                           # CUDAExtension build
├── pyproject.toml
├── csrc/                              # CUDA source (kernels only)
│   ├── bindings.cpp
│   ├── common.cuh
│   ├── kernel_v0_naive_float4.cu
│   ├── kernel_tma_ws.cuh              # shared template for v1/v2
│   ├── kernel_v1_tma_ws_db.cu
│   └── kernel_v2_tma_ws_ms.cu
├── tma_dwconv/                        # Python package
│   ├── __init__.py
│   └── reference.py
├── tests/                             # pytest correctness tests
└── bench/                             # benchmark scripts
```



## Build & test

```bash
# build the extension in-place
pip install -e .

# unit tests
pytest tests/ -v

# benchmark all versions + cuDNN
python -m bench.run_all
```
