import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUTLASS include path. The plan expects cutlass to live alongside this project:
#   /ssd1/hanhaowen/cutlass   (sibling of /ssd1/hanhaowen/tma-depthwise-conv)
# You can override it with the CUTLASS_DIR environment variable.
HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_DIR = os.environ.get(
    "CUTLASS_DIR",
    os.path.normpath(os.path.join(HERE, "..", "cutlass")),
)

cutlass_includes = [
    os.path.join(CUTLASS_DIR, "include"),
    os.path.join(CUTLASS_DIR, "tools", "util", "include"),
]

setup(
    name="tma_dwconv",
    version="0.0.1",
    packages=find_packages(exclude=["tests", "bench"]),
    ext_modules=[
        CUDAExtension(
            name="tma_dwconv._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/kernel_v0_naive_float4.cu",
                "csrc/kernel_v1_tma_ws_db.cu",
                "csrc/kernel_v2_tma_ws_ms.cu",
            ],
            include_dirs=cutlass_includes,
            libraries=["cuda"],  # driver API: cuTensorMapEncodeTiled
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--generate-code=arch=compute_100a,code=sm_100a",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
