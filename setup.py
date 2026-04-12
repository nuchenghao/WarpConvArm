import sys
import os
from setuptools import setup

try:
    import torch
    import torch.utils.cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

workspace_dir = os.path.dirname(os.path.abspath(__file__))
ext_modules = []
cmdclass = {}

if _HAS_TORCH:
    include_dirs = [
        os.path.join(workspace_dir, "warpconvnet/csrc"),
    ]

    cxx_args = ["-std=c++20", "-O3"]

    # On Apple Silicon, link against libdispatch (Grand Central Dispatch)
    extra_link_args = []
    if sys.platform == "darwin":
        extra_link_args = ["-framework", "Foundation"]

    ext_modules = [
        CppExtension(
            name="warpconvnet._C",
            sources=[
                "warpconvnet/csrc/warpconvnet_pybind.cpp",
                "warpconvnet/csrc/hashmap_kernels.cpp",
                "warpconvnet/csrc/coords_launch.cpp",
                "warpconvnet/csrc/bindings/coords_bindings.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_args},
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
else:
    print("PyTorch not found — skipping C++ extension build.")


# python3 setup.py build_ext --inplace
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
