import sys
import os
import shlex
import sysconfig
import setuptools._distutils.sysconfig as distutils_sysconfig
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


def _dedupe_linker_flags(flags: str) -> str:
    deduped = []
    seen = set()
    for token in shlex.split(flags):
        if token.startswith("-Wl,-rpath,") or token.startswith("-L"):
            if token in seen:
                continue
            seen.add(token)
        deduped.append(token)
    return " ".join(deduped)


def _dedupe_linker_command(command):
    if isinstance(command, str):
        return _dedupe_linker_flags(command)

    deduped = []
    seen = set()
    for token in command:
        if token.startswith("-Wl,-rpath,") or token.startswith("-L"):
            if token in seen:
                continue
            seen.add(token)
        deduped.append(token)
    return deduped


def _strip_optimization_command(command):
    if isinstance(command, str):
        return _strip_optimization_flags(command)

    filtered = []
    for token in command:
        if token.startswith("-O"):
            continue
        filtered.append(token)
    return filtered


def _strip_optimization_flags(flags: str) -> str:
    filtered = []
    for token in shlex.split(flags):
        if token.startswith("-O"):
            continue
        filtered.append(token)
    return " ".join(filtered)


def _dedupe_sysconfig_linker_flags(config_module) -> None:
    config_vars = config_module.get_config_vars()
    for key in ("LDFLAGS", "LDSHARED", "LDCXXSHARED"):
        value = config_vars.get(key)
        if value:
            config_vars[key] = _dedupe_linker_flags(value)


def _strip_sysconfig_optimization_flags(config_module) -> None:
    config_vars = config_module.get_config_vars()
    for key in (
        "OPT",
        "CFLAGS",
        "CXXFLAGS",
        "PY_CFLAGS",
        "PY_CFLAGS_NODIST",
        "CONFIGURE_CFLAGS",
        "CONFIGURE_CFLAGS_NODIST",
    ):
        value = config_vars.get(key)
        if value:
            config_vars[key] = _strip_optimization_flags(value)


class CleanBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_ninja", False)
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        if sys.platform == "darwin":
            original_spawn = self.compiler.spawn

            def clean_spawn(cmd, **kwargs):
                if isinstance(cmd, (list, tuple)):
                    cmd = list(cmd)
                    if "-c" in cmd:
                        cmd = _strip_optimization_command(cmd)
                        cmd.append("-O3")
                    else:
                        cmd = _dedupe_linker_command(cmd)
                return original_spawn(cmd, **kwargs)

            self.compiler.spawn = clean_spawn

            for attr in ("compiler", "compiler_so", "compiler_cxx"):
                value = getattr(self.compiler, attr, None)
                if value:
                    setattr(self.compiler, attr, _strip_optimization_command(value))

            for attr in ("linker_so", "linker_so_cxx"):
                value = getattr(self.compiler, attr, None)
                if value:
                    setattr(self.compiler, attr, _dedupe_linker_command(value))

            executables = getattr(self.compiler, "executables", None)
            if isinstance(executables, dict):
                for key in ("compiler", "compiler_so", "compiler_cxx"):
                    value = executables.get(key)
                    if value:
                        executables[key] = _strip_optimization_command(value)

                for key in ("linker_so", "linker_so_cxx"):
                    value = executables.get(key)
                    if value:
                        executables[key] = _dedupe_linker_command(value)

        super().build_extensions()

if _HAS_TORCH:
    if sys.platform == "darwin":
        _strip_sysconfig_optimization_flags(sysconfig)
        _strip_sysconfig_optimization_flags(distutils_sysconfig)
        _dedupe_sysconfig_linker_flags(sysconfig)
        _dedupe_sysconfig_linker_flags(distutils_sysconfig)

    include_dirs = [
        os.path.join(workspace_dir, "warpconvnet/csrc/include"),
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
                "warpconvnet/csrc/discrete_kernels.cpp",
                "warpconvnet/csrc/coords_launch.cpp",
                "warpconvnet/csrc/bindings/coords_bindings.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args={"cxx": cxx_args},
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": CleanBuildExtension}
else:
    print("PyTorch not found — skipping C++ extension build.")


# python3 setup.py build_ext --inplace
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
