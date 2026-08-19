# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import platform
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

_build_support_spec = importlib.util.spec_from_file_location(
    "fastsafetensors_build_support",
    Path(__file__).parent / "fastsafetensors" / "build_support.py",
)
if _build_support_spec is None or _build_support_spec.loader is None:
    raise RuntimeError("cannot load fastsafetensors build support")
_build_support = importlib.util.module_from_spec(_build_support_spec)
_build_support_spec.loader.exec_module(_build_support)
compiler_accepts = _build_support.compiler_accepts
wants_gb10_target = _build_support.wants_gb10_target


class FastSafeTensorsBuildExt(build_ext):
    """Apply GB10 tuning only after compiler initialization."""

    def build_extensions(self):
        if wants_gb10_target():
            flag = "-mcpu=gb10"
            if compiler_accepts(self.compiler, flag):
                for extension in self.extensions:
                    extension.extra_compile_args.append(flag)
                self.announce(f"enabling supported compiler flag {flag}", 2)
            else:
                self.announce(f"unsupported {flag}; using portable flags", 2)
        super().build_extensions()


def MyExtension(name, sources, mod_name, *args, **kwargs):
    import pybind11

    pybind11_path = os.path.dirname(pybind11.__file__)

    kwargs["define_macros"] = [("__MOD_NAME__", mod_name)]
    kwargs["libraries"] = ["stdc++"]
    kwargs["include_dirs"] = kwargs.get("include_dirs", []) + [
        f"{pybind11_path}/include"
    ]
    kwargs["language"] = "c++"
    kwargs["extra_compile_args"] = ["-fvisibility=hidden", "-std=c++17"]

    # Windows-specific configuration for DirectStorage + D3D12/CUDA interop
    if platform.system() == "Windows":
        sources.append("fastsafetensors/cpp/dstorage_reader.cpp")
        kwargs["libraries"] = []
        # c++20 required for designated initializers at ext.hpp
        kwargs["extra_compile_args"] = ["/std:c++20"]
        # DirectStorage, D3D12, and DXGI DLLs are loaded at runtime so importing
        # the extension does not require GPU/DirectX runtime DLLs to be present.

        # CUDA interop headers: if CUDA_HOME/CUDA_PATH is set, add include path
        # for cudaExternalMemory types used by the interop bridge.
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cuda_include = os.path.join(cuda_home, "include")
            if os.path.isdir(cuda_include):
                kwargs["include_dirs"].append(cuda_include)

    return Extension(name, sources, *args, **kwargs)


package_data_patterns = ["*.hpp", "*.h", "cpp.pyi"]

setup(
    cmdclass={"build_ext": FastSafeTensorsBuildExt},
    packages=[
        "fastsafetensors",
        "fastsafetensors.copier",
        "fastsafetensors.cpp",
        "fastsafetensors.frameworks",
    ],
    include_package_data=True,
    package_data={"fastsafetensors.cpp": package_data_patterns},
    ext_modules=[
        MyExtension(
            name="fastsafetensors.cpp",
            sources=["fastsafetensors/cpp/ext.cpp"],
            include_dirs=["fastsafetensors/cpp"],
            mod_name="cpp",
        )
    ],
)
