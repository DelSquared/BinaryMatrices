
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


ROOT = Path(__file__).resolve().parent
CUDA_HOME = Path(
    os.environ.get("CUDA_HOME", "/usr/local/cuda")
).resolve()

NVCC = shutil.which("nvcc")
if NVCC is None:
    candidate = CUDA_HOME / "bin" / "nvcc"
    if candidate.exists():
        NVCC = str(candidate)

if NVCC is None:
    raise RuntimeError("nvcc not found. Set CUDA_HOME or install CUDA Toolkit.")

CUDA_INCLUDE = CUDA_HOME / "include"

# Find the CUDA library directory.
CUDA_LIB_CANDIDATES = [
    CUDA_HOME / "lib64",
    CUDA_HOME / "targets" / "x86_64-linux" / "lib",
]

CUDA_LIB = next(
    (p for p in CUDA_LIB_CANDIDATES if p.exists()),
    None,
)
if CUDA_LIB is None:
    raise RuntimeError(f"CUDA library directory not found under {CUDA_HOME}")

CUDA_RUNTIME_STATIC = CUDA_LIB / "libcudart_static.a"
if not CUDA_RUNTIME_STATIC.exists():
    raise RuntimeError(
        f"Static CUDA runtime not found: {CUDA_RUNTIME_STATIC}"
    )

BRIDGE_SOURCE = ROOT / "src" / "python_bridge.cpp"
CUDA_SOURCE = ROOT / "src" / "binary_matrix_cuda.cu"

for path in (BRIDGE_SOURCE, CUDA_SOURCE):
    if not path.exists():
        raise RuntimeError(f"Required source file not found: {path}")


class StaticCudaBuildExt(build_ext):
    def build_extensions(self):
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)

        # Compile the CUDA kernel.
        cuda_object = build_temp / "binary_matrix_cuda.o"

        subprocess.run(
            [
                NVCC,
                "-c",
                str(CUDA_SOURCE),
                "-o",
                str(cuda_object),
                "-std=c++17",
                "-O3",
                "-Xcompiler=-fPIC",
                f"-I{ROOT / 'include'}",
            ],
            check=True,
        )

        # Compile the C++ bridge.
        bridge_object = build_temp / "python_bridge.o"

        compiler = os.environ.get("CXX", "g++")
        subprocess.run(
            [
                compiler,
                "-c",
                str(BRIDGE_SOURCE),
                "-o",
                str(bridge_object),
                "-std=c++17",
                "-O3",
                "-fPIC",
                f"-I{ROOT / 'include'}",
            ],
            check=True,
        )

        # Package the bridge into a static archive.
        archive = build_temp / "libpython_bridge.a"
        ar = shutil.which("ar")
        if ar is None:
            raise RuntimeError("'ar' was not found; install binutils.")

        if archive.exists():
            archive.unlink()

        subprocess.run(
            [ar, "rcs", str(archive), str(bridge_object)],
            check=True,
        )

        # Link both the CUDA kernel object and bridge archive.
        for ext in self.extensions:
            ext.extra_objects = list(ext.extra_objects or [])
            ext.extra_objects.extend([str(cuda_object), str(archive)])

            ext.include_dirs = list(ext.include_dirs or [])
            ext.include_dirs.append(str(CUDA_INCLUDE))

            # Link the CUDA runtime statically, plus pthread/dl support.
            ext.extra_link_args = list(ext.extra_link_args or [])
            ext.extra_link_args.extend([
                str(CUDA_RUNTIME_STATIC),
                "-lpthread",
                "-ldl",
                "-lrt",
                "-Wl,--exclude-libs,ALL",
            ])

        super().build_extensions()


extension = Extension(
    name="binary_matrices_cuda",
    sources=[
        str(ROOT / "python" / "binary_matrices_cuda.pyx"),
    ],
    include_dirs=[
        str(ROOT / "include"),
        np.get_include(),
        str(CUDA_INCLUDE),
    ],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
    # Do not also add -lcudart: libcudart_static.a is linked explicitly.
    libraries=[],
)

setup(
    name="binary-matrices-cuda",
    version="0.1.0",
    description="CUDA packed binary matrix operations",
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": 3},
    ),
    cmdclass={"build_ext": StaticCudaBuildExt},
    zip_safe=False,
)

