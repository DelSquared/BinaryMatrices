
import os
import sys
import shutil

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize


# ------------------------------------------------------------
# Compiler selection
# ------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"

CC = os.environ.get("CC", "gcc")
CXX = os.environ.get("CXX", "g++")

USE_MINGW = IS_WINDOWS and (
    "gcc" in os.path.basename(CC).lower()
    or "g++" in os.path.basename(CXX).lower()
)

if USE_MINGW:
    if shutil.which(CC) is None:
        raise RuntimeError(
            f"C compiler not found: {CC}. "
            "Check your GCC installation and PATH."
        )

    if shutil.which(CXX) is None:
        raise RuntimeError(
            f"C++ compiler not found: {CXX}. "
            "Check your GCC installation and PATH."
        )

    compile_args = [
        "-O2",
        "-std=c++17",
    ]

    link_args = [
        "-static-libgcc",
        "-static-libstdc++",
        "-static",
    ]

elif IS_WINDOWS:
    # Microsoft Visual C++
    compile_args = [
        "/O2",
        "/std:c++17",
    ]

    link_args = []

else:
    compile_args = [
        "-O2",
        "-std=c++17",
    ]

    link_args = []


# ------------------------------------------------------------
# Cython extension
# ------------------------------------------------------------

extension = Extension(
    name="binary_matrices",
    sources=["binary_matrices.pyx"],
    include_dirs=[
        np.get_include(),
        os.path.abspath("."),
    ],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)


# ------------------------------------------------------------
# Package
# ------------------------------------------------------------

setup(
    name="binary-matrices",
    version="0.1.0",
    description="Cython wrapper for packed binary matrix operations",
    ext_modules=cythonize(
        [extension],
        compiler_directives={
            "language_level": 3,
        },
    ),
    zip_safe=False,
)
