
# BinaryMatricesCUDA

An experimental CUDA implementation of packed binary matrix multiplication
with a C++ core and Cython Python interface.

## Requirements

- Linux
- NVIDIA GPU and compatible driver
- CUDA Toolkit, including `nvcc`
- CMake 3.24+
- C++17 compiler
- Python 3.10+
- NumPy, Cython, setuptools, wheel

Check the GPU and CUDA compiler:

```bash
nvidia-smi
nvcc --version
cmake --version
```

## Build the CUDA library

From the repository root, select an architecture supported by your
GPU and installed CUDA Toolkit. For example, use 86 for many Ampere
GPUs, or 89 for many Ada GPUs.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

## Build the Python extension

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy Cython setuptools wheel

python -m pip install -e .
```

The CUDA library must be built before installing the Python extension.

## Verify correctness

```bash
python tests/test_correctness.py
```

## Run the benchmark

```bash
python tests/benchmark.py
```

## Python example

```python
import numpy as np
from binary_matrices_cuda import matmul_binary, matmul_scores

a = np.array([
    [1, -1, 1],
    [-1, -1, 1],
], dtype=np.int8)

b = np.array([
    [1, -1],
    [-1, 1],
    [1, 1],
], dtype=np.int8)

scores = matmul_scores(a, b)
binary = matmul_binary(a, b)

print(scores)
print(binary)
```

## Representation

Input values -1 and +1 are encoded as bits 0 and 1.

The implementation packs 64 values per uint64 word. It calculates
signed dot products using:

    dot(a, b) = K - 2 * popcount(a XOR b)

Binary outputs use the convention score >= 0 -> +1, otherwise -1.

## Current limitations

- Matrices are packed on the CPU on every call.
- GPU memory is allocated and released on every call.
- Inputs and outputs are transferred for every operation.
- This is a correctness-first baseline, not a tuned GEMM kernel.
- The initial kernel assigns one CUDA thread to each output element.

Future work includes persistent GPU-resident matrices, reusable packed
weights, tiled kernels, optimized memory access, fused output thresholding,
and comparisons against the original CPU implementation.

