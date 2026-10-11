# BinaryMatrix

A header-only C++17 library for packed binary matrix operations tailored for Binary Neural Networks (BNNs).

## Key Features

* **Bit-Packed Storage**: Packs 64 binary values into each `std::uint64_t` word.
* **Aligned Memory**: Uses 64-byte aligned storage allocations to support SIMD instructions.
* **Runtime SIMD Dispatch**: Detects CPU features at runtime to automatically dispatch to AVX2 or AVX-512 VPOPCNTDQ kernels without requiring `-march=native`.
* **Optimized Inference**: Includes dedicated `BinaryWeightMatrix` storage that permanently transposes and packs weight vectors to accelerate repeated inference.
* **Flexible Output Modes**: Supports direct bit-packed binary multiplication and integer accumulator outputs.
* **C++17 Compatible**: Works with standard C++17 compilers across Linux, macOS, and MSVC.

---

## Data Representation & Mathematics

Binary values are encoded as single bits within 64-bit unsigned integers (`uint64_t`):
* **Bit 0** $\rightarrow +1$
* **Bit 1** $\rightarrow -1$

Multiplication between two binary values translates directly to bitwise equality:
* $(+1) \times (+1) = +1$
* $(-1) \times (-1) = +1$
* $(+1) \times (-1) = -1$

For a vector of $K$ elements, the dot product is calculated using bitwise XOR and population count:

$$\text{dot}(a, b) = K - 2 \times \text{popcount}(a \oplus b)$$

The output bit representation maps the dot product threshold as follows:
* $\text{dot} \ge 0 \rightarrow +1$ (Bit `0`)
* $\text{dot} < 0 \rightarrow -1$ (Bit `1`)

---

## Performance Benchmarks

Benchmark results comparing standard clipped matrix multiplication (`int` arithmetic + thresholding) against packed `BinaryMatrix` and pre-packed `BinaryWeightMatrix` inference paths on AVX2 hardware:

| Matrix Dimensions ($M \times K \times N$) | Implementation | Execution Time | Speedup |
| :--- | :--- | :--- | :--- |
| **32 × 128 * 128 × 64** | Standard clipped | 19.888 ms | 1.00x |
| | Packed `BinaryMatrix` | 3.054 ms | 6.51x |
| | Pre-packed weights | 2.645 ms | **7.52x** |
| **64 × 256 * 256 × 128** | Standard clipped | 163.151 ms | 1.00x |
| | Packed `BinaryMatrix` | 21.339 ms | 7.65x |
| | Pre-packed weights | 20.218 ms | **8.07x** |
| **128 × 512 * 512 × 256** | Standard clipped | 1589.098 ms | 1.00x |
| | Packed `BinaryMatrix` | 162.553 ms | 9.78x |
| | Pre-packed weights | 155.713 ms | **10.21x** |
| **37 × 193 * 193 × 71** *(Non-aligned)* | Standard clipped | 38.743 ms | 1.00x |
| | Packed `BinaryMatrix` | 5.700 ms | 6.80x |
| | Pre-packed weights | 5.161 ms | **7.51x** |

---

## Build & Requirements

### Prerequisites
* C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)

### Compilation Examples
Standard build with default runtime SIMD detection:
```bash
g++ -O3 -std=c++17 main.cpp -o bnn
```

Target-native build for additional local optimizations:
```bash
g++ -O3 -march=native -std=c++17 main.cpp -o bnn
```

---

## Quick Start Examples

### 1. Basic Binary Matrix Multiplication
```cpp
#include "binary_matrix.hpp"
#include <iostream>

int main()
{
    // Create matrices
    bnn::BinaryMatrix A(2, 4);
    bnn::BinaryMatrix B(4, 3);

    // Set signed values (+1 or -1)
    A.set_value(0, 0, +1);
    A.set_value(0, 1, -1);
    A.set_value(0, 2, +1);
    A.set_value(0, 3, -1);

    B.set_value(0, 0, +1);
    B.set_value(1, 0, -1);
    B.set_value(2, 0, +1);
    B.set_value(3, 0, -1);

    // Perform packed multiplication
    bnn::BinaryMatrix C = bnn::multiply(A, B);

    return 0;
}
```

### 2. High-Performance Inference with Pre-packed Weights
```cpp
#include "binary_matrix.hpp"

void run_inference()
{
    constexpr std::size_t K = 1024;
    constexpr std::size_t N = 256;

    // Create and randomize weights
    bnn::BinaryMatrix weights_matrix(K, N);
    weights_matrix.randomize();

    // Pre-pack weights into contiguous output vectors
    bnn::BinaryWeightMatrix weights(weights_matrix);

    // Activation input
    bnn::BinaryMatrix activations(32, K);
    activations.randomize();

    // Fast inference pass
    bnn::BinaryMatrix output = bnn::multiply(activations, weights);
}
```

### 3. Integer Accumulator Output
To retrieve the raw integer dot-product totals instead of bit-packed outputs:
```cpp
#include "binary_matrix.hpp"
#include <vector>

void accumulate_example(const bnn::BinaryMatrix& A, const bnn::BinaryWeightMatrix& weights)
{
    std::vector<std::int32_t> accumulators;
    
    // Fills vector with raw integer dot products [-K, ..., +K]
    bnn::multiply_accumulate(A, weights, accumulators);
}
```

---

## API Summary

### Classes
* `bnn::BinaryMatrix(rows, cols)`: Standard bit-packed matrix.
* `bnn::BinaryWeightMatrix(matrix)`: Pre-transposed, contiguously packed weight matrix optimized for inference.

### Primary Operations
* `bnn::multiply(A, B)`: Multiplies two binary matrices.
* `bnn::multiply(activations, weights)`: Multiplies activations against pre-packed weights.
* `bnn::multiply_accumulate(A, weights, output_vec)`: Multiplies and outputs un-clipped integer accumulators into a `std::vector<int32_t>`.
* `bnn::dot(matrix, row, weights, output_idx)`: Single dot product evaluation.

### Conversions & Utilities
* `bnn::from_signed(array, rows, cols)`: Constructs a `BinaryMatrix` from a signed input buffer containing `+1` and `-1`.
* `bnn::to_signed<T>(matrix)`: Exports a `BinaryMatrix` to a `std::vector<T>` of signed values.
* `matrix.transpose()`: Returns a transposed `BinaryMatrix`.
* `matrix.randomize(rng)`: Randomizes matrix bit values while maintaining padding bits.

### Hardware & Backend Inspection
* `bnn::simd_backend()`: Returns active backend string (`"Scalar"`, `"AVX2"`, or `"AVX-512 VPOPCNTDQ"`).
* `bnn::has_avx2()`: Returns `true` if AVX2 kernels are enabled.
* `bnn::has_avx512_vpopcntdq()`: Returns `true` if AVX-512 VPOPCNTDQ kernels are enabled.