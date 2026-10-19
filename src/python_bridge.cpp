
#include "binary_matrix_cuda.hpp"

#include <cstddef>
#include <cstdint>

void bm_cuda_matmul_scores(
    const int8_t* a,
    const int8_t* b,
    int32_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    bmcuda::matmul_scores(a, b, output, m, k, n);
}

void bm_cuda_matmul_binary(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    bmcuda::matmul_binary(a, b, output, m, k, n);
}

