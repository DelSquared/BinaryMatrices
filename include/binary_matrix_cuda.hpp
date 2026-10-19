
#pragma once

#include <cstddef>
#include <cstdint>

namespace bmcuda {

void matmul_scores(
    const int8_t* a,
    const int8_t* b,
    int32_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

void matmul_binary(
    const int8_t* a,
    const int8_t* b,
    int8_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
);

}  // namespace bmcuda

