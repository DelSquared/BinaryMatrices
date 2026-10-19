
#include "binary_matrix_cuda.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bmcuda {
namespace {

using Word = std::uint64_t;

constexpr std::size_t WORD_BITS = 64;
constexpr int THREADS = 256;

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(status)
        );
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes != 0) {
            cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const {
        return ptr_;
    }

private:
    void* ptr_ = nullptr;
};

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Matrix size overflow");
    }

    return a * b;
}

std::size_t words_for(std::size_t k) {
    return k / WORD_BITS + (k % WORD_BITS != 0);
}

void validate_binary(
    const std::int8_t* values,
    std::size_t count,
    const char* name
) {
    if (count != 0 && values == nullptr) {
        throw std::invalid_argument(
            std::string(name) + " is null"
        );
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] != -1 && values[i] != 1) {
            throw std::invalid_argument(
                std::string(name) +
                " must contain only -1 and +1"
            );
        }
    }
}

// Encoding convention:
//     +1 -> 0
//     -1 -> 1
//
// Pack A by row:
//     packed_a[row * words + word]
//
// Pack B by output column:
//     packed_b[col * words + word]
//
// Both matrices use the same bit encoding. Consequently,
// XOR is 0 for matching signs and 1 for differing signs.
void pack_matrices(
    const std::int8_t* a,
    const std::int8_t* b,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::vector<Word>& packed_a,
    std::vector<Word>& packed_b
) {
    const std::size_t words = words_for(k);

    packed_a.assign(checked_product(m, words), Word{0});
    packed_b.assign(checked_product(n, words), Word{0});

    // Pack A: row-major input.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            if (a[i * k + p] == -1) {
                packed_a[i * words + p / WORD_BITS] |=
                    Word{1} << (p % WORD_BITS);
            }
        }
    }

    // Pack B by output column for convenient kernel access.
    // Original B is row-major with shape (k, n).
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t p = 0; p < k; ++p) {
            if (b[p * n + j] == -1) {
                packed_b[j * words + p / WORD_BITS] |=
                    Word{1} << (p % WORD_BITS);
            }
        }
    }
}

// For bipolar values {-1, +1} encoded as {1, 0}:
//
//     dot(a, b) = K - 2 * popcount(a XOR b)
//
// Padding bits are initialized to zero in both packed matrices.
// Since they match, they contribute nothing to the mismatch count.
__device__ __forceinline__
std::int32_t packed_dot_score(
    const Word* a,
    const Word* b,
    std::size_t k,
    std::size_t words
) {
    unsigned int different = 0;

    for (std::size_t w = 0; w < words; ++w) {
        const Word x = a[w];
        const Word y = b[w];

        different += __popcll(
            static_cast<unsigned long long>(x ^ y)
        );
    }

    const long long score =
        static_cast<long long>(k) -
        2LL * static_cast<long long>(different);

    return static_cast<std::int32_t>(score);
}

// Each thread computes one matrix output.
// The integer score is preserved exactly.
__global__ void packed_matmul_scores_kernel(
    const Word* a,
    const Word* b,
    std::int32_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n,
    std::size_t words,
    std::size_t total
) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    // Grid-stride loop also avoids requiring a huge grid
    // for very large output matrices.
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t index = first;
         index < total;
         index += stride) {

        const std::size_t row = index / n;
        const std::size_t col = index % n;

        output[index] = packed_dot_score(
            a + row * words,
            b + col * words,
            k,
            words
        );
    }
}

// Binary output is computed directly on the GPU.
// This avoids allocating an intermediate score array and
// avoids a separate CPU thresholding pass.
__global__ void packed_matmul_binary_kernel(
    const Word* a,
    const Word* b,
    std::int8_t* output,
    std::size_t k,
    std::size_t n,
    std::size_t words,
    std::size_t total
) {
    const std::size_t first =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t index = first;
         index < total;
         index += stride) {

        const std::size_t row = index / n;
        const std::size_t col = index % n;

        const std::int32_t score = packed_dot_score(
            a + row * words,
            b + col * words,
            k,
            words
        );

        output[index] = score >= 0 ? 1 : -1;
    }
}

void compute_output(
    const std::int8_t* a,
    const std::int8_t* b,
    void* output,
    bool binary_output,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    const std::size_t a_count = checked_product(m, k);
    const std::size_t b_count = checked_product(k, n);
    const std::size_t out_count = checked_product(m, n);

    if (out_count == 0) {
        return;
    }

    if (output == nullptr) {
        throw std::invalid_argument("Output is null");
    }

    // The public score API returns int32, so every possible
    // score must fit within that type.
    if (k > static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(
            "Inner dimension exceeds int32 score range"
        );
    }

    validate_binary(a, a_count, "A");
    validate_binary(b, b_count, "B");

    std::vector<Word> packed_a;
    std::vector<Word> packed_b;

    pack_matrices(
        a, b, m, k, n,
        packed_a, packed_b
    );

    const std::size_t words = words_for(k);

    DeviceBuffer device_a(
        checked_product(packed_a.size(), sizeof(Word))
    );

    DeviceBuffer device_b(
        checked_product(packed_b.size(), sizeof(Word))
    );

    const std::size_t output_element_size =
        binary_output ? sizeof(std::int8_t)
                      : sizeof(std::int32_t);

    DeviceBuffer device_output(
        checked_product(out_count, output_element_size)
    );

    if (!packed_a.empty()) {
        cuda_check(
            cudaMemcpy(
                device_a.get(),
                packed_a.data(),
                checked_product(packed_a.size(), sizeof(Word)),
                cudaMemcpyHostToDevice
            ),
            "Copy A to GPU"
        );
    }

    if (!packed_b.empty()) {
        cuda_check(
            cudaMemcpy(
                device_b.get(),
                packed_b.data(),
                checked_product(packed_b.size(), sizeof(Word)),
                cudaMemcpyHostToDevice
            ),
            "Copy B to GPU"
        );
    }

    // Avoid overflow in (out_count + THREADS - 1).
    const std::size_t blocks =
        out_count / THREADS +
        (out_count % THREADS != 0);

    // CUDA's x grid dimension is limited by the device.
    // Use a conservative upper bound and let the kernel's
    // grid-stride loop cover the remaining outputs.
    constexpr std::size_t MAX_GRID_X = 2147483647ULL;

    const unsigned int grid_x = static_cast<unsigned int>(
        blocks < MAX_GRID_X ? blocks : MAX_GRID_X
    );

    if (binary_output) {
        packed_matmul_binary_kernel<<<grid_x, THREADS>>>(
            static_cast<const Word*>(device_a.get()),
            static_cast<const Word*>(device_b.get()),
            static_cast<std::int8_t*>(device_output.get()),
            k, n, words, out_count
        );
    } else {
        packed_matmul_scores_kernel<<<grid_x, THREADS>>>(
            static_cast<const Word*>(device_a.get()),
            static_cast<const Word*>(device_b.get()),
            static_cast<std::int32_t*>(device_output.get()),
            m, k, n, words, out_count
        );
    }

    cuda_check(cudaGetLastError(), "Launch matrix kernel");
    cuda_check(cudaDeviceSynchronize(), "Execute matrix kernel");

    cuda_check(
        cudaMemcpy(
            output,
            device_output.get(),
            checked_product(out_count, output_element_size),
            cudaMemcpyDeviceToHost
        ),
        "Copy output from GPU"
    );
}

}  // namespace

void matmul_scores(
    const std::int8_t* a,
    const std::int8_t* b,
    std::int32_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    compute_output(
        a, b, output, false, m, k, n
    );
}

void matmul_binary(
    const std::int8_t* a,
    const std::int8_t* b,
    std::int8_t* output,
    std::size_t m,
    std::size_t k,
    std::size_t n
) {
    compute_output(
        a, b, output, true, m, k, n
    );
}

}  // namespace bmcuda


// The next meaningful optimization would be a GPU-resident packed-matrix interface that lets you pack once and reuse the packed data across multiple multiplications. That is likely to be more beneficial for repeated inference workloads than changing from 64-bit to 32-bit words alone.
