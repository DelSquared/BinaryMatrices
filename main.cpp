#include "binary_matrix.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace bnn;


// ============================================================================
// Test configuration
// ============================================================================

static constexpr std::uint64_t RANDOM_SEED =
    0xBADC0FFEEULL;


// ============================================================================
// Benchmark configuration
// ============================================================================
//
// Increase these if you want more stable measurements.
//
// The benchmark uses several matrix sizes because binary kernels can behave
// differently depending on whether K/N are multiples of 64.
//
// ============================================================================

static constexpr std::size_t BENCHMARK_WARMUP_RUNS = 3;
static constexpr std::size_t BENCHMARK_RUNS = 10;


// Prevent the compiler from optimizing benchmark results away.
#if defined(_MSC_VER)

static volatile std::int64_t g_benchmark_sink = 0;

#else

static volatile std::int64_t g_benchmark_sink = 0;

#endif


// ============================================================================
// Test helpers
// ============================================================================

struct TestStats
{
    std::size_t passed = 0;
    std::size_t failed = 0;
};


static TestStats g_stats;


// ----------------------------------------------------------------------------
// Failure helper
// ----------------------------------------------------------------------------

[[noreturn]]
static void test_failure(
    const std::string& message)
{
    ++g_stats.failed;

    std::cerr
        << "\nTEST FAILED:\n"
        << message
        << "\n";

    std::exit(EXIT_FAILURE);
}


// ----------------------------------------------------------------------------
// Expect condition
// ----------------------------------------------------------------------------

static void expect(
    bool condition,
    const std::string& message)
{
    if (!condition)
        test_failure(message);

    ++g_stats.passed;
}


// ----------------------------------------------------------------------------
// Convert packed matrix to ordinary signed representation.
// ----------------------------------------------------------------------------

static std::vector<int> signed_values(
    const BinaryMatrix& matrix)
{
    std::vector<int> result(
        matrix.rows() * matrix.cols());

    for (std::size_t r = 0;
         r < matrix.rows();
         ++r)
    {
        for (std::size_t c = 0;
             c < matrix.cols();
             ++c)
        {
            result[
                r * matrix.cols() + c] =
                matrix.get_bit(r, c)
                    ? -1
                    : +1;
        }
    }

    return result;
}


// ----------------------------------------------------------------------------
// Print matrix.
// ----------------------------------------------------------------------------

static void print_matrix(
    const std::vector<int>& matrix,
    std::size_t rows,
    std::size_t cols)
{
    for (std::size_t r = 0;
         r < rows;
         ++r)
    {
        std::cout << "[ ";

        for (std::size_t c = 0;
             c < cols;
             ++c)
        {
            std::cout
                << std::setw(3)
                << matrix[r * cols + c]
                << ' ';
        }

        std::cout << "]\n";
    }
}


// ============================================================================
// Reference implementation
//
// This is deliberately straightforward.
//
// C[i][j] = sign(
//                sum_k A[i][k] * B[k][j]
//            )
//
// +1 remains +1
// -1 remains -1
// zero becomes +1
//
// ============================================================================

static std::vector<int> reference_multiply(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N)
{
    std::vector<int> C(M * N);

    for (std::size_t i = 0;
         i < M;
         ++i)
    {
        for (std::size_t j = 0;
             j < N;
             ++j)
        {
            int accumulator = 0;

            for (std::size_t k = 0;
                 k < K;
                 ++k)
            {
                accumulator +=
                    A[i * K + k] *
                    B[k * N + j];
            }

            C[i * N + j] =
                accumulator < 0
                    ? -1
                    : +1;
        }
    }

    return C;
}


// ============================================================================
// Reference implementation WITHOUT clipping.
//
// Useful for checking multiply_accumulate().
// ============================================================================

static std::vector<int> reference_accumulate(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N)
{
    std::vector<int> C(M * N, 0);

    for (std::size_t i = 0;
         i < M;
         ++i)
    {
        for (std::size_t j = 0;
             j < N;
             ++j)
        {
            int accumulator = 0;

            for (std::size_t k = 0;
                 k < K;
                 ++k)
            {
                accumulator +=
                    A[i * K + k] *
                    B[k * N + j];
            }

            C[i * N + j] = accumulator;
        }
    }

    return C;
}


// ============================================================================
// Compare matrices.
// ============================================================================

static void compare_vectors(
    const std::vector<int>& expected,
    const std::vector<int>& actual,
    std::size_t rows,
    std::size_t cols,
    const std::string& test_name)
{
    if (expected.size() != actual.size())
    {
        test_failure(
            test_name +
            ": vector sizes differ");
    }

    for (std::size_t i = 0;
         i < expected.size();
         ++i)
    {
        if (expected[i] != actual[i])
        {
            const std::size_t row =
                i / cols;

            const std::size_t col =
                i % cols;

            std::cerr
                << "\nMismatch in "
                << test_name
                << "\n"
                << "  position: ("
                << row
                << ", "
                << col
                << ")\n"
                << "  expected: "
                << expected[i]
                << "\n"
                << "  actual:   "
                << actual[i]
                << "\n";

            std::cerr
                << "\nExpected:\n";

            print_matrix(
                expected,
                rows,
                cols);

            std::cerr
                << "\nActual:\n";

            print_matrix(
                actual,
                rows,
                cols);

            test_failure(
                test_name +
                ": result mismatch");
        }
    }

    ++g_stats.passed;
}


// ============================================================================
// Generate random +/-1 matrix.
// ============================================================================

static std::vector<int> random_signed_matrix(
    std::size_t rows,
    std::size_t cols,
    std::mt19937_64& rng)
{
    std::vector<int> result(
        rows * cols);

    std::uniform_int_distribution<int> distribution(
        0,
        1);

    for (int& value : result)
    {
        value =
            distribution(rng)
                ? -1
                : +1;
    }

    return result;
}


// ============================================================================
// Convert signed matrix to BinaryMatrix.
// ============================================================================

static BinaryMatrix make_binary(
    const std::vector<int>& data,
    std::size_t rows,
    std::size_t cols)
{
    return from_signed(
        data.data(),
        rows,
        cols);
}


// ============================================================================
// Basic sanity test.
// ============================================================================

static void test_basic()
{
    std::cout
        << "Running basic test...\n";

    constexpr std::size_t M = 2;
    constexpr std::size_t K = 4;
    constexpr std::size_t N = 3;

    const std::vector<int> A = {
        +1, +1, +1, +1,
        +1, -1, +1, -1
    };

    const std::vector<int> B = {
        +1, +1, -1,
        +1, -1, +1,
        +1, +1, +1,
        -1, +1, -1
    };

    const auto expected =
        reference_multiply(
            A, B, M, K, N);

    const BinaryMatrix binary_A =
        make_binary(A, M, K);

    const BinaryMatrix binary_B =
        make_binary(B, K, N);

    const BinaryMatrix binary_C =
        multiply(binary_A, binary_B);

    const auto actual =
        signed_values(binary_C);

    compare_vectors(
        expected,
        actual,
        M,
        N,
        "basic multiplication");
}


// ============================================================================
// Zero-dot-product test.
// ============================================================================

static void test_zero_dot()
{
    std::cout
        << "Running zero-dot test...\n";

    constexpr std::size_t M = 1;
    constexpr std::size_t K = 2;
    constexpr std::size_t N = 1;

    const std::vector<int> A = {
        +1, +1
    };

    const std::vector<int> B = {
        +1,
        -1
    };

    const auto expected =
        reference_multiply(
            A, B, M, K, N);

    expect(
        expected[0] == +1,
        "Zero dot product must map to +1");

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    const auto result =
        multiply(binary_A, binary_B);

    const auto actual =
        signed_values(result);

    compare_vectors(
        expected,
        actual,
        M,
        N,
        "zero dot product");
}


// ============================================================================
// Accumulator test.
// ============================================================================

static void test_accumulation()
{
    std::cout
        << "Running accumulation test...\n";

    constexpr std::size_t M = 3;
    constexpr std::size_t K = 7;
    constexpr std::size_t N = 5;

    std::mt19937_64 rng(
        RANDOM_SEED);

    const auto A =
        random_signed_matrix(
            M, K, rng);

    const auto B =
        random_signed_matrix(
            K, N, rng);

    const auto expected =
        reference_accumulate(
            A, B, M, K, N);

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    BinaryWeightMatrix weights(
        binary_B);

    std::vector<std::int32_t> actual;

    multiply_accumulate(
        binary_A,
        weights,
        actual);

    std::vector<int> actual_int(
        actual.begin(),
        actual.end());

    compare_vectors(
        expected,
        actual_int,
        M,
        N,
        "integer accumulation");
}


// ============================================================================
// Pre-packed weights vs normal matrix multiplication.
// ============================================================================

static void test_prepacked_weights()
{
    std::cout
        << "Running pre-packed weight test...\n";

    constexpr std::size_t M = 9;
    constexpr std::size_t K = 37;
    constexpr std::size_t N = 71;

    std::mt19937_64 rng(
        RANDOM_SEED + 1);

    const auto A =
        random_signed_matrix(
            M, K, rng);

    const auto B =
        random_signed_matrix(
            K, N, rng);

    const auto expected =
        reference_multiply(
            A, B, M, K, N);

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    const auto result_normal =
        multiply(
            binary_A,
            binary_B);

    const auto actual_normal =
        signed_values(result_normal);

    compare_vectors(
        expected,
        actual_normal,
        M,
        N,
        "normal multiplication");

    BinaryWeightMatrix weights(
        binary_B);

    const auto result_packed =
        multiply(
            binary_A,
            weights);

    const auto actual_packed =
        signed_values(result_packed);

    compare_vectors(
        expected,
        actual_packed,
        M,
        N,
        "pre-packed multiplication");
}


// ============================================================================
// Padding tests.
// ============================================================================

static void test_padding_dimensions()
{
    std::cout
        << "Running padding/dimension tests...\n";

    const std::vector<std::size_t> dimensions = {
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        65,
        66,
        95,
        127,
        128,
        129,
        191,
        192,
        193
    };

    std::mt19937_64 rng(
        RANDOM_SEED + 2);

    for (const std::size_t K : dimensions)
    {
        constexpr std::size_t M = 5;
        constexpr std::size_t N = 7;

        const auto A =
            random_signed_matrix(
                M, K, rng);

        const auto B =
            random_signed_matrix(
                K, N, rng);

        const auto expected =
            reference_multiply(
                A, B, M, K, N);

        const auto binary_A =
            make_binary(A, M, K);

        const auto binary_B =
            make_binary(B, K, N);

        BinaryWeightMatrix weights(
            binary_B);

        const auto result =
            multiply(
                binary_A,
                weights);

        const auto actual =
            signed_values(result);

        compare_vectors(
            expected,
            actual,
            M,
            N,
            "padding K=" +
                std::to_string(K));
    }
}


// ============================================================================
// Exhaustive test.
// ============================================================================

static void test_exhaustive_small()
{
    std::cout
        << "Running exhaustive small-matrix tests...\n";

    constexpr std::size_t M = 2;
    constexpr std::size_t K = 3;
    constexpr std::size_t N = 2;

    constexpr std::size_t A_BITS = M * K;
    constexpr std::size_t B_BITS = K * N;

    constexpr std::uint64_t A_COUNT =
        1ULL << A_BITS;

    constexpr std::uint64_t B_COUNT =
        1ULL << B_BITS;

    std::vector<int> A(
        A_BITS);

    std::vector<int> B(
        B_BITS);

    for (std::uint64_t a_bits = 0;
         a_bits < A_COUNT;
         ++a_bits)
    {
        for (std::size_t i = 0;
             i < A_BITS;
             ++i)
        {
            A[i] =
                (a_bits & (1ULL << i))
                    ? -1
                    : +1;
        }

        const auto binary_A =
            make_binary(A, M, K);

        for (std::uint64_t b_bits = 0;
             b_bits < B_COUNT;
             ++b_bits)
        {
            for (std::size_t i = 0;
                 i < B_BITS;
                 ++i)
            {
                B[i] =
                    (b_bits & (1ULL << i))
                        ? -1
                        : +1;
            }

            const auto expected =
                reference_multiply(
                    A, B, M, K, N);

            const auto binary_B =
                make_binary(B, K, N);

            const auto actual_matrix =
                multiply(
                    binary_A,
                    binary_B);

            const auto actual =
                signed_values(actual_matrix);

            if (expected != actual)
            {
                std::cerr
                    << "\nExhaustive test failed!\n"
                    << "A bits = 0x"
                    << std::hex
                    << a_bits
                    << "\nB bits = 0x"
                    << b_bits
                    << std::dec
                    << "\n";

                test_failure(
                    "exhaustive multiplication");
            }
        }
    }

    ++g_stats.passed;

    std::cout
        << "  Exhaustive combinations checked: "
        << (A_COUNT * B_COUNT)
        << "\n";
}


// ============================================================================
// Randomized test.
// ============================================================================

static void test_randomized()
{
    std::cout
        << "Running randomized tests...\n";

    std::mt19937_64 rng(
        RANDOM_SEED + 3);

    std::uniform_int_distribution<std::size_t>
        dimension_distribution(1, 160);

    constexpr std::size_t ITERATIONS = 500;

    for (std::size_t iteration = 0;
         iteration < ITERATIONS;
         ++iteration)
    {
        const std::size_t M =
            dimension_distribution(rng);

        const std::size_t K =
            dimension_distribution(rng);

        const std::size_t N =
            dimension_distribution(rng);

        const auto A =
            random_signed_matrix(
                M, K, rng);

        const auto B =
            random_signed_matrix(
                K, N, rng);

        const auto expected =
            reference_multiply(
                A, B, M, K, N);

        const auto binary_A =
            make_binary(A, M, K);

        const auto binary_B =
            make_binary(B, K, N);

        const auto result =
            multiply(
                binary_A,
                binary_B);

        const auto actual =
            signed_values(result);

        if (expected != actual)
        {
            std::cerr
                << "\nRandom test failed.\n"
                << "Iteration: "
                << iteration
                << "\nDimensions: "
                << M
                << " x "
                << K
                << " * "
                << K
                << " x "
                << N
                << "\n";

            compare_vectors(
                expected,
                actual,
                M,
                N,
                "randomized multiplication");
        }
    }

    ++g_stats.passed;

    std::cout
        << "  Random cases checked: "
        << ITERATIONS
        << "\n";
}


// ============================================================================
// All-positive test.
// ============================================================================

static void test_all_positive()
{
    std::cout
        << "Running all-positive test...\n";

    constexpr std::size_t M = 4;
    constexpr std::size_t K = 129;
    constexpr std::size_t N = 9;

    std::vector<int> A(
        M * K,
        +1);

    std::vector<int> B(
        K * N,
        +1);

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    const auto result =
        multiply(
            binary_A,
            binary_B);

    const auto actual =
        signed_values(result);

    const std::vector<int> expected(
        M * N,
        +1);

    compare_vectors(
        expected,
        actual,
        M,
        N,
        "all-positive");
}


// ============================================================================
// All-negative test.
// ============================================================================

static void test_all_negative()
{
    std::cout
        << "Running all-negative test...\n";

    constexpr std::size_t M = 3;
    constexpr std::size_t K = 65;
    constexpr std::size_t N = 11;

    std::vector<int> A(
        M * K,
        -1);

    std::vector<int> B(
        K * N,
        -1);

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    const auto result =
        multiply(
            binary_A,
            binary_B);

    const auto actual =
        signed_values(result);

    const std::vector<int> expected(
        M * N,
        +1);

    compare_vectors(
        expected,
        actual,
        M,
        N,
        "all-negative");
}


// ============================================================================
// Alternating values.
// ============================================================================

static void test_alternating()
{
    std::cout
        << "Running alternating-pattern test...\n";

    constexpr std::size_t M = 8;
    constexpr std::size_t K = 127;
    constexpr std::size_t N = 13;

    std::vector<int> A(
        M * K);

    std::vector<int> B(
        K * N);

    for (std::size_t i = 0;
         i < A.size();
         ++i)
    {
        A[i] =
            (i & 1)
                ? -1
                : +1;
    }

    for (std::size_t i = 0;
         i < B.size();
         ++i)
    {
        B[i] =
            (i & 1)
                ? +1
                : -1;
    }

    const auto expected =
        reference_multiply(
            A, B, M, K, N);

    const auto binary_A =
        make_binary(A, M, K);

    const auto binary_B =
        make_binary(B, K, N);

    const auto result =
        multiply(
            binary_A,
            binary_B);

    const auto actual =
        signed_values(result);

    compare_vectors(
        expected,
        actual,
        M,
        N,
        "alternating pattern");
}


// ============================================================================
// Bit conversion test.
// ============================================================================

static void test_bit_conversion()
{
    std::cout
        << "Running bit conversion test...\n";

    constexpr std::size_t ROWS = 3;
    constexpr std::size_t COLS = 131;

    std::mt19937_64 rng(
        RANDOM_SEED + 4);

    const auto input =
        random_signed_matrix(
            ROWS,
            COLS,
            rng);

    const auto matrix =
        make_binary(
            input,
            ROWS,
            COLS);

    for (std::size_t r = 0;
         r < ROWS;
         ++r)
    {
        for (std::size_t c = 0;
             c < COLS;
             ++c)
        {
            const int actual =
                matrix.get_value(r, c);

            const int expected =
                input[r * COLS + c];

            if (actual != expected)
            {
                test_failure(
                    "individual bit conversion failed");
            }
        }
    }

    ++g_stats.passed;
}


// ============================================================================
// Transpose test.
// ============================================================================

static void test_transpose()
{
    std::cout
        << "Running transpose test...\n";

    constexpr std::size_t ROWS = 73;
    constexpr std::size_t COLS = 131;

    std::mt19937_64 rng(
        RANDOM_SEED + 5);

    const auto input =
        random_signed_matrix(
            ROWS,
            COLS,
            rng);

    const auto matrix =
        make_binary(
            input,
            ROWS,
            COLS);

    const auto transposed =
        matrix.transpose();

    expect(
        transposed.rows() == COLS,
        "transpose row count incorrect");

    expect(
        transposed.cols() == ROWS,
        "transpose column count incorrect");

    for (std::size_t r = 0;
         r < ROWS;
         ++r)
    {
        for (std::size_t c = 0;
             c < COLS;
             ++c)
        {
            const int original =
                matrix.get_value(r, c);

            const int transposed_value =
                transposed.get_value(c, r);

            if (original != transposed_value)
            {
                test_failure(
                    "transpose content mismatch");
            }
        }
    }

    ++g_stats.passed;
}


// ============================================================================
// BENCHMARK HELPERS
// ============================================================================

struct BenchmarkResult
{
    double milliseconds = 0.0;
    std::int64_t checksum = 0;
};


// ----------------------------------------------------------------------------
// Standard clipped matmul benchmark.
//
// This represents the conventional implementation:
//
//     accumulator += A * B
//
// followed by:
//
//     accumulator < 0 ? -1 : +1
//
// Data remains ordinary int values.
// ----------------------------------------------------------------------------

static BenchmarkResult benchmark_standard_clipped(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N,
    std::size_t runs)
{
    BenchmarkResult result;

    std::int64_t checksum = 0;

    // Warmup.
    for (std::size_t run = 0;
         run < BENCHMARK_WARMUP_RUNS;
         ++run)
    {
        for (std::size_t i = 0;
             i < M;
             ++i)
        {
            for (std::size_t j = 0;
                 j < N;
                 ++j)
            {
                int accumulator = 0;

                for (std::size_t k = 0;
                     k < K;
                     ++k)
                {
                    accumulator +=
                        A[i * K + k] *
                        B[k * N + j];
                }

                checksum +=
                    accumulator < 0
                        ? -1
                        : +1;
            }
        }
    }

    const auto start =
        std::chrono::steady_clock::now();

    for (std::size_t run = 0;
         run < runs;
         ++run)
    {
        for (std::size_t i = 0;
             i < M;
             ++i)
        {
            for (std::size_t j = 0;
                 j < N;
                 ++j)
            {
                int accumulator = 0;

                for (std::size_t k = 0;
                     k < K;
                     ++k)
                {
                    accumulator +=
                        A[i * K + k] *
                        B[k * N + j];
                }

                checksum +=
                    accumulator < 0
                        ? -1
                        : +1;
            }
        }
    }

    const auto end =
        std::chrono::steady_clock::now();

    result.milliseconds =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    result.checksum =
        checksum;

    return result;
}


// ----------------------------------------------------------------------------
// Binary packed matmul benchmark.
//
// This includes the BinaryMatrix -> BinaryMatrix multiplication itself.
//
// The matrices are converted before timing.
// ----------------------------------------------------------------------------

static BenchmarkResult benchmark_binary(
    const BinaryMatrix& A,
    const BinaryMatrix& B,
    std::size_t runs)
{
    BenchmarkResult result;

    std::int64_t checksum = 0;

    // Warmup.
    for (std::size_t run = 0;
         run < BENCHMARK_WARMUP_RUNS;
         ++run)
    {
        const BinaryMatrix C =
            multiply(A, B);

        for (std::size_t r = 0;
             r < C.rows();
             ++r)
        {
            const word_t* row =
                C.row_data(r);

            for (std::size_t w = 0;
                 w < C.words_per_row();
                 ++w)
            {
                checksum +=
                    static_cast<std::int64_t>(
                        row[w]);
            }
        }
    }

    const auto start =
        std::chrono::steady_clock::now();

    for (std::size_t run = 0;
         run < runs;
         ++run)
    {
        const BinaryMatrix C =
            multiply(A, B);

        for (std::size_t r = 0;
             r < C.rows();
             ++r)
        {
            const word_t* row =
                C.row_data(r);

            for (std::size_t w = 0;
                 w < C.words_per_row();
                 ++w)
            {
                checksum +=
                    static_cast<std::int64_t>(
                        row[w]);
            }
        }
    }

    const auto end =
        std::chrono::steady_clock::now();

    result.milliseconds =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    result.checksum =
        checksum;

    return result;
}


// ----------------------------------------------------------------------------
// Pre-packed BinaryWeightMatrix benchmark.
//
// This is the important deployment case:
//
//     weights are packed once
//
// and then reused for many inference calls.
//
// ----------------------------------------------------------------------------

static BenchmarkResult benchmark_prepacked(
    const BinaryMatrix& A,
    const BinaryWeightMatrix& weights,
    std::size_t runs)
{
    BenchmarkResult result;

    std::int64_t checksum = 0;

    // Warmup.
    for (std::size_t run = 0;
         run < BENCHMARK_WARMUP_RUNS;
         ++run)
    {
        const BinaryMatrix C =
            multiply(A, weights);

        for (std::size_t r = 0;
             r < C.rows();
             ++r)
        {
            const word_t* row =
                C.row_data(r);

            for (std::size_t w = 0;
                 w < C.words_per_row();
                 ++w)
            {
                checksum +=
                    static_cast<std::int64_t>(
                        row[w]);
            }
        }
    }

    const auto start =
        std::chrono::steady_clock::now();

    for (std::size_t run = 0;
         run < runs;
         ++run)
    {
        const BinaryMatrix C =
            multiply(A, weights);

        for (std::size_t r = 0;
             r < C.rows();
             ++r)
        {
            const word_t* row =
                C.row_data(r);

            for (std::size_t w = 0;
                 w < C.words_per_row();
                 ++w)
            {
                checksum +=
                    static_cast<std::int64_t>(
                        row[w]);
            }
        }
    }

    const auto end =
        std::chrono::steady_clock::now();

    result.milliseconds =
        std::chrono::duration<double, std::milli>(
            end - start).count();

    result.checksum =
        checksum;

    return result;
}


// ============================================================================
// Verify benchmark implementations before timing.
//
// This ensures that the benchmark is comparing equivalent computations.
// ============================================================================

static void verify_benchmark_case(
    const std::vector<int>& A,
    const std::vector<int>& B,
    std::size_t M,
    std::size_t K,
    std::size_t N)
{
    const auto expected =
        reference_multiply(
            A,
            B,
            M,
            K,
            N);

    const BinaryMatrix binary_A =
        make_binary(
            A,
            M,
            K);

    const BinaryMatrix binary_B =
        make_binary(
            B,
            K,
            N);

    const BinaryMatrix binary_result =
        multiply(
            binary_A,
            binary_B);

    const auto actual_binary =
        signed_values(
            binary_result);

    if (expected != actual_binary)
    {
        test_failure(
            "benchmark binary result does not "
            "match standard matmul");
    }

    const BinaryWeightMatrix weights(
        binary_B);

    const BinaryMatrix packed_result =
        multiply(
            binary_A,
            weights);

    const auto actual_packed =
        signed_values(
            packed_result);

    if (expected != actual_packed)
    {
        test_failure(
            "benchmark pre-packed result does not "
            "match standard matmul");
    }
}


// ============================================================================
// Print benchmark row.
// ============================================================================

static void print_benchmark_row(
    const std::string& name,
    double milliseconds,
    double baseline_milliseconds)
{
    const double speedup =
        baseline_milliseconds /
        milliseconds;

    std::cout
        << std::left
        << std::setw(28)
        << name
        << std::right
        << std::setw(14)
        << std::fixed
        << std::setprecision(3)
        << milliseconds
        << " ms"
        << std::setw(14)
        << std::setprecision(2)
        << speedup
        << "x\n";
}


// ============================================================================
// Run one benchmark case.
// ============================================================================

static void run_benchmark_case(
    std::size_t M,
    std::size_t K,
    std::size_t N,
    std::mt19937_64& rng)
{
    const auto A =
        random_signed_matrix(
            M,
            K,
            rng);

    const auto B =
        random_signed_matrix(
            K,
            N,
            rng);

    // Verify correctness before measuring.
    verify_benchmark_case(
        A,
        B,
        M,
        K,
        N);

    const BinaryMatrix binary_A =
        make_binary(
            A,
            M,
            K);

    const BinaryMatrix binary_B =
        make_binary(
            B,
            K,
            N);

    const BinaryWeightMatrix weights(
        binary_B);


    // ------------------------------------------------------------------------
    // Benchmark standard implementation.
    // ------------------------------------------------------------------------

    const BenchmarkResult standard =
        benchmark_standard_clipped(
            A,
            B,
            M,
            K,
            N,
            BENCHMARK_RUNS);


    // ------------------------------------------------------------------------
    // Benchmark ordinary BinaryMatrix path.
    // ------------------------------------------------------------------------

    const BenchmarkResult binary =
        benchmark_binary(
            binary_A,
            binary_B,
            BENCHMARK_RUNS);


    // ------------------------------------------------------------------------
    // Benchmark permanently packed weight path.
    // ------------------------------------------------------------------------

    const BenchmarkResult prepacked =
        benchmark_prepacked(
            binary_A,
            weights,
            BENCHMARK_RUNS);


    // Keep checksums alive.
    g_benchmark_sink +=
        standard.checksum;

    g_benchmark_sink +=
        binary.checksum;

    g_benchmark_sink +=
        prepacked.checksum;


    std::cout
        << "\n"
        << "Matrix size: "
        << M
        << " x "
        << K
        << " * "
        << K
        << " x "
        << N
        << "\n";

    std::cout
        << "Runs: "
        << BENCHMARK_RUNS
        << "\n";

    std::cout
        << "------------------------------------------------------------\n";

    std::cout
        << std::left
        << std::setw(28)
        << "Implementation"
        << std::right
        << std::setw(14)
        << "Time"
        << std::setw(14)
        << "Speedup"
        << "\n";

    std::cout
        << "------------------------------------------------------------\n";

    print_benchmark_row(
        "Standard clipped",
        standard.milliseconds,
        standard.milliseconds);

    print_benchmark_row(
        "Packed BinaryMatrix",
        binary.milliseconds,
        standard.milliseconds);

    print_benchmark_row(
        "Pre-packed weights",
        prepacked.milliseconds,
        standard.milliseconds);

    std::cout
        << "------------------------------------------------------------\n";

    std::cout
        << "Checksum: "
        << g_benchmark_sink
        << "\n";
}


// ============================================================================
// Performance benchmarks.
//
// These deliberately include both dimensions that are multiples of 64 and
// dimensions that are not.
//
// The standard implementation operates on ordinary signed integers.
//
// The binary implementations operate on packed bits.
//
// ============================================================================

static void run_benchmarks()
{
    std::cout
        << "\n\n"
        << "========================================\n"
        << " Performance Benchmarks\n"
        << "========================================\n";

    std::cout
        << "\nThese timings measure matrix multiplication only.\n"
        << "Matrix conversion/weight packing is performed before timing.\n"
        << "Standard clipped = ordinary int matmul + sign/clipping.\n"
        << "Packed BinaryMatrix = packed binary multiplication.\n"
        << "Pre-packed weights = BinaryWeightMatrix inference path.\n";

    std::mt19937_64 rng(
        RANDOM_SEED + 100);

    // ------------------------------------------------------------------------
    // Small neural-network-style layer.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        32,
        128,
        64,
        rng);

    // ------------------------------------------------------------------------
    // Larger layer.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        64,
        256,
        128,
        rng);

    // ------------------------------------------------------------------------
    // Larger still.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        128,
        512,
        256,
        rng);

    // ------------------------------------------------------------------------
    // Non-64-aligned K.
    //
    // This exercises the partial-word path.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        64,
        257,
        128,
        rng);

    // ------------------------------------------------------------------------
    // Non-64-aligned N.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        64,
        256,
        131,
        rng);

    // ------------------------------------------------------------------------
    // Both K and N non-aligned.
    // ------------------------------------------------------------------------

    run_benchmark_case(
        37,
        193,
        71,
        rng);

    std::cout
        << "\nBenchmark complete.\n";
}


// ============================================================================
// Main
// ============================================================================

int main()
{
    std::cout
        << "========================================\n"
        << " Binary Matrix BNN Test Suite\n"
        << "========================================\n\n";

    std::cout
        << "SIMD backend: "
        << bnn::simd_backend()
        << "\n";

    std::cout
        << "AVX2: "
        << (bnn::has_avx2()
            ? "yes"
            : "no")
        << "\n";

    std::cout
        << "AVX-512 VPOPCNTDQ: "
        << (bnn::has_avx512_vpopcntdq()
            ? "yes"
            : "no")
        << "\n\n";


    // ========================================================================
    // Correctness tests
    // ========================================================================

    test_basic();

    test_zero_dot();

    test_accumulation();

    test_prepacked_weights();

    test_padding_dimensions();

    test_exhaustive_small();

    test_randomized();

    test_all_positive();

    test_all_negative();

    test_alternating();

    test_bit_conversion();

    test_transpose();


    // ========================================================================
    // Benchmark section
    // ========================================================================

    run_benchmarks();


    // ========================================================================
    // Final test summary
    // ========================================================================

    std::cout
        << "\n========================================\n"
        << " Tests complete\n"
        << "========================================\n";

    std::cout
        << "Assertions passed: "
        << g_stats.passed
        << "\n";

    std::cout
        << "Assertions failed: "
        << g_stats.failed
        << "\n";

    if (g_stats.failed != 0)
    {
        return EXIT_FAILURE;
    }

    std::cout
        << "\nALL TESTS PASSED\n";

    return EXIT_SUCCESS;
}
