
import sys
import time
import statistics
import traceback

import numpy as np

from binary_matrices import (
    BinaryMatrix,
    BinaryWeightMatrix,
    multiply,
    multiply_accumulate,
    dot,
    simd_backend,
    has_avx2,
    has_avx512_vpopcntdq,
)


# ============================================================
# Configuration
# ============================================================

SEED = 12345
RANDOM_CASES = 100

BENCHMARK_SIZES = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
]

BENCHMARK_WARMUP = 3
BENCHMARK_REPEATS = 9

rng = np.random.default_rng(SEED)

tests_passed = 0
tests_failed = 0


# ============================================================
# Test helpers
# ============================================================

def check(condition, message):
    global tests_passed, tests_failed

    if condition:
        tests_passed += 1
        print(f"[PASS] {message}")
    else:
        tests_failed += 1
        print(f"[FAIL] {message}")


def check_equal(actual, expected, message):
    try:
        np.testing.assert_array_equal(actual, expected)
        check(True, message)
    except AssertionError:
        check(False, message)
        print("  Expected:")
        print(expected)
        print("  Actual:")
        print(actual)


def expect_exception(function, exception_type, message):
    try:
        function()
    except exception_type:
        check(True, message)
    except Exception as exc:
        check(
            False,
            f"{message} (got {type(exc).__name__}: {exc})",
        )
    else:
        check(False, message)


def random_binary(rows, cols, generator=None):
    if generator is None:
        generator = rng

    return (
        generator.integers(
            0, 2, size=(rows, cols), dtype=np.int8
        ) * 2 - 1
    ).astype(np.int8)


# ============================================================
# NumPy reference implementations
# ============================================================

def reference_dot_product(a, b):
    """Calculate signed integer dot products without int8 overflow."""
    return a.astype(np.int64) @ b.astype(np.int64)


def reference_binary_multiply(a, b):
    """Threshold signed dot products at zero."""
    scores = reference_dot_product(a, b)
    return np.where(scores >= 0, 1, -1).astype(np.int8)


# ============================================================
# Correctness tests
# ============================================================

def test_import_and_backend():
    print("\n=== Import and SIMD backend ===")

    check(callable(simd_backend), "SIMD backend function exists")
    check(callable(has_avx2), "AVX2 query exists")
    check(
        callable(has_avx512_vpopcntdq),
        "AVX-512 query exists",
    )

    try:
        backend = simd_backend()
        print("SIMD backend:", backend)
        print("AVX2 supported:", has_avx2())
        print("AVX-512 VPOPCNTDQ supported:", has_avx512_vpopcntdq())

        check(
            isinstance(backend, str),
            "SIMD backend returns a string",
        )
    except Exception as exc:
        check(False, f"SIMD backend query failed: {exc}")


def test_construction_and_conversion():
    print("\n=== Construction and NumPy conversion ===")

    original = np.array(
        [[1, -1, 1], [-1, -1, 1]],
        dtype=np.int8,
    )

    matrix = BinaryMatrix.from_numpy(original)

    check_equal(matrix.to_numpy(), original, "NumPy round trip")
    check_equal(matrix.shape, original.shape, "Matrix shape")
    check(matrix.nbytes >= 0, "Packed storage size is non-negative")

    empty = BinaryMatrix(0, 0)
    check_equal(empty.shape, (0, 0), "Empty matrix construction")

    expect_exception(
        lambda: BinaryMatrix.from_numpy(np.array([1, -1])),
        ValueError,
        "Reject one-dimensional input",
    )

    expect_exception(
        lambda: BinaryMatrix.from_numpy(np.array([[0, 1]])),
        ValueError,
        "Reject zero as a matrix value",
    )

    expect_exception(
        lambda: BinaryMatrix.from_numpy(np.array([[1, 2]])),
        ValueError,
        "Reject values other than -1 and +1",
    )

    expect_exception(
        lambda: BinaryMatrix(-1, 2),
        ValueError,
        "Reject negative dimensions",
    )


def test_get_set_fill():
    print("\n=== Element access and filling ===")

    matrix = BinaryMatrix.from_numpy(
        np.array([[1, -1], [-1, 1]], dtype=np.int8)
    )

    check(matrix.get(0, 0) == 1, "Read positive element")
    check(matrix.get(0, 1) == -1, "Read negative element")

    matrix.set(0, 1, 1)

    expected = np.array([[1, 1], [-1, 1]], dtype=np.int8)

    check_equal(matrix.to_numpy(), expected, "Set an element")

    matrix.fill(-1)

    check_equal(
        matrix.to_numpy(),
        np.full((2, 2), -1, dtype=np.int8),
        "Fill with -1",
    )

    matrix.fill(1)

    check_equal(
        matrix.to_numpy(),
        np.ones((2, 2), dtype=np.int8),
        "Fill with +1",
    )

    expect_exception(
        lambda: matrix.set(0, 0, 0),
        ValueError,
        "Reject invalid element value",
    )

    expect_exception(
        lambda: matrix.get(10, 0),
        IndexError,
        "Reject invalid row index",
    )

    expect_exception(
        lambda: matrix.get(0, 10),
        IndexError,
        "Reject invalid column index",
    )


def test_transpose():
    print("\n=== Transposition ===")

    original = random_binary(3, 5)
    matrix = BinaryMatrix.from_numpy(original)
    transposed = matrix.transpose()

    check_equal(
        transposed.to_numpy(),
        original.T,
        "Transpose matches NumPy",
    )

    check_equal(
        transposed.shape,
        (5, 3),
        "Transpose shape",
    )

    check_equal(
        transposed.transpose().to_numpy(),
        original,
        "Double transpose restores original",
    )


def test_multiplication():
    print("\n=== Binary matrix multiplication ===")

    a = np.array(
        [[1, -1, 1], [-1, -1, 1]],
        dtype=np.int8,
    )

    b = np.array(
        [[1, -1], [-1, 1], [1, 1]],
        dtype=np.int8,
    )

    actual = multiply(
        BinaryMatrix.from_numpy(a),
        BinaryMatrix.from_numpy(b),
    ).to_numpy()

    expected = reference_binary_multiply(a, b)

    check_equal(
        actual,
        expected,
        "Binary multiplication matches NumPy",
    )

    check_equal(
        actual,
        np.array([[1, -1], [1, 1]], dtype=np.int8),
        "Known multiplication example",
    )

    expect_exception(
        lambda: multiply(
            BinaryMatrix.from_numpy(random_binary(2, 3)),
            BinaryMatrix.from_numpy(random_binary(4, 2)),
        ),
        ValueError,
        "Reject incompatible dimensions",
    )


def test_weights_and_accumulation():
    print("\n=== Packed weights and integer accumulation ===")

    a = random_binary(4, 7)
    b = random_binary(7, 5)

    activation = BinaryMatrix.from_numpy(a)
    weight_matrix = BinaryMatrix.from_numpy(b)
    weights = BinaryWeightMatrix(weight_matrix)

    check_equal(weights.shape, b.shape, "Packed weight dimensions")

    expected_scores = reference_dot_product(a, b)

    actual_scores = multiply_accumulate(activation, weights)

    check_equal(
        actual_scores,
        expected_scores,
        "Integer accumulation matches NumPy",
    )

    actual_binary = multiply(activation, weights).to_numpy()

    check_equal(
        actual_binary,
        reference_binary_multiply(a, b),
        "Multiplication with packed weights",
    )

    for row in range(a.shape[0]):
        for output in range(b.shape[1]):
            actual_dot = dot(activation, row, weights, output)
            expected_dot = int(expected_scores[row, output])

            check(
                actual_dot == expected_dot,
                f"Dot product row={row}, output={output}",
            )

    expect_exception(
        lambda: dot(activation, -1, weights, 0),
        IndexError,
        "Reject negative activation row",
    )

    expect_exception(
        lambda: dot(activation, 0, weights, b.shape[1]),
        IndexError,
        "Reject invalid output index",
    )


def test_tie_threshold():
    print("\n=== Zero-score threshold ===")

    a = np.array([[1, -1]], dtype=np.int8)
    b = np.array([[1], [1]], dtype=np.int8)

    activation = BinaryMatrix.from_numpy(a)
    weights = BinaryWeightMatrix(BinaryMatrix.from_numpy(b))

    score = dot(activation, 0, weights, 0)
    result = multiply(activation, weights).to_numpy()

    check(score == 0, "Zero-score dot product")

    check_equal(
        result,
        np.array([[1]], dtype=np.int8),
        "Zero score is classified as +1",
    )


def test_random_matrices():
    print("\n=== Randomized correctness tests ===")

    for case in range(RANDOM_CASES):
        m = int(rng.integers(1, 9))
        k = int(rng.integers(1, 80))
        n = int(rng.integers(1, 9))

        a = random_binary(m, k)
        b = random_binary(k, n)

        cpp_a = BinaryMatrix.from_numpy(a)
        cpp_b = BinaryMatrix.from_numpy(b)

        expected_binary = reference_binary_multiply(a, b)
        expected_scores = reference_dot_product(a, b)

        actual_binary = multiply(cpp_a, cpp_b).to_numpy()

        if not np.array_equal(actual_binary, expected_binary):
            check(
                False,
                f"Random binary multiplication case {case + 1}",
            )
            print("A shape:", a.shape, "B shape:", b.shape)
            print("Expected:\n", expected_binary)
            print("Actual:\n", actual_binary)
            continue

        check(
            True,
            f"Random binary multiplication case {case + 1}",
        )

        weights = BinaryWeightMatrix(cpp_b)
        actual_scores = multiply_accumulate(cpp_a, weights)

        check_equal(
            actual_scores,
            expected_scores,
            f"Random accumulation case {case + 1}",
        )


# ============================================================
# Performance benchmark
# ============================================================

def benchmark_binary_matrices(
    sizes=None,
    warmup=BENCHMARK_WARMUP,
    repeats=BENCHMARK_REPEATS,
    seed=2026,
):
    """
    Compare the compiled library against NumPy.

    Kernel measurements reuse existing C++ objects and packed weights.
    End-to-end measurements include conversion and packing.
    """

    if sizes is None:
        sizes = BENCHMARK_SIZES

    benchmark_rng = np.random.default_rng(seed)

    def make_binary(shape):
        return random_binary(*shape, generator=benchmark_rng)

    def measure_ms(function):
        # Warm up each implementation before timing.
        for _ in range(warmup):
            function()

        timings = []

        for _ in range(repeats):
            start = time.perf_counter()
            function()
            timings.append((time.perf_counter() - start) * 1000.0)

        return statistics.median(timings)

    def report(label, numpy_ms, cpp_ms):
        speedup = numpy_ms / cpp_ms if cpp_ms > 0 else float("inf")
        winner = "C++" if speedup > 1 else "NumPy"

        print(
            f"{label:<25}"
            f"{numpy_ms:>12.3f}"
            f"{cpp_ms:>12.3f}"
            f"{speedup:>11.2f}x"
            f"   {winner}"
        )

    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: CYTHON/C++ VS PURE NUMPY")
    print("=" * 80)
    print("Timings are median milliseconds; lower is better.")
    print("Speedup > 1x means the compiled library is faster.")

    for m, k, n in sizes:
        print(f"\nA=({m}, {k}), B=({k}, {n})")
        print("-" * 80)
        print(
            f"{'Operation':<25}"
            f"{'NumPy (ms)':>12}"
            f"{'C++ (ms)':>12}"
            f"{'Speedup':>12}   Faster"
        )
        print("-" * 80)

        a = make_binary((m, k))
        b = make_binary((k, n))

        # Prepare C++ objects outside kernel-only measurements.
        cpp_a = BinaryMatrix.from_numpy(a)
        cpp_b = BinaryMatrix.from_numpy(b)
        weights = BinaryWeightMatrix(cpp_b)

        def numpy_binary():
            scores = a.astype(np.int64) @ b.astype(np.int64)
            return np.where(scores >= 0, 1, -1).astype(np.int8)

        def numpy_accumulate():
            return a.astype(np.int64) @ b.astype(np.int64)

        def cpp_binary():
            return multiply(cpp_a, cpp_b).to_numpy()

        def cpp_accumulate():
            return multiply_accumulate(cpp_a, weights)

        # Verify that the benchmark compares equivalent results.
        np.testing.assert_array_equal(
            cpp_binary(),
            numpy_binary(),
        )

        np.testing.assert_array_equal(
            cpp_accumulate(),
            numpy_accumulate(),
        )

        # Kernel-only timings.
        report(
            "Binary multiplication",
            measure_ms(numpy_binary),
            measure_ms(cpp_binary),
        )

        report(
            "Integer accumulation",
            measure_ms(numpy_accumulate),
            measure_ms(cpp_accumulate),
        )

        # End-to-end: conversion and packing included.
        def cpp_binary_end_to_end():
            matrix_a = BinaryMatrix.from_numpy(a)
            matrix_b = BinaryMatrix.from_numpy(b)
            return multiply(matrix_a, matrix_b).to_numpy()

        def cpp_accumulate_end_to_end():
            matrix_a = BinaryMatrix.from_numpy(a)
            matrix_b = BinaryMatrix.from_numpy(b)
            packed_weights = BinaryWeightMatrix(matrix_b)
            return multiply_accumulate(matrix_a, packed_weights)

        report(
            "Binary end-to-end",
            measure_ms(numpy_binary),
            measure_ms(cpp_binary_end_to_end),
        )

        report(
            "Accumulation end-to-end",
            measure_ms(numpy_accumulate),
            measure_ms(cpp_accumulate_end_to_end),
        )

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


# ============================================================
# Main test runner
# ============================================================

def main():
    global tests_passed, tests_failed

    print("=" * 64)
    print("BinaryMatrices Cython wrapper verification")
    print("=" * 64)
    print("NumPy version:", np.__version__)
    print("Random seed:", SEED)
    print("Random cases:", RANDOM_CASES)

    test_functions = [
        test_import_and_backend,
        test_construction_and_conversion,
        test_get_set_fill,
        test_transpose,
        test_multiplication,
        test_weights_and_accumulation,
        test_tie_threshold,
        test_random_matrices,
    ]

    for test_function in test_functions:
        try:
            test_function()
        except Exception:
            tests_failed += 1
            print(f"\n[ERROR] {test_function.__name__}")
            traceback.print_exc()

    print("\n" + "=" * 64)
    print("CORRECTNESS SUMMARY")
    print("=" * 64)
    print("Passed:", tests_passed)
    print("Failed:", tests_failed)

    if tests_failed:
        print("\nSome correctness tests failed. Benchmark skipped.")
        return 1

    print("\nAll correctness tests passed.")
    benchmark_binary_matrices()

    return 0


if __name__ == "__main__":
    sys.exit(main())
