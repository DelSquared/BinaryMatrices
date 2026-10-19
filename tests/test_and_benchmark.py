
import statistics
import sys
import time
import traceback

import numpy as np

from binary_matrices_cuda import matmul_binary, matmul_scores


# ============================================================
# Configuration
# ============================================================

SEED = 12345
RANDOM_CASES = 100

BENCHMARK_SIZES = [64, 128, 256, 512, 1024]
WARMUP = 3
REPEATS = 9

rng = np.random.default_rng(SEED)

passed = 0
failed = 0


# ============================================================
# Test helpers
# ============================================================

def check(condition, message):
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {message}")
    else:
        failed += 1
        print(f"[FAIL] {message}")


def random_binary(shape, generator=None):
    if generator is None:
        generator = rng

    return (
        generator.integers(0, 2, size=shape, dtype=np.int8) * 2 - 1
    ).astype(np.int8)


def reference_scores(a, b):
    """Reference implementation using int64 to avoid overflow."""
    return a.astype(np.int64) @ b.astype(np.int64)


def reference_binary(a, b):
    scores = reference_scores(a, b)
    return np.where(scores >= 0, 1, -1).astype(np.int8)


def check_array(actual, expected, message):
    global passed, failed

    try:
        np.testing.assert_array_equal(actual, expected)
        check(True, message)
    except AssertionError as exc:
        check(False, message)
        print(exc)


def expect_value_error(function, message):
    try:
        function()
    except ValueError:
        check(True, message)
    except Exception as exc:
        check(False, f"{message}: unexpected {type(exc).__name__}: {exc}")
    else:
        check(False, message)


# ============================================================
# Correctness tests
# ============================================================

def test_known_example():
    print("\n=== Known example ===")

    a = np.array([
        [1, -1, 1],
        [-1, -1, 1],
    ], dtype=np.int8)

    b = np.array([
        [1, -1],
        [-1, 1],
        [1, 1],
    ], dtype=np.int8)

    expected_scores = np.array([
        [3, -1],
        [1, 1],
    ], dtype=np.int32)

    expected_binary = np.where(
        expected_scores >= 0, 1, -1
    ).astype(np.int8)

    check_array(
        matmul_scores(a, b),
        expected_scores,
        "Known integer scores",
    )

    check_array(
        matmul_binary(a, b),
        expected_binary,
        "Known binary output",
    )


def test_edge_cases():
    print("\n=== Edge cases ===")

    # K = 1
    a = np.array([[1], [-1]], dtype=np.int8)
    b = np.array([[1, -1, 1]], dtype=np.int8)

    check_array(
        matmul_scores(a, b),
        reference_scores(a, b).astype(np.int32),
        "Single-element inner dimension",
    )

    # Zero-score threshold.
    a = np.array([[1, -1]], dtype=np.int8)
    b = np.array([[1], [1]], dtype=np.int8)

    check_array(
        matmul_scores(a, b),
        np.array([[0]], dtype=np.int32),
        "Zero dot product",
    )

    check_array(
        matmul_binary(a, b),
        np.array([[1]], dtype=np.int8),
        "Zero score maps to +1",
    )

    # Non-contiguous NumPy views.
    a0 = random_binary((8, 16))
    b0 = random_binary((16, 8))

    a_view = a0[::2, ::2]
    b_view = b0[::2, ::2]

    check_array(
        matmul_scores(a_view, b_view),
        reference_scores(a_view, b_view).astype(np.int32),
        "Non-contiguous NumPy inputs",
    )

    # Non-multiple-of-64 dimensions.
    for k in (7, 63, 64, 65, 127, 129):
        a = random_binary((3, k))
        b = random_binary((k, 5))

        check_array(
            matmul_scores(a, b),
            reference_scores(a, b).astype(np.int32),
            f"Packed-word boundary K={k}",
        )


def test_input_validation():
    print("\n=== Input validation ===")

    expect_value_error(
        lambda: matmul_binary(
            np.ones((2, 3), dtype=np.int8),
            np.ones((4, 2), dtype=np.int8),
        ),
        "Reject incompatible dimensions",
    )

    expect_value_error(
        lambda: matmul_binary(
            np.array([[0, 1]], dtype=np.int8),
            np.ones((2, 1), dtype=np.int8),
        ),
        "Reject zero-valued binary input",
    )

    expect_value_error(
        lambda: matmul_binary(
            np.array([[2, 1]], dtype=np.int8),
            np.ones((2, 1), dtype=np.int8),
        ),
        "Reject values outside {-1, +1}",
    )

    expect_value_error(
        lambda: matmul_binary(
            np.array([1, -1], dtype=np.int8),
            np.array([[1], [-1]], dtype=np.int8),
        ),
        "Reject one-dimensional input",
    )


def test_randomized():
    print("\n=== Randomized correctness ===")

    for case in range(RANDOM_CASES):
        m = int(rng.integers(1, 17))
        k = int(rng.integers(1, 258))
        n = int(rng.integers(1, 17))

        a = random_binary((m, k))
        b = random_binary((k, n))

        expected_scores = reference_scores(a, b).astype(np.int32)
        expected_binary = np.where(
            expected_scores >= 0, 1, -1
        ).astype(np.int8)

        try:
            actual_scores = matmul_scores(a, b)
            actual_binary = matmul_binary(a, b)

            np.testing.assert_array_equal(
                actual_scores, expected_scores
            )
            np.testing.assert_array_equal(
                actual_binary, expected_binary
            )

            check(True, f"Random case {case + 1}/{RANDOM_CASES}")

        except Exception as exc:
            check(False, f"Random case {case + 1}/{RANDOM_CASES}")
            print(f"  Shapes: A={a.shape}, B={b.shape}")
            print(f"  Error: {exc}")


def run_correctness_tests():
    print("=" * 64)
    print("BinaryMatricesCUDA: correctness verification")
    print("=" * 64)
    print("NumPy version:", np.__version__)
    print("Random seed:", SEED)

    test_known_example()
    test_edge_cases()
    test_input_validation()
    test_randomized()

    print("\n" + "=" * 64)
    print("CORRECTNESS SUMMARY")
    print("=" * 64)
    print("Passed:", passed)
    print("Failed:", failed)

    return failed == 0


# ============================================================
# Benchmark helpers
# ============================================================

def median_ms(function, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        function()

    samples = []

    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)

    return statistics.median(samples)


def benchmark():
    print("\n" + "=" * 82)
    print("PERFORMANCE BENCHMARK: CUDA vs NUMPY")
    print("=" * 82)
    print("Times are median milliseconds; lower is better.")
    print("Speedup > 1x means CUDA is faster.")
    print("CUDA measurements include CPU packing, allocations,")
    print("host-to-device transfers, kernel execution, and result copies.")
    print()

    print(
        f"{'Shape':<20}"
        f"{'Operation':<16}"
        f"{'NumPy ms':>12}"
        f"{'CUDA ms':>12}"
        f"{'Speedup':>12}"
    )
    print("-" * 72)

    benchmark_rng = np.random.default_rng(2026)

    for size in BENCHMARK_SIZES:
        m = k = n = size

        a = random_binary((m, k), benchmark_rng)
        b = random_binary((k, n), benchmark_rng)

        def numpy_scores():
            return reference_scores(a, b)

        def numpy_binary():
            return reference_binary(a, b)

        def cuda_scores():
            return matmul_scores(a, b)

        def cuda_binary():
            return matmul_binary(a, b)

        # Verify each benchmark workload before measuring it.
        np.testing.assert_array_equal(
            cuda_scores(),
            numpy_scores().astype(np.int32),
        )

        np.testing.assert_array_equal(
            cuda_binary(),
            numpy_binary(),
        )

        operations = [
            ("Integer scores", numpy_scores, cuda_scores),
            ("Binary output", numpy_binary, cuda_binary),
        ]

        for name, numpy_fn, cuda_fn in operations:
            numpy_time = median_ms(numpy_fn)
            cuda_time = median_ms(cuda_fn)

            speedup = (
                numpy_time / cuda_time
                if cuda_time > 0 else float("inf")
            )

            print(
                f"{m}x{k}x{n:<12}"
                f"{name:<16}"
                f"{numpy_time:>12.3f}"
                f"{cuda_time:>12.3f}"
                f"{speedup:>11.2f}x"
            )

    print("\nBenchmark complete.")


# ============================================================
# Entry point
# ============================================================

def main():
    try:
        correctness_ok = run_correctness_tests()
    except Exception:
        print("\nA correctness test raised an unexpected exception:")
        traceback.print_exc()
        return 1

    if not correctness_ok:
        print("\nCorrectness tests failed. Benchmark skipped.")
        return 1

    print("\nAll correctness tests passed.")
    benchmark()

    return 0


if __name__ == "__main__":
    sys.exit(main())

