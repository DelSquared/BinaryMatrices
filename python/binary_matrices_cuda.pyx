
# cython: language_level=3

import numpy as np
cimport numpy as cnp

from libc.stdint cimport int8_t, int32_t
from libc.stddef cimport size_t

cnp.import_array()


cdef extern from "python_bridge.hpp":
    void bm_cuda_matmul_scores(
        const int8_t* a,
        const int8_t* b,
        int32_t* output,
        size_t m,
        size_t k,
        size_t n
    ) except +

    void bm_cuda_matmul_binary(
        const int8_t* a,
        const int8_t* b,
        int8_t* output,
        size_t m,
        size_t k,
        size_t n
    ) except +


def _prepare_matrix(values, name):
    array = np.asarray(values)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")

    if not np.all((array == -1) | (array == 1)):
        raise ValueError(f"{name} must contain only -1 and +1")

    return np.ascontiguousarray(array, dtype=np.int8)


def matmul_scores(a, b):
    """Compute signed integer dot-product scores on the GPU."""
    a = _prepare_matrix(a, "a")
    b = _prepare_matrix(b, "b")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: {a.shape} and {b.shape}"
        )

    m, k = a.shape
    _, n = b.shape

    if k > np.iinfo(np.int32).max:
        raise OverflowError("Inner dimension exceeds int32 range")

    output = np.empty((m, n), dtype=np.int32)

    if m == 0 or n == 0:
        return output

    bm_cuda_matmul_scores(
        <const int8_t*>cnp.PyArray_DATA(a),
        <const int8_t*>cnp.PyArray_DATA(b),
        <int32_t*>cnp.PyArray_DATA(output),
        <size_t>m,
        <size_t>k,
        <size_t>n
    )

    return output


def matmul_binary(a, b):
    """Compute binary output: scores >= 0 map to +1, otherwise -1."""
    a = _prepare_matrix(a, "a")
    b = _prepare_matrix(b, "b")

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: {a.shape} and {b.shape}"
        )

    m, k = a.shape
    _, n = b.shape

    output = np.empty((m, n), dtype=np.int8)

    if m == 0 or n == 0:
        return output

    bm_cuda_matmul_binary(
        <const int8_t*>cnp.PyArray_DATA(a),
        <const int8_t*>cnp.PyArray_DATA(b),
        <int8_t*>cnp.PyArray_DATA(output),
        <size_t>m,
        <size_t>k,
        <size_t>n
    )

    return output

