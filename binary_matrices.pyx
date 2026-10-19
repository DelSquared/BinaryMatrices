
# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
cimport numpy as cnp

from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libc.stdint cimport int32_t, int64_t

cnp.import_array()


# ============================================================
# C++ declarations
# ============================================================

cdef extern from "binary_matrix.hpp" namespace "bnn":

    cdef cppclass CppMatrix "bnn::BinaryMatrix":
        CppMatrix() except +
        CppMatrix(size_t rows, size_t cols) except +

        size_t rows() const
        size_t cols() const
        size_t storage_bytes() const

        cpp_bool get_bit(size_t row, size_t col) const
        int get_value(size_t row, size_t col) const

        void set_value(
            size_t row,
            size_t col,
            int value
        ) except +

        void set_bit(
            size_t row,
            size_t col,
            cpp_bool value
        ) except +

        void fill_value(int value) except +
        void randomize() except +

        CppMatrix transpose() const


    cdef cppclass CppWeights "bnn::BinaryWeightMatrix":
        CppWeights() except +
        CppWeights(const CppMatrix& matrix) except +

        size_t input_size() const
        size_t output_size() const


    CppMatrix cpp_multiply "bnn::multiply"(
        const CppMatrix& a,
        const CppMatrix& b
    ) except +

    CppMatrix cpp_multiply_weights "bnn::multiply"(
        const CppMatrix& a,
        const CppWeights& weights
    ) except +

    void cpp_accumulate "bnn::multiply_accumulate"(
        const CppMatrix& a,
        const CppWeights& weights,
        vector[int32_t]& output
    ) except +

    int64_t cpp_dot "bnn::dot"(
        const CppMatrix& activation,
        size_t activation_row,
        const CppWeights& weights,
        size_t output
    ) except +

    const char* cpp_simd_backend "bnn::simd_backend"()
    cpp_bool cpp_has_avx2 "bnn::has_avx2"()
    cpp_bool cpp_has_avx512 "bnn::has_avx512_vpopcntdq"()


# ============================================================
# NumPy conversion helpers
# ============================================================

cdef object _as_signed_array(object values):
    """Validate a 2D array containing only -1 and +1."""

    arr = np.asarray(values)

    if arr.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional matrix"
        )

    if not np.all((arr == 1) | (arr == -1)):
        raise ValueError(
            "Matrix values must be +1 or -1"
        )

    return np.ascontiguousarray(arr, dtype=np.int8)


cdef CppMatrix* _matrix_from_array(object values) except *:
    """Convert a NumPy matrix into packed C++ storage."""

    cdef object arr = _as_signed_array(values)

    cdef cnp.ndarray[
        cnp.int8_t,
        ndim=2,
        mode="c"
    ] data = arr

    cdef Py_ssize_t r, c

    cdef CppMatrix* result = new CppMatrix(
        <size_t>data.shape[0],
        <size_t>data.shape[1]
    )

    try:
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                result.set_value(
                    <size_t>r,
                    <size_t>c,
                    <int>data[r, c]
                )
    except:
        del result
        raise

    return result


cdef object _matrix_to_array(const CppMatrix* matrix):
    """Convert packed C++ storage into a NumPy int8 matrix."""

    cdef Py_ssize_t r, c

    cdef Py_ssize_t rows = <Py_ssize_t>matrix.rows()
    cdef Py_ssize_t cols = <Py_ssize_t>matrix.cols()

    cdef cnp.ndarray[
        cnp.int8_t,
        ndim=2
    ] result = np.empty(
        (rows, cols),
        dtype=np.int8
    )

    for r in range(rows):
        for c in range(cols):
            result[r, c] = matrix.get_value(
                <size_t>r,
                <size_t>c
            )

    return result


# ============================================================
# BinaryMatrix
# ============================================================

cdef class BinaryMatrix:
    """
    Binary matrix stored using the original C++ implementation.

    Logical values are restricted to -1 and +1.
    """

    cdef CppMatrix* _ptr

    def __cinit__(self, rows=0, cols=0):
        if rows < 0 or cols < 0:
            raise ValueError(
                "Dimensions must be non-negative"
            )

        self._ptr = new CppMatrix(
            <size_t>rows,
            <size_t>cols
        )

    def __dealloc__(self):
        if self._ptr != NULL:
            del self._ptr
            self._ptr = NULL

    @classmethod
    def from_numpy(cls, values):
        """Create a BinaryMatrix from a NumPy array."""

        cdef object arr = _as_signed_array(values)

        cdef BinaryMatrix result = cls(
            arr.shape[0],
            arr.shape[1]
        )

        cdef Py_ssize_t r, c

        cdef cnp.ndarray[
            cnp.int8_t,
            ndim=2,
            mode="c"
        ] data = arr

        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                result._ptr.set_value(
                    <size_t>r,
                    <size_t>c,
                    <int>data[r, c]
                )

        return result

    @property
    def shape(self):
        return (
            self._ptr.rows(),
            self._ptr.cols()
        )

    @property
    def nbytes(self):
        """Return the packed storage size in bytes."""
        return self._ptr.storage_bytes()

    def get(self, row, col):
        """Get one element as -1 or +1."""

        if row < 0 or row >= self._ptr.rows():
            raise IndexError("Row index out of range")

        if col < 0 or col >= self._ptr.cols():
            raise IndexError("Column index out of range")

        return self._ptr.get_value(
            <size_t>row,
            <size_t>col
        )

    def set(self, row, col, value):
        """Set one element to -1 or +1."""

        if row < 0 or row >= self._ptr.rows():
            raise IndexError("Row index out of range")

        if col < 0 or col >= self._ptr.cols():
            raise IndexError("Column index out of range")

        if value not in (-1, 1):
            raise ValueError(
                "Value must be -1 or +1"
            )

        self._ptr.set_value(
            <size_t>row,
            <size_t>col,
            <int>value
        )

    def fill(self, value):
        """Fill the entire matrix with -1 or +1."""

        if value not in (-1, 1):
            raise ValueError(
                "Value must be -1 or +1"
            )

        self._ptr.fill_value(<int>value)

    def randomize(self):
        """Randomize the matrix using the C++ implementation."""
        self._ptr.randomize()

    def transpose(self):
        """Return a transposed copy."""

        cdef BinaryMatrix result = BinaryMatrix()

        result._ptr[0] = self._ptr.transpose()

        return result

    def to_numpy(self):
        """Return a NumPy int8 array containing -1 and +1."""
        return _matrix_to_array(self._ptr)

    def __repr__(self):
        return (
            f"BinaryMatrix(shape={self.shape}, "
            f"nbytes={self.nbytes})"
        )


# ============================================================
# BinaryWeightMatrix
# ============================================================

cdef class BinaryWeightMatrix:
    """Pre-packed weights for repeated binary operations."""

    cdef CppWeights* _ptr

    def __cinit__(self, BinaryMatrix matrix):
        self._ptr = new CppWeights(
            matrix._ptr[0]
        )

    def __dealloc__(self):
        if self._ptr != NULL:
            del self._ptr
            self._ptr = NULL

    @property
    def shape(self):
        """Return (input_size, output_size)."""

        return (
            self._ptr.input_size(),
            self._ptr.output_size()
        )

    def __repr__(self):
        return f"BinaryWeightMatrix(shape={self.shape})"


# ============================================================
# Matrix multiplication
# ============================================================

def multiply(BinaryMatrix a, b):
    """
    Multiply binary matrices.

    b may be a BinaryMatrix or BinaryWeightMatrix.
    Returns a BinaryMatrix.
    """

    cdef BinaryMatrix result = BinaryMatrix()

    if isinstance(b, BinaryWeightMatrix):
        cdef_weights = <BinaryWeightMatrix>b

        if a._ptr.cols() != cdef_weights._ptr.input_size():
            raise ValueError(
                "Incompatible matrix dimensions"
            )

        result._ptr[0] = cpp_multiply_weights(
            a._ptr[0],
            cdef_weights._ptr[0]
        )

    elif isinstance(b, BinaryMatrix):
        cdef_matrix = <BinaryMatrix>b

        if a._ptr.cols() != cdef_matrix._ptr.rows():
            raise ValueError(
                "Incompatible matrix dimensions"
            )

        result._ptr[0] = cpp_multiply(
            a._ptr[0],
            cdef_matrix._ptr[0]
        )

    else:
        raise TypeError(
            "b must be a BinaryMatrix or BinaryWeightMatrix"
        )

    return result


def multiply_accumulate(
    BinaryMatrix a,
    BinaryWeightMatrix weights
):
    """
    Return signed integer dot products as a NumPy int32 array.
    """

    cdef vector[int32_t] output

    cdef Py_ssize_t r, c

    cdef size_t rows = a._ptr.rows()
    cdef size_t cols = weights._ptr.output_size()

    if a._ptr.cols() != weights._ptr.input_size():
        raise ValueError(
            "Incompatible matrix dimensions"
        )

    cpp_accumulate(
        a._ptr[0],
        weights._ptr[0],
        output
    )

    if output.size() != rows * cols:
        raise RuntimeError(
            "Unexpected output size from C++ multiplication"
        )

    result = np.empty(
        (rows, cols),
        dtype=np.int32
    )

    for r in range(rows):
        for c in range(cols):
            result[r, c] = output[r * cols + c]

    return result


def dot(
    BinaryMatrix a,
    row,
    BinaryWeightMatrix weights,
    output
):
    """Return one signed integer dot product."""

    if a._ptr.cols() != weights._ptr.input_size():
        raise ValueError(
            "Incompatible matrix dimensions"
        )

    if row < 0 or row >= a._ptr.rows():
        raise IndexError(
            "Activation row out of range"
        )

    if output < 0 or output >= weights._ptr.output_size():
        raise IndexError(
            "Output index out of range"
        )

    return cpp_dot(
        a._ptr[0],
        <size_t>row,
        weights._ptr[0],
        <size_t>output
    )


# ============================================================
# CPU SIMD information
# ============================================================

def simd_backend():
    """Return the backend name reported by the C++ library."""

    return (<bytes>cpp_simd_backend()).decode("ascii")


def has_avx2():
    """Return whether the library reports AVX2 support."""
    return bool(cpp_has_avx2())


def has_avx512_vpopcntdq():
    """Return whether the library reports AVX-512 VPOPCNTDQ support."""
    return bool(cpp_has_avx512())
