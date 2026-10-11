#pragma once

// ============================================================================
// binary_matrix.hpp
//
// Packed binary matrices for Binary Neural Networks.
//
// Representation:
//
//     bit 0 -> +1
//     bit 1 -> -1
//
// Therefore:
//
//     a * b = +1 if bits are equal
//           = -1 if bits differ
//
// For a vector of K elements:
//
//     dot(a,b) = K - 2 * popcount(a XOR b)
//
// This header provides:
//
//   - Arbitrary matrix dimensions.
//   - 64 binary values packed into each uint64_t.
//   - 64-byte aligned storage.
//   - Runtime CPU feature detection.
//   - Scalar fallback.
//   - AVX2 kernels.
//   - AVX-512 VPOPCNTDQ kernels.
//   - Multi-output SIMD kernels.
//   - Permanently packed/transposed weights.
//   - Direct packed binary output.
//   - Integer accumulator output.
//   - C++17 compatible.
//   - No requirement for -march=native.
//
// Recommended:
//
//     g++ -O3 -std=c++17 main.cpp -o bnn
//
// For maximum optimization when you know the target:
//
//     g++ -O3 -march=native -std=c++17 main.cpp -o bnn
//
// Runtime dispatch remains active even when -march=native is not used.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || \
    defined(__i386__)  || defined(_M_IX86)

    #include <immintrin.h>

    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif

#endif


namespace bnn {

// ============================================================================
// Constants
// ============================================================================

static constexpr std::size_t BITS_PER_WORD = 64;
static constexpr std::size_t SIMD_ALIGNMENT = 64;

using word_t = std::uint64_t;


// ============================================================================
// 64-byte aligned allocator
// ============================================================================

template <typename T, std::size_t Alignment>
class AlignedAllocator
{
public:

    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind
    {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(
        const AlignedAllocator<U, Alignment>&) noexcept
    {
    }

    [[nodiscard]]
    T* allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();

        const std::size_t bytes = n * sizeof(T);

#if defined(__cpp_aligned_new)

        return static_cast<T*>(
            ::operator new(
                bytes,
                std::align_val_t(Alignment)));

#else

        void* ptr = nullptr;

#if defined(_MSC_VER)

        ptr = _aligned_malloc(bytes, Alignment);

        if (!ptr)
            throw std::bad_alloc();

#else

        if (posix_memalign(&ptr, Alignment, bytes) != 0)
            throw std::bad_alloc();

#endif

        return static_cast<T*>(ptr);

#endif
    }

    void deallocate(
        T* ptr,
        std::size_t) noexcept
    {
#if defined(__cpp_aligned_new)

        ::operator delete(
            ptr,
            std::align_val_t(Alignment));

#else

#if defined(_MSC_VER)

        _aligned_free(ptr);

#else

        std::free(ptr);

#endif

#endif
    }

    template <typename U>
    bool operator==(
        const AlignedAllocator<U, Alignment>&) const noexcept
    {
        return true;
    }

    template <typename U>
    bool operator!=(
        const AlignedAllocator<U, Alignment>&) const noexcept
    {
        return false;
    }
};


using word_vector =
    std::vector<
        word_t,
        AlignedAllocator<word_t, SIMD_ALIGNMENT>
    >;


// ============================================================================
// Internal utilities
// ============================================================================

namespace detail {

inline std::size_t checked_mul(
    std::size_t a,
    std::size_t b)
{
    if (a != 0 &&
        b > std::numeric_limits<std::size_t>::max() / a)
    {
        throw std::length_error(
            "BinaryMatrix allocation size overflow");
    }

    return a * b;
}


inline constexpr std::size_t words_for_bits(
    std::size_t bits) noexcept
{
    return (bits + 63) / 64;
}


inline constexpr word_t low_bits_mask(
    std::size_t bits) noexcept
{
    if (bits == 0)
        return 0;

    if (bits >= 64)
        return ~word_t{0};

    return (word_t{1} << bits) - 1;
}


inline unsigned ctz64(
    word_t x) noexcept
{
#if defined(_MSC_VER)

    unsigned long index = 0;

#if defined(_M_X64)

    _BitScanForward64(
        &index,
        x);

#else

    if (static_cast<std::uint32_t>(x) != 0)
    {
        _BitScanForward(
            &index,
            static_cast<std::uint32_t>(x));
    }
    else
    {
        _BitScanForward(
            &index,
            static_cast<std::uint32_t>(x >> 32));

        index += 32;
    }

#endif

    return static_cast<unsigned>(index);

#elif defined(__GNUC__) || defined(__clang__)

    return static_cast<unsigned>(
        __builtin_ctzll(x));

#else

    unsigned n = 0;

    while ((x & 1u) == 0)
    {
        x >>= 1;
        ++n;
    }

    return n;

#endif
}


inline std::uint32_t popcount64_soft(
    word_t x) noexcept
{
    std::uint32_t count = 0;

    while (x != 0)
    {
        x &= x - 1;
        ++count;
    }

    return count;
}


// ============================================================================
// CPU feature detection
// ============================================================================

enum class SimdLevel
{
    Scalar,
    AVX2,
    AVX512_VPOPCNTDQ
};


#if defined(__x86_64__) || defined(_M_X64) || \
    defined(__i386__)  || defined(_M_IX86)

inline std::uint64_t xgetbv(
    std::uint32_t index) noexcept
{
#if defined(_MSC_VER)

    return _xgetbv(index);

#elif defined(__GNUC__) || defined(__clang__)

    std::uint32_t eax;
    std::uint32_t edx;

    __asm__ volatile(
        "xgetbv"
        : "=a"(eax),
          "=d"(edx)
        : "c"(index));

    return
        (static_cast<std::uint64_t>(edx) << 32) |
        eax;

#else

    (void)index;
    return 0;

#endif
}


inline void cpuid(
    int out[4],
    int leaf,
    int subleaf = 0) noexcept
{
#if defined(_MSC_VER)

    __cpuidex(out, leaf, subleaf);

#elif defined(__GNUC__) || defined(__clang__)

#if defined(__i386__) && defined(__PIC__)

    std::uint32_t ebx;

    __asm__ volatile(
        "xchgl %%ebx, %1\n\t"
        "cpuid\n\t"
        "xchgl %%ebx, %1"
        : "=a"(out[0]),
          "=&r"(ebx),
          "=c"(out[2]),
          "=d"(out[3])
        : "a"(leaf),
          "c"(subleaf));

    out[1] = static_cast<int>(ebx);

#else

    __asm__ volatile(
        "cpuid"
        : "=a"(out[0]),
          "=b"(out[1]),
          "=c"(out[2]),
          "=d"(out[3])
        : "a"(leaf),
          "c"(subleaf));

#endif

#else

    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;

#endif
}


inline SimdLevel detect_simd_level() noexcept
{
    int info[4] = {};

    cpuid(info, 0);

    const int max_leaf = info[0];

    if (max_leaf < 1)
        return SimdLevel::Scalar;

    cpuid(info, 1);

    const std::uint32_t ecx =
        static_cast<std::uint32_t>(info[2]);

    const bool osxsave =
        (ecx & (1u << 27)) != 0;

    const bool avx =
        (ecx & (1u << 28)) != 0;

    if (!osxsave || !avx)
        return SimdLevel::Scalar;

    const std::uint64_t xcr0 =
        xgetbv(0);

    constexpr std::uint64_t AVX_STATE =
        (1ull << 1) |
        (1ull << 2);

    if ((xcr0 & AVX_STATE) != AVX_STATE)
        return SimdLevel::Scalar;

    if (max_leaf < 7)
        return SimdLevel::Scalar;

    cpuid(info, 7, 0);

    const std::uint32_t ebx =
        static_cast<std::uint32_t>(info[1]);

    const std::uint32_t ecx7 =
        static_cast<std::uint32_t>(info[2]);

    const bool avx2 =
        (ebx & (1u << 5)) != 0;

    const bool avx512f =
        (ebx & (1u << 16)) != 0;

    const bool avx512vpopcntdq =
        (ecx7 & (1u << 14)) != 0;

    constexpr std::uint64_t AVX512_STATE =
        (1ull << 1) |
        (1ull << 2) |
        (1ull << 5) |
        (1ull << 6) |
        (1ull << 7);

    const bool os_avx512 =
        (xcr0 & AVX512_STATE) == AVX512_STATE;

    if (avx512f &&
        avx512vpopcntdq &&
        os_avx512)
    {
        return SimdLevel::AVX512_VPOPCNTDQ;
    }

    if (avx2)
        return SimdLevel::AVX2;

    return SimdLevel::Scalar;
}


#else

inline SimdLevel detect_simd_level() noexcept
{
    return SimdLevel::Scalar;
}

#endif


inline SimdLevel simd_level() noexcept
{
    static const SimdLevel level =
        detect_simd_level();

    return level;
}


inline const char* simd_level_name() noexcept
{
    switch (simd_level())
    {
        case SimdLevel::AVX512_VPOPCNTDQ:
            return "AVX-512 VPOPCNTDQ";

        case SimdLevel::AVX2:
            return "AVX2";

        default:
            return "Scalar";
    }
}


// ============================================================================
// Scalar dot product
// ============================================================================

inline std::int64_t dot_scalar(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    std::uint64_t mismatches = 0;

    for (std::size_t i = 0;
         i < words;
         ++i)
    {
        word_t x =
            a[i] ^ b[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            x &= low_bits_mask(
                valid_bits & 63);
        }

        mismatches +=
            popcount64_soft(x);
    }

    return
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(mismatches);
}


// ============================================================================
// AVX2 single dot product
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx2")))
inline std::int64_t dot_avx2(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    std::uint64_t mismatches = 0;

    std::size_t i = 0;

    for (; i + 4 <= words; i += 4)
    {
        const __m256i va =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    a + i));

        const __m256i vb =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    b + i));

        const __m256i vx =
            _mm256_xor_si256(
                va,
                vb);

        alignas(32) word_t tmp[4];

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(tmp),
            vx);

        mismatches +=
            popcount64_soft(tmp[0]);

        mismatches +=
            popcount64_soft(tmp[1]);

        mismatches +=
            popcount64_soft(tmp[2]);

        mismatches +=
            popcount64_soft(tmp[3]);
    }

    for (; i < words; ++i)
    {
        word_t x =
            a[i] ^ b[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            x &= low_bits_mask(
                valid_bits & 63);
        }

        mismatches +=
            popcount64_soft(x);
    }

    return
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(mismatches);
}

#elif defined(_MSC_VER) && defined(__AVX2__)

inline std::int64_t dot_avx2(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    std::uint64_t mismatches = 0;

    std::size_t i = 0;

    for (; i + 4 <= words; i += 4)
    {
        const __m256i va =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    a + i));

        const __m256i vb =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    b + i));

        const __m256i vx =
            _mm256_xor_si256(
                va,
                vb);

        alignas(32) word_t tmp[4];

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(tmp),
            vx);

        mismatches += popcount64_soft(tmp[0]);
        mismatches += popcount64_soft(tmp[1]);
        mismatches += popcount64_soft(tmp[2]);
        mismatches += popcount64_soft(tmp[3]);
    }

    for (; i < words; ++i)
    {
        word_t x = a[i] ^ b[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            x &= low_bits_mask(
                valid_bits & 63);
        }

        mismatches +=
            popcount64_soft(x);
    }

    return
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(mismatches);
}

#endif


// ============================================================================
// AVX-512 single dot product
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx512f,avx512vpopcntdq")))
inline std::int64_t dot_avx512(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    std::uint64_t mismatches = 0;

    std::size_t i = 0;

    for (; i + 8 <= words; i += 8)
    {
        const __m512i va =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    a + i));

        const __m512i vb =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    b + i));

        const __m512i vx =
            _mm512_xor_si512(
                va,
                vb);

        const __m512i counts =
            _mm512_popcnt_epi64(vx);

        alignas(64) word_t tmp[8];

        _mm512_store_si512(
            reinterpret_cast<void*>(tmp),
            counts);

        mismatches += tmp[0];
        mismatches += tmp[1];
        mismatches += tmp[2];
        mismatches += tmp[3];
        mismatches += tmp[4];
        mismatches += tmp[5];
        mismatches += tmp[6];
        mismatches += tmp[7];
    }

    for (; i < words; ++i)
    {
        word_t x =
            a[i] ^ b[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            x &= low_bits_mask(
                valid_bits & 63);
        }

        mismatches +=
            popcount64_soft(x);
    }

    return
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(mismatches);
}

#elif defined(_MSC_VER) && \
      defined(__AVX512F__) && \
      defined(__AVX512VPOPCNTDQ__)

inline std::int64_t dot_avx512(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    std::uint64_t mismatches = 0;

    std::size_t i = 0;

    for (; i + 8 <= words; i += 8)
    {
        const __m512i va =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    a + i));

        const __m512i vb =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    b + i));

        const __m512i vx =
            _mm512_xor_si512(
                va,
                vb);

        const __m512i counts =
            _mm512_popcnt_epi64(vx);

        alignas(64) word_t tmp[8];

        _mm512_store_si512(
            reinterpret_cast<void*>(tmp),
            counts);

        mismatches += tmp[0];
        mismatches += tmp[1];
        mismatches += tmp[2];
        mismatches += tmp[3];
        mismatches += tmp[4];
        mismatches += tmp[5];
        mismatches += tmp[6];
        mismatches += tmp[7];
    }

    for (; i < words; ++i)
    {
        word_t x =
            a[i] ^ b[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            x &= low_bits_mask(
                valid_bits & 63);
        }

        mismatches +=
            popcount64_soft(x);
    }

    return
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(mismatches);
}

#endif


// ============================================================================
// Four simultaneous AVX2 dot products
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx2")))
inline void binary_dot_4_avx2(
    const word_t* activation,
    const word_t* w0,
    const word_t* w1,
    const word_t* w2,
    const word_t* w3,
    std::size_t words,
    std::size_t valid_bits,
    std::int64_t& d0,
    std::int64_t& d1,
    std::int64_t& d2,
    std::int64_t& d3) noexcept
{
    std::uint64_t m0 = 0;
    std::uint64_t m1 = 0;
    std::uint64_t m2 = 0;
    std::uint64_t m3 = 0;

    std::size_t i = 0;

    for (; i + 4 <= words; i += 4)
    {
        const __m256i va =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    activation + i));

        const __m256i v0 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w0 + i));

        const __m256i v1 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w1 + i));

        const __m256i v2 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w2 + i));

        const __m256i v3 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w3 + i));

        alignas(32) word_t t0[4];
        alignas(32) word_t t1[4];
        alignas(32) word_t t2[4];
        alignas(32) word_t t3[4];

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t0),
            _mm256_xor_si256(va, v0));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t1),
            _mm256_xor_si256(va, v1));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t2),
            _mm256_xor_si256(va, v2));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t3),
            _mm256_xor_si256(va, v3));

        m0 += popcount64_soft(t0[0]);
        m0 += popcount64_soft(t0[1]);
        m0 += popcount64_soft(t0[2]);
        m0 += popcount64_soft(t0[3]);

        m1 += popcount64_soft(t1[0]);
        m1 += popcount64_soft(t1[1]);
        m1 += popcount64_soft(t1[2]);
        m1 += popcount64_soft(t1[3]);

        m2 += popcount64_soft(t2[0]);
        m2 += popcount64_soft(t2[1]);
        m2 += popcount64_soft(t2[2]);
        m2 += popcount64_soft(t2[3]);

        m3 += popcount64_soft(t3[0]);
        m3 += popcount64_soft(t3[1]);
        m3 += popcount64_soft(t3[2]);
        m3 += popcount64_soft(t3[3]);
    }

    for (; i < words; ++i)
    {
        const word_t a = activation[i];

        word_t x0 = a ^ w0[i];
        word_t x1 = a ^ w1[i];
        word_t x2 = a ^ w2[i];
        word_t x3 = a ^ w3[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            const word_t mask =
                low_bits_mask(valid_bits & 63);

            x0 &= mask;
            x1 &= mask;
            x2 &= mask;
            x3 &= mask;
        }

        m0 += popcount64_soft(x0);
        m1 += popcount64_soft(x1);
        m2 += popcount64_soft(x2);
        m3 += popcount64_soft(x3);
    }

    d0 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m0);

    d1 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m1);

    d2 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m2);

    d3 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m3);
}


#elif defined(_MSC_VER) && defined(__AVX2__)

inline void binary_dot_4_avx2(
    const word_t* activation,
    const word_t* w0,
    const word_t* w1,
    const word_t* w2,
    const word_t* w3,
    std::size_t words,
    std::size_t valid_bits,
    std::int64_t& d0,
    std::int64_t& d1,
    std::int64_t& d2,
    std::int64_t& d3) noexcept
{
    std::uint64_t m0 = 0;
    std::uint64_t m1 = 0;
    std::uint64_t m2 = 0;
    std::uint64_t m3 = 0;

    std::size_t i = 0;

    for (; i + 4 <= words; i += 4)
    {
        const __m256i va =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    activation + i));

        const __m256i v0 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w0 + i));

        const __m256i v1 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w1 + i));

        const __m256i v2 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w2 + i));

        const __m256i v3 =
            _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    w3 + i));

        alignas(32) word_t t0[4];
        alignas(32) word_t t1[4];
        alignas(32) word_t t2[4];
        alignas(32) word_t t3[4];

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t0),
            _mm256_xor_si256(va, v0));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t1),
            _mm256_xor_si256(va, v1));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t2),
            _mm256_xor_si256(va, v2));

        _mm256_store_si256(
            reinterpret_cast<__m256i*>(t3),
            _mm256_xor_si256(va, v3));

        m0 += popcount64_soft(t0[0]);
        m0 += popcount64_soft(t0[1]);
        m0 += popcount64_soft(t0[2]);
        m0 += popcount64_soft(t0[3]);

        m1 += popcount64_soft(t1[0]);
        m1 += popcount64_soft(t1[1]);
        m1 += popcount64_soft(t1[2]);
        m1 += popcount64_soft(t1[3]);

        m2 += popcount64_soft(t2[0]);
        m2 += popcount64_soft(t2[1]);
        m2 += popcount64_soft(t2[2]);
        m2 += popcount64_soft(t2[3]);

        m3 += popcount64_soft(t3[0]);
        m3 += popcount64_soft(t3[1]);
        m3 += popcount64_soft(t3[2]);
        m3 += popcount64_soft(t3[3]);
    }

    for (; i < words; ++i)
    {
        const word_t a = activation[i];

        word_t x0 = a ^ w0[i];
        word_t x1 = a ^ w1[i];
        word_t x2 = a ^ w2[i];
        word_t x3 = a ^ w3[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            const word_t mask =
                low_bits_mask(valid_bits & 63);

            x0 &= mask;
            x1 &= mask;
            x2 &= mask;
            x3 &= mask;
        }

        m0 += popcount64_soft(x0);
        m1 += popcount64_soft(x1);
        m2 += popcount64_soft(x2);
        m3 += popcount64_soft(x3);
    }

    d0 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m0);

    d1 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m1);

    d2 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m2);

    d3 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m3);
}

#endif


// ============================================================================
// Four simultaneous AVX-512 dot products
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)

__attribute__((target("avx512f,avx512vpopcntdq")))
inline void binary_dot_4_avx512(
    const word_t* activation,
    const word_t* w0,
    const word_t* w1,
    const word_t* w2,
    const word_t* w3,
    std::size_t words,
    std::size_t valid_bits,
    std::int64_t& d0,
    std::int64_t& d1,
    std::int64_t& d2,
    std::int64_t& d3) noexcept
{
    std::uint64_t m0 = 0;
    std::uint64_t m1 = 0;
    std::uint64_t m2 = 0;
    std::uint64_t m3 = 0;

    std::size_t i = 0;

    for (; i + 8 <= words; i += 8)
    {
        const __m512i va =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    activation + i));

        const __m512i v0 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w0 + i));

        const __m512i v1 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w1 + i));

        const __m512i v2 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w2 + i));

        const __m512i v3 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w3 + i));

        const __m512i c0 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v0));

        const __m512i c1 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v1));

        const __m512i c2 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v2));

        const __m512i c3 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v3));

        alignas(64) word_t t0[8];
        alignas(64) word_t t1[8];
        alignas(64) word_t t2[8];
        alignas(64) word_t t3[8];

        _mm512_store_si512(
            reinterpret_cast<void*>(t0),
            c0);

        _mm512_store_si512(
            reinterpret_cast<void*>(t1),
            c1);

        _mm512_store_si512(
            reinterpret_cast<void*>(t2),
            c2);

        _mm512_store_si512(
            reinterpret_cast<void*>(t3),
            c3);

        m0 += t0[0] + t0[1] + t0[2] + t0[3]
            + t0[4] + t0[5] + t0[6] + t0[7];

        m1 += t1[0] + t1[1] + t1[2] + t1[3]
            + t1[4] + t1[5] + t1[6] + t1[7];

        m2 += t2[0] + t2[1] + t2[2] + t2[3]
            + t2[4] + t2[5] + t2[6] + t2[7];

        m3 += t3[0] + t3[1] + t3[2] + t3[3]
            + t3[4] + t3[5] + t3[6] + t3[7];
    }

    for (; i < words; ++i)
    {
        const word_t a = activation[i];

        word_t x0 = a ^ w0[i];
        word_t x1 = a ^ w1[i];
        word_t x2 = a ^ w2[i];
        word_t x3 = a ^ w3[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            const word_t mask =
                low_bits_mask(valid_bits & 63);

            x0 &= mask;
            x1 &= mask;
            x2 &= mask;
            x3 &= mask;
        }

        m0 += popcount64_soft(x0);
        m1 += popcount64_soft(x1);
        m2 += popcount64_soft(x2);
        m3 += popcount64_soft(x3);
    }

    d0 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m0);

    d1 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m1);

    d2 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m2);

    d3 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m3);
}


#elif defined(_MSC_VER) && \
      defined(__AVX512F__) && \
      defined(__AVX512VPOPCNTDQ__)

inline void binary_dot_4_avx512(
    const word_t* activation,
    const word_t* w0,
    const word_t* w1,
    const word_t* w2,
    const word_t* w3,
    std::size_t words,
    std::size_t valid_bits,
    std::int64_t& d0,
    std::int64_t& d1,
    std::int64_t& d2,
    std::int64_t& d3) noexcept
{
    std::uint64_t m0 = 0;
    std::uint64_t m1 = 0;
    std::uint64_t m2 = 0;
    std::uint64_t m3 = 0;

    std::size_t i = 0;

    for (; i + 8 <= words; i += 8)
    {
        const __m512i va =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    activation + i));

        const __m512i v0 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w0 + i));

        const __m512i v1 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w1 + i));

        const __m512i v2 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w2 + i));

        const __m512i v3 =
            _mm512_loadu_si512(
                reinterpret_cast<const void*>(
                    w3 + i));

        const __m512i c0 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v0));

        const __m512i c1 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v1));

        const __m512i c2 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v2));

        const __m512i c3 =
            _mm512_popcnt_epi64(
                _mm512_xor_si512(va, v3));

        alignas(64) word_t t0[8];
        alignas(64) word_t t1[8];
        alignas(64) word_t t2[8];
        alignas(64) word_t t3[8];

        _mm512_store_si512(
            reinterpret_cast<void*>(t0),
            c0);

        _mm512_store_si512(
            reinterpret_cast<void*>(t1),
            c1);

        _mm512_store_si512(
            reinterpret_cast<void*>(t2),
            c2);

        _mm512_store_si512(
            reinterpret_cast<void*>(t3),
            c3);

        m0 += t0[0] + t0[1] + t0[2] + t0[3]
            + t0[4] + t0[5] + t0[6] + t0[7];

        m1 += t1[0] + t1[1] + t1[2] + t1[3]
            + t1[4] + t1[5] + t1[6] + t1[7];

        m2 += t2[0] + t2[1] + t2[2] + t2[3]
            + t2[4] + t2[5] + t2[6] + t2[7];

        m3 += t3[0] + t3[1] + t3[2] + t3[3]
            + t3[4] + t3[5] + t3[6] + t3[7];
    }

    for (; i < words; ++i)
    {
        const word_t a = activation[i];

        word_t x0 = a ^ w0[i];
        word_t x1 = a ^ w1[i];
        word_t x2 = a ^ w2[i];
        word_t x3 = a ^ w3[i];

        if (i + 1 == words &&
            (valid_bits & 63) != 0)
        {
            const word_t mask =
                low_bits_mask(valid_bits & 63);

            x0 &= mask;
            x1 &= mask;
            x2 &= mask;
            x3 &= mask;
        }

        m0 += popcount64_soft(x0);
        m1 += popcount64_soft(x1);
        m2 += popcount64_soft(x2);
        m3 += popcount64_soft(x3);
    }

    d0 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m0);

    d1 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m1);

    d2 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m2);

    d3 =
        static_cast<std::int64_t>(valid_bits) -
        2 * static_cast<std::int64_t>(m3);
}

#endif


// ============================================================================
// Dispatched single dot
// ============================================================================

inline std::int64_t binary_dot(
    const word_t* a,
    const word_t* b,
    std::size_t words,
    std::size_t valid_bits) noexcept
{
    switch (simd_level())
    {
        case SimdLevel::AVX512_VPOPCNTDQ:

#if defined(__GNUC__) || defined(__clang__) || \
    (defined(_MSC_VER) && \
     defined(__AVX512F__) && \
     defined(__AVX512VPOPCNTDQ__))

            return dot_avx512(
                a,
                b,
                words,
                valid_bits);

#else

            return dot_scalar(
                a,
                b,
                words,
                valid_bits);

#endif

        case SimdLevel::AVX2:

#if defined(__GNUC__) || defined(__clang__) || \
    (defined(_MSC_VER) && defined(__AVX2__))

            return dot_avx2(
                a,
                b,
                words,
                valid_bits);

#else

            return dot_scalar(
                a,
                b,
                words,
                valid_bits);

#endif

        default:

            return dot_scalar(
                a,
                b,
                words,
                valid_bits);
    }
}


// ============================================================================
// Dispatched four-way dot
// ============================================================================

inline void binary_dot_4(
    const word_t* activation,
    const word_t* w0,
    const word_t* w1,
    const word_t* w2,
    const word_t* w3,
    std::size_t words,
    std::size_t valid_bits,
    std::int64_t& d0,
    std::int64_t& d1,
    std::int64_t& d2,
    std::int64_t& d3) noexcept
{
    switch (simd_level())
    {
        case SimdLevel::AVX512_VPOPCNTDQ:

#if defined(__GNUC__) || defined(__clang__) || \
    (defined(_MSC_VER) && \
     defined(__AVX512F__) && \
     defined(__AVX512VPOPCNTDQ__))

            binary_dot_4_avx512(
                activation,
                w0,
                w1,
                w2,
                w3,
                words,
                valid_bits,
                d0,
                d1,
                d2,
                d3);

            return;

#endif

            break;


        case SimdLevel::AVX2:

#if defined(__GNUC__) || defined(__clang__) || \
    (defined(_MSC_VER) && defined(__AVX2__))

            binary_dot_4_avx2(
                activation,
                w0,
                w1,
                w2,
                w3,
                words,
                valid_bits,
                d0,
                d1,
                d2,
                d3);

            return;

#endif

            break;

        default:
            break;
    }

    d0 = binary_dot(
        activation,
        w0,
        words,
        valid_bits);

    d1 = binary_dot(
        activation,
        w1,
        words,
        valid_bits);

    d2 = binary_dot(
        activation,
        w2,
        words,
        valid_bits);

    d3 = binary_dot(
        activation,
        w3,
        words,
        valid_bits);
}


inline bool dot_to_bit(
    std::int64_t dot_value) noexcept
{
    return dot_value < 0;
}


} // namespace detail


// ============================================================================
// BinaryMatrix
// ============================================================================

class BinaryMatrix
{
public:

    using value_type = word_t;
    using container_type = word_vector;

private:

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::size_t words_per_row_ = 0;

    container_type data_;


    word_t* row_ptr(
        std::size_t row) noexcept
    {
        return data_.data() +
               row * words_per_row_;
    }


    const word_t* row_ptr(
        std::size_t row) const noexcept
    {
        return data_.data() +
               row * words_per_row_;
    }


    void clear_padding_bits() noexcept
    {
        if (rows_ == 0 ||
            cols_ == 0 ||
            words_per_row_ == 0)
        {
            return;
        }

        const std::size_t remainder =
            cols_ & 63;

        if (remainder == 0)
            return;

        const word_t mask =
            detail::low_bits_mask(remainder);

        for (std::size_t r = 0;
             r < rows_;
             ++r)
        {
            row_ptr(r)[
                words_per_row_ - 1] &= mask;
        }
    }


    void check_index(
        std::size_t r,
        std::size_t c) const
    {
        if (r >= rows_ || c >= cols_)
        {
            throw std::out_of_range(
                "BinaryMatrix index out of range");
        }
    }


public:

    BinaryMatrix() = default;


    BinaryMatrix(
        std::size_t rows,
        std::size_t cols)
        : rows_(rows),
          cols_(cols),
          words_per_row_(
              detail::words_for_bits(cols)),
          data_(
              detail::checked_mul(
                  rows,
                  words_per_row_),
              0)
    {
    }


    [[nodiscard]]
    std::size_t rows() const noexcept
    {
        return rows_;
    }


    [[nodiscard]]
    std::size_t cols() const noexcept
    {
        return cols_;
    }


    [[nodiscard]]
    std::size_t words_per_row() const noexcept
    {
        return words_per_row_;
    }


    [[nodiscard]]
    std::size_t storage_words() const noexcept
    {
        return data_.size();
    }


    [[nodiscard]]
    std::size_t storage_bytes() const noexcept
    {
        return data_.size() * sizeof(word_t);
    }


    [[nodiscard]]
    word_t* data() noexcept
    {
        return data_.data();
    }


    [[nodiscard]]
    const word_t* data() const noexcept
    {
        return data_.data();
    }


    [[nodiscard]]
    word_t* row_data(
        std::size_t row)
    {
        if (row >= rows_)
            throw std::out_of_range(
                "row out of range");

        return row_ptr(row);
    }


    [[nodiscard]]
    const word_t* row_data(
        std::size_t row) const
    {
        if (row >= rows_)
            throw std::out_of_range(
                "row out of range");

        return row_ptr(row);
    }


    [[nodiscard]]
    bool get_bit(
        std::size_t row,
        std::size_t col) const
    {
        check_index(row, col);

        const std::size_t word =
            col >> 6;

        const std::size_t bit =
            col & 63;

        return
            ((row_ptr(row)[word] >> bit) & 1ULL)
            != 0;
    }


    void set_bit(
        std::size_t row,
        std::size_t col,
        bool value)
    {
        check_index(row, col);

        const std::size_t word =
            col >> 6;

        const std::size_t bit =
            col & 63;

        const word_t mask =
            word_t{1} << bit;

        if (value)
            row_ptr(row)[word] |= mask;
        else
            row_ptr(row)[word] &= ~mask;
    }


    [[nodiscard]]
    int get_value(
        std::size_t row,
        std::size_t col) const
    {
        return get_bit(row, col)
            ? -1
            : +1;
    }


    void set_value(
        std::size_t row,
        std::size_t col,
        int value)
    {
        if (value != +1 &&
            value != -1)
        {
            throw std::invalid_argument(
                "BinaryMatrix values must be +1 or -1");
        }

        set_bit(
            row,
            col,
            value == -1);
    }


    void fill_bit(bool value) noexcept
    {
        std::fill(
            data_.begin(),
            data_.end(),
            value
                ? ~word_t{0}
                : word_t{0});

        clear_padding_bits();
    }


    void fill_value(int value)
    {
        if (value != +1 &&
            value != -1)
        {
            throw std::invalid_argument(
                "BinaryMatrix values must be +1 or -1");
        }

        fill_bit(value == -1);
    }


    template <typename RNG>
    void randomize(RNG& rng)
    {
        for (word_t& word : data_)
        {
            word =
                std::uniform_int_distribution<word_t>{}(rng);
        }

        clear_padding_bits();
    }


    void randomize()
    {
        std::random_device rd;
        std::mt19937_64 rng(rd());

        randomize(rng);
    }


    [[nodiscard]]
    BinaryMatrix transpose() const
    {
        BinaryMatrix result(
            cols_,
            rows_);

        for (std::size_t r = 0;
             r < rows_;
             ++r)
        {
            const word_t* src =
                row_ptr(r);

            for (std::size_t w = 0;
                 w < words_per_row_;
                 ++w)
            {
                word_t x = src[w];

                if (w + 1 == words_per_row_ &&
                    (cols_ & 63) != 0)
                {
                    x &= detail::low_bits_mask(
                        cols_ & 63);
                }

                while (x != 0)
                {
                    const unsigned bit =
                        detail::ctz64(x);

                    const std::size_t c =
                        w * 64 + bit;

                    result.row_ptr(c)[
                        r >> 6] |=
                        word_t{1} <<
                        (r & 63);

                    x &= x - 1;
                }
            }
        }

        return result;
    }


    static BinaryMatrix from_packed(
        std::size_t rows,
        std::size_t cols,
        const word_t* packed)
    {
        BinaryMatrix result(
            rows,
            cols);

        if (!result.data_.empty())
        {
            std::memcpy(
                result.data_.data(),
                packed,
                result.storage_bytes());
        }

        result.clear_padding_bits();

        return result;
    }


    [[nodiscard]]
    word_t& word(
        std::size_t row,
        std::size_t word_index)
    {
        if (row >= rows_ ||
            word_index >= words_per_row_)
        {
            throw std::out_of_range(
                "BinaryMatrix word index out of range");
        }

        return row_ptr(row)[word_index];
    }


    [[nodiscard]]
    const word_t& word(
        std::size_t row,
        std::size_t word_index) const
    {
        if (row >= rows_ ||
            word_index >= words_per_row_)
        {
            throw std::out_of_range(
                "BinaryMatrix word index out of range");
        }

        return row_ptr(row)[word_index];
    }
};


// ============================================================================
// BinaryWeightMatrix
// ============================================================================

class BinaryWeightMatrix
{
private:

    std::size_t input_size_ = 0;
    std::size_t output_size_ = 0;
    std::size_t words_per_vector_ = 0;

    word_vector data_;


    word_t* output_ptr(
        std::size_t output) noexcept
    {
        return
            data_.data() +
            output * words_per_vector_;
    }


    const word_t* output_ptr(
        std::size_t output) const noexcept
    {
        return
            data_.data() +
            output * words_per_vector_;
    }


    void clear_padding_bits() noexcept
    {
        if (input_size_ == 0 ||
            output_size_ == 0)
        {
            return;
        }

        if ((input_size_ & 63) == 0)
            return;

        const word_t mask =
            detail::low_bits_mask(
                input_size_ & 63);

        for (std::size_t o = 0;
             o < output_size_;
             ++o)
        {
            output_ptr(o)[
                words_per_vector_ - 1] &= mask;
        }
    }


public:

    BinaryWeightMatrix() = default;


    explicit BinaryWeightMatrix(
        const BinaryMatrix& matrix)
        : input_size_(matrix.rows()),
          output_size_(matrix.cols()),
          words_per_vector_(
              detail::words_for_bits(
                  matrix.rows())),
          data_(
              detail::checked_mul(
                  matrix.cols(),
                  words_per_vector_),
              0)
    {
        for (std::size_t input = 0;
             input < input_size_;
             ++input)
        {
            const word_t* src =
                matrix.row_data(input);

            for (std::size_t w = 0;
                 w < matrix.words_per_row();
                 ++w)
            {
                word_t x = src[w];

                if (w + 1 ==
                    matrix.words_per_row() &&
                    (matrix.cols() & 63) != 0)
                {
                    x &= detail::low_bits_mask(
                        matrix.cols() & 63);
                }

                while (x != 0)
                {
                    const unsigned bit =
                        detail::ctz64(x);

                    const std::size_t output =
                        w * 64 + bit;

                    if (output < output_size_)
                    {
                        output_ptr(output)[
                            input >> 6] |=
                            word_t{1} <<
                            (input & 63);
                    }

                    x &= x - 1;
                }
            }
        }

        clear_padding_bits();
    }


    template <typename T>
    static BinaryWeightMatrix from_signed(
        const T* input,
        std::size_t input_size,
        std::size_t output_size)
    {
        BinaryWeightMatrix result;

        result.input_size_ =
            input_size;

        result.output_size_ =
            output_size;

        result.words_per_vector_ =
            detail::words_for_bits(
                input_size);

        result.data_.resize(
            detail::checked_mul(
                output_size,
                result.words_per_vector_));

        for (std::size_t i = 0;
             i < input_size;
             ++i)
        {
            for (std::size_t o = 0;
                 o < output_size;
                 ++o)
            {
                const T value =
                    input[
                        i * output_size + o];

                if (value != T{+1} &&
                    value != T{-1})
                {
                    throw std::invalid_argument(
                        "BinaryWeightMatrix expects "
                        "only +1 and -1");
                }

                if (value == T{-1})
                {
                    result.output_ptr(o)[
                        i >> 6] |=
                        word_t{1} <<
                        (i & 63);
                }
            }
        }

        result.clear_padding_bits();

        return result;
    }


    [[nodiscard]]
    std::size_t input_size() const noexcept
    {
        return input_size_;
    }


    [[nodiscard]]
    std::size_t output_size() const noexcept
    {
        return output_size_;
    }


    [[nodiscard]]
    std::size_t words_per_vector() const noexcept
    {
        return words_per_vector_;
    }


    [[nodiscard]]
    const word_t* output_data(
        std::size_t output) const
    {
        if (output >= output_size_)
        {
            throw std::out_of_range(
                "output index out of range");
        }

        return output_ptr(output);
    }


    [[nodiscard]]
    word_t* output_data(
        std::size_t output)
    {
        if (output >= output_size_)
        {
            throw std::out_of_range(
                "output index out of range");
        }

        return output_ptr(output);
    }


    [[nodiscard]]
    const word_t* data() const noexcept
    {
        return data_.data();
    }


    [[nodiscard]]
    word_t* data() noexcept
    {
        return data_.data();
    }
};


// ============================================================================
// Single dot against packed weights
// ============================================================================

inline std::int64_t dot(
    const BinaryMatrix& activation,
    std::size_t activation_row,
    const BinaryWeightMatrix& weights,
    std::size_t output)
{
    if (activation.cols() !=
        weights.input_size())
    {
        throw std::invalid_argument(
            "activation and weights dimensions do not match");
    }

    return detail::binary_dot(
        activation.row_data(activation_row),
        weights.output_data(output),
        activation.words_per_row(),
        activation.cols());
}


// ============================================================================
// Matrix multiplication
// ============================================================================
//
// IMPORTANT:
// The BinaryWeightMatrix overload is declared BEFORE the BinaryMatrix
// overload because the latter calls the former.
//
// ============================================================================

[[nodiscard]]
inline BinaryMatrix multiply(
    const BinaryMatrix& activations,
    const BinaryWeightMatrix& weights);


// ============================================================================
// BinaryMatrix x BinaryMatrix
//
//     A[M x K] * B[K x N]
//
// Binary result:
//
//     dot >= 0 -> +1 -> 0
//     dot <  0 -> -1 -> 1
// ============================================================================

[[nodiscard]]
inline BinaryMatrix multiply(
    const BinaryMatrix& A,
    const BinaryMatrix& B)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument(
            "BinaryMatrix multiply: incompatible dimensions");
    }

    BinaryWeightMatrix weights(B);

    return multiply(
        A,
        weights);
}


// ============================================================================
// High-performance BNN inference.
//
// Uses permanently packed weights.
//
// Output is packed directly into uint64_t words.
//
// IMPORTANT FIX:
//
// The result bits are placed directly at their final positions.
// ============================================================================

[[nodiscard]]
inline BinaryMatrix multiply(
    const BinaryMatrix& activations,
    const BinaryWeightMatrix& weights)
{
    if (activations.cols() !=
        weights.input_size())
    {
        throw std::invalid_argument(
            "activation and weights dimensions do not match");
    }

    BinaryMatrix result(
        activations.rows(),
        weights.output_size());

    const std::size_t output_words =
        detail::words_for_bits(
            weights.output_size());

    const std::size_t K =
        activations.cols();

    const std::size_t words =
        activations.words_per_row();

    for (std::size_t r = 0;
         r < activations.rows();
         ++r)
    {
        const word_t* activation =
            activations.row_data(r);

        word_t* output =
            result.row_data(r);

        for (std::size_t output_word = 0;
             output_word < output_words;
             ++output_word)
        {
            word_t packed_result = 0;

            const std::size_t first =
                output_word * 64;

            const std::size_t last =
                std::min(
                    first + 64,
                    weights.output_size());

            std::size_t o = first;

            for (; o + 4 <= last; o += 4)
            {
                std::int64_t d0;
                std::int64_t d1;
                std::int64_t d2;
                std::int64_t d3;

                detail::binary_dot_4(
                    activation,
                    weights.output_data(o + 0),
                    weights.output_data(o + 1),
                    weights.output_data(o + 2),
                    weights.output_data(o + 3),
                    words,
                    K,
                    d0,
                    d1,
                    d2,
                    d3);

                if (d0 < 0)
                    packed_result |=
                        word_t{1} << (o - first + 0);

                if (d1 < 0)
                    packed_result |=
                        word_t{1} << (o - first + 1);

                if (d2 < 0)
                    packed_result |=
                        word_t{1} << (o - first + 2);

                if (d3 < 0)
                    packed_result |=
                        word_t{1} << (o - first + 3);
            }

            for (; o < last; ++o)
            {
                const std::int64_t d =
                    detail::binary_dot(
                        activation,
                        weights.output_data(o),
                        words,
                        K);

                if (d < 0)
                {
                    packed_result |=
                        word_t{1} <<
                        (o - first);
                }
            }

            output[output_word] =
                packed_result;
        }
    }

    return result;
}


// ============================================================================
// Integer accumulator multiplication
// ============================================================================

inline void multiply_accumulate(
    const BinaryMatrix& activations,
    const BinaryWeightMatrix& weights,
    std::vector<std::int32_t>& output)
{
    if (activations.cols() !=
        weights.input_size())
    {
        throw std::invalid_argument(
            "activation and weights dimensions do not match");
    }

    const std::size_t M =
        activations.rows();

    const std::size_t N =
        weights.output_size();

    const std::size_t K =
        activations.cols();

    output.resize(
        detail::checked_mul(
            M,
            N));

    for (std::size_t r = 0;
         r < M;
         ++r)
    {
        const word_t* activation =
            activations.row_data(r);

        for (std::size_t o = 0;
             o < N;
             ++o)
        {
            const std::int64_t d =
                detail::binary_dot(
                    activation,
                    weights.output_data(o),
                    activations.words_per_row(),
                    K);

            output[
                r * N + o] =
                static_cast<std::int32_t>(d);
        }
    }
}


// ============================================================================
// Integer accumulator using ordinary BinaryMatrix weights
// ============================================================================

inline void multiply_accumulate(
    const BinaryMatrix& A,
    const BinaryMatrix& B,
    std::vector<std::int32_t>& output)
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument(
            "BinaryMatrix multiply_accumulate: "
            "incompatible dimensions");
    }

    BinaryWeightMatrix weights(B);

    multiply_accumulate(
        A,
        weights,
        output);
}


// ============================================================================
// Conversion from ordinary +/-1 data
// ============================================================================

template <typename T>
[[nodiscard]]
BinaryMatrix from_signed(
    const T* input,
    std::size_t rows,
    std::size_t cols)
{
    BinaryMatrix result(
        rows,
        cols);

    for (std::size_t r = 0;
         r < rows;
         ++r)
    {
        for (std::size_t c = 0;
             c < cols;
             ++c)
        {
            const T value =
                input[
                    r * cols + c];

            if (value != T{+1} &&
                value != T{-1})
            {
                throw std::invalid_argument(
                    "from_signed expects "
                    "only +1 and -1");
            }

            if (value == T{-1})
            {
                result.set_bit(
                    r,
                    c,
                    true);
            }
        }
    }

    return result;
}


// ============================================================================
// Conversion to ordinary +/-1 data
// ============================================================================

template <typename T = std::int8_t>
[[nodiscard]]
std::vector<T> to_signed(
    const BinaryMatrix& matrix)
{
    std::vector<T> result(
        detail::checked_mul(
            matrix.rows(),
            matrix.cols()));

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
                    ? T{-1}
                    : T{+1};
        }
    }

    return result;
}


// ============================================================================
// Packed word count
// ============================================================================

[[nodiscard]]
inline std::size_t packed_words(
    std::size_t bits) noexcept
{
    return detail::words_for_bits(bits);
}


// ============================================================================
// SIMD information
// ============================================================================

inline const char* simd_backend() noexcept
{
    return detail::simd_level_name();
}


inline bool has_avx2() noexcept
{
    return
        detail::simd_level() ==
            detail::SimdLevel::AVX2 ||
        detail::simd_level() ==
            detail::SimdLevel::AVX512_VPOPCNTDQ;
}


inline bool has_avx512_vpopcntdq() noexcept
{
    return
        detail::simd_level() ==
        detail::SimdLevel::AVX512_VPOPCNTDQ;
}


} // namespace bnn

// ============================================================================
// APPENDED DOCUMENTATION
// ============================================================================
//
// This section is intentionally kept inside C++ comments so that this single
// header file is self-documenting and does not require a separate README.
//
// ============================================================================
//
// OVERVIEW
// ============================================================================
//
// binary_matrix.hpp provides packed binary matrix operations for Binary Neural
// Networks (BNNs).
//
// Values are represented using one bit:
//
//     bit 0 -> +1
//     bit 1 -> -1
//
// For two binary values:
//
//     +1 * +1 = +1
//     -1 * -1 = +1
//     +1 * -1 = -1
//     -1 * +1 = -1
//
// Therefore multiplication can be implemented using XOR:
//
//     equal bits    -> +1
//     different     -> -1
//
// For K binary values:
//
//     dot(a,b) = K - 2 * popcount(a XOR b)
//
// This allows binary neural-network matrix multiplication to replace large
// numbers of floating-point multiplications with bitwise operations and
// population counts.
//
// ============================================================================
//
// FEATURES
// ============================================================================
//
// The implementation provides:
//
//   - Arbitrary matrix dimensions.
//   - 64 binary values packed into each uint64_t.
//   - 64-byte aligned dynamic storage.
//   - Runtime CPU feature detection.
//   - Portable scalar fallback.
//   - AVX2 kernels.
//   - AVX-512 VPOPCNTDQ kernels.
//   - Four-output-at-a-time SIMD kernels.
//   - Permanently packed/transposed weight storage.
//   - Direct packed binary output.
//   - Integer accumulator output.
//   - Conversion between packed binary and ordinary +/-1 arrays.
//   - C++17 compatibility.
//   - No mandatory -march=native requirement.
//
// ============================================================================
//
// BASIC COMPILATION
// ============================================================================
//
// Recommended:
//
//     g++ -O3 -std=c++17 main.cpp -o bnn
//
// The implementation performs runtime CPU detection, so compiling without
// -march=native is supported.
//
// When the deployment target is known and compiler-specific optimization is
// desired:
//
//     g++ -O3 -march=native -std=c++17 main.cpp -o bnn
//
// On MSVC, compile as C++17 or later with optimization enabled.
//
// ============================================================================
//
// BASIC DATA MODEL
// ============================================================================
//
// BinaryMatrix stores:
//
//     rows x cols
//
// binary values.
//
// Each uint64_t stores 64 binary values:
//
//     bit 0   -> element 0
//     bit 1   -> element 1
//     ...
//     bit 63  -> element 63
//
// The least-significant bit corresponds to the first element in each packed
// group.
//
// A zero bit represents +1.
//
// A one bit represents -1.
//
// Example:
//
//     +1 +1 -1 +1 -1
//
// is represented conceptually as:
//
//     0  0  1  0  1
//
// ============================================================================
//
// BINARY DOT PRODUCT
// ============================================================================
//
// Given:
//
//     a = [+1, -1, +1, -1]
//     b = [+1, +1, -1, -1]
//
// The products are:
//
//     [+1, -1, -1, +1]
//
// Therefore:
//
//     dot(a,b) = 0
//
// The packed implementation computes:
//
//     xor = packed_a XOR packed_b
//
// followed by:
//
//     mismatches = popcount(xor)
//
// and finally:
//
//     dot = K - 2 * mismatches
//
// This works because every equal pair contributes +1 and every differing pair
// contributes -1.
//
// ============================================================================
//
// BINARY MATRIX MULTIPLICATION
// ============================================================================
//
// For:
//
//     A[M x K] * B[K x N]
//
// the result contains M x N binary dot products.
//
// The ordinary BinaryMatrix overload is:
//
//     BinaryMatrix C = bnn::multiply(A, B);
//
// The result is binary:
//
//     dot >= 0 -> +1
//     dot <  0 -> -1
//
// Internally the negative result is stored as bit 1 and the positive result
// as bit 0.
//
// ============================================================================
//
// PRE-PACKED WEIGHTS
// ============================================================================
//
// For repeated inference, BinaryWeightMatrix should generally be used.
//
// A normal BinaryMatrix stores rows:
//
//     [input_dimension][output_dimension]
//
// BinaryWeightMatrix permanently reorganizes the data so that every output
// neuron has its complete packed input vector stored contiguously.
//
// This avoids repeatedly transposing or reorganizing weights during inference.
//
// Construction:
//
//     bnn::BinaryWeightMatrix weights(B);
//
// where B has dimensions:
//
//     K x N
//
// Then:
//
//     bnn::BinaryMatrix result =
//         bnn::multiply(activations, weights);
//
// The same packed weights can be reused for many activation matrices.
//
// ============================================================================
//
// EXAMPLE: SIMPLE BINARY MATRIX MULTIPLICATION
// ============================================================================
//
//     #include "binary_matrix.hpp"
//
//     int main()
//     {
//         bnn::BinaryMatrix A(2, 4);
//         bnn::BinaryMatrix B(4, 3);
//
//         A.set_value(0, 0, +1);
//         A.set_value(0, 1, -1);
//         A.set_value(0, 2, +1);
//         A.set_value(0, 3, -1);
//
//         B.set_value(0, 0, +1);
//         B.set_value(1, 0, -1);
//         B.set_value(2, 0, +1);
//         B.set_value(3, 0, -1);
//
//         bnn::BinaryMatrix C =
//             bnn::multiply(A, B);
//
//         return 0;
//     }
//
// ============================================================================
//
// EXAMPLE: RANDOM BINARY MATRICES
// ============================================================================
//
//     #include "binary_matrix.hpp"
//     #include <random>
//
//     int main()
//     {
//         std::mt19937_64 rng(12345);
//
//         bnn::BinaryMatrix A(128, 256);
//         bnn::BinaryMatrix B(256, 512);
//
//         A.randomize(rng);
//         B.randomize(rng);
//
//         bnn::BinaryMatrix C =
//             bnn::multiply(A, B);
//
//         return 0;
//     }
//
// ============================================================================
//
// EXAMPLE: PRE-PACKING WEIGHTS
// ============================================================================
//
//     bnn::BinaryMatrix weights_matrix(256, 512);
//
//     weights_matrix.randomize();
//
//     bnn::BinaryWeightMatrix weights(
//         weights_matrix);
//
//     bnn::BinaryMatrix activations(
//         32,
//         256);
//
//     activations.randomize();
//
//     bnn::BinaryMatrix output =
//         bnn::multiply(
//             activations,
//             weights);
//
// The BinaryWeightMatrix can then be reused:
//
//     output1 = bnn::multiply(activations1, weights);
//     output2 = bnn::multiply(activations2, weights);
//     output3 = bnn::multiply(activations3, weights);
//
// This is the preferred pattern for repeated BNN inference.
//
// ============================================================================
//
// INTEGER ACCUMULATORS
// ============================================================================
//
// If the signed dot-product values are required instead of immediately
// converting them back into binary values, use multiply_accumulate:
//
//     std::vector<std::int32_t> output;
//
//     bnn::multiply_accumulate(
//         activations,
//         weights,
//         output);
//
// For:
//
//     activations = M x K
//     weights     = K x N
//
// the resulting vector has:
//
//     M * N
//
// elements stored in row-major order:
//
//     output[r * N + o]
//
// Each value is the signed binary dot product:
//
//     [-K, -K+2, ..., K-2, K]
//
// ============================================================================
//
// DIRECT DOT PRODUCT
// ============================================================================
//
// A single activation row can be compared with a single packed output:
//
//     std::int64_t value =
//         bnn::dot(
//             activations,
//             activation_row,
//             weights,
//             output);
//
// For example:
//
//     std::int64_t d =
//         bnn::dot(
//             A,
//             0,
//             weights,
//             7);
//
// computes the dot product between row 0 of A and output neuron 7.
//
// ============================================================================
//
// CONVERTING FROM +/-1 DATA
// ============================================================================
//
// Ordinary signed data can be converted into a BinaryMatrix.
//
// Example:
//
//     const std::int8_t values[] =
//     {
//         +1, -1, +1, -1,
//         -1, +1, +1, -1
//     };
//
//     bnn::BinaryMatrix matrix =
//         bnn::from_signed(
//             values,
//             2,
//             4);
//
// Only +1 and -1 are accepted.
//
// Any other value throws std::invalid_argument.
//
// ============================================================================
//
// CONVERTING BACK TO +/-1 DATA
// ============================================================================
//
// A BinaryMatrix can be converted back into an ordinary signed vector:
//
//     std::vector<std::int8_t> values =
//         bnn::to_signed<std::int8_t>(matrix);
//
// By default, to_signed uses int8_t.
//
// Each zero bit becomes +1.
//
// Each one bit becomes -1.
//
// ============================================================================
//
// ACCESSING INDIVIDUAL VALUES
// ============================================================================
//
// Read a binary value:
//
//     bool negative =
//         matrix.get_bit(row, column);
//
// A false bit represents +1.
//
// A true bit represents -1.
//
// To read the signed representation:
//
//     int value =
//         matrix.get_value(row, column);
//
// To modify the packed representation:
//
//     matrix.set_bit(row, column, true);
//
// or:
//
//     matrix.set_value(row, column, -1);
//
// Only +1 and -1 are accepted by set_value.
//
// ============================================================================
//
// RANDOMIZATION
// ============================================================================
//
// BinaryMatrix can be filled with random packed bits:
//
//     matrix.randomize();
//
// Or a caller-supplied RNG can be used:
//
//     std::mt19937_64 rng(1234);
//
//     matrix.randomize(rng);
//
// Padding bits outside the logical matrix dimensions are automatically cleared.
//
// ============================================================================
//
// TRANSPOSE
// ============================================================================
//
// BinaryMatrix supports:
//
//     BinaryMatrix T = matrix.transpose();
//
// If matrix has dimensions:
//
//     M x N
//
// the returned matrix has dimensions:
//
//     N x M
//
// The operation works directly on the packed representation.
//
// ============================================================================
//
// PACKED STORAGE
// ============================================================================
//
// The number of uint64_t words required for a logical bit count can be queried
// using:
//
//     std::size_t words =
//         bnn::packed_words(bits);
//
// For example:
//
//     bnn::packed_words(64)  == 1
//     bnn::packed_words(65)  == 2
//     bnn::packed_words(128) == 2
//
// The final word may contain unused padding bits when the logical dimension is
// not a multiple of 64.
//
// Those padding bits are cleared by the matrix implementation.
//
// ============================================================================
//
// SIMD DISPATCH
// ============================================================================
//
// The implementation detects CPU capabilities at runtime.
//
// The available backend is exposed through:
//
//     const char* backend =
//         bnn::simd_backend();
//
// Possible results are:
//
//     "Scalar"
//     "AVX2"
//     "AVX-512 VPOPCNTDQ"
//
// AVX2 is used when available.
//
// AVX-512 VPOPCNTDQ is selected when the required CPU and operating-system
// state are available.
//
// Otherwise the scalar implementation is used.
//
// ============================================================================
//
// AVX2
// ============================================================================
//
// The AVX2 implementation processes four uint64_t words at a time using
// 256-bit registers.
//
// The four-output kernel also computes four output neurons simultaneously:
//
//     binary_dot_4(...)
//
// This reduces repeated activation loads when several output neurons are
// processed together.
//
// ============================================================================
//
// AVX-512 VPOPCNTDQ
// ============================================================================
//
// On CPUs supporting AVX-512 VPOPCNTDQ, the implementation uses:
//
//     _mm512_popcnt_epi64
//
// to calculate eight 64-bit population counts simultaneously.
//
// The four-output kernel similarly evaluates four output vectors in parallel.
//
// Runtime dispatch ensures that these instructions are only used when the
// detected CPU and operating-system state support them.
//
// ============================================================================
//
// RUNTIME DISPATCH AND COMPILER FLAGS
// ============================================================================
//
// GCC and Clang use function-level target attributes for the SIMD kernels:
//
//     target("avx2")
//
// and:
//
//     target("avx512f,avx512vpopcntdq")
//
// This allows a single binary to contain multiple implementations without
// requiring the entire application to be compiled for AVX2 or AVX-512.
//
// Consequently:
//
//     g++ -O3 -std=c++17 main.cpp -o bnn
//
// is sufficient for runtime dispatch on supported GCC/Clang configurations.
//
// Using:
//
//     -march=native
//
// can still allow the compiler to optimize surrounding application code for
// the local CPU.
//
// ============================================================================
//
// MEMORY ALIGNMENT
// ============================================================================
//
// Packed storage uses a custom allocator with:
//
//     SIMD_ALIGNMENT = 64
//
// This provides 64-byte aligned base allocations.
//
// SIMD kernels themselves use unaligned loads where appropriate, so the
// implementation does not rely on every individual pointer being aligned to
// the SIMD register width.
//
// ============================================================================
//
// MEMORY REPRESENTATION OF WEIGHTS
// ============================================================================
//
// BinaryWeightMatrix stores output vectors contiguously:
//
//     output 0 -> packed input vector
//     output 1 -> packed input vector
//     output 2 -> packed input vector
//     ...
//
// If the input dimension is K, each output requires:
//
//     ceil(K / 64)
//
// uint64_t words.
//
// Therefore the storage requirement is approximately:
//
//     N * ceil(K / 64) * 8 bytes
//
// for a K-input, N-output layer.
//
// ============================================================================
//
// COMPLEXITY
// ============================================================================
//
// A conventional binary dot product conceptually performs K elementwise
// multiplications.
//
// The packed implementation performs approximately:
//
//     ceil(K / 64)
//
// XOR operations and population counts.
//
// Therefore the number of logical binary operations represented by each
// machine-word operation is up to 64.
//
// SIMD implementations process multiple packed words simultaneously.
//
// ============================================================================
//
// OUTPUT THRESHOLD
// ============================================================================
//
// Binary multiplication uses the following threshold:
//
//     dot >= 0 -> +1
//     dot <  0 -> -1
//
// Equivalently, the result bit is:
//
//     0 -> +1
//     1 -> -1
//
// In particular, a dot product of exactly zero produces +1.
//
// If a different threshold or activation function is required, use:
//
//     multiply_accumulate()
//
// and apply the desired post-processing externally.
//
// ============================================================================
//
// INTEGER RANGE
// ============================================================================
//
// Dot products are accumulated internally as int64_t.
//
// For K binary inputs:
//
//     -K <= dot <= K
//
// This is sufficient for ordinary matrix dimensions represented by the
// implementation.
//
// multiply_accumulate() exposes the final values as int32_t.
//
// Applications using extremely large dimensions should ensure that their
// intended K values fit comfortably within int32_t before using the
// accumulator interface.
//
// ============================================================================
//
// THREADING
// ============================================================================
//
// This header does not create worker threads itself.
//
// Matrix multiplication is currently performed by the calling thread.
//
// Applications can parallelize independent rows, batches, or layers externally
// if desired.
//
// The packed BinaryWeightMatrix is suitable for read-only sharing between
// concurrent inference workers because the multiplication routines do not
// modify the weights.
//
// ============================================================================
//
// ERROR HANDLING
// ============================================================================
//
// Invalid dimensions throw std::invalid_argument.
//
// Examples include:
//
//     A.cols() != B.rows()
//
// or:
//
//     activations.cols() != weights.input_size()
//
// Invalid matrix indices throw std::out_of_range.
//
// Invalid binary values throw std::invalid_argument.
//
// Allocation-size overflow is checked and can throw:
//
//     std::length_error
//
// or:
//
//     std::bad_array_new_length
//
// / std::bad_alloc
//
// depending on the failure mode.
//
// ============================================================================
//
// IMPORTANT SEMANTIC DETAIL
// ============================================================================
//
// The packed representation is:
//
//     bit 0 -> +1
//     bit 1 -> -1
//
// This convention must be preserved when importing weights or activations
// generated by external code.
//
// In particular, this is NOT the convention:
//
//     bit 0 -> -1
//     bit 1 -> +1
//
// Mixing conventions between activation and weight generation will invert the
// interpretation of the data.
//
// ============================================================================
//
// PADDING BITS
// ============================================================================
//
// If a dimension is not divisible by 64, the final uint64_t contains unused
// bits.
//
// For example, a 70-bit vector requires two words:
//
//     word 0 -> bits 0..63
//     word 1 -> bits 64..69 are valid
//               bits 70..127 are padding
//
// Padding bits are cleared by the matrix implementation.
//
// Dot-product kernels also mask the final word when necessary so that padding
// bits cannot contribute to the result.
//
// ============================================================================
//
// API SUMMARY
// ============================================================================
//
// Main types:
//
//     bnn::BinaryMatrix
//     bnn::BinaryWeightMatrix
//
// Main operations:
//
//     bnn::multiply(A, B)
//
//     bnn::multiply(A, packed_weights)
//
//     bnn::multiply_accumulate(
//         A,
//         packed_weights,
//         output)
//
//     bnn::dot(
//         activation,
//         activation_row,
//         weights,
//         output)
//
// Conversion:
//
//     bnn::from_signed(...)
//     bnn::to_signed(...)
//
// Utilities:
//
//     bnn::packed_words(bits)
//
// SIMD information:
//
//     bnn::simd_backend()
//     bnn::has_avx2()
//     bnn::has_avx512_vpopcntdq()
//
// Matrix operations:
//
//     matrix.transpose()
//     matrix.get_bit()
//     matrix.set_bit()
//     matrix.get_value()
//     matrix.set_value()
//     matrix.randomize()
//
// ============================================================================
//
// MINIMAL END-TO-END EXAMPLE
// ============================================================================
//
//     #include "binary_matrix.hpp"
//     #include <iostream>
//
//     int main()
//     {
//         constexpr std::size_t K = 1024;
//         constexpr std::size_t N = 256;
//
//         bnn::BinaryMatrix weights_matrix(K, N);
//         weights_matrix.randomize();
//
//         bnn::BinaryWeightMatrix weights(
//             weights_matrix);
//
//         bnn::BinaryMatrix input(1, K);
//         input.randomize();
//
//         bnn::BinaryMatrix output =
//             bnn::multiply(
//                 input,
//                 weights);
//
//         std::cout
//             << "SIMD backend: "
//             << bnn::simd_backend()
//             << '\\n';
//
//         std::cout
//             << "Output dimensions: "
//             << output.rows()
//             << " x "
//             << output.cols()
//             << '\\n';
//
//         return 0;
//     }
//
// ============================================================================
//
// END OF DOCUMENTATION
// ============================================================================
