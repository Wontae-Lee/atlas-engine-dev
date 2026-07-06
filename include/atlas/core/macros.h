#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__)

#define ATLAS_HOST __host__

#define ATLAS_DEVICE __device__

#define ATLAS_ALL_DEVICE __host__ __device__

#define ATLAS_FORCE_INLINE __forceinline__

#define ATLAS_UNROLL _Pragma("unroll")

#else

#define ATLAS_HOST

#define ATLAS_DEVICE

#define ATLAS_ALL_DEVICE

#define ATLAS_UNROLL

#if defined(_MSC_VER)
#define ATLAS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATLAS_FORCE_INLINE inline
#endif

#endif

#define ATLAS_NODISCARD [[nodiscard]]

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define ATLAS_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_MAYBE_UNUSED __attribute__((unused))
#else
#define ATLAS_MAYBE_UNUSED
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#if !defined(ATLAS_DEBUG)
#if !defined(NDEBUG)
#define ATLAS_DEBUG 1
#else
#define ATLAS_DEBUG 0
#endif
#endif

#if ATLAS_DEBUG
#define ATLAS_IF_DEBUG(code) \
    do { code; } while (0)
#else
#define ATLAS_IF_DEBUG(code) \
    do {                     \
    } while (0)
#endif
