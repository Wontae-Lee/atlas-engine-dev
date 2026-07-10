#pragma once

/**
 * @file macros.h
 * @brief Compiler- and compilation-mode-portable function qualifiers and helper macros.
 *
 * Every function-call-site macro here expands to the correct CUDA execution-space
 * qualifier when the translation unit is compiled by nvcc (@c __CUDACC__ defined) and to
 * nothing (or the nearest host-only equivalent) when compiled by a plain host C++
 * compiler. This lets the same headers be included from both @c .cu and @c .cpp/.h
 * translation units without duplicating declarations: a leaf type annotated with
 * @c ATLAS_ALL_DEVICE compiles as an ordinary host type in a host TU and as a callable
 * device type in a device TU.
 *
 * @note @c <cstdio> and @c <cstdlib> are included so any macro that later grows to call
 *       @c std::printf / @c std::abort in a diagnostic path has its declarations
 *       available at every include site.
 */

#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__)

/** @brief Marks a function callable from host code. Expands to CUDA @c __host__ under nvcc. */
#define ATLAS_HOST __host__

/** @brief Marks a function callable from device (kernel) code. Expands to CUDA @c __device__. */
#define ATLAS_DEVICE __device__

/**
 * @brief Marks a function callable from both host and device.
 *
 * This is the qualifier the trivially-copyable tagged-union leaves rely on: it lets the
 * same member function be invoked on the host during setup and captured by value into a
 * device lambda for use inside a kernel. Expands to @c __host__ @c __device__ under nvcc.
 */
#define ATLAS_ALL_DEVICE __host__ __device__

/** @brief Requests aggressive inlining. Expands to nvcc's @c __forceinline__. */
#define ATLAS_FORCE_INLINE __forceinline__

/** @brief Requests loop unrolling of the immediately following loop. Expands to @c \#pragma @c unroll. */
#define ATLAS_UNROLL _Pragma("unroll")

#else

/** @brief Host-compilation stand-in for @c ATLAS_HOST: no qualifier is needed off-device. */
#define ATLAS_HOST

/** @brief Host-compilation stand-in for @c ATLAS_DEVICE: expands to nothing. */
#define ATLAS_DEVICE

/**
 * @brief Host-compilation stand-in for @c ATLAS_ALL_DEVICE.
 *
 * Expands to nothing so that an @c ATLAS_ALL_DEVICE-qualified function is simply an
 * ordinary host function when the header is compiled by a non-nvcc compiler.
 */
#define ATLAS_ALL_DEVICE

/** @brief Host-compilation stand-in for @c ATLAS_UNROLL: no CUDA pragma exists, so expands to nothing. */
#define ATLAS_UNROLL

#if defined(_MSC_VER)
/** @brief Aggressive-inline request on MSVC. */
#define ATLAS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
/** @brief Aggressive-inline request on GCC/Clang: @c inline plus the always_inline attribute. */
#define ATLAS_FORCE_INLINE inline __attribute__((always_inline))
#else
/** @brief Portable fallback: plain @c inline when no compiler-specific hint is known. */
#define ATLAS_FORCE_INLINE inline
#endif

#endif

/** @brief Marks a return value that must not be discarded. Expands to @c [[nodiscard]]. */
#define ATLAS_NODISCARD [[nodiscard]]

#if defined(__cplusplus) && (__cplusplus >= 201703L)
/** @brief Suppresses unused-entity warnings via @c [[maybe_unused]] when C++17 or newer is available. */
#define ATLAS_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
/** @brief Pre-C++17 GCC/Clang fallback: the @c unused attribute. */
#define ATLAS_MAYBE_UNUSED __attribute__((unused))
#else
/** @brief Portable fallback: no annotation when the compiler offers none. */
#define ATLAS_MAYBE_UNUSED
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
/** @brief No-alias pointer qualifier. Expands to @c __restrict__ on GCC/Clang/nvcc. */
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
/** @brief No-alias pointer qualifier on MSVC: @c __restrict. */
#define RESTRICT __restrict
#else
/** @brief Portable fallback: no aliasing guarantee is expressed. */
#define RESTRICT
#endif

#if !defined(ATLAS_DEBUG)
#if !defined(NDEBUG)
/**
 * @brief Debug-build indicator, set to 1 when @c NDEBUG is absent.
 *
 * Only defined here if the build has not already forced a value. When @c NDEBUG is
 * undefined the build is treated as a debug build; otherwise @c ATLAS_DEBUG is 0.
 */
#define ATLAS_DEBUG 1
#else
/** @brief Debug-build indicator, set to 0 for release builds (@c NDEBUG defined). */
#define ATLAS_DEBUG 0
#endif
#endif

#if ATLAS_DEBUG
/**
 * @brief Runs @p code only in debug builds.
 *
 * The body is wrapped in a @c do/while(0) so the macro is a single statement that
 * requires a trailing semicolon and composes correctly inside unbraced @c if/else.
 *
 * @param code Statement(s) to execute when @c ATLAS_DEBUG is 1.
 */
#define ATLAS_IF_DEBUG(code) \
    do { code; } while (0)
#else
/**
 * @brief Release-build form of @c ATLAS_IF_DEBUG: expands to an empty @c do/while(0).
 *
 * @param code Ignored in release builds; the argument is not evaluated.
 */
#define ATLAS_IF_DEBUG(code) \
    do {                     \
    } while (0)
#endif