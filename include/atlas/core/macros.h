#pragma once

/**
 * @file macros.h
 * @brief Defines platform abstraction macros, compiler hints, and utility helpers.
 *
 * @details
 * This header provides a unified set of macros used across Atlas to:
 * - abstract host/device compilation differences (CPU vs CUDA),
 * - enforce inlining and optimization hints,
 * - control debug behavior,
 * - provide portable attributes (nodiscard, unused),
 * - wrap CUDA error handling safely,
 * - define deterministic hashing constants for random generation.
 *
 * The goal is to centralize all low-level compilation and platform concerns
 * so higher-level code remains clean and portable.
 *
 * ---
 */

#include <cstdio>
#include <cstdlib>

#if defined(__CUDACC__)

/**
 * @brief Marks a function as host-only (CPU callable).
 */
#define ATLAS_HOST __host__

/**
 * @brief Marks a function as device-only (GPU callable).
 */
#define ATLAS_DEVICE __device__

/**
 * @brief Marks a function as both host and device callable.
 */
#define ATLAS_ALL_DEVICE __host__ __device__

/**
 * @brief Forces aggressive inlining (CUDA).
 */
#define ATLAS_FORCE_INLINE __forceinline__

/**
 * @brief Loop unrolling hint for CUDA.
 */
#define ATLAS_UNROLL _Pragma("unroll")

#else

/**
 * @brief Host annotation (no-op on non-CUDA builds).
 */
#define ATLAS_HOST

/**
 * @brief Device annotation (no-op on non-CUDA builds).
 */
#define ATLAS_DEVICE

/**
 * @brief Host/device annotation (no-op on non-CUDA builds).
 */
#define ATLAS_ALL_DEVICE

/**
 * @brief Loop unrolling hint (disabled on CPU builds).
 */
#define ATLAS_UNROLL

/**
 * @brief Force inline depending on compiler.
 */
#if defined(_MSC_VER)
#define ATLAS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATLAS_FORCE_INLINE inline
#endif

#endif

/**
 * @brief Marks a return value as required to be used.
 */
#define ATLAS_NODISCARD [[nodiscard]]

/**
 * @brief Marks a variable as possibly unused to suppress warnings.
 */
#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define ATLAS_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_MAYBE_UNUSED __attribute__((unused))
#else
#define ATLAS_MAYBE_UNUSED
#endif

/**
 * @brief Restrict qualifier for pointer aliasing optimization.
 *
 * @details
 * Expands to compiler-specific restrict keyword to inform the compiler
 * that pointers do not alias, enabling better optimization.
 */
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

/**
 * @brief Debug mode detection.
 *
 * @details
 * Automatically enables ATLAS_DEBUG if not explicitly defined and
 * NDEBUG is not set.
 */
#if !defined(ATLAS_DEBUG)
#if !defined(NDEBUG)
#define ATLAS_DEBUG 1
#else
#define ATLAS_DEBUG 0
#endif
#endif

/**
 * @brief Execute code only in debug builds.
 *
 * @param code Code block executed only when ATLAS_DEBUG is enabled.
 */
#if ATLAS_DEBUG
#define ATLAS_IF_DEBUG(code) \
    do { code; } while (0)
#else
#define ATLAS_IF_DEBUG(code) \
    do {                     \
    } while (0)
#endif

/**
 * @brief CUDA error checking macro (host-side only).
 *
 * @details
 * Wraps CUDA API calls and aborts execution if an error occurs.
 * Provides file/line information and readable error string.
 *
 * Disabled in:
 * - device code (__CUDA_ARCH__)
 * - non-CUDA builds
 *
 * @param call CUDA API call to check.
 */
#if defined(ATLAS_TASKING_CUDA) && !defined(__CUDA_ARCH__)

#include <cuda_runtime.h>

#define ATLAS_DEVICE_CHECK(call)                       \
    do {                                               \
        cudaError_t err__ = (call);                    \
        if (err__ != cudaSuccess) {                    \
            std::fprintf(stderr,                       \
                         "[CUDA ERROR] %s:%d\n  %s\n", \
                         __FILE__,                     \
                         __LINE__,                     \
                         cudaGetErrorString(err__));   \
            std::abort();                              \
        }                                              \
    } while (0)

#else

/**
 * @brief No-op CUDA check for unsupported environments.
 */
#define ATLAS_DEVICE_CHECK(call) \
    do { (void)sizeof(call); } while (0)

#endif
