#pragma once

/**
 * @file macros.h
 * @brief Core portability, build-configuration, and safety macros used throughout Atlas.
 *
 * @details
 * This header centralizes small platform- and toolchain-dependent attributes so the rest of
 * the codebase can remain mostly free of compiler-specific `#if` logic while still enabling:
 *
 * - CUDA host/device qualifiers when compiled with NVCC.
 * - Cross-compiler force-inlining and loop unrolling hints.
 * - Convenience attributes such as `[[nodiscard]]`.
 * - A `restrict`-like pointer annotation for aliasing assumptions.
 * - A project-wide debug flag compatible with the standard `NDEBUG` convention.
 * - A CUDA runtime error checking macro that compiles away cleanly when CUDA is not available.
 *
 * @note
 * The macros in this file are intended to be small, pervasive building blocks. Keep additions
 * minimal and carefully documented to avoid surprising behavior across toolchains.
 */

#include <cstdio>   // std::fprintf
#include <cstdlib>  // std::abort

// =========================================================
// CUDA / compiler attributes
// =========================================================

#if defined(__CUDACC__)

/**
 * @def ATLAS_HOST
 * @brief Marks a function as callable from the host (CPU) in CUDA builds.
 *
 * @details
 * Expands to `__host__` when compiling with NVCC and is used to annotate functions
 * intended to run on the CPU in CUDA-enabled builds.
 */
#define ATLAS_HOST __host__

/**
 * @def ATLAS_DEVICE
 * @brief Marks a function as callable from the device (GPU) in CUDA builds.
 *
 * @details
 * Expands to `__device__` when compiling with NVCC and is used to annotate functions
 * intended to run on the GPU.
 */
#define ATLAS_DEVICE __device__

/**
 * @def ATLAS_ALL_DEVICE
 * @brief Marks a function as callable from both host and device in CUDA builds.
 *
 * @details
 * Expands to `__host__ __device__` when compiling with NVCC.
 */
#define ATLAS_ALL_DEVICE __host__ __device__

/**
 * @def ATLAS_FORCE_INLINE
 * @brief Strong inlining hint for CUDA device/host compilation.
 *
 * @details
 * Expands to `__forceinline__` under NVCC. This is a stronger hint than plain `inline`
 * and is useful for tiny hot-path utilities.
 *
 * @warning
 * Overuse may increase compile time and code size.
 */
#define ATLAS_FORCE_INLINE __forceinline__

/**
 * @def ATLAS_UNROLL
 * @brief Loop unrolling hint for CUDA compilation.
 *
 * @details
 * Expands to `_Pragma("unroll")`. This should typically appear immediately before a loop:
 *
 * @code
 * ATLAS_UNROLL
 * for (int i = 0; i < 4; ++i) { ... }
 * @endcode
 */
#define ATLAS_UNROLL _Pragma("unroll")

#else  // !__CUDACC__

/**
 * @def ATLAS_HOST
 * @brief No-op for non-CUDA builds.
 */
#define ATLAS_HOST

/**
 * @def ATLAS_DEVICE
 * @brief No-op for non-CUDA builds.
 */
#define ATLAS_DEVICE

/**
 * @def ATLAS_ALL_DEVICE
 * @brief No-op for non-CUDA builds.
 */
#define ATLAS_ALL_DEVICE

/**
 * @def ATLAS_UNROLL
 * @brief No-op for non-CUDA builds.
 */
#define ATLAS_UNROLL

/**
 * @def ATLAS_FORCE_INLINE
 * @brief Cross-compiler force-inline hint for non-CUDA builds.
 *
 * @details
 * - MSVC: expands to `__forceinline`
 * - GCC/Clang: expands to `inline __attribute__((always_inline))`
 * - Fallback: expands to plain `inline`
 */
#if defined(_MSC_VER)
#define ATLAS_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_FORCE_INLINE inline __attribute__((always_inline))
#else
#define ATLAS_FORCE_INLINE inline
#endif

#endif // defined(__CUDACC__)

/**
 * @def ATLAS_NODISCARD
 * @brief Convenience macro for the `[[nodiscard]]` attribute.
 *
 * @details
 * Use this for return values that should not be ignored (e.g., computed results,
 * status codes). Ignoring the return value may indicate a bug.
 */
#define ATLAS_NODISCARD [[nodiscard]]

/**
 * @def ATLAS_MAYBE_UNUSED
 * @brief Marks a variable, function, parameter, or type as intentionally unused.
 *
 * @details
 * This macro suppresses "unused" warnings in a portable way.
 *
 * - C++17 and later: expands to `[[maybe_unused]]`
 * - GCC/Clang: expands to `__attribute__((unused))`
 * - MSVC: no direct equivalent attribute for all cases, so expands to empty
 * - Fallback: expands to empty
 *
 * ### Common use cases
 * - Unused parameters in debug-only code paths
 * - Variables only used in assertions / logging
 *
 * @code
 * void foo(int x, int y ATLAS_MAYBE_UNUSED) {
 *     (void)x;
 * }
 *
 * ATLAS_MAYBE_UNUSED static int kDebugCounter = 0;
 * @endcode
 */
#if defined(__cplusplus) && (__cplusplus >= 201703L)
#define ATLAS_MAYBE_UNUSED [[maybe_unused]]
#elif defined(__GNUC__) || defined(__clang__)
#define ATLAS_MAYBE_UNUSED __attribute__((unused))
#else
#define ATLAS_MAYBE_UNUSED
#endif


/**
 * @def RESTRICT
 * @brief Compiler-specific `restrict` qualifier for pointer parameters.
 *
 * @details
 * Provides a `restrict`-like non-aliasing hint:
 * - GCC/Clang/NVCC: `__restrict__`
 * - MSVC: `__restrict`
 * - Otherwise: empty
 *
 * @warning
 * This is an optimization hint that can affect correctness if violated. Passing
 * aliased pointers where `RESTRICT` is used results in undefined behavior for
 * compilers that honor `restrict` semantics.
 */
#if defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// =========================================================
// Build configuration
// =========================================================

/**
 * @def ATLAS_DEBUG
 * @brief Build-time debug flag (0 or 1).
 *
 * @details
 * If `ATLAS_DEBUG` is not explicitly defined:
 * - When `NDEBUG` is not defined, `ATLAS_DEBUG` becomes 1.
 * - When `NDEBUG` is defined, `ATLAS_DEBUG` becomes 0.
 *
 * This aligns with the common C/C++ convention:
 * - Debug builds typically omit `NDEBUG`
 * - Release builds typically define `NDEBUG`
 */
#if !defined(ATLAS_DEBUG)
#if !defined(NDEBUG)
#define ATLAS_DEBUG 1
#else
#define ATLAS_DEBUG 0
#endif
#endif

/**
 * @def ATLAS_IF_DEBUG(code)
 * @brief Executes `code` only when `ATLAS_DEBUG == 1`.
 *
 * @param code Statement or statement block to run in debug builds.
 *
 * @details
 * In release builds (`ATLAS_DEBUG == 0`), `code` is not evaluated.
 *
 * @code
 * ATLAS_IF_DEBUG(std::fprintf(stderr, "x=%d\n", x));
 * @endcode
 */
#if ATLAS_DEBUG
#define ATLAS_IF_DEBUG(code) \
    do { code; } while (0)
#else
#define ATLAS_IF_DEBUG(code) \
    do {                     \
    } while (0)
#endif

/* =========================================================
 * ATLAS_DEVICE_CHECK (CUDA runtime error checking)
 * =========================================================
 *
 * @def ATLAS_DEVICE_CHECK(call)
 * @brief Checks a CUDA runtime call for errors and aborts on failure (host only).
 *
 * @details
 * This macro provides a light-weight and uniform way to check CUDA runtime API calls.
 *
 * ### Usage
 * @code
 * int dev = 0;
 * ATLAS_DEVICE_CHECK(cudaGetDevice(&dev));
 * ATLAS_DEVICE_CHECK(cudaDeviceSynchronize());
 * @endcode
 *
 * ### Behavior
 * - If `ATLAS_TASKING_CUDA` is defined **and** we are compiling **host code**
 *   (i.e., not inside device compilation where `__CUDA_ARCH__` is defined),
 *   then:
 *   - includes `<cuda_runtime.h>`
 *   - evaluates `(call)` once
 *   - checks the returned `cudaError_t`
 *   - on failure prints a diagnostic including file/line and the CUDA error string
 *   - terminates the process via `std::abort()`
 *
 * - Otherwise (non-CUDA builds or device compilation path):
 *   - compiles as a safe no-op that does not require CUDA headers/libraries
 *
 * @warning
 * In the no-op configuration, `call` is **not evaluated**. Do not pass expressions
 * with side effects unless you are certain CUDA checking is enabled in that build.
 *
 * @note
 * The host-only guard `!defined(__CUDA_ARCH__)` ensures the CUDA runtime API is not used
 * in device compilation, where it is unavailable.
 */
#if defined(ATLAS_TASKING_CUDA) && !defined(__CUDA_ARCH__)

    // Host-side CUDA build path: CUDA runtime API is available here.
    #include <cuda_runtime.h>

    #define ATLAS_DEVICE_CHECK(call)                                         \
        do {                                                                 \
            cudaError_t err__ = (call);                                      \
            if (err__ != cudaSuccess) {                                      \
                std::fprintf(stderr,                                         \
                    "[CUDA ERROR] %s:%d\n  %s\n",                            \
                    __FILE__, __LINE__,                                      \
                    cudaGetErrorString(err__));                              \
                std::abort();                                                \
            }                                                                \
        } while (0)

#else

    // Non-CUDA build or device compilation path: swallow expression in a way that
    // does not require CUDA headers and avoids "unused" warnings.
    #define ATLAS_DEVICE_CHECK(call) \
        do { (void)sizeof(call); } while (0)

#endif
