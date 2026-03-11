#pragma once
#include <atlas/parallel/parallel_for.h>

#if defined(ATLAS_TASKING_CUDA)
namespace atlas {

/**
 * @file parallel_fill.h
 * @brief Parallel fill algorithm with backend dispatch (Thrust host/device or serial fallback).
 *
 * @details
 * This header provides `atlas::parallel_fill<P>(first, last, value)` which fills a range
 * `[first, last)` with `value` using the backend implied by:
 * - The build configuration (`ATLAS_TASKING_CUDA`)
 * - The compile-time execution policy `P` (`ExecutionPolicy::host`, `device`, or `serial`)
 *
 * CUDA build behavior:
 * - `ExecutionPolicy::host`   -> `thrust::fill(thrust::host, ...)`
 * - `ExecutionPolicy::device` -> `thrust::fill(thrust::device, ...)`
 * - `ExecutionPolicy::serial` -> `std::fill(...)`
 *
 * Non-CUDA build behavior:
 * - `ExecutionPolicy::host`   -> TBB parallel fill (index-based)
 * - `ExecutionPolicy::device` -> same as host (device policy maps to CPU fallback)
 * - `ExecutionPolicy::serial` -> `std::fill(...)`
 *
 * The implementation is split into small backend helpers in `atlas::detail`.
 *
 * @tparam P        Compile-time execution policy.
 * @tparam Iterator Iterator type of the range.
 * @tparam T        Value type to assign.
 *
 * @note
 * - The helpers early-out on empty ranges.
 * - In CUDA builds, this header assumes the required Thrust headers are available
 *   through includes brought in by `atlas/parallel/parallel_for.h` (or elsewhere).
 * - In the serial path, `std::fill` requires `<algorithm>`; ensure it is included
 *   transitively or include it explicitly if needed.
 */

// ------------------------------------------------------------
// Backend implementations (CUDA build)
// ------------------------------------------------------------

namespace detail {

    /**
     * @brief Host backend fill using Thrust host execution policy.
     *
     * @param first Range begin.
     * @param last  Range end.
     * @param value Value to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_host_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return; // Empty range.
        thrust::fill(thrust::host, first, last, value);
    }

    /**
     * @brief Device backend fill using Thrust device execution policy.
     *
     * @param first Range begin (device iterator / pointer-like).
     * @param last  Range end.
     * @param value Value to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return; // Empty range.
        thrust::fill(thrust::device, first, last, value);
    }

    /**
     * @brief Serial fill fallback using `std::fill`.
     *
     * @param first Range begin.
     * @param last  Range end.
     * @param value Value to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return; // Empty range.
        std::fill(first, last, value);
    }

} // namespace detail

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

/**
 * @brief Fills `[first, last)` with `value` using the selected execution policy.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam Iterator Iterator type.
 * @tparam T        Value type.
 *
 * @param first Range begin.
 * @param last  Range end.
 * @param value Value to assign to each element.
 *
 * @details
 * Dispatches at compile time based on `P`.
 */
template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_fill_host_impl(first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_fill_device_impl(first, last, value);
    } else {
        detail::parallel_fill_serial_impl(first, last, value);
    }
}

} // namespace atlas

#else  // --------------------------------------------------------
// Non-CUDA build (TBB host backend)
// --------------------------------------------------------

namespace atlas {

/**
 * @file parallel_fill.h
 * @brief Parallel fill algorithm with backend dispatch (TBB on CPU, serial fallback).
 *
 * @details
 * In non-CUDA builds, `ExecutionPolicy::host` uses a TBB-based parallel loop, and
 * `ExecutionPolicy::device` maps to the same implementation (CPU fallback).
 *
 * This keeps call sites identical across CUDA and non-CUDA configurations.
 */

namespace detail {

    /**
     * @brief Host backend fill using TBB parallel_for (index-based).
     *
     * @details
     * Computes `n = distance(first, last)` and assigns `first[i] = value` in parallel.
     *
     * @note
     * This requires that `first[i]` is valid, i.e., the iterator is random-access.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_host_impl(Iterator first, Iterator last, const T& value) {
        using diff_t = typename std::iterator_traits<Iterator>::difference_type;

        diff_t n = std::distance(first, last);
        if (n <= 0) return;

        tbb::parallel_for<diff_t>(
            diff_t(0),
            n,
            [first, value](diff_t i) {
                first[i] = value; // Write element i.
            });
    }

    /**
     * @brief "Device" backend fill in non-CUDA builds (maps to host).
     *
     * @details
     * When CUDA is not enabled, the device policy is treated as a CPU fallback.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        detail::parallel_fill_host_impl(first, last, value);
    }

    /**
     * @brief Serial fill fallback using `std::fill`.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        std::fill(first, last, value);
    }

} // namespace detail

/**
 * @brief Fills `[first, last)` with `value` using the selected execution policy.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam Iterator Iterator type.
 * @tparam T        Value type.
 */
template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_fill_host_impl(first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_fill_device_impl(first, last, value);
    } else {
        detail::parallel_fill_serial_impl(first, last, value);
    }
}

} // namespace atlas
#endif // ATLAS_TASKING_CUDA
