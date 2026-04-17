#pragma once

/**
 * @file parallel_fill.h
 * @brief Declares a backend-portable parallel fill utility parameterized by execution policy.
 *
 * @details
 * This header defines @ref atlas::parallel_fill, a small utility that fills a
 * contiguous or iterator-addressable range with a constant value using a
 * backend-dependent execution strategy selected at compile time through
 * @ref ExecutionPolicy.
 *
 * ## Purpose
 * The function provides a uniform Atlas-level interface for range filling across
 * different execution environments:
 * - host-parallel execution,
 * - device-oriented execution,
 * - serial fallback execution.
 *
 * This allows higher-level code to remain backend-agnostic while still using
 * appropriate implementations for the active tasking backend.
 *
 * ## Execution-policy behavior
 * The function template is specialized by the compile-time execution policy `P`:
 * - `ExecutionPolicy::host`
 *   - uses a host-parallel implementation,
 * - `ExecutionPolicy::device`
 *   - uses a device-oriented implementation,
 * - any other policy
 *   - falls back to a serial implementation.
 *
 * ## CUDA-enabled builds
 * When `ATLAS_TASKING_CUDA` is defined:
 * - host execution uses `thrust::fill(thrust::host, ...)`,
 * - device execution uses `thrust::fill(thrust::device, ...)`,
 * - serial execution uses `std::fill`.
 *
 * ## Non-CUDA builds
 * When CUDA tasking is not enabled:
 * - host execution uses a TBB-based parallel loop,
 * - device execution falls back to the host-parallel implementation,
 * - serial execution uses `std::fill`.
 *
 * ## Iterator expectations
 * The host-parallel fallback implementation in non-CUDA builds uses indexed
 * access through `first[i]`, so it expects iterators with random-access-like
 * semantics. The CUDA/Thrust path follows the iterator requirements of
 * `thrust::fill`.
 *
 * ## Empty-range behavior
 * All implementations treat an empty range as a no-op.
 *
 * ---
 */

#include <atlas/parallel/parallel_for.h>

#if defined(ATLAS_TASKING_CUDA)
namespace atlas {

namespace detail {

    /**
     * @brief Fill a range using the host execution backend in CUDA-enabled builds.
     *
     * @details
     * This implementation delegates to `thrust::fill` with the Thrust host
     * execution policy.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_host_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        thrust::fill(thrust::host, first, last, value);
    }

    /**
     * @brief Fill a range using the device execution backend in CUDA-enabled builds.
     *
     * @details
     * This implementation delegates to `thrust::fill` with the Thrust device
     * execution policy.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        thrust::fill(thrust::device, first, last, value);
    }

    /**
     * @brief Fill a range using a serial fallback implementation.
     *
     * @details
     * This implementation delegates to `std::fill`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        std::fill(first, last, value);
    }

} // namespace detail

/**
 * @brief Fill a range with a constant value according to the selected execution policy.
 *
 * @details
 * This function dispatches at compile time to the backend implementation
 * corresponding to the execution policy template parameter `P`.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> @ref detail::parallel_fill_host_impl
 * - `ExecutionPolicy::device` -> @ref detail::parallel_fill_device_impl
 * - otherwise                 -> @ref detail::parallel_fill_serial_impl
 *
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 * @param value Fill value to assign to each element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam Iterator Iterator type over the target range.
 * @tparam T Value type to assign.
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

#else

namespace atlas {

namespace detail {

    /**
     * @brief Fill a range using the host-parallel backend in non-CUDA builds.
     *
     * @details
     * This implementation computes the range length and then uses `tbb::parallel_for`
     * over index space to assign the fill value to each element.
     *
     * The implementation assumes random-access-like iterator semantics because it
     * writes elements using `first[i]`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
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
                first[i] = value;
            });
    }

    /**
     * @brief Fill a range using the device execution path in non-CUDA builds.
     *
     * @details
     * Since no dedicated device backend is available in this configuration, the
     * device path falls back to the host-parallel implementation.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        detail::parallel_fill_host_impl(first, last, value);
    }

    /**
     * @brief Fill a range using a serial fallback implementation.
     *
     * @details
     * This implementation delegates to `std::fill`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     * @param value Fill value to assign to each element.
     *
     * @tparam Iterator Iterator type over the target range.
     * @tparam T Value type to assign.
     */
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        std::fill(first, last, value);
    }

} // namespace detail

/**
 * @brief Fill a range with a constant value according to the selected execution policy.
 *
 * @details
 * This function dispatches at compile time to the backend implementation
 * corresponding to the execution policy template parameter `P`.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> @ref detail::parallel_fill_host_impl
 * - `ExecutionPolicy::device` -> @ref detail::parallel_fill_device_impl
 * - otherwise                 -> @ref detail::parallel_fill_serial_impl
 *
 * In non-CUDA builds, the device execution path falls back to the host-parallel
 * implementation.
 *
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 * @param value Fill value to assign to each element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam Iterator Iterator type over the target range.
 * @tparam T Value type to assign.
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
#endif