#pragma once
#include <atlas/core/macros.h>
#include <type_traits>

/**
 * @file parallel_for.h
 * @brief Unified `parallel_for` abstraction with compile-time execution policy dispatch.
 *
 * @details
 * This header provides a small parallel loop API that can target:
 * - **Serial execution** (always available)
 * - **Host parallel execution**
 * - **Device execution** (CUDA builds), with a CPU fallback in non-CUDA builds
 *
 * The dispatch is controlled by:
 * - The compile-time enum `atlas::ExecutionPolicy`
 * - The build flag `ATLAS_TASKING_CUDA`
 *
 * Two overload families are provided:
 * 1) **Index-range overload**:
 *    `parallel_for<P>(start, end, func)` where `start` and `end` are integral indices.
 *
 * 2) **Iterator-range overload**:
 *    `parallel_for<P>(first, last, func)` where `first/last` are iterators and `func`
 *    is applied to each dereferenced element.
 *
 * Backend selection:
 * - CUDA build:
 *   - `host`   -> Thrust host policy (`thrust::for_each(thrust::host, ...)`)
 *   - `device` -> Thrust device policy (`thrust::for_each(thrust::device, ...)`)
 *   - `serial` -> plain for-loop / iterator loop
 *
 * - Non-CUDA build:
 *   - `host`   -> oneTBB (`tbb::parallel_for`)
 *   - `device` -> same as host (device policy maps to CPU fallback)
 *   - `serial` -> plain for-loop / iterator loop
 *
 * @note
 * - The iterator overload on the non-CUDA path requires **random-access iterators** because it
 *   indexes with `first[i]` inside TBB blocked ranges.
 * - The iterator overload on the CUDA path delegates to `thrust::for_each` and therefore expects
 *   iterators compatible with Thrust execution policies.
 */

namespace atlas {

/**
 * @brief Execution policy used to select serial/host/device backends.
 */
enum class ExecutionPolicy {
    serial, ///< Execute serially on the calling thread.
    host,   ///< Execute in parallel on host (Thrust host or TBB).
    device  ///< Execute on device in CUDA builds; otherwise maps to host backend.
};

/**
 * @brief SFINAE helper: enables a template only if `T` is an integral type.
 *
 * @tparam T Candidate index type.
 */
template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral<T>::value, int>;

} // namespace atlas

// ============================================================
// CUDA backend (Thrust)
// ============================================================
#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

namespace detail {

    /**
     * @brief Host parallel implementation for index ranges using Thrust host policy.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;

        // Counting iterators materialize [start, end) lazily.
        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);

        // Execute on host using Thrust.
        thrust::for_each(thrust::host, first, last, func);
    }

    /**
     * @brief Device parallel implementation for index ranges using Thrust device policy.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;

        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);

        // Execute on device using Thrust.
        thrust::for_each(thrust::device, first, last, func);
    }

    /**
     * @brief Serial implementation for index ranges.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;

        for (IndexType i = start; i < end; ++i) {
            func(i);
        }
    }

    /**
     * @brief Host parallel implementation for iterator ranges using Thrust host policy.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_host_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::host, first, last, func);
    }

    /**
     * @brief Device parallel implementation for iterator ranges using Thrust device policy.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::device, first, last, func);
    }

    /**
     * @brief Serial implementation for iterator ranges.
     *
     * @details
     * Applies `func(*it)` to each element.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
    }

} // namespace detail

// ------------------------------------------------------------
// Public API: index-range overload
// ------------------------------------------------------------

/**
 * @brief Executes `func(i)` for each integer index `i` in `[start, end)`.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam IndexType Integral index type.
 * @tparam Function Callable type; should be invocable as `func(IndexType)`.
 *
 * @param start Inclusive start index.
 * @param end   Exclusive end index.
 * @param func  Function invoked per index.
 *
 * @note
 * In CUDA builds:
 * - `host` uses `thrust::for_each(thrust::host, ...)`
 * - `device` uses `thrust::for_each(thrust::device, ...)`
 */
template <ExecutionPolicy P, typename IndexType, typename Function, is_integral_index<IndexType> = 0>
ATLAS_FORCE_INLINE void
parallel_for(IndexType start, IndexType end, const Function& func) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_for_host_impl(start, end, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_for_device_impl(start, end, func);
    } else {
        detail::parallel_for_serial_impl(start, end, func);
    }
}

// ------------------------------------------------------------
// Public API: iterator-range overload
// ------------------------------------------------------------

/**
 * @brief Applies `func(x)` to each element `x` in `[first, last)`.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam InputIt  Iterator type (must be compatible with Thrust for parallel policies).
 * @tparam Function Callable type; should accept the iterator's dereference type.
 *
 * @param first Range begin.
 * @param last  Range end.
 * @param func  Function invoked per element.
 *
 * @note
 * This overload is disabled for integral `InputIt` to avoid ambiguity with the index-range overload.
 */
template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral<InputIt>::value, int> = 0>
ATLAS_FORCE_INLINE void
parallel_for(InputIt first, InputIt last, const Function& func) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_for_host_iter_impl(first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_for_device_iter_impl(first, last, func);
    } else {
        detail::parallel_for_serial_iter_impl(first, last, func);
    }
}

} // namespace atlas

// ============================================================
// Non-CUDA backend (oneTBB)
// ============================================================
#else

#include <iterator>
#include <tbb/tbb.h>

namespace atlas {

namespace detail {

    /**
     * @brief Host parallel implementation for index ranges using oneTBB.
     *
     * @details
     * Uses `tbb::parallel_for(start, end, func)` which calls `func(i)` for each `i`.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, func);
    }

    /**
     * @brief Device parallel implementation in non-CUDA builds (maps to host).
     *
     * @details
     * When CUDA is not enabled, the "device" policy is treated as a CPU parallel fallback.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, func);
    }

    /**
     * @brief Serial implementation for index ranges.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        for (IndexType i = start; i < end; ++i) {
            func(i);
        }
    }

    // ---- iterator overload support ----

    /**
     * @brief Trait to check whether an iterator is random-access.
     *
     * @details
     * Used to enforce requirements for the TBB iterator overload (it indexes with `first[i]`).
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief Host parallel implementation for iterator ranges using oneTBB blocked ranges.
     *
     * @details
     * Requires random-access iterators so that `first[i]` is valid.
     * Executes:
     * `func(first[i])` for each index `i` in `[0, n)`, where `n = distance(first, last)`.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_host_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::parallel_for (TBB iterator overload) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        tbb::parallel_for(
            tbb::blocked_range<diff_t>(0, n),
            [&](const tbb::blocked_range<diff_t>& r) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    func(first[i]);
                }
            });
    }

    /**
     * @brief Device iterator implementation in non-CUDA builds (maps to host).
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        detail::parallel_for_host_iter_impl(first, last, func);
    }

    /**
     * @brief Serial implementation for iterator ranges.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
    }

} // namespace detail

// ------------------------------------------------------------
// Public API: index-range overload
// ------------------------------------------------------------

/**
 * @brief Executes `func(i)` for each integer index `i` in `[start, end)`.
 *
 * @tparam P        Execution policy.
 * @tparam IndexType Integral index type.
 * @tparam Function Callable type invocable as `func(IndexType)`.
 */
template <ExecutionPolicy P, typename IndexType, typename Function, is_integral_index<IndexType> = 0>
ATLAS_FORCE_INLINE void
parallel_for(IndexType start, IndexType end, const Function& func) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_for_host_impl(start, end, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_for_device_impl(start, end, func);
    } else {
        detail::parallel_for_serial_impl(start, end, func);
    }
}

// ------------------------------------------------------------
// Public API: iterator-range overload
// ------------------------------------------------------------

/**
 * @brief Applies `func(x)` to each element `x` in `[first, last)`.
 *
 * @tparam P        Execution policy.
 * @tparam InputIt  Iterator type.
 * @tparam Function Callable type.
 *
 * @note
 * In the non-CUDA build, the parallel iterator overload requires random-access iterators.
 */
template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral<InputIt>::value, int> = 0>
ATLAS_FORCE_INLINE void
parallel_for(InputIt first, InputIt last, const Function& func) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_for_host_iter_impl(first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_for_device_iter_impl(first, last, func);
    } else {
        detail::parallel_for_serial_iter_impl(first, last, func);
    }
}

} // namespace atlas
#endif // ATLAS_TASKING_CUDA
