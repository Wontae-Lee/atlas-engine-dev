#pragma once
#include <atlas/core/macros.h>
#include <type_traits>

namespace atlas {

/**
 * @brief Execution backend selector used by Atlas parallel dispatch helpers.
 *
 * @details
 * `ExecutionPolicy` determines which implementation path `parallel_for()`
 * chooses at compile time.
 *
 * Semantics:
 * - @ref serial : force a plain sequential loop
 * - @ref host   : use the host-parallel backend
 * - @ref device : use the device-oriented backend
 *
 * Backend mapping depends on the configured tasking system:
 * - CUDA build:
 *   - `host`   -> Thrust host execution policy
 *   - `device` -> Thrust device execution policy
 * - TBB build:
 *   - `host`   -> TBB parallel execution
 *   - `device` -> currently mapped to the same TBB implementation
 */
enum class ExecutionPolicy {
    serial, ///< Run work sequentially on the calling thread.
    host,   ///< Run work through the configured host-parallel backend.
    device  ///< Run work through the configured device/backend-parallel path.
};

/**
 * @brief SFINAE helper that enables an overload only for integral index types.
 *
 * @tparam T Candidate index type.
 *
 * @details
 * This alias is used to constrain the index-based `parallel_for()` overloads
 * so they participate in overload resolution only when `T` is an integral type.
 *
 * When `T` is integral, the alias resolves to `int`.
 * Otherwise, substitution fails and the overload is removed.
 */
template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral_v<T>, int>;

} // namespace atlas

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

namespace detail {

    /**
     * @brief Host-parallel implementation for index-range traversal under CUDA builds.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * This implementation uses Thrust host execution with counting iterators,
     * avoiding the need to materialize an explicit index container.
     *
     * No work is performed when `start >= end`.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;

        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);

        thrust::for_each(thrust::host, first, last, func);
    }

    /**
     * @brief Device-parallel implementation for index-range traversal under CUDA builds.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * This implementation uses Thrust device execution with counting iterators.
     * Each logical index is generated virtually and passed to `func`.
     *
     * No work is performed when `start >= end`.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;

        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);

        thrust::for_each(thrust::device, first, last, func);
    }

    /**
     * @brief Serial implementation for index-range traversal.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * This is the fallback sequential loop used when the selected execution
     * policy is @ref atlas::ExecutionPolicy::serial.
     *
     * No work is performed when `start >= end`.
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
     * @brief Host-parallel implementation for iterator-range traversal under CUDA builds.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * This implementation forwards directly to `thrust::for_each` with the
     * Thrust host execution policy.
     *
     * No work is performed when `first == last`.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_host_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::host, first, last, func);
    }

    /**
     * @brief Device-parallel implementation for iterator-range traversal under CUDA builds.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * This implementation forwards directly to `thrust::for_each` with the
     * Thrust device execution policy.
     *
     * No work is performed when `first == last`.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::device, first, last, func);
    }

    /**
     * @brief Serial implementation for iterator-range traversal.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * This implementation performs a plain sequential iterator walk and applies
     * `func(*first)` for every element.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
    }

} // namespace detail

/**
 * @brief Execute a callable over an integral half-open index range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam IndexType Integral loop index type.
 * @tparam Function Callable receiving one index argument.
 *
 * @param start First index in the half-open range.
 * @param end One-past-the-last index in the half-open range.
 * @param func Callable applied to every index in `[start, end)`.
 *
 * @details
 * This overload is selected only when `IndexType` is integral.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> host-parallel implementation
 * - `ExecutionPolicy::device` -> device-parallel implementation
 * - `ExecutionPolicy::serial` -> sequential implementation
 *
 * The dispatch is resolved with `if constexpr`, so the unused branches are
 * removed at compile time.
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

/**
 * @brief Execute a callable over an iterator half-open range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Iterator type.
 * @tparam Function Callable receiving dereferenced iterator values.
 *
 * @param first First iterator in the half-open range.
 * @param last One-past-the-last iterator in the half-open range.
 * @param func Callable applied to each element in `[first, last)`.
 *
 * @details
 * This overload is enabled only when `InputIt` is not an integral type,
 * which keeps it disjoint from the index-range overload.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> host-parallel iterator implementation
 * - `ExecutionPolicy::device` -> device-parallel iterator implementation
 * - `ExecutionPolicy::serial` -> sequential iterator implementation
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

#else

#include <iterator>
#include <tbb/tbb.h>

namespace atlas {

namespace detail {

    /**
     * @brief Host-parallel implementation for index-range traversal under TBB builds.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * This implementation uses `tbb::parallel_for` with unit step size.
     *
     * No work is performed when `start >= end`.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, IndexType(1), [&](IndexType i) { func(i); });
    }

    /**
     * @brief Device-policy implementation for index-range traversal under TBB builds.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * In the TBB backend, `device` currently maps to the same implementation
     * as `host`, because execution remains CPU/TBB-based.
     *
     * No work is performed when `start >= end`.
     */
    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, IndexType(1), [&](IndexType i) { func(i); });
    }

    /**
     * @brief Serial implementation for index-range traversal.
     *
     * @tparam IndexType Integral loop index type.
     * @tparam Function Callable receiving one index at a time.
     *
     * @param start First index in the half-open range.
     * @param end One-past-the-last index in the half-open range.
     * @param func Callable applied to every index in `[start, end)`.
     *
     * @details
     * This implementation performs a plain sequential loop.
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
     * @brief Trait checking whether an iterator is random-access.
     *
     * @tparam It Iterator type.
     *
     * @details
     * The TBB iterator-range implementation relies on indexable access into
     * the iterator range, so it requires a random-access iterator category.
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief Host-parallel implementation for iterator-range traversal under TBB builds.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * This implementation requires a random-access iterator because it converts
     * the iterator range into an index range and accesses elements via `first[i]`
     * inside a `tbb::blocked_range`.
     *
     * No work is performed when `first == last`.
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
     * @brief Device-policy implementation for iterator-range traversal under TBB builds.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * In the TBB backend, `device` currently reuses the same implementation
     * as the host-parallel iterator path.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        detail::parallel_for_host_iter_impl(first, last, func);
    }

    /**
     * @brief Serial implementation for iterator-range traversal.
     *
     * @tparam InputIt Iterator type.
     * @tparam Function Callable receiving dereferenced iterator values.
     *
     * @param first First iterator in the half-open range.
     * @param last One-past-the-last iterator in the half-open range.
     * @param func Callable applied to each element in `[first, last)`.
     *
     * @details
     * This implementation performs a plain sequential iterator walk.
     */
    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
    }

} // namespace detail

/**
 * @brief Execute a callable over an integral half-open index range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam IndexType Integral loop index type.
 * @tparam Function Callable receiving one index argument.
 *
 * @param start First index in the half-open range.
 * @param end One-past-the-last index in the half-open range.
 * @param func Callable applied to every index in `[start, end)`.
 *
 * @details
 * This overload is selected only when `IndexType` is integral.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> TBB host-parallel implementation
 * - `ExecutionPolicy::device` -> TBB-mapped device-policy implementation
 * - `ExecutionPolicy::serial` -> sequential implementation
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

/**
 * @brief Execute a callable over an iterator half-open range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Iterator type.
 * @tparam Function Callable receiving dereferenced iterator values.
 *
 * @param first First iterator in the half-open range.
 * @param last One-past-the-last iterator in the half-open range.
 * @param func Callable applied to each element in `[first, last)`.
 *
 * @details
 * This overload is enabled only when `InputIt` is not an integral type.
 *
 * Under the TBB backend, the parallel iterator implementation requires
 * random-access iterators.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> host-parallel iterator implementation
 * - `ExecutionPolicy::device` -> TBB-mapped device-policy iterator implementation
 * - `ExecutionPolicy::serial` -> sequential iterator implementation
 */
template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
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
#endif