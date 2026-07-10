#pragma once

#include <atlas/core/macros.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <type_traits>

namespace atlas {

/**
 * @brief Selects where a parallel algorithm executes and which backend runs it.
 *
 * Every algorithm in this header is templated on an ExecutionPolicy non-type
 * parameter and dispatches at compile time (`if constexpr`) to the matching
 * Thrust execution policy, so the branch for the unused backends is never
 * instantiated. This keeps a single call site usable from both host and device
 * builds without runtime overhead.
 */
enum class ExecutionPolicy {
    serial, ///< Single-threaded on the host (`thrust::seq`); deterministic, no parallelism.
    host,   ///< Multi-threaded on the host (`thrust::host`); CPU parallel backend.
    device  ///< On the GPU (`thrust::device`); the callable must be device-callable.
};

/**
 * @brief SFINAE alias that enables an overload only for integral index types.
 *
 * Resolves to `int` (a valid non-type template argument) when @p T is integral
 * and is otherwise ill-formed, removing the counting-range `parallel_for`
 * overload from the candidate set. It is what disambiguates the index-range
 * overload from the iterator-range overload below.
 *
 * @tparam T The candidate index type being tested for integrality.
 */
template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral_v<T>, int>;

/**
 * @brief Applies @p func to every index in the half-open range [start, end).
 *
 * Backed by a Thrust counting iterator, so no index buffer is materialized; the
 * indices are generated on the fly. The callable is invoked once per index with
 * that index as its sole argument. On the device policy @p func must be a
 * `__device__`-callable object (typically an extended `__host__ __device__`
 * lambda) capturing everything it needs by value.
 *
 * @tparam P         Execution backend to run on.
 * @tparam IndexType Integral index type; also the type passed to @p func.
 * @tparam Function  Callable invocable as `func(IndexType)`.
 * @param start Inclusive lower bound of the iteration range.
 * @param end   Exclusive upper bound of the iteration range.
 * @param func  Body invoked for each index; taken by const reference.
 * @note Returns immediately (no work, no launch) when the range is empty or
 *       inverted (`start >= end`).
 */
template <ExecutionPolicy P, typename IndexType, typename Function, is_integral_index<IndexType> = 0>
ATLAS_FORCE_INLINE void
parallel_for(IndexType start, IndexType end, const Function& func) {
    if (start >= end) return;

    const auto first = thrust::make_counting_iterator<IndexType>(start);
    const auto last  = thrust::make_counting_iterator<IndexType>(end);

    if constexpr (P == ExecutionPolicy::host) {
        thrust::for_each(thrust::host, first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::for_each(thrust::device, first, last, func);
    } else {
        thrust::for_each(thrust::seq, first, last, func);
    }
}

/**
 * @brief Applies @p func to each element in the iterator range [first, last).
 *
 * The counterpart to the index-range overload, selected when the range is given
 * by iterators rather than integral bounds (the `!is_integral` SFINAE guard
 * keeps the two overloads mutually exclusive). Each dereferenced element is
 * passed to @p func. For the device policy the iterators must refer to
 * device-resident memory and @p func must be device-callable.
 *
 * @tparam P        Execution backend to run on.
 * @tparam InputIt  Iterator type; deduced, and required to be non-integral.
 * @tparam Function Callable invocable with the iterator's value type.
 * @param first Beginning of the range.
 * @param last  One past the end of the range.
 * @param func  Body invoked for each element; taken by const reference.
 * @note Returns immediately when the range is empty (`first == last`).
 */
template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
ATLAS_FORCE_INLINE void
parallel_for(InputIt first, InputIt last, const Function& func) {
    if (first == last) return;

    if constexpr (P == ExecutionPolicy::host) {
        thrust::for_each(thrust::host, first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::for_each(thrust::device, first, last, func);
    } else {
        thrust::for_each(thrust::seq, first, last, func);
    }
}

}