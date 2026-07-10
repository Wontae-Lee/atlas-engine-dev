#pragma once

#include <atlas/core/macros.h>

#include <type_traits>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#else
#include <algorithm>
#include <cstddef>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace atlas {

/**
 * @brief Where a parallel algorithm runs.
 *
 * The three values name intents, not hardware. Under the host backend @c device still
 * resolves — it simply runs the callable on CPU threads, which is why every device lambda
 * in the engine is annotated @c ATLAS_ALL_DEVICE rather than @c __device__.
 */
enum class ExecutionPolicy {
    serial, ///< Single-threaded on the host; deterministic, no parallelism.
    host,   ///< Multi-threaded on the host.
    device  ///< On the GPU under the CUDA backend; on host threads otherwise.
};

/** @brief Enables an overload only for an integral index type. */
template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral_v<T>, int>;

/**
 * @brief Invoke @p func once for every index in the half-open range `[start, end)`.
 *
 * The workhorse of the engine: every kernel is a `parallel_for<ExecutionPolicy::device>`
 * over a particle or cell index, capturing raw pointers by value.
 *
 * Under the CUDA backend this is a @c thrust::for_each over a counting iterator, and the
 * @c device policy launches a kernel, so @p func must be device-callable. Under the host
 * backend @c host and @c device both become a @c tbb::parallel_for over a
 * @c tbb::blocked_range, letting TBB pick the grain size.
 *
 * @tparam P Where to run.
 * @tparam IndexType Integral index type.
 * @tparam Function Callable invoked as `func(index)`.
 * @param start First index, inclusive.
 * @param end One past the last index. `end <= start` returns without invoking @p func.
 * @param func The body. Invoked concurrently, so it must not mutate shared state except
 *             through atomics or through indices it alone owns.
 */
template <ExecutionPolicy P, typename IndexType, typename Function, is_integral_index<IndexType> = 0>
ATLAS_FORCE_INLINE void
parallel_for(IndexType start, IndexType end, const Function& func) {
    if (start >= end) return;

#if defined(ATLAS_BACKEND_CUDA)
    const auto first = thrust::make_counting_iterator<IndexType>(start);
    const auto last  = thrust::make_counting_iterator<IndexType>(end);

    if constexpr (P == ExecutionPolicy::host) {
        thrust::for_each(thrust::host, first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::for_each(thrust::device, first, last, func);
    } else {
        thrust::for_each(thrust::seq, first, last, func);
    }
#else
    if constexpr (P == ExecutionPolicy::serial) {
        for (IndexType i = start; i < end; ++i) {
            func(i);
        }
    } else {
        tbb::parallel_for(tbb::blocked_range<IndexType>(start, end),
                          [&](const tbb::blocked_range<IndexType>& range) {
                              for (IndexType i = range.begin(); i != range.end(); ++i) {
                                  func(i);
                              }
                          });
    }
#endif
}

/**
 * @brief Invoke @p func once for every element in the iterator range `[first, last)`.
 *
 * Under the host backend the range must be random-access so TBB can split it; every
 * caller in the engine passes contiguous buffer iterators.
 *
 * @tparam P Where to run.
 * @tparam InputIt Iterator type; must not be integral, which would select the index
 *                 overload.
 * @tparam Function Callable invoked as `func(*it)`.
 * @param first Start of the range.
 * @param last One past the end. An empty range returns without invoking @p func.
 * @param func The body, invoked concurrently.
 */
template <ExecutionPolicy P, typename InputIt, typename Function,
          std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
ATLAS_FORCE_INLINE void
parallel_for(InputIt first, InputIt last, const Function& func) {
    if (first == last) return;

#if defined(ATLAS_BACKEND_CUDA)
    if constexpr (P == ExecutionPolicy::host) {
        thrust::for_each(thrust::host, first, last, func);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::for_each(thrust::device, first, last, func);
    } else {
        thrust::for_each(thrust::seq, first, last, func);
    }
#else
    if constexpr (P == ExecutionPolicy::serial) {
        std::for_each(first, last, func);
    } else {
        const auto count = static_cast<std::ptrdiff_t>(last - first);

        tbb::parallel_for(tbb::blocked_range<std::ptrdiff_t>(0, count),
                          [&](const tbb::blocked_range<std::ptrdiff_t>& range) {
                              for (std::ptrdiff_t i = range.begin(); i != range.end(); ++i) {
                                  func(*(first + i));
                              }
                          });
    }
#endif
}

}
