#pragma once

#include <atlas/parallel/parallel_for.h>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/fill.h>
#else
#include <algorithm>
#include <cstddef>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#endif

namespace atlas {

/**
 * @brief Writes @p value into every element of the range [first, last).
 *
 * An ExecutionPolicy-dispatched fill that lets one call site cover a host or device range
 * without naming a backend policy. Under CUDA it is @c thrust::fill, and the @c device
 * case requires @p first / @p last to be device iterators. Under the host backend the
 * parallel policies split the range across TBB threads.
 *
 * @tparam P        Execution backend to run on.
 * @tparam Iterator Forward iterator over the range to fill; random access under the host
 *                  backend's parallel policies.
 * @tparam T        Type of the fill value; must be assignable to `*first`.
 * @param first Beginning of the range.
 * @param last  One past the end of the range.
 * @param value Value copied into each element; taken by const reference.
 * @note Returns immediately when the range is empty (`first == last`), avoiding a
 *       needless kernel launch for the device policy.
 */
template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if (first == last) return;

#if defined(ATLAS_BACKEND_CUDA)
    if constexpr (P == ExecutionPolicy::host) {
        thrust::fill(thrust::host, first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::fill(thrust::device, first, last, value);
    } else {
        thrust::fill(thrust::seq, first, last, value);
    }
#else
    if constexpr (P == ExecutionPolicy::serial) {
        std::fill(first, last, value);
    } else {
        const auto count = static_cast<std::ptrdiff_t>(last - first);

        tbb::parallel_for(tbb::blocked_range<std::ptrdiff_t>(0, count),
                          [&](const tbb::blocked_range<std::ptrdiff_t>& range) {
                              std::fill(first + range.begin(), first + range.end(), value);
                          });
    }
#endif
}

}
