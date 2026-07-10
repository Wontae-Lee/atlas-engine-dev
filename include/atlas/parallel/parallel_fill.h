#pragma once

#include <atlas/parallel/parallel_for.h>

#include <thrust/fill.h>

namespace atlas {

/**
 * @brief Writes @p value into every element of the range [first, last).
 *
 * A thin ExecutionPolicy-dispatched wrapper over `thrust::fill` that lets one
 * call site fill a host or device range without naming a Thrust policy. The
 * backend is chosen at compile time via `if constexpr`; the `device` case
 * requires @p first / @p last to be device iterators.
 *
 * @tparam P        Execution backend to run on.
 * @tparam Iterator Forward iterator over the range to fill.
 * @tparam T        Type of the fill value; must be assignable to `*first`.
 * @param first Beginning of the range.
 * @param last  One past the end of the range.
 * @param value Value copied into each element; taken by const reference.
 * @note Returns immediately when the range is empty (`first == last`),
 *       avoiding a needless kernel launch for the device policy.
 */
template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if (first == last) return;

    if constexpr (P == ExecutionPolicy::host) {
        thrust::fill(thrust::host, first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        thrust::fill(thrust::device, first, last, value);
    } else {
        thrust::fill(thrust::seq, first, last, value);
    }
}

}