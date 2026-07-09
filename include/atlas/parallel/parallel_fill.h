#pragma once

#include <atlas/parallel/parallel_for.h>

#include <thrust/fill.h>

namespace atlas {

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