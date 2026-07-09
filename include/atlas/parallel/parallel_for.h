#pragma once

#include <atlas/core/macros.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <type_traits>

namespace atlas {

enum class ExecutionPolicy {
    serial,
    host,
    device
};

template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral_v<T>, int>;

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