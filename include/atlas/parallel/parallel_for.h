#ifndef INCLUDE_ATLAS_PARALLEL_PARALLEL_FOR_H
#define INCLUDE_ATLAS_PARALLEL_PARALLEL_FOR_H
#include <atlas/math/detail/config.h>
#include <type_traits>
namespace atlas {

enum class ExecutionPolicy { serial,
                             host,
                             device };

template <typename T>
using is_integral_index = std::enable_if_t<std::is_integral<T>::value, int>;
}

#if defined(ATLAS_TASKING_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace atlas {
namespace detail {

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);
        thrust::for_each(thrust::host, first, last, func);
    }

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        auto first = thrust::make_counting_iterator<IndexType>(start);
        auto last  = thrust::make_counting_iterator<IndexType>(end);
        thrust::for_each(thrust::device, first, last, func);
    }

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        for (IndexType i = start; i < end; ++i) func(i);
    }

}

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

}

#else
#include <tbb/tbb.h>
namespace atlas {
namespace detail {

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_host_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, func);
    }

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_device_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        tbb::parallel_for(start, end, func);
    }

    template <typename IndexType, typename Function, is_integral_index<IndexType> = 0>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_impl(IndexType start, IndexType end, const Function& func) {
        if (start >= end) return;
        for (IndexType i = start; i < end; ++i) func(i);
    }

}

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
}

#endif
#endif