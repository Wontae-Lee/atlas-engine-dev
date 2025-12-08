#pragma once
#include <atlas/core/macros.h>
#include <type_traits>

namespace atlas {
enum class ExecutionPolicy {
    serial,
    host,
    device
};

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
        for (IndexType i = start; i < end; ++i) {
            func(i);
        }
    }

    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_host_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::host, first, last, func);
    }

    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        if (first == last) return;
        thrust::for_each(thrust::device, first, last, func);
    }

    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
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
}
#else
#include <iterator>
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
        for (IndexType i = start; i < end; ++i) {
            func(i);
        }
    }

    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

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

    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_device_iter_impl(InputIt first, InputIt last, const Function& func) {
        parallel_for_host_iter_impl(first, last, func);
    }

    template <typename InputIt, typename Function>
    ATLAS_FORCE_INLINE void
    parallel_for_serial_iter_impl(InputIt first, InputIt last, const Function& func) {
        for (; first != last; ++first) {
            func(*first);
        }
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
}
#endif