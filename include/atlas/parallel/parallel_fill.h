#pragma once
#include <atlas/parallel/parallel_for.h>
#if defined(ATLAS_TASKING_CUDA)
namespace atlas {
namespace detail {
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_host_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        thrust::fill(thrust::host, first, last, value);
    }

    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        thrust::fill(thrust::device, first, last, value);
    }

    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        std::fill(first, last, value);
    }
}

template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_fill_host_impl(first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_fill_device_impl(first, last, value);
    } else {
        detail::parallel_fill_serial_impl(first, last, value);
    }
}
}
#else
namespace atlas {
namespace detail {
    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_host_impl(Iterator first, Iterator last, const T& value) {
        using diff_t = typename std::iterator_traits<Iterator>::difference_type;
        diff_t n     = std::distance(first, last);
        if (n <= 0) return;
        tbb::parallel_for<diff_t>(
            0,
            n,
            [first, value](diff_t i) {
                first[i] = value;
            });
    }

    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_device_impl(Iterator first, Iterator last, const T& value) {
        parallel_fill_host_impl(first, last, value);
    }

    template <typename Iterator, typename T>
    ATLAS_FORCE_INLINE void
    parallel_fill_serial_impl(Iterator first, Iterator last, const T& value) {
        if (first == last) return;
        std::fill(first, last, value);
    }
}

template <ExecutionPolicy P, typename Iterator, typename T>
ATLAS_FORCE_INLINE void
parallel_fill(Iterator first, Iterator last, const T& value) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_fill_host_impl(first, last, value);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_fill_device_impl(first, last, value);
    } else {
        detail::parallel_fill_serial_impl(first, last, value);
    }
}
}
#endif