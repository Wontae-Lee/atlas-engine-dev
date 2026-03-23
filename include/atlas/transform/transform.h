#pragma once
#include <atlas/parallel/parallel_for.h>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace detail {

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt first, InputIt last,
                        OutputIt d_first,
                        UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::host, first, last, d_first, op);
    }

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::device, first, last, d_first, op);
    }

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::seq, first, last, d_first, op);
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2,
                        OutputIt d_first,
                        BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::host, first1, last1, first2, d_first, op);
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::device, first1, last1, first2, d_first, op);
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::seq, first1, last1, first2, d_first, op);
    }

}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt first, InputIt last,
          OutputIt d_first,
          UnaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first, last, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first, last, d_first, op);
    } else {
        return detail::transform_serial_impl(first, last, d_first, op);
    }
}

template <ExecutionPolicy P, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt1 first1, InputIt1 last1,
          InputIt2 first2,
          OutputIt d_first,
          BinaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first1, last1, first2, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first1, last1, first2, d_first, op);
    } else {
        return detail::transform_serial_impl(first1, last1, first2, d_first, op);
    }
}

#else

#include <iterator>
#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt first, InputIt last,
                        OutputIt d_first,
                        UnaryOp op) {
        if (first == last) return d_first;

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::transform (TBB backend) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        tbb::parallel_for(
            tbb::blocked_range<diff_t>(0, n),
            [&](const tbb::blocked_range<diff_t>& r) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    d_first[i] = op(first[i]);
                }
            });

        return d_first + n;
    }

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        return transform_host_impl(first, last, d_first, op);
    }

    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        for (; first != last; ++first, ++d_first) {
            *d_first = op(*first);
        }
        return d_first;
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2,
                        OutputIt d_first,
                        BinaryOp op) {
        if (first1 == last1) return d_first;

        static_assert(is_random_access_iterator<InputIt1>::value,
                      "atlas::transform (TBB backend) requires a random-access iterator for first range.");

        using diff_t   = typename std::iterator_traits<InputIt1>::difference_type;
        const diff_t n = std::distance(first1, last1);

        tbb::parallel_for(
            tbb::blocked_range<diff_t>(0, n),
            [&](const tbb::blocked_range<diff_t>& r) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    d_first[i] = op(first1[i], first2[i]);
                }
            });

        return d_first + n;
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        return transform_host_impl(first1, last1, first2, d_first, op);
    }

    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        for (; first1 != last1; ++first1, ++first2, ++d_first) {
            *d_first = op(*first1, *first2);
        }
        return d_first;
    }

}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt first, InputIt last,
          OutputIt d_first,
          UnaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first, last, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first, last, d_first, op);
    } else {
        return detail::transform_serial_impl(first, last, d_first, op);
    }
}

template <ExecutionPolicy P, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt1 first1, InputIt1 last1,
          InputIt2 first2,
          OutputIt d_first,
          BinaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first1, last1, first2, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first1, last1, first2, d_first, op);
    } else {
        return detail::transform_serial_impl(first1, last1, first2, d_first, op);
    }
}

#endif

}