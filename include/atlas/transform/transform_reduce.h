#pragma once
#include <atlas/parallel/parallel_for.h>

namespace atlas {
#if defined(ATLAS_TASKING_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace detail {
    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T

    transform_reduce_host_impl(InputIt first, InputIt last,
                               T init,
                               UnaryOp unary_op,
                               BinaryOp binary_op) {
        if (first == last) {
            return init;
        }
        return thrust::transform_reduce(
            thrust::host,
            first,
            last,
            unary_op,
            init,
            binary_op);
    }

    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T

    transform_reduce_device_impl(InputIt first, InputIt last,
                                 T init,
                                 UnaryOp unary_op,
                                 BinaryOp binary_op) {
        if (first == last) {
            return init;
        }
        return thrust::transform_reduce(
            thrust::device,
            first,
            last,
            unary_op,
            init,
            binary_op);
    }

    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T

    transform_reduce_serial_impl(InputIt first, InputIt last,
                                 T init,
                                 UnaryOp unary_op,
                                 BinaryOp binary_op) {
        if (first == last) {
            return init;
        }
        return thrust::transform_reduce(
            thrust::seq,
            first,
            last,
            unary_op,
            init,
            binary_op);
    }
}

template <ExecutionPolicy P, typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T
transform_reduce(InputIt first, InputIt last,
                 T init,
                 UnaryOp unary_op,
                 BinaryOp binary_op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_reduce_host_impl(first, last, init, unary_op, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_reduce_device_impl(first, last, init, unary_op, binary_op);
    } else {
        return detail::transform_reduce_serial_impl(first, last, init, unary_op, binary_op);
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
template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T

transform_reduce_host_impl(InputIt first, InputIt last,
                           T init,
                           UnaryOp unary_op,
                           BinaryOp binary_op) {
    if (first == last) {
        return init;
    }
    static_assert(is_random_access_iterator<InputIt>::value,
                  "atlas::transform_reduce (TBB backend) requires a random-access iterator.");
    using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
    const diff_t n = std::distance(first, last);
    return tbb::parallel_reduce(
        tbb::blocked_range<diff_t>(0, n),
        init,
        [&](const tbb::blocked_range<diff_t>& r, T local) {
            for (diff_t i = r.begin(); i != r.end(); ++i) {
                local = binary_op(local, unary_op(first[i]));
            }
            return local;
        },
        [&](const T& a, const T& b) {
            return binary_op(a, b);
        });
}

template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T

transform_reduce_device_impl(InputIt first, InputIt last,
                             T init,
                             UnaryOp unary_op,
                             BinaryOp binary_op) {
    return transform_reduce_host_impl(first, last, init, unary_op, binary_op);
}

template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T

transform_reduce_serial_impl(InputIt first, InputIt last,
                             T init,
                             UnaryOp unary_op,
                             BinaryOp binary_op) {
    T result = init;
    for (auto it = first; it != last; ++it) {
        result = binary_op(result, unary_op(*it));
    }
    return result;
}
}

template <ExecutionPolicy P, typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T

transform_reduce(InputIt first, InputIt last,
                 T init,
                 UnaryOp unary_op,
                 BinaryOp binary_op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_reduce_host_impl(first, last, init, unary_op, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_reduce_device_impl(first, last, init, unary_op, binary_op);
    } else {
        return detail::transform_reduce_serial_impl(first, last, init, unary_op, binary_op);
    }
}
#endif
}