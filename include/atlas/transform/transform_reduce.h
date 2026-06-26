#pragma once
#include <atlas/parallel/parallel_for.h>

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#else

#include <iterator>
#include <optional>
#include <type_traits>

#endif

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

namespace detail {

    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T
    transform_reduce_host_impl(InputIt first, InputIt last,
                               T init,
                               UnaryOp unary_op,
                               BinaryOp binary_op) {
        if (first == last) return init;
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
        if (first == last) return init;
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
        if (first == last) return init;
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
        if (first == last) return init;

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::transform_reduce (TBB backend) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        const auto partial = tbb::parallel_reduce(
            tbb::blocked_range<diff_t>(0, n),
            std::optional<T> {},
            [&](const tbb::blocked_range<diff_t>& r, std::optional<T> local) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    const T value = unary_op(first[i]);
                    if (local) {
                        *local = binary_op(*local, value);
                    } else {
                        local.emplace(value);
                    }
                }
                return local;
            },
            [&](const std::optional<T>& a, const std::optional<T>& b) -> std::optional<T> {
                if (!a) return b;
                if (!b) return a;
                return std::optional<T> { binary_op(*a, *b) };
            });

        return partial ? binary_op(init, *partial) : init;
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