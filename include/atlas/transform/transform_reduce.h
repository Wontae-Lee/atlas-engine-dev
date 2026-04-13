#pragma once
#include <atlas/parallel/parallel_for.h>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace detail {

    /**
     * @brief Host-backend implementation of transform-reduce for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * This is the CUDA-build host execution path. It delegates directly to
     * `thrust::transform_reduce` with the `thrust::host` execution policy.
     *
     * Computation model:
     * @code
     * result = init;
     * for each x in [first, last):
     *     result = binary_op(result, unary_op(x));
     * @endcode
     *
     * If the range is empty, `init` is returned unchanged.
     */
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

    /**
     * @brief Device-backend implementation of transform-reduce for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * This is the CUDA-build device execution path. It delegates directly to
     * `thrust::transform_reduce` with the `thrust::device` execution policy.
     *
     * If the range is empty, `init` is returned unchanged.
     */
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

    /**
     * @brief Serial implementation of transform-reduce for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * Even in CUDA builds, the serial path is implemented through Thrust using
     * the `thrust::seq` execution policy.
     *
     * If the range is empty, `init` is returned unchanged.
     */
    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T
    transform_reduce_serial_impl(InputIt first, InputIt last,
                                 T init,
                                 UnaryOp unary_op,
                                 Binary_op) {
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
    /**
     * @brief Apply a unary transform to an input range and reduce the results.
     *
     * @tparam P Compile-time execution policy.
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * This function dispatches at compile time according to @p P:
     * - `ExecutionPolicy::host`   -> host implementation
     * - `ExecutionPolicy::device` -> device implementation
     * - `ExecutionPolicy::serial` -> serial implementation
     *
     * The semantic effect is equivalent to:
     * @code
     * T result = init;
     * for (it = first; it != last; ++it) {
     *     result = binary_op(result, unary_op(*it));
     * }
     * return result;
     * @endcode
     */
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

    /**
     * @brief Trait indicating whether an iterator is random-access.
     *
     * @tparam It Iterator type.
     *
     * @details
     * The TBB host/device implementation indexes the input range with `first[i]`,
     * so random-access iterators are required.
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief Host-backend implementation of transform-reduce for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * This implementation uses `tbb::parallel_reduce` over an index range
     * `[0, n)` where `n = distance(first, last)`.
     *
     * Requirements:
     * - `InputIt` must be a random-access iterator.
     *
     * Reduction logic inside each block:
     * @code
     * local = init;
     * for i in block:
     *     local = binary_op(local, unary_op(first[i]));
     * @endcode
     *
     * Partial results from different blocks are then merged using the same
     * `binary_op`.
     *
     * If the input range is empty, `init` is returned unchanged.
     */
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

    /**
     * @brief Device-policy implementation of transform-reduce for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * In the TBB backend, the `device` policy is currently mapped to the same
     * implementation as the host policy.
     */
    template <typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
    ATLAS_FORCE_INLINE T
    transform_reduce_device_impl(InputIt first, InputIt last,
                                 T init,
                                 UnaryOp unary_op,
                                 BinaryOp binary_op) {
        return transform_reduce_host_impl(first, last, init, unary_op, binary_op);
    }

    /**
     * @brief Serial implementation of transform-reduce for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * This is the straightforward sequential reference implementation:
     * @code
     * result = init;
     * for each element:
     *     result = binary_op(result, unary_op(element));
     * @endcode
     */
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
    /**
     * @brief Apply a unary transform to an input range and reduce the results.
     *
     * @tparam P Compile-time execution policy.
     * @tparam InputIt Input iterator type.
     * @tparam T Reduction value type.
     * @tparam UnaryOp Unary transform operation type.
     * @tparam BinaryOp Binary reduction operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param init Initial reduction value.
     * @param unary_op Unary operation applied to each input element before reduction.
     * @param binary_op Binary operation used to combine transformed values.
     * @return Final reduced value.
     *
     * @details
     * Compile-time dispatch:
     * - `ExecutionPolicy::host`   -> host-parallel implementation
     * - `ExecutionPolicy::device` -> device-mapped implementation
     * - `ExecutionPolicy::serial` -> sequential implementation
     *
     * This interface mirrors the standard transform-reduce pattern while hiding
     * the backend-specific execution strategy.
     */
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_reduce_host_impl(first, last, init, unary_op, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_reduce_device_impl(first, last, init, unary_op, binary_op);
    } else {
        return detail::transform_reduce_serial_impl(first, last, init, unary_op, binary_op);
    }
}

#endif

} // namespace atlas