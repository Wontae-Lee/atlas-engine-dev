#pragma once
#include <atlas/parallel/parallel_for.h>

/**
 * @file transform_reduce.h
 * @brief Backend-dispatched transform-reduce algorithm (Thrust/TBB/serial).
 *
 * @details
 * This header provides `atlas::transform_reduce<P>(first, last, init, unary_op, binary_op)`,
 * a backend-agnostic transform-reduce utility that matches the common pattern:
 *
 * 1) Transform each input element with `unary_op`
 * 2) Combine transformed values using `binary_op`, starting from `init`
 *
 * Conceptually:
 * \f[
 *   r = \text{fold}\_{binary\_op}(init,\; unary\_op(x_0),\; unary\_op(x_1),\; \dots)
 * \f]
 *
 * Backend selection:
 * - **CUDA build (`ATLAS_TASKING_CUDA`)**:
 *   - `host`   -> `thrust::transform_reduce(thrust::host, ...)`
 *   - `device` -> `thrust::transform_reduce(thrust::device, ...)`
 *   - `serial` -> `thrust::transform_reduce(thrust::seq, ...)`
 *
 * - **Non-CUDA build**:
 *   - `host`   -> `tbb::parallel_reduce` over a blocked index range
 *   - `device` -> same as host (device policy maps to CPU fallback)
 *   - `serial` -> simple loop
 *
 * @tparam P         Execution policy (`ExecutionPolicy::host`, `device`, or `serial`).
 * @tparam InputIt   Input iterator type.
 * @tparam T         Accumulator / result type.
 * @tparam UnaryOp   Unary transform callable, invoked as `unary_op(x)`.
 * @tparam BinaryOp  Binary reduction callable, invoked as `binary_op(a, b)`.
 *
 * @param first    Range begin.
 * @param last     Range end.
 * @param init     Initial accumulator value.
 * @param unary_op Transform operation applied to each element.
 * @param binary_op Reduction operation used to combine partial results.
 *
 * @return Reduced value of type `T`.
 *
 * @note
 * - Correct parallel execution generally requires `binary_op` to be associative (and preferably
 *   commutative) and to have `init` as a neutral element, especially for the TBB backend.
 * - For the non-CUDA host backend, this implementation requires **random-access iterators**
 *   because it indexes with `first[i]` for performance and simplicity.
 * - Empty ranges return `init`.
 */

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace detail {

    /**
     * @brief Thrust host backend implementation.
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
     * @brief Thrust device backend implementation.
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
     * @brief Thrust serial backend implementation.
     *
     * @note
     * Uses `thrust::seq` execution policy.
     */
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

} // namespace detail

/**
 * @brief Applies `unary_op` to each element in `[first, last)` and reduces via `binary_op`.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam InputIt  Input iterator type.
 * @tparam T        Accumulator / result type.
 * @tparam UnaryOp  Unary transform functor.
 * @tparam BinaryOp Binary reduction functor.
 */
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

#else // --------------------------------------------------------
// Non-CUDA backend (oneTBB)
// --------------------------------------------------------

#include <iterator>
#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    /**
     * @brief Trait to check whether `InputIt` is a random-access iterator.
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief TBB host backend implementation using `tbb::parallel_reduce`.
     *
     * @details
     * Performs an index-based blocked reduction:
     * - Each worker reduces a subrange into a local accumulator `local`.
     * - Partials are combined using `binary_op`.
     *
     * @note
     * Requires random-access iterators because the loop uses `first[i]`.
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
     * @brief "Device" backend implementation in non-CUDA builds (maps to host).
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
     * @brief Serial transform-reduce implementation.
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

} // namespace detail

/**
 * @brief Applies `unary_op` to each element in `[first, last)` and reduces via `binary_op`.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam InputIt  Input iterator type.
 * @tparam T        Accumulator / result type.
 * @tparam UnaryOp  Unary transform functor.
 * @tparam BinaryOp Binary reduction functor.
 */
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

#endif // ATLAS_TASKING_CUDA

} // namespace atlas
