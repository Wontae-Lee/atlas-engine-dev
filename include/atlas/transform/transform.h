#pragma once

/**
 * @file transform.h
 * @brief Declares backend-portable unary and binary transform utilities parameterized by execution policy.
 *
 * @details
 * This header defines @ref atlas::transform, a backend-portable algorithm for
 * applying:
 * - a unary operation over one input range, or
 * - a binary operation over two input ranges,
 * and writing the transformed result into an output range.
 *
 * ## Purpose
 * The utility provides a uniform Atlas-level interface for elementwise transforms
 * across multiple execution environments:
 * - host-parallel execution,
 * - device-oriented execution,
 * - serial fallback execution.
 *
 * This allows higher-level code to remain backend-agnostic while still using the
 * most appropriate implementation for the selected execution policy.
 *
 * ## Execution-policy dispatch
 * The public @ref transform overloads dispatch at compile time according to the
 * template execution policy @p P:
 * - `ExecutionPolicy::host`
 * - `ExecutionPolicy::device`
 * - a serial fallback path for all other policies
 *
 * ## CUDA-enabled builds
 * When `ATLAS_TASKING_CUDA` is defined:
 * - host execution uses `thrust::transform(thrust::host, ...)`,
 * - device execution uses `thrust::transform(thrust::device, ...)`,
 * - serial execution uses `thrust::transform(thrust::seq, ...)`.
 *
 * ## Non-CUDA builds
 * When CUDA tasking is not enabled:
 * - host execution uses a TBB-based parallel loop,
 * - device execution falls back to the same host-parallel implementation,
 * - serial execution uses a standard sequential loop.
 *
 * ## Iterator expectations
 * In the non-CUDA host-parallel path, the implementation uses indexed access
 * such as:
 * - `first[i]`
 * - `first2[i]`
 * - `d_first[i]`
 *
 * Therefore, the TBB-backed host implementation requires random-access iterators
 * for the input ranges. The serial fallback is less restrictive and advances the
 * iterators incrementally.
 *
 * ## Return value
 * Both unary and binary overloads return an iterator pointing one past the last
 * written output element.
 *
 * ---
 */

#include <atlas/parallel/parallel_for.h>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace detail {

/**
 * @brief Apply a unary transform using the host backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the Thrust host execution policy.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
template <typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_host_impl(InputIt first, InputIt last,
                    OutputIt d_first,
                    UnaryOp op) {
    if (first == last) return d_first;
    return thrust::transform(thrust::host, first, last, d_first, op);
}

/**
 * @brief Apply a unary transform using the device backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the Thrust device execution policy.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
template <typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_device_impl(InputIt first, InputIt last,
                      OutputIt d_first,
                      UnaryOp op) {
    if (first == last) return d_first;
    return thrust::transform(thrust::device, first, last, d_first, op);
}

/**
 * @brief Apply a unary transform using the serial backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the sequential Thrust execution policy.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
template <typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_serial_impl(InputIt first, InputIt last,
                      OutputIt d_first,
                      UnaryOp op) {
    if (first == last) return d_first;
    return thrust::transform(thrust::seq, first, last, d_first, op);
}

/**
 * @brief Apply a binary transform using the host backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the Thrust host execution policy.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_host_impl(InputIt1 first1, InputIt1 last1,
                    InputIt2 first2,
                    OutputIt d_first,
                    BinaryOp op) {
    if (first1 == last1) return d_first;
    return thrust::transform(thrust::host, first1, last1, first2, d_first, op);
}

/**
 * @brief Apply a binary transform using the device backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the Thrust device execution policy.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_device_impl(InputIt1 first1, InputIt1 last1,
                      InputIt2 first2,
                      OutputIt d_first,
                      BinaryOp op) {
    if (first1 == last1) return d_first;
    return thrust::transform(thrust::device, first1, last1, first2, d_first, op);
}

/**
 * @brief Apply a binary transform using the serial backend in CUDA-enabled builds.
 *
 * @details
 * Delegates to `thrust::transform` with the sequential Thrust execution policy.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_serial_impl(InputIt1 first1, InputIt1 last1,
                      InputIt2 first2,
                      OutputIt d_first,
                      BinaryOp op) {
    if (first1 == last1) return d_first;
    return thrust::transform(thrust::seq, first1, last1, first2, d_first, op);
}

} // namespace detail

/**
 * @brief Apply a unary transform according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to
 * @p P:
 * - `ExecutionPolicy::host`   -> host backend
 * - `ExecutionPolicy::device` -> device backend
 * - otherwise                 -> serial backend
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
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

/**
 * @brief Apply a binary transform according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to
 * @p P:
 * - `ExecutionPolicy::host`   -> host backend
 * - `ExecutionPolicy::device` -> device backend
 * - otherwise                 -> serial backend
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
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

/**
 * @brief Type trait checking whether an iterator is random-access.
 *
 * @tparam It Iterator type.
 */
template <typename It>
using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                  typename std::iterator_traits<It>::iterator_category>;

/**
 * @brief Apply a unary transform using the host-parallel backend in non-CUDA builds.
 *
 * @details
 * Uses `tbb::parallel_for` over index space and requires a random-access input
 * iterator because the implementation uses `first[i]` and `d_first[i]`.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
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

/**
 * @brief Apply a unary transform using the device path in non-CUDA builds.
 *
 * @details
 * Falls back to the host-parallel implementation.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
template <typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_device_impl(InputIt first, InputIt last,
                      OutputIt d_first,
                      UnaryOp op) {
    return transform_host_impl(first, last, d_first, op);
}

/**
 * @brief Apply a unary transform using the serial backend.
 *
 * @details
 * Uses a standard sequential loop over the input range.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
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

/**
 * @brief Apply a binary transform using the host-parallel backend in non-CUDA builds.
 *
 * @details
 * Uses `tbb::parallel_for` over index space and requires a random-access first
 * input iterator because the implementation uses `first1[i]`, `first2[i]`, and
 * `d_first[i]`.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
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

/**
 * @brief Apply a binary transform using the device path in non-CUDA builds.
 *
 * @details
 * Falls back to the host-parallel implementation.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform_device_impl(InputIt1 first1, InputIt1 last1,
                      InputIt2 first2,
                      OutputIt d_first,
                      BinaryOp op) {
    return transform_host_impl(first1, last1, first2, d_first, op);
}

/**
 * @brief Apply a binary transform using the serial backend.
 *
 * @details
 * Uses a standard sequential loop over both input ranges.
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
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

} // namespace detail

/**
 * @brief Apply a unary transform according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to
 * @p P:
 * - `ExecutionPolicy::host`   -> host-parallel backend
 * - `ExecutionPolicy::device` -> device path, falling back to host in non-CUDA builds
 * - otherwise                 -> serial backend
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Unary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp Unary operator type.
 */
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

/**
 * @brief Apply a binary transform according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to
 * @p P:
 * - `ExecutionPolicy::host`   -> host-parallel backend
 * - `ExecutionPolicy::device` -> device path, falling back to host in non-CUDA builds
 * - otherwise                 -> serial backend
 *
 * @param first1 Iterator to the beginning of the first input range.
 * @param last1 Iterator to the end of the first input range.
 * @param first2 Iterator to the beginning of the second input range.
 * @param d_first Iterator to the beginning of the output range.
 * @param op Binary transform operation.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt1 First input iterator type.
 * @tparam InputIt2 Second input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam BinaryOp Binary operator type.
 */
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

} // namespace atlas