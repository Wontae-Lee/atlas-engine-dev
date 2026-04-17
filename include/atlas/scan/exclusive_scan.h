#pragma once

#include <atlas/parallel/parallel_for.h>

#include <functional>
#include <iterator>
#include <numeric>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace detail {

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        // An empty input range produces no output and returns the original result iterator.
        if (first == last) {
            return result;
        }

        // Dispatch the exclusive scan to Thrust's host execution policy.
        //
        // The output sequence is defined as:
        //   result[0] = init
        //   result[i] = binary_op(...binary_op(init, input[0])..., input[i-1])
        //
        // In other words, each output element excludes the current input element.
        return thrust::exclusive_scan(
            thrust::host,
            first,
            last,
            result,
            init,
            binary_op);
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // Preserve empty-range behavior consistently across all backends.
        if (first == last) {
            return result;
        }

        // Dispatch the exclusive scan to Thrust's device execution policy.
        //
        // This is the GPU-oriented implementation path when CUDA support is enabled.
        return thrust::exclusive_scan(
            thrust::device,
            first,
            last,
            result,
            init,
            binary_op);
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // Preserve empty-range behavior consistently across all backends.
        if (first == last) {
            return result;
        }

        // Dispatch the exclusive scan to Thrust's explicit sequential policy.
        //
        // This path is useful when the caller wants deterministic serial execution
        // even in CUDA-enabled builds.
        return thrust::exclusive_scan(
            thrust::seq,
            first,
            last,
            result,
            init,
            binary_op);
    }

} // namespace detail

#else

#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        // An empty input range produces no output and returns the original result iterator.
        if (first == last) {
            return result;
        }

        // Use the standard library exclusive scan on host builds.
        //
        // This is the normal CPU implementation path when CUDA support is not enabled.
        return std::exclusive_scan(first, last, result, init, binary_op);
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // In non-CUDA builds there is no distinct device backend.
        //
        // Fall back to the host implementation so the public API remains available
        // regardless of whether CUDA support is compiled in.
        return exclusive_scan_host_impl(first, last, result, init, binary_op);
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // Explicit sequential implementation of exclusive scan.
        //
        // This version does not rely on std::exclusive_scan so the serial path is
        // fully under project control and independent of backend-specific behavior.
        T sum = init;

        for (; first != last; ++first, ++result) {
            // Write the accumulated prefix *before* consuming the current input element.
            *result = sum;

            // Then incorporate the current input into the running prefix.
            sum = binary_op(sum, *first);
        }

        return result;
    }

} // namespace detail

#endif

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    // Public policy-dispatch entry point.
    //
    // The compile-time execution policy selects which backend implementation is used:
    // - host   -> host backend
    // - device -> device backend if available, otherwise host fallback
    // - serial -> explicit sequential backend
    if constexpr (P == ExecutionPolicy::host) {
        return detail::exclusive_scan_host_impl(first, last, result, init, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::exclusive_scan_device_impl(first, last, result, init, binary_op);
    } else {
        return detail::exclusive_scan_serial_impl(first, last, result, init, binary_op);
    }
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    // Convenience overload using addition as the scan operator.
    //
    // The value_type is derived from the input iterator so callers do not need
    // to specify the binary operation explicitly for the common additive case.
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, std::plus<value_type> {});
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    // Convenience overload using:
    // - default-initialized value_type as the initial prefix value
    // - addition as the scan operator
    //
    // This matches the most common "plain prefix sum" usage pattern.
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, std::plus<value_type> {});
}

} // namespace atlas