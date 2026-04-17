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
    if (first == last) {
        return result;
    }

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
    if (first == last) {
        return result;
    }

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
    if (first == last) {
        return result;
    }

    return thrust::exclusive_scan(
        thrust::seq,
        first,
        last,
        result,
        init,
        binary_op);
}

}

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
    if (first == last) {
        return result;
    }

    return std::exclusive_scan(first, last, result, init, binary_op);
}

template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan_device_impl(InputIt first, InputIt last,
                           OutputIt result,
                           T init,
                           BinaryOp binary_op) {
    return exclusive_scan_host_impl(first, last, result, init, binary_op);
}

template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan_serial_impl(InputIt first, InputIt last,
                           OutputIt result,
                           T init,
                           BinaryOp binary_op) {
    T sum = init;
    for (; first != last; ++first, ++result) {
        *result = sum;
        sum     = binary_op(sum, *first);
    }
    return result;
}

}

#endif

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
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
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, std::plus<value_type> {});
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, std::plus<value_type> {});
}

}
