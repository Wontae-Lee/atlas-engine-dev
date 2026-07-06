#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <execution>
#include <iterator>
#include <numeric>
#include <thrust/functional.h>
#include <thrust/scan.h>

namespace atlas {

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    if (first == last) return result;

    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        return thrust::exclusive_scan(thrust::device, first, last, result, init, binary_op);
    } else {
        const auto count = static_cast<std::ptrdiff_t>(last - first);
        auto* raw_first  = atlas::raw_pointer_cast(&*first);
        auto* raw_result = atlas::raw_pointer_cast(&*result);
        if constexpr (P == ExecutionPolicy::serial) {
            std::exclusive_scan(raw_first, raw_first + count, raw_result, init, binary_op);
        } else {
            std::exclusive_scan(std::execution::par, raw_first, raw_first + count, raw_result, init, binary_op);
        }
        return result + count;
    }
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, thrust::plus<value_type> {});
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, thrust::plus<value_type> {});
}

}
