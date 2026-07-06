#pragma once

#include <atlas/parallel/parallel_for.h>

#include <thrust/transform.h>

namespace atlas {

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt first, InputIt last,
          OutputIt d_first,
          UnaryOp op) {
    if (first == last) return d_first;

    if constexpr (P == ExecutionPolicy::host) {
        return thrust::transform(thrust::host, first, last, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return thrust::transform(thrust::device, first, last, d_first, op);
    } else {
        return thrust::transform(thrust::seq, first, last, d_first, op);
    }
}

template <ExecutionPolicy P, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt1 first1, InputIt1 last1,
          InputIt2 first2,
          OutputIt d_first,
          BinaryOp op) {
    if (first1 == last1) return d_first;

    if constexpr (P == ExecutionPolicy::host) {
        return thrust::transform(thrust::host, first1, last1, first2, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return thrust::transform(thrust::device, first1, last1, first2, d_first, op);
    } else {
        return thrust::transform(thrust::seq, first1, last1, first2, d_first, op);
    }
}

}
