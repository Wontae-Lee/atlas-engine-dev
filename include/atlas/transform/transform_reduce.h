#pragma once

#include <atlas/parallel/parallel_for.h>

#include <thrust/transform_reduce.h>

namespace atlas {

template <ExecutionPolicy P, typename InputIt, typename T, typename UnaryOp, typename BinaryOp>
ATLAS_FORCE_INLINE T
transform_reduce(InputIt first, InputIt last,
                 T init,
                 UnaryOp unary_op,
                 BinaryOp binary_op) {
    if (first == last) return init;

    if constexpr (P == ExecutionPolicy::host) {
        return thrust::transform_reduce(thrust::host, first, last, unary_op, init, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return thrust::transform_reduce(thrust::device, first, last, unary_op, init, binary_op);
    } else {
        return thrust::transform_reduce(thrust::seq, first, last, unary_op, init, binary_op);
    }
}

}
