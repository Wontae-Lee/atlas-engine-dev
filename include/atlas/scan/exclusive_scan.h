#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <execution>
#include <iterator>
#include <numeric>
#include <thrust/functional.h>
#include <thrust/scan.h>

namespace atlas {

/**
 * @brief Exclusive prefix scan over [@p first, @p last), dispatched by execution policy.
 *
 * Writes to @p result the running reduction of everything strictly *before* each input
 * position: @c result[0] is @p init, @c result[i] is @c binary_op(init, in[0..i-1]).
 * Atlas uses this to turn per-cell / per-particle counts into the base offsets that pack
 * survivors or emitted candidates into a dense array (see the DSMC solver and the fluid
 * step). @p result may alias @p first for an in-place scan.
 *
 * The policy @p P selects the backend at compile time:
 *   - @c device (with a CUDA thrust backend) runs on the GPU via
 *     @c thrust::exclusive_scan; @p first / @p last / @p result must be device iterators.
 *   - @c serial runs @c std::exclusive_scan sequentially on the host.
 *   - @c host runs @c std::exclusive_scan with @c std::execution::par.
 * For the two host paths the iterators are first lowered to raw pointers via
 * @c atlas::raw_pointer_cast(&*it), so this branch requires contiguous, host-addressable
 * ranges.
 *
 * @tparam P Compile-time execution policy selecting the backend.
 * @tparam InputIt Input iterator type; must be contiguous/host-addressable on the host
 *                 paths, a device iterator on the device path.
 * @tparam OutputIt Output iterator type; same constraints as @p InputIt.
 * @tparam T Type of the scan's initial/identity value.
 * @tparam BinaryOp Associative combining operation applied left-to-right.
 * @param first Iterator to the first input element.
 * @param last Iterator one past the last input element.
 * @param result Iterator to the first output slot; needs room for @c (last-first) writes.
 * @param init Value emitted at @c result[0] and folded into every later prefix.
 * @param binary_op Associative reduction operator (e.g. @c thrust::plus).
 * @return Iterator one past the last written output element (@c result advanced by the
 *         input length). An empty range returns @p result unchanged.
 * @note The device branch is gated on @c THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
 *       so that a non-CUDA build of an @c ExecutionPolicy::device call still compiles by
 *       falling through to the host-side parallel scan.
 */
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
        // Host backends need bare pointers; lower any fancy/device iterator to raw here.
        auto* raw_first  = atlas::raw_pointer_cast(&*first);
        auto* raw_result = atlas::raw_pointer_cast(&*result);
        if constexpr (P == ExecutionPolicy::serial) {
            std::exclusive_scan(raw_first, raw_first + count, raw_result, init, binary_op);
        } else {
            std::exclusive_scan(std::execution::par, raw_first, raw_first + count, raw_result, init, binary_op);
        }
        // thrust returns the past-the-end output iterator; reconstruct it for parity.
        return result + count;
    }
}

/**
 * @brief Exclusive prefix-sum overload defaulting the combine op to addition.
 *
 * Equivalent to the five-argument form with @c binary_op = @c thrust::plus over the
 * input's value type. This is the common "counts to offsets" case.
 *
 * @tparam P Compile-time execution policy selecting the backend.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T Type of the initial value.
 * @param first Iterator to the first input element.
 * @param last Iterator one past the last input element.
 * @param result Iterator to the first output slot.
 * @param init Value emitted at @c result[0].
 * @return Iterator one past the last written output element.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, thrust::plus<value_type> {});
}

/**
 * @brief Exclusive prefix-sum overload defaulting both the op (addition) and @c init.
 *
 * The initial value is a value-initialized element of the input's value type (0 for
 * arithmetic types), so @c result[0] is 0 and each later slot is the sum of all prior
 * inputs.
 *
 * @tparam P Compile-time execution policy selecting the backend.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @param first Iterator to the first input element.
 * @param last Iterator one past the last input element.
 * @param result Iterator to the first output slot.
 * @return Iterator one past the last written output element.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, thrust::plus<value_type> {});
}

}