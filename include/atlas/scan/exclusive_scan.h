#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <functional>
#include <iterator>
#include <numeric>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/functional.h>
#include <thrust/scan.h>
#endif

namespace atlas {

/**
 * @brief The addition functor the defaulted @ref exclusive_scan overloads pass down.
 *
 * @c thrust::plus under the CUDA backend and @c std::plus otherwise. They behave
 * identically, but only thrust's carries the @c __host__ @c __device__ annotations a
 * device scan needs without leaning on @c --expt-relaxed-constexpr.
 *
 * @tparam T Operand type.
 */
#if defined(ATLAS_BACKEND_CUDA)
template <typename T>
using scan_plus = thrust::plus<T>;
#else
template <typename T>
using scan_plus = std::plus<T>;
#endif

namespace detail {

    /**
     * @brief Sequential exclusive scan over a contiguous, host-addressable range.
     *
     * Kept out of @ref exclusive_scan so the CUDA backend's device branch never
     * instantiates it with a device iterator.
     *
     * @tparam InputIt Input iterator over contiguous host storage.
     * @tparam OutputIt Output iterator over contiguous host storage; may alias the input.
     * @tparam T Type of the initial value.
     * @tparam BinaryOp Associative combining operation.
     * @param first Beginning of the input.
     * @param last One past the end of the input.
     * @param result Beginning of the output.
     * @param init Value written at `result[0]`.
     * @param binary_op Associative reduction operator.
     * @return Iterator one past the last written output element.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    host_exclusive_scan(InputIt first, InputIt last, OutputIt result, T init, BinaryOp binary_op) {
        const auto count = static_cast<std::ptrdiff_t>(last - first);

        // Host backends need bare pointers; lower any fancy iterator to raw here.
        auto* raw_first  = atlas::raw_pointer_cast(&*first);
        auto* raw_result = atlas::raw_pointer_cast(&*result);

        std::exclusive_scan(raw_first, raw_first + count, raw_result, init, binary_op);

        return result + count;
    }

}

/**
 * @brief Exclusive prefix scan over [@p first, @p last), dispatched by execution policy.
 *
 * Writes to @p result the running reduction of everything strictly *before* each input
 * position: @c result[0] is @p init, @c result[i] is @c binary_op(init, in[0..i-1]).
 * Atlas uses this to turn per-cell / per-particle counts into the base offsets that pack
 * survivors or emitted candidates into a dense array (see the DSMC solver and the fluid
 * step). @p result may alias @p first for an in-place scan.
 *
 * Under the CUDA backend, @c device runs on the GPU via @c thrust::exclusive_scan and the
 * iterators must be device iterators. Every other case lowers the iterators to raw
 * pointers via @c atlas::raw_pointer_cast(&*it) and runs @c std::exclusive_scan, so it
 * requires contiguous, host-addressable ranges.
 *
 * @tparam P Compile-time execution policy selecting the backend.
 * @tparam InputIt Input iterator type; contiguous and host-addressable on the host paths,
 *                 a device iterator on the device path.
 * @tparam OutputIt Output iterator type; same constraints as @p InputIt.
 * @tparam T Type of the scan's initial/identity value.
 * @tparam BinaryOp Associative combining operation applied left-to-right.
 * @param first Iterator to the first input element.
 * @param last Iterator one past the last input element.
 * @param result Iterator to the first output slot; needs room for @c (last-first) writes.
 * @param init Value emitted at @c result[0] and folded into every later prefix.
 * @param binary_op Associative reduction operator (e.g. @ref scan_plus).
 * @return Iterator one past the last written output element (@p result advanced by the
 *         input length). An empty range returns @p result unchanged.
 * @note The host scan is sequential. The ranges Atlas scans are per-cell counts, and a
 *       parallel scan over a few thousand cells does not pay for its own synchronization.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    if (first == last) return result;

#if defined(ATLAS_BACKEND_CUDA)
    if constexpr (P == ExecutionPolicy::device) {
        return thrust::exclusive_scan(thrust::device, first, last, result, init, binary_op);
    } else {
        return detail::host_exclusive_scan(first, last, result, init, binary_op);
    }
#else
    return detail::host_exclusive_scan(first, last, result, init, binary_op);
#endif
}

/**
 * @brief Exclusive prefix-sum overload defaulting the combine op to addition.
 *
 * Equivalent to the five-argument form with @c binary_op = @ref scan_plus over the input's
 * value type. This is the common "counts to offsets" case.
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
    return exclusive_scan<P>(first, last, result, init, scan_plus<value_type> {});
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
    return exclusive_scan<P>(first, last, result, value_type {}, scan_plus<value_type> {});
}

}
