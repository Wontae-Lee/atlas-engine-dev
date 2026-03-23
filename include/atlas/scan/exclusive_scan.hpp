#pragma once

#include <functional>
#include <iterator>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace detail {

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Algorithm: Exclusive Prefix Scan (a.k.a. exclusive prefix sum)
        //
        // Definition:
        //   out[i] = init ⊕ in[0] ⊕ ... ⊕ in[i-1]
        //
        // Properties:
        //   - "Exclusive" means the current input element in[i] is NOT
        //     included in out[i]. (It contributes to out[i+1].)
        //   - Requires an associative binary_op for predictable results
        //     under parallel execution.
        //
        // Backend:
        //   - Thrust host execution policy (CPU path).
        // ------------------------------------------------------------
        if (first == last) {
            // Empty range -> nothing to write, return output begin.
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
    OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Algorithm: Exclusive Prefix Scan on GPU (Thrust device)
        //
        // High-level idea:
        //   - Performs a parallel scan across the input range.
        //   - Implemented by Thrust using an optimized GPU scan
        //     (typically a Blelloch-style / upsweep-downsweep variant
        //      under the hood, depending on Thrust/CUB backend).
        //
        // Output semantics remain:
        //   out[i] = init ⊕ in[0] ⊕ ... ⊕ in[i-1]
        //
        // Requirements:
        //   - Iterators must be device-accessible for thrust::device.
        // ------------------------------------------------------------
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
    OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Algorithm: Exclusive Prefix Scan (Serial baseline)
        //
        // This uses Thrust's sequential policy. It is useful for:
        //   - Deterministic debugging reference
        //   - Environments where only a serial implementation is desired
        //
        // Same semantics:
        //   out[i] = init ⊕ in[0] ⊕ ... ⊕ in[i-1]
        // ------------------------------------------------------------
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

} // namespace detail

#else // !ATLAS_TASKING_CUDA

#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt

    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Algorithm: Exclusive Prefix Scan via TBB parallel_scan
        //
        // Exclusive scan definition:
        //   out[i] = init ⊕ in[0] ⊕ ... ⊕ in[i-1]
        //
        // TBB strategy (parallel_scan):
        //   - Splits the index range into blocks.
        //   - pre_scan:
        //       computes partial sums for each block without writing output.
        //   - final_scan:
        //       replays the block and writes outputs using the correct
        //       incoming prefix sum (carried in `sum`).
        //   - reverse_join / assign:
        //       combine partial sums from splits so final_scan has the
        //       correct prefix values.
        //
        // Requirements:
        //   - Random-access iterators are required because the body uses
        //     indexing (first[i], result[i]).
        //
        // Notes:
        //   - Associativity of binary_op is important for meaningful
        //     parallel results.
        // ------------------------------------------------------------
        if (first == last) {
            return result;
        }

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::exclusive_scan (TBB backend) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        struct ScanBody {
            // Base pointers/iterators and operator.
            InputIt first;
            OutputIt result;
            BinaryOp binary_op;

            // init: the global initial value
            // sum : the running prefix value for the current block
            T init;
            T sum;

            ScanBody(InputIt f, OutputIt r, BinaryOp op, T init_val)
                : first(f)
                , result(r)
                , binary_op(op)
                , init(init_val)
                , sum(init_val) { }

            ScanBody(ScanBody& other, tbb::split)
                : first(other.first)
                , result(other.result)
                , binary_op(other.binary_op)
                , init(other.init)
                , sum(other.init) { }

            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::pre_scan_tag) {
                // ----------------------------------------------------
                // pre_scan phase:
                //   - Computes the reduction (sum) of this block.
                //   - Does NOT write to output.
                //
                // Effect:
                //   - sum becomes init ⊕ (all inputs in this block)
                // ----------------------------------------------------
                T temp = sum;
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    temp = binary_op(temp, first[i]);
                }
                sum = temp;
            }

            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::final_scan_tag) {
                // ----------------------------------------------------
                // final_scan phase:
                //   - Writes exclusive outputs for this block using the
                //     correct incoming prefix `sum`.
                //
                // For each i:
                //   out[i] = sum
                //   sum    = sum ⊕ in[i]
                // ----------------------------------------------------
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    T temp    = sum;
                    sum       = binary_op(sum, first[i]);
                    result[i] = temp;
                }
            }

            void
            reverse_join(ScanBody& rhs) {
                // ----------------------------------------------------
                // Join phase:
                //   - Combine sums when merging split tasks.
                //
                // reverse_join means rhs corresponds to the "left" side
                // in TBB's combination order.
                // ----------------------------------------------------
                sum = binary_op(rhs.sum, sum);
            }

            void
            assign(ScanBody& rhs) {
                // ----------------------------------------------------
                // Assign phase:
                //   - Copy sum from rhs (used internally by TBB).
                // ----------------------------------------------------
                sum = rhs.sum;
            }
        };

        ScanBody body(first, result, binary_op, init);
        tbb::parallel_scan(tbb::blocked_range<diff_t>(0, n), body);

        // Output iterator advanced by the number of written elements.
        return result + n;
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt

    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Backend behavior: TBB build does not provide a true device path.
        //
        // Policy:
        //   - ExecutionPolicy::device falls back to host implementation.
        //   - Keeps API uniform across CUDA/TBB builds.
        // ------------------------------------------------------------
        return exclusive_scan_host_impl(first, last, result, init, binary_op);
    }

    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt

    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        // ------------------------------------------------------------
        // Algorithm: Exclusive Prefix Scan (manual serial loop)
        //
        // This is a straightforward textbook implementation:
        //   sum = init
        //   for i:
        //     out[i] = sum
        //     sum    = sum ⊕ in[i]
        //
        // Useful as:
        //   - a simple fallback
        //   - a reference for correctness
        // ------------------------------------------------------------
        T sum = init;
        for (; first != last; ++first, ++result) {
            *result = sum;
            sum     = binary_op(sum, *first);
        }
        return result;
    }

} // namespace detail

#endif // ATLAS_TASKING_CUDA

// ---------------------------
// public API (definitions)
// ---------------------------

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op) {
    // ------------------------------------------------------------
    // Dispatch wrapper for exclusive_scan based on ExecutionPolicy.
    //
    // Policies:
    //   - host   : CPU parallel (TBB) or thrust::host (CUDA build)
    //   - device : GPU scan (CUDA build) or host fallback (TBB build)
    //   - serial : guaranteed serial scan (reference/fallback)
    //
    // Result:
    //   - Returns output iterator advanced by N.
    // ------------------------------------------------------------
    if constexpr (P == ExecutionPolicy::host) {
        return detail::exclusive_scan_host_impl(first, last, result, init, binary_op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::exclusive_scan_device_impl(first, last, result, init, binary_op);
    } else {
        return detail::exclusive_scan_serial_impl(first, last, result, init, binary_op);
    }
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    // ------------------------------------------------------------
    // Convenience overload: uses addition as the binary operation.
    //
    // This matches the common "exclusive prefix sum" use case.
    // ------------------------------------------------------------
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, std::plus<value_type> {});
}

template <ExecutionPolicy P, typename InputIt, typename OutputIt>
OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    // ------------------------------------------------------------
    // Convenience overload:
    //   - init = value_type{}
    //   - op   = std::plus<value_type>
    //
    // Produces the standard exclusive prefix sum with zero init.
    // ------------------------------------------------------------
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, std::plus<value_type> {});
}

} // namespace atlas