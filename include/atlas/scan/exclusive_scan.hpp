#pragma once

#include <functional>
#include <iterator>

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace detail {

    /**
     * @brief Host-backend exclusive scan implementation for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * This implementation dispatches to `thrust::exclusive_scan` with the
     * Thrust host execution policy.
     *
     * For an input sequence:
     * @code
     * x0, x1, x2, ...
     * @endcode
     *
     * and initial value `init`, the output becomes:
     * @code
     * init,
     * binary_op(init, x0),
     * binary_op(binary_op(init, x0), x1),
     * ...
     * @endcode
     *
     * If the input range is empty, the function returns @p result unchanged.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
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

    /**
     * @brief Device-backend exclusive scan implementation for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * This implementation dispatches to `thrust::exclusive_scan` with the
     * Thrust device execution policy.
     *
     * If the input range is empty, the function returns @p result unchanged.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
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

    /**
     * @brief Serial exclusive scan implementation for CUDA builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * Even in CUDA builds, the serial path is delegated to Thrust using the
     * sequential execution policy `thrust::seq`.
     *
     * If the input range is empty, the function returns @p result unchanged.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
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

} // namespace detail

#else

#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    /**
     * @brief Trait indicating whether an iterator is random-access.
     *
     * @tparam It Iterator type.
     *
     * @details
     * The TBB parallel-scan implementation indexes into the input and output
     * ranges via `first[i]` and `result[i]`, so it requires random-access
     * iterators.
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief Host-backend exclusive scan implementation for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * This implementation uses `tbb::parallel_scan` over the integer range
     * `[0, n)`, where `n = distance(first, last)`.
     *
     * The algorithm is structured with a scan body that supports:
     * - pre-scan accumulation,
     * - final-scan output emission,
     * - split/join for parallel composition.
     *
     * Requirements:
     * - `InputIt` must be a random-access iterator.
     *
     * If the input range is empty, the function returns @p result unchanged.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op) {
        if (first == last) {
            return result;
        }

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::exclusive_scan (TBB backend) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        /**
         * @brief TBB scan body used to implement parallel exclusive scan.
         *
         * @details
         * The body stores:
         * - the input base iterator,
         * - the output base iterator,
         * - the binary operation,
         * - the initial value,
         * - the running partial sum.
         *
         * During:
         * - `pre_scan_tag`, it only propagates prefix totals,
         * - `final_scan_tag`, it also writes exclusive-scan outputs.
         */
        struct ScanBody {
            InputIt first;
            ///< Base iterator of the input range.

            OutputIt result;
            ///< Base iterator of the output range.

            BinaryOp binary_op;
            ///< Binary operation used for accumulation.

            T init;
            ///< Initial value of the scan.

            T sum;
            ///< Running prefix sum for the current body instance.

            /**
             * @brief Construct a root scan body.
             *
             * @param f Input-range base iterator.
             * @param r Output-range base iterator.
             * @param op Binary accumulation operation.
             * @param init_val Initial prefix value.
             */
            ScanBody(InputIt f, OutputIt r, BinaryOp op, T init_val)
                : first(f)
                , result(r)
                , binary_op(op)
                , init(init_val)
                , sum(init_val) { }

            /**
             * @brief Split constructor used by TBB during parallel decomposition.
             *
             * @param other Source body being split.
             * @param Unused split tag required by TBB.
             *
             * @details
             * A split body starts with the same input/output anchors and the
             * same initial value, but its running sum is reset to `init`.
             */
            ScanBody(ScanBody& other, tbb::split)
                : first(other.first)
                , result(other.result)
                , binary_op(other.binary_op)
                , init(other.init)
                , sum(other.init) { }

            /**
             * @brief Pre-scan phase: accumulate partial prefix totals only.
             *
             * @param r Index block assigned by TBB.
             * @param Unused tag selecting the pre-scan phase.
             */
            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::pre_scan_tag) {
                T temp = sum;
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    temp = binary_op(temp, first[i]);
                }
                sum = temp;
            }

            /**
             * @brief Final-scan phase: emit exclusive outputs and update prefix totals.
             *
             * @param r Index block assigned by TBB.
             * @param Unused tag selecting the final-scan phase.
             *
             * @details
             * For each input element:
             * - write the current running sum to the output,
             * - then advance the running sum with the current input value.
             */
            void
            operator()(const tbb::blocked_range<diff_t>& r, tbb::final_scan_tag) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    T temp    = sum;
                    sum       = binary_op(sum, first[i]);
                    result[i] = temp;
                }
            }

            /**
             * @brief Combine partial sums from the right-hand body into this body.
             *
             * @param rhs Right-hand scan body.
             *
             * @details
             * TBB uses `reverse_join` to combine scan segments in the correct
             * prefix order.
             */
            void
            reverse_join(ScanBody& rhs) {
                sum = binary_op(rhs.sum, sum);
            }

            /**
             * @brief Assign state from another scan body.
             *
             * @param rhs Source scan body.
             */
            void
            assign(ScanBody& rhs) {
                sum = rhs.sum;
            }
        };

        ScanBody body(first, result, binary_op, init);
        tbb::parallel_scan(tbb::blocked_range<diff_t>(0, n), body);

        return result + n;
    }

    /**
     * @brief Device-policy exclusive scan implementation for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * In the TBB backend, the `device` policy currently maps to the same
     * implementation as the host policy.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op) {
        return exclusive_scan_host_impl(first, last, result, init, binary_op);
    }

    /**
     * @brief Serial exclusive scan implementation for TBB builds.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Initial-value type.
     * @tparam BinaryOp Associative binary operation type.
     *
     * @param first Beginning of the input range.
     * @param last End of the input range.
     * @param result Beginning of the output range.
     * @param init Initial prefix value.
     * @param binary_op Binary operation used to accumulate the scan.
     * @return Iterator one past the last written output element.
     *
     * @details
     * This implementation performs the standard sequential exclusive-scan loop:
     * - emit the current sum,
     * - update the sum with the current input value.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    OutputIt
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

} // namespace detail

#endif

/**
 * @brief Compute an exclusive prefix scan over an input range using a specified execution policy.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T Initial-value type.
 * @tparam BinaryOp Associative binary operation type.
 *
 * @param first Beginning of the input range.
 * @param last End of the input range.
 * @param result Beginning of the output range.
 * @param init Initial prefix value.
 * @param binary_op Binary operation used to accumulate the scan.
 * @return Iterator one past the last written output element.
 *
 * @details
 * This is the primary policy-dispatch overload.
 *
 * Dispatch behavior:
 * - `ExecutionPolicy::host`   -> host backend implementation
 * - `ExecutionPolicy::device` -> device backend implementation
 * - `ExecutionPolicy::serial` -> serial implementation
 *
 * The scan is exclusive, meaning the output at each position does not include
 * the corresponding input element itself.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
OutputIt
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

/**
 * @brief Compute an exclusive prefix scan using the default addition operator.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T Initial-value type.
 *
 * @param first Beginning of the input range.
 * @param last End of the input range.
 * @param result Beginning of the output range.
 * @param init Initial prefix value.
 * @return Iterator one past the last written output element.
 *
 * @details
 * This overload uses `std::plus<value_type>` as the binary operation, where
 * `value_type` is the input iterator's value type.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, init, std::plus<value_type> {});
}

/**
 * @brief Compute an exclusive prefix scan using value-initialized zero and default addition.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 *
 * @param first Beginning of the input range.
 * @param last End of the input range.
 * @param result Beginning of the output range.
 * @return Iterator one past the last written output element.
 *
 * @details
 * This overload uses:
 * - `value_type{}` as the initial prefix value,
 * - `std::plus<value_type>` as the accumulation operator.
 *
 * It is convenient for the common case of additive scans starting from zero.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result) {
    using value_type = typename std::iterator_traits<InputIt>::value_type;
    return exclusive_scan<P>(first, last, result, value_type {}, std::plus<value_type> {});
}

} // namespace atlas