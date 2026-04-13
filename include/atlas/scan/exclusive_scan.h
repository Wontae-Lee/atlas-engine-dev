#pragma once

/**
 * @file exclusive_scan.h
 * @brief Declares backend-portable exclusive-scan utilities parameterized by execution policy.
 *
 * @details
 * This header defines the Atlas interface for performing an exclusive prefix scan
 * over an input range and writing the result to an output range.
 *
 * The API is centered around:
 * - backend-specific implementation helpers in @ref atlas::detail,
 * - public overloads of @ref atlas::exclusive_scan parameterized by
 *   @ref ExecutionPolicy.
 *
 * ## Exclusive-scan semantics
 * Given an input range:
 * \f[
 * x_0, x_1, x_2, \dots, x_{n-1}
 * \f]
 * an initial value \f$init\f$, and a binary operation \f$\oplus\f$, an exclusive scan
 * produces:
 * \f[
 * y_0 = init,
 * \quad
 * y_1 = init \oplus x_0,
 * \quad
 * y_2 = init \oplus x_0 \oplus x_1,
 * \quad \dots
 * \f]
 *
 * That is, each output element contains the accumulated value of all preceding
 * input elements, excluding the current one.
 *
 * ## Execution-policy dispatch
 * The public @ref exclusive_scan functions dispatch at compile time according to
 * the supplied @ref ExecutionPolicy:
 * - host execution path,
 * - device execution path,
 * - serial fallback path.
 *
 * The concrete backend implementations are defined in
 * `exclusive_scan.hpp`.
 *
 * ## Overload set
 * Three public overload families are provided:
 * - explicit binary operation and initial value,
 * - default binary operation with explicit initial value,
 * - default binary operation with default-initialized accumulation state.
 *
 * This mirrors the common C++ scan API style while preserving Atlas's
 * backend-portable execution-policy design.
 *
 * ## Iterator expectations
 * The exact iterator requirements depend on the selected backend implementation
 * in `exclusive_scan.hpp`, but callers should generally supply valid input and
 * output iterator ranges with sufficient output storage.
 *
 * ---
 */

#include <atlas/parallel/parallel_for.h>
#include <functional>
#include <iterator>

namespace atlas {

namespace detail {

    /**
     * @brief Host-backend implementation of exclusive scan.
     *
     * @details
     * Performs an exclusive prefix scan over `[first, last)` and writes the result
     * beginning at @p result using:
     * - initial accumulator value @p init,
     * - binary reduction operator @p binary_op.
     *
     * The exact backend and parallelization strategy are implementation-defined in
     * `exclusive_scan.hpp`.
     *
     * @param first Iterator to the beginning of the input range.
     * @param last Iterator to the end of the input range.
     * @param result Iterator to the beginning of the output range.
     * @param init Initial accumulator value.
     * @param binary_op Binary accumulation operator.
     * @return Iterator one past the last written output element.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Accumulator/value type.
     * @tparam BinaryOp Binary operator type.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op);

    /**
     * @brief Device-backend implementation of exclusive scan.
     *
     * @details
     * Performs an exclusive prefix scan over `[first, last)` and writes the result
     * beginning at @p result using:
     * - initial accumulator value @p init,
     * - binary reduction operator @p binary_op.
     *
     * The exact backend and device execution strategy are implementation-defined in
     * `exclusive_scan.hpp`.
     *
     * @param first Iterator to the beginning of the input range.
     * @param last Iterator to the end of the input range.
     * @param result Iterator to the beginning of the output range.
     * @param init Initial accumulator value.
     * @param binary_op Binary accumulation operator.
     * @return Iterator one past the last written output element.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Accumulator/value type.
     * @tparam BinaryOp Binary operator type.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

    /**
     * @brief Serial-backend implementation of exclusive scan.
     *
     * @details
     * Performs a serial exclusive prefix scan over `[first, last)` and writes the
     * result beginning at @p result using:
     * - initial accumulator value @p init,
     * - binary reduction operator @p binary_op.
     *
     * @param first Iterator to the beginning of the input range.
     * @param last Iterator to the end of the input range.
     * @param result Iterator to the beginning of the output range.
     * @param init Initial accumulator value.
     * @param binary_op Binary accumulation operator.
     * @return Iterator one past the last written output element.
     *
     * @tparam InputIt Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T Accumulator/value type.
     * @tparam BinaryOp Binary operator type.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

} // namespace detail

/**
 * @brief Perform an exclusive scan according to the selected execution policy.
 *
 * @details
 * This overload performs an exclusive prefix scan over the input range
 * `[first, last)` and writes the result beginning at @p result.
 *
 * The accumulation uses:
 * - the explicitly supplied initial value @p init,
 * - the explicitly supplied binary operator @p binary_op.
 *
 * Dispatch is selected at compile time through the execution policy template
 * parameter @p P.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param result Iterator to the beginning of the output range.
 * @param init Initial accumulator value.
 * @param binary_op Binary accumulation operator.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T Accumulator/value type.
 * @tparam BinaryOp Binary operator type.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op);

/**
 * @brief Perform an exclusive scan using the default binary operation.
 *
 * @details
 * This overload performs an exclusive prefix scan over the input range
 * `[first, last)` and writes the result beginning at @p result.
 *
 * The accumulation uses:
 * - the explicitly supplied initial value @p init,
 * - the default associative binary operation appropriate for the implementation
 *   in `exclusive_scan.hpp`, commonly addition.
 *
 * Dispatch is selected at compile time through the execution policy template
 * parameter @p P.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param result Iterator to the beginning of the output range.
 * @param init Initial accumulator value.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T Accumulator/value type.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init);

/**
 * @brief Perform an exclusive scan using default initialization and default binary operation.
 *
 * @details
 * This overload performs an exclusive prefix scan over the input range
 * `[first, last)` and writes the result beginning at @p result.
 *
 * The accumulation uses:
 * - a default-initialized accumulator value,
 * - the default associative binary operation appropriate for the implementation
 *   in `exclusive_scan.hpp`, commonly addition.
 *
 * Dispatch is selected at compile time through the execution policy template
 * parameter @p P.
 *
 * @param first Iterator to the beginning of the input range.
 * @param last Iterator to the end of the input range.
 * @param result Iterator to the beginning of the output range.
 * @return Iterator one past the last written output element.
 *
 * @tparam P Compile-time execution policy.
 * @tparam InputIt Input iterator type.
 * @tparam OutputIt Output iterator type.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result);

} // namespace atlas

#include <atlas/scan/exclusive_scan.hpp>