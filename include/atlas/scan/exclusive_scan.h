#pragma once
#include <atlas/parallel/parallel_for.h> // ExecutionPolicy, ATLAS_FORCE_INLINE, etc.
#include <functional>
#include <iterator>

namespace atlas {

/**
 * @file exclusive_scan.h
 * @brief Policy-dispatched exclusive prefix scan (exclusive prefix reduction).
 *
 * @details
 * This header declares `atlas::exclusive_scan`, a small wrapper API for computing an
 * **exclusive prefix scan** (also known as an exclusive prefix sum/reduction).
 *
 * Given an input range \f$[x_0, x_1, \dots, x_{n-1}]\f$, an initial value \f$init\f$, and a binary
 * operator \f$\oplus\f$, the exclusive scan produces an output range
 * \f$[y_0, y_1, \dots, y_{n-1}]\f$ such that:
 * \f[
 *   y_0 = init,\qquad
 *   y_i = init \oplus x_0 \oplus x_1 \oplus \cdots \oplus x_{i-1}\quad (i \ge 1).
 * \f]
 *
 * The implementation is selected at compile time via `ExecutionPolicy`:
 * - `ExecutionPolicy::host`   → host backend implementation
 * - `ExecutionPolicy::device` → device backend implementation (may map to CUDA/Thrust when enabled)
 * - otherwise                 → serial backend implementation
 *
 * The backend definitions are provided in the included implementation header:
 * `atlas/scan/exclusive_scan.hpp`.
 *
 * ---
 *
 * @note
 * - For parallel backends, `binary_op` should be **associative** to ensure deterministic and correct results.
 * - The output range must be large enough to store `distance(first, last)` elements.
 * - All overloads return an iterator pointing **one past the last output element** (`result + n`),
 *   matching common standard-library conventions.
 *
 * @see atlas/scan/exclusive_scan.hpp
 */

// detail namespace declarations (definitions live in the included .hpp/.inl)
namespace detail {

    /**
     * @brief Host backend implementation of `exclusive_scan`.
     *
     * @details
     * Implements an exclusive scan for host execution. The concrete backend may be:
     * - A Thrust host execution policy when CUDA tasking is enabled, or
     * - A TBB-based parallel scan / serial fallback when CUDA tasking is disabled,
     * depending on your platform configuration.
     *
     * ---
     *
     * @tparam InputIt  Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T        Accumulator / init type.
     * @tparam BinaryOp Binary associative operator type.
     *
     * @param first     Begin iterator of the input range.
     * @param last      End iterator of the input range.
     * @param result    Begin iterator of the output range.
     * @param init      Initial value \f$init\f$.
     * @param binary_op Binary operator \f$\oplus\f$ used to combine values.
     * @return Output iterator advanced by the number of processed elements.
     *
     * @note
     * - Implementations typically early-out when `first == last`.
     * - For parallel implementations, `binary_op` must be associative.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_host_impl(InputIt first, InputIt last,
                             OutputIt result,
                             T init,
                             BinaryOp binary_op);

    /**
     * @brief Device backend implementation of `exclusive_scan`.
     *
     * @details
     * Implements an exclusive scan for device execution. Depending on build configuration,
     * this may dispatch to a GPU backend (e.g., Thrust device policy) or fall back to the host
     * backend when device execution is not available.
     *
     * ---
     *
     * @tparam InputIt  Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T        Accumulator / init type.
     * @tparam BinaryOp Binary associative operator type.
     *
     * @param first     Begin iterator of the input range.
     * @param last      End iterator of the input range.
     * @param result    Begin iterator of the output range.
     * @param init      Initial value \f$init\f$.
     * @param binary_op Binary operator \f$\oplus\f$ used to combine values.
     * @return Output iterator advanced by the number of processed elements.
     *
     * @note
     * - Iterator types must be compatible with the selected device backend.
     * - Implementations typically early-out when `first == last`.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_device_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

    /**
     * @brief Serial implementation of `exclusive_scan`.
     *
     * @details
     * Computes the exclusive scan in a single pass:
     * - write the current accumulator to output
     * - update the accumulator with the current input element
     *
     * ---
     *
     * @tparam InputIt  Input iterator type.
     * @tparam OutputIt Output iterator type.
     * @tparam T        Accumulator / init type.
     * @tparam BinaryOp Binary associative operator type.
     *
     * @param first     Begin iterator of the input range.
     * @param last      End iterator of the input range.
     * @param result    Begin iterator of the output range.
     * @param init      Initial value \f$init\f$.
     * @param binary_op Binary operator \f$\oplus\f$ used to combine values.
     * @return Output iterator advanced by the number of processed elements.
     *
     * @note
     * - Works with minimal iterator requirements (input/output iterators).
     * - Provides deterministic behavior regardless of `binary_op` associativity.
     */
    template <typename InputIt, typename OutputIt, typename T, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    exclusive_scan_serial_impl(InputIt first, InputIt last,
                               OutputIt result,
                               T init,
                               BinaryOp binary_op);

} // namespace detail

// ---------------------------
// public API (declarations)
// ---------------------------

/**
 * @brief Computes an exclusive scan over `[first, last)` using an explicit binary operator.
 *
 * @details
 * This is the most general overload. It computes an exclusive prefix reduction using the
 * provided initial value `init` and binary operator `binary_op`.
 *
 * Backend selection is performed at compile time using the template parameter `P`:
 * - If `P == ExecutionPolicy::host`, dispatches to `detail::exclusive_scan_host_impl`.
 * - If `P == ExecutionPolicy::device`, dispatches to `detail::exclusive_scan_device_impl`.
 * - Otherwise, dispatches to `detail::exclusive_scan_serial_impl`.
 *
 * The output written has the same length as the input range (`n = distance(first, last)`),
 * and the return value is `result + n` (or an equivalent advanced output iterator).
 *
 * ---
 *
 * @tparam P        Execution policy tag.
 * @tparam InputIt  Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T        Accumulator / init type.
 * @tparam BinaryOp Binary associative operator type.
 *
 * @param first     Begin iterator of the input range.
 * @param last      End iterator of the input range.
 * @param result    Begin iterator of the output range.
 * @param init      Initial value \f$init\f$.
 * @param binary_op Binary operator \f$\oplus\f$ used to combine values.
 * @return Output iterator advanced by the number of processed elements.
 *
 * @pre `result` points to a writable range of at least `distance(first, last)` elements.
 *
 * @note
 * - For parallel backends, `binary_op` should be associative; otherwise results may depend on
 *   evaluation order and may vary between runs/platforms.
 * - If `first == last`, no output is written and the function returns `result`.
 *
 * @see exclusive_scan<P>(InputIt, InputIt, OutputIt, T)
 * @see exclusive_scan<P>(InputIt, InputIt, OutputIt)
 * @see detail::exclusive_scan_host_impl
 * @see detail::exclusive_scan_device_impl
 * @see detail::exclusive_scan_serial_impl
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init,
               BinaryOp binary_op);

/**
 * @brief Computes an exclusive scan over `[first, last)` using `std::plus` and an explicit init value.
 *
 * @details
 * This overload uses `std::plus<value_type>` as the reduction operator, where `value_type` is
 * the input iterator's `value_type`. The initial value is provided explicitly via `init`.
 *
 * ---
 *
 * @tparam P        Execution policy tag.
 * @tparam InputIt  Input iterator type.
 * @tparam OutputIt Output iterator type.
 * @tparam T        Init/accumulator type.
 *
 * @param first  Begin iterator of the input range.
 * @param last   End iterator of the input range.
 * @param result Begin iterator of the output range.
 * @param init   Initial value \f$init\f$.
 * @return Output iterator advanced by the number of processed elements.
 *
 * @note
 * - Requires `value_type` to support `operator+` (or a suitable `std::plus` specialization).
 * - If `first == last`, returns `result`.
 *
 * @see exclusive_scan<P>(InputIt, InputIt, OutputIt, T, BinaryOp)
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename T>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result,
               T init);

/**
 * @brief Computes an exclusive scan over `[first, last)` using `std::plus` and a value-initialized init.
 *
 * @details
 * This overload uses:
 * - `value_type{}` as the initial value, and
 * - `std::plus<value_type>{}` as the reduction operator.
 *
 * This is the classic "exclusive prefix sum" form for additive types where `value_type{}` behaves
 * as the additive identity.
 *
 * ---
 *
 * @tparam P        Execution policy tag.
 * @tparam InputIt  Input iterator type.
 * @tparam OutputIt Output iterator type.
 *
 * @param first  Begin iterator of the input range.
 * @param last   End iterator of the input range.
 * @param result Begin iterator of the output range.
 * @return Output iterator advanced by the number of processed elements.
 *
 * @note
 * - `value_type{}` should represent the identity for `std::plus` to get standard prefix-sum semantics.
 * - If `value_type{}` is not a true identity (e.g., non-zero default), results follow the formal definition.
 * - If `first == last`, returns `result`.
 *
 * @see exclusive_scan<P>(InputIt, InputIt, OutputIt, T)
 * @see exclusive_scan<P>(InputIt, InputIt, OutputIt, T, BinaryOp)
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt>
ATLAS_FORCE_INLINE OutputIt
exclusive_scan(InputIt first, InputIt last,
               OutputIt result);

} // namespace atlas

// Template implementation must be included
#include <atlas/scan/exclusive_scan.hpp>