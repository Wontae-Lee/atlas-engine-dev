#pragma once

#include <iterator>

#ifdef ATLAS_TASKING_CUDA
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace atlas {

/**
 * @file remove.h
 * @brief `remove_if` wrapper that dispatches to CUDA (Thrust) or CPU (TBB) backends.
 *
 * @details
 * This header provides a small, unified `atlas::remove_if` algorithm intended to work in both:
 * - **CUDA builds** (`ATLAS_TASKING_CUDA`): forwards to `thrust::remove_if` on the device.
 * - **CPU builds**: performs a parallel compaction using Intel oneTBB (`tbb::parallel_for` and
 *   `tbb::parallel_scan`).
 *
 * The interface is similar to `std::remove_if`: it compacts elements in `[first,last)` by removing
 * those that satisfy `pred`, and returns the new logical end iterator.
 *
 * ---
 *
 * Behavior summary:
 * - Elements for which `pred(value)` is **true** are removed.
 * - Elements for which `pred(value)` is **false** are kept and compacted to the front.
 * - The relative order of kept elements is preserved (stable compaction) in both implementations.
 *
 * @note
 * - This algorithm requires **random-access iterators** in the CPU path because it uses `first + i`.
 * - On CPU, this implementation uses temporary storage (`keep`, `positions`, `temp`) proportional to `n`.
 * - The returned iterator marks the new end; elements beyond it are left in a valid but unspecified state.
 */

// ------------------------------------------------------------
// Execution policy tag
// ------------------------------------------------------------

/**
 * @brief Tag type representing a "device policy" for algorithms.
 *
 * @details
 * This is a lightweight dispatch token used to select the appropriate backend.
 * In CUDA builds it indicates device execution (Thrust). In CPU builds it still exists
 * to keep call sites uniform.
 */
struct device_policy_t { };

/**
 * @brief Global device policy instance.
 *
 * @details
 * Use as: `atlas::remove_if(atlas::device, first, last, pred);`
 */
static constexpr device_policy_t device {};

// ------------------------------------------------------------
// CUDA backend (Thrust)
// ------------------------------------------------------------

/**
 * @brief Removes elements satisfying `pred` from a device range using Thrust.
 *
 * @tparam Policy    Execution policy tag (currently unused other than overload selection).
 * @tparam Iterator  Iterator type (typically Thrust device iterator / raw device pointer wrapper).
 * @tparam Predicate Unary predicate callable on device.
 *
 * @param first  Beginning of the range.
 * @param last   End of the range.
 * @param pred   Predicate; elements with `pred(x) == true` are removed.
 *
 * @return Iterator pointing to the new logical end after compaction.
 *
 * @note
 * This is a thin wrapper over `thrust::remove_if(thrust::device, ...)`.
 */
template <typename Policy, typename Iterator, typename Predicate>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Iterator
    remove_if(Policy, Iterator first, Iterator last, Predicate pred) {
    // Dispatch to Thrust device execution.
    return thrust::remove_if(thrust::device, first, last, pred);
}

} // namespace atlas

#else // --------------------------------------------------------
// CPU backend (TBB)
// --------------------------------------------------------

#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <vector>

namespace atlas {

/**
 * @file remove.h
 * @brief `remove_if` wrapper that dispatches to CUDA (Thrust) or CPU (TBB) backends.
 *
 * @details
 * CPU implementation performs a parallel "keep-mask + prefix-sum + scatter" compaction:
 * 1) Build a keep mask: `keep[i] = !pred(first[i])`
 * 2) Compute exclusive positions via prefix sum (scan)
 * 3) Scatter kept elements to a temporary buffer at their computed positions
 * 4) Move the compacted result back into `[first, first + total_kept)`
 *
 * This yields stable ordering of kept elements.
 */

/** @brief Tag type representing a "device policy" for algorithms (CPU build still provides it). */
struct device_policy_t { };

/** @brief Global device policy instance (kept for API uniformity). */
static constexpr device_policy_t device {};

/**
 * @brief Removes elements satisfying `pred` from a range using a parallel CPU compaction.
 *
 * @tparam Policy    Execution policy tag (currently unused other than overload selection).
 * @tparam Iterator  Random-access iterator type.
 * @tparam Predicate Unary predicate callable on CPU.
 *
 * @param first  Beginning of the range.
 * @param last   End of the range.
 * @param pred   Predicate; elements with `pred(x) == true` are removed.
 *
 * @return Iterator pointing to the new logical end after compaction (`first + total_kept`).
 *
 * @details
 * Implementation outline:
 * - Compute `n = distance(first,last)`.
 * - Build `keep[i]` in parallel where `keep[i] = 1` means keep the element.
 * - Run a parallel scan to compute the destination index for each kept element.
 * - Scatter kept elements into a temporary buffer.
 * - Move results back into the original range.
 *
 * @note
 * - Requires random-access iterators because it uses `first + i`.
 * - Uses O(n) additional memory for masks and positions, plus O(k) for the compacted buffer
 *   where `k` is the number of kept elements.
 * - Stable: kept elements preserve their original relative order because the scan order defines
 *   unique increasing destinations.
 */
template <typename Policy, typename Iterator, typename Predicate>
inline Iterator
remove_if(Policy, Iterator first, Iterator last, Predicate pred) {
    using diff_type  = typename std::iterator_traits<Iterator>::difference_type;
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    // Number of elements in the input range.
    diff_type n = std::distance(first, last);
    if (n <= 0) {
        // Empty range: nothing to remove.
        return last;
    }

    // keep[i] == 1 => keep element i; keep[i] == 0 => remove element i
    std::vector<unsigned char> keep(static_cast<std::size_t>(n));

    // Build the keep mask in parallel.
    tbb::parallel_for(
        diff_type(0),
        n,
        [&](diff_type i) {
            // Remove if pred(...) is true, so keep is the negation.
            keep[static_cast<std::size_t>(i)] = pred(*(first + i)) ? 0 : 1;
        });

    // positions[i] stores the compacted destination index for element i (if kept),
    // or -1 for removed elements (only meaningful in the final scan pass).
    std::vector<diff_type> positions(static_cast<std::size_t>(n));

    // Parallel prefix-sum (scan) over keep[] to compute output positions.
    diff_type total_kept = tbb::parallel_scan(
        tbb::blocked_range<diff_type>(0, n),
        diff_type(0),
        [&](const tbb::blocked_range<diff_type>& r, diff_type sum, bool is_final) {
            for (diff_type i = r.begin(); i != r.end(); ++i) {
                if (keep[static_cast<std::size_t>(i)]) {
                    // If kept, its destination is the current running count.
                    if (is_final) {
                        positions[static_cast<std::size_t>(i)] = sum;
                    }
                    ++sum;
                } else if (is_final) {
                    // Mark removed elements as invalid destinations.
                    positions[static_cast<std::size_t>(i)] = diff_type(-1);
                }
            }
            return sum; // Return updated partial sum for this block.
        },
        std::plus<diff_type>());

    if (total_kept == 0) {
        // Everything removed: new end is 'first'.
        return first;
    }

    // Temporary buffer for compacted results (size = number of kept elements).
    std::vector<value_type> temp(static_cast<std::size_t>(total_kept));

    // Scatter kept elements into the temporary buffer in parallel.
    tbb::parallel_for(
        diff_type(0),
        n,
        [&](diff_type i) {
            if (keep[static_cast<std::size_t>(i)]) {
                diff_type dst                       = positions[static_cast<std::size_t>(i)];
                temp[static_cast<std::size_t>(dst)] = *(first + i);
            }
        });

    // Move compacted results back to the original range (sequential writeback).
    for (diff_type i = 0; i < total_kept; ++i) {
        *(first + i) = std::move(temp[static_cast<std::size_t>(i)]);
    }

    // New logical end after removal.
    return first + total_kept;
}

} // namespace atlas

#endif // ATLAS_TASKING_CUDA
