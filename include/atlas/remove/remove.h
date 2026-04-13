#pragma once

/**
 * @file remove_if.h
 * @brief Declares a backend-portable `remove_if` utility and its lightweight execution-policy tag.
 *
 * @details
 * This header provides a unified Atlas-level `remove_if` interface that can be
 * used across different execution backends while preserving a consistent call
 * pattern.
 *
 * The API is centered around:
 * - @ref atlas::device_policy_t, a lightweight execution-policy tag used to
 *   select the device-oriented removal path,
 * - @ref atlas::device, a global constexpr instance of that policy tag,
 * - @ref atlas::remove_if, a backend-portable wrapper that removes elements
 *   satisfying a predicate from a range and returns the new logical end.
 *
 * ## High-level semantics
 * The function follows the familiar "erase-remove" style convention:
 * - elements for which the predicate returns `true` are removed logically,
 * - retained elements are compacted toward the beginning of the range,
 * - the function returns an iterator to the new logical end of the kept range,
 * - elements beyond the returned iterator remain unspecified for logical use.
 *
 * ## Backend split
 * The implementation is selected at compile time:
 * - when `ATLAS_TASKING_CUDA` is enabled, the function delegates to
 *   `thrust::remove_if` using Thrust's device execution backend,
 * - otherwise, a host-side parallel implementation based on Intel oneTBB is used.
 *
 * ## Host fallback algorithm
 * In the non-CUDA path, the implementation proceeds in several stages:
 * 1. evaluate the predicate for each element and store a keep-mask,
 * 2. compute compacted destination indices via a parallel prefix scan,
 * 3. scatter retained values into a temporary buffer,
 * 4. move the compacted values back into the original range,
 * 5. return the iterator to the new logical end.
 *
 * This approach allows parallel compaction while preserving the relative order
 * of retained elements.
 *
 * ## Stability
 * The fallback implementation preserves the original order of the retained
 * elements because destination positions are assigned by prefix order.
 *
 * ## Iterator requirements
 * The host fallback uses expressions such as:
 * - `first + i`
 * - `std::distance(first, last)`
 *
 * so it effectively requires iterators with random-access-like behavior.
 *
 * ---
 */

#include <iterator>

#ifdef ATLAS_TASKING_CUDA
#include <thrust/execution_policy.h>
#include <thrust/remove.h>

namespace atlas {

/**
 * @brief Lightweight execution-policy tag representing Atlas device execution.
 *
 * @details
 * This tag type is used to select the backend-portable device-oriented overload
 * of algorithms such as @ref remove_if.
 *
 * In CUDA-enabled builds, this maps naturally to device execution through Thrust.
 */
struct device_policy_t { };

/**
 * @brief Global constexpr instance of the device execution policy.
 *
 * @details
 * This object is intended to be passed as the first argument to Atlas algorithms
 * that accept an execution-policy selector.
 */
static constexpr device_policy_t device {};

/**
 * @brief Remove elements satisfying a predicate using the CUDA/Thrust backend.
 *
 * @details
 * This overload forwards directly to `thrust::remove_if` with the Thrust device
 * execution policy.
 *
 * The function compacts the elements for which `pred(element)` is `false` toward
 * the beginning of the range and returns the new logical end.
 *
 * @param Policy Execution policy type. The value is accepted for API uniformity.
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 * @param pred Unary predicate returning `true` for elements to remove.
 * @return Iterator to the new logical end of the retained range.
 *
 * @tparam Policy Execution-policy tag type.
 * @tparam Iterator Iterator type over the target range.
 * @tparam Predicate Unary predicate type.
 */
template <typename Policy, typename Iterator, typename Predicate>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Iterator
    remove_if(Policy, Iterator first, Iterator last, Predicate pred) {

    return thrust::remove_if(thrust::device, first, last, pred);
}

} // namespace atlas

#else

#include <functional>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_scan.h>
#include <vector>

namespace atlas {

/**
 * @brief Lightweight execution-policy tag representing Atlas device-style execution.
 *
 * @details
 * In non-CUDA builds, this tag still exists so higher-level code can keep a
 * uniform API even though the implementation is carried out by a host-side
 * parallel backend.
 */
struct device_policy_t { };

/**
 * @brief Global constexpr instance of the device execution policy.
 *
 * @details
 * This object is intended to be passed as the first argument to Atlas algorithms
 * that accept an execution-policy selector.
 */
static constexpr device_policy_t device {};

/**
 * @brief Remove elements satisfying a predicate using the host-side parallel backend.
 *
 * @details
 * This function performs a stable parallel compaction of the input range:
 * - elements for which `pred(element)` returns `true` are removed logically,
 * - elements for which the predicate returns `false` are retained,
 * - retained elements are moved to the front of the range in original order,
 * - the function returns an iterator to the new logical end.
 *
 * ## Algorithm
 * The implementation consists of four major phases:
 * 1. **Predicate evaluation**
 *    A byte mask named `keep` is filled in parallel, where:
 *    - `1` means keep the element,
 *    - `0` means remove the element.
 *
 * 2. **Prefix-scan compaction indexing**
 *    A parallel scan computes the compacted destination position of each kept
 *    element and records those positions in `positions`.
 *
 * 3. **Parallel scatter**
 *    Retained elements are copied into a temporary contiguous buffer according
 *    to their computed compacted positions.
 *
 * 4. **Write-back**
 *    The temporary compacted sequence is moved back into the original range.
 *
 * ## Stability
 * This implementation preserves the relative order of retained elements.
 *
 * ## Complexity
 * - Linear work in the number of input elements,
 * - additional temporary storage proportional to the input size,
 * - parallel execution for predicate evaluation, scan, and scatter phases.
 *
 * ## Iterator assumptions
 * This implementation assumes iterator operations compatible with:
 * - `std::distance(first, last)`,
 * - `first + i`,
 * - dereference of `*(first + i)`.
 *
 * In practice, this means the iterator is expected to behave like a random-access iterator.
 *
 * @tparam Policy Execution policy type. The value is accepted for API uniformity.
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 * @param pred Unary predicate returning `true` for elements to remove.
 * @return Iterator to the new logical end of the retained range.
 *
 * @tparam Policy Execution-policy tag type.
 * @tparam Iterator Iterator type over the target range.
 * @tparam Predicate Unary predicate type.
 */
template <typename Policy, typename Iterator, typename Predicate>
inline Iterator
remove_if(Policy, Iterator first, Iterator last, Predicate pred) {
    /**
     * @brief Signed distance type associated with the iterator.
     */
    using diff_type = typename std::iterator_traits<Iterator>::difference_type;

    /**
     * @brief Value type stored in the iterator range.
     */
    using value_type = typename std::iterator_traits<Iterator>::value_type;

    /**
     * @brief Total number of elements in the input range.
     */
    diff_type n = std::distance(first, last);

    if (n <= 0) {

        return last;
    }

    /**
     * @brief Per-element keep mask.
     *
     * @details
     * Entry `i` is:
     * - `1` if element `i` should be retained,
     * - `0` if element `i` should be removed.
     */
    std::vector<unsigned char> keep(static_cast<std::size_t>(n));

    /**
     * @brief Evaluate the predicate in parallel and build the keep mask.
     *
     * @details
     * An element is kept when the predicate returns `false`.
     */
    tbb::parallel_for(
        diff_type(0),
        n,
        [&](diff_type i) {
            keep[static_cast<std::size_t>(i)] = pred(*(first + i)) ? 0 : 1;
        });

    /**
     * @brief Destination positions for retained elements.
     *
     * @details
     * For kept elements, `positions[i]` stores the compacted destination index.
     * For removed elements, the position is later set to `-1`.
     */
    std::vector<diff_type> positions(static_cast<std::size_t>(n));

    /**
     * @brief Total number of retained elements after compaction.
     *
     * @details
     * Computed by a parallel prefix scan over the keep mask.
     */
    diff_type total_kept = tbb::parallel_scan(
        tbb::blocked_range<diff_type>(0, n),
        diff_type(0),
        [&](const tbb::blocked_range<diff_type>& r, diff_type sum, bool is_final) {
            for (diff_type i = r.begin(); i != r.end(); ++i) {
                if (keep[static_cast<std::size_t>(i)]) {

                    if (is_final) {
                        positions[static_cast<std::size_t>(i)] = sum;
                    }
                    ++sum;
                } else if (is_final) {

                    positions[static_cast<std::size_t>(i)] = diff_type(-1);
                }
            }
            return sum;
        },
        std::plus<diff_type>());

    /**
     * @brief Fast path for the case where no elements are retained.
     *
     * @details
     * When every element is removed, the new logical end is the beginning of the range.
     */
    if (total_kept == 0) {

        return first;
    }

    /**
     * @brief Temporary compacted storage for retained elements.
     *
     * @details
     * This buffer holds the kept values in their final compacted order before
     * they are moved back into the original range.
     */
    std::vector<value_type> temp(static_cast<std::size_t>(total_kept));

    /**
     * @brief Scatter retained elements into their compacted positions in parallel.
     */
    tbb::parallel_for(
        diff_type(0),
        n,
        [&](diff_type i) {
            if (keep[static_cast<std::size_t>(i)]) {
                diff_type dst                       = positions[static_cast<std::size_t>(i)];
                temp[static_cast<std::size_t>(dst)] = *(first + i);
            }
        });

    /**
     * @brief Move the compacted retained elements back into the original range prefix.
     *
     * @details
     * After this loop, the valid retained range is `[first, first + total_kept)`.
     */
    for (diff_type i = 0; i < total_kept; ++i) {
        *(first + i) = std::move(temp[static_cast<std::size_t>(i)]);
    }

    return first + total_kept;
}

} // namespace atlas

#endif