#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <execution>
#include <iterator>
#include <numeric>
#include <thrust/sort.h>
#include <vector>

namespace atlas {

/**
 * @brief Sorts a host key/value pair of raw arrays in ascending key order.
 *
 * Reorders @p keys ascending and applies the identical permutation to @p values
 * so the pairing is preserved, operating on plain host pointers (the fallback
 * path used by @ref parallel_sort_by_key when the device backend is not CUDA).
 * Rather than swapping key/value pairs in place, it sorts an index permutation
 * and then scatters snapshots of the original arrays through it; this needs only
 * a `less-than` on keys and avoids requiring the values to be comparable or
 * cheaply swappable. Two full-length copies (@c keys_copy, @c values_copy) are
 * allocated as the scatter source, so the routine uses O(count) extra memory.
 *
 * @tparam Parallel When true, use the C++17 parallel execution policy for both
 *                  the sort and the scatter; when false, run single-threaded.
 * @tparam Key   Element type of the keys; must be less-than comparable.
 * @tparam Value Element type of the values carried alongside the keys.
 * @param keys   In/out array of @p count keys; sorted ascending on return.
 * @param values In/out array of @p count values; permuted to match @p keys.
 * @param count  Number of elements in each array.
 * @note The underlying `std::sort` is not stable, so equal keys may have their
 *       relative order changed.
 * @warning @p keys and @p values must both point to at least @p count host
 *          elements; the pointers are read and written directly.
 */
template <bool Parallel, typename Key, typename Value>
ATLAS_FORCE_INLINE void
host_sort_by_key(Key* keys, Value* values, const std::size_t count) {
    std::vector<std::size_t> permutation(count);
    std::iota(permutation.begin(), permutation.end(), std::size_t { 0 });

    // Sort indices by the key they point at, leaving the source arrays untouched.
    const auto by_key = [keys](const std::size_t a, const std::size_t b) { return keys[a] < keys[b]; };

    // Snapshot the inputs so the scatter reads a stable source while it overwrites the originals.
    const std::vector<Key> keys_copy(keys, keys + count);
    const std::vector<Value> values_copy(values, values + count);

    if constexpr (Parallel) {
        std::sort(std::execution::par, permutation.begin(), permutation.end(), by_key);
        // A separate destination-index range is needed because std::for_each hands the
        // parallel body an element value, not its position; here that element is the position.
        std::vector<std::size_t> indices(count);
        std::iota(indices.begin(), indices.end(), std::size_t { 0 });
        std::for_each(std::execution::par, indices.begin(), indices.end(), [&](const std::size_t i) {
            keys[i]   = keys_copy[permutation[i]];
            values[i] = values_copy[permutation[i]];
        });
    } else {
        std::sort(permutation.begin(), permutation.end(), by_key);
        for (std::size_t i = 0; i < count; ++i) {
            keys[i]   = keys_copy[permutation[i]];
            values[i] = values_copy[permutation[i]];
        }
    }
}

/**
 * @brief Sorts the range [first, last) into ascending order on the chosen backend.
 *
 * Dispatches at compile time. Only the pairing of `device` policy *and* a CUDA
 * Thrust device backend takes the `thrust::sort(thrust::device, ...)` GPU path;
 * every other combination (host, serial, or `device` when CUDA is unavailable)
 * falls through to a host `std::sort`. On that host path the iterators are first
 * lowered to raw pointers via @ref raw_pointer_cast so `std::sort` operates on
 * contiguous host memory — which requires the range to be contiguous and
 * host-accessible. `serial` runs single-threaded; `host` uses the C++17 parallel
 * policy.
 *
 * @tparam P        Execution backend to run on.
 * @tparam RandomIt Random-access iterator over the range to sort.
 * @param first Beginning of the range.
 * @param last  One past the end of the range.
 * @note Returns immediately when the range is empty (`first == last`).
 * @warning The compile-time device branch is only taken when the range lives in
 *          device memory; requesting `device` on a non-CUDA build silently sorts
 *          on the host, so callers must not pass a device pointer in that case.
 */
template <ExecutionPolicy P, typename RandomIt>
ATLAS_FORCE_INLINE void
parallel_sort(RandomIt first, RandomIt last) {
    if (first == last) return;

    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        thrust::sort(thrust::device, first, last);
    } else {
        // Lower iterators to raw host pointers so std::sort sees a contiguous range.
        auto* raw_first = atlas::raw_pointer_cast(&*first);
        auto* raw_last  = raw_first + static_cast<std::ptrdiff_t>(last - first);
        if constexpr (P == ExecutionPolicy::serial) {
            std::sort(raw_first, raw_last);
        } else {
            std::sort(std::execution::par, raw_first, raw_last);
        }
    }
}

/**
 * @brief Sorts keys ascending and reorders a parallel value range to match.
 *
 * The keyed counterpart to @ref parallel_sort, used to bring a payload array
 * (for example particle indices) along with its sort keys (for example cell
 * ids). Mirrors the same dispatch: `device` on a CUDA backend goes straight to
 * `thrust::sort_by_key` on the GPU; otherwise the iterators are lowered to raw
 * host pointers and forwarded to @ref host_sort_by_key, whose `Parallel`
 * template argument is set from whether @p P is anything other than `serial`.
 * The values range is assumed to be at least as long as the keys range.
 *
 * @tparam P       Execution backend to run on.
 * @tparam KeyIt   Random-access iterator over the keys.
 * @tparam ValueIt Random-access iterator over the values, parallel to the keys.
 * @param keys_first   Beginning of the key range; sorted ascending on return.
 * @param keys_last    One past the end of the key range.
 * @param values_first Beginning of the value range; permuted to track the keys.
 * @note Returns immediately when the key range is empty.
 * @warning As with @ref parallel_sort, the host fallback requires contiguous,
 *          host-accessible memory, so requesting `device` on a non-CUDA build
 *          must not be given device pointers.
 */
template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
ATLAS_FORCE_INLINE void
parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
    if (keys_first == keys_last) return;

    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    } else {
        // Host fallback: recover contiguous pointers and delegate to the index-permutation sort.
        auto* keys              = atlas::raw_pointer_cast(&*keys_first);
        auto* values            = atlas::raw_pointer_cast(&*values_first);
        const std::size_t count = static_cast<std::size_t>(keys_last - keys_first);
        host_sort_by_key<P != ExecutionPolicy::serial>(keys, values, count);
    }
}

}