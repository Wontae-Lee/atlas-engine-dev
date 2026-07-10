#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#if defined(ATLAS_BACKEND_CUDA)
#include <thrust/sort.h>
#endif

namespace atlas {

/**
 * @brief Sorts a host key/value pair of raw arrays in ascending key order.
 *
 * Reorders @p keys ascending and applies the identical permutation to @p values so the
 * pairing is preserved, operating on plain host pointers. Rather than swapping key/value
 * pairs in place, it sorts an index permutation and then scatters snapshots of the
 * original arrays through it; this needs only a less-than on keys and avoids requiring the
 * values to be comparable or cheaply swappable. Two full-length copies (@c keys_copy,
 * @c values_copy) are allocated as the scatter source, so the routine uses O(count) extra
 * memory.
 *
 * @tparam Parallel When true, sort and scatter through TBB; when false, run
 *                  single-threaded.
 * @tparam Key   Element type of the keys; must be less-than comparable.
 * @tparam Value Element type of the values carried alongside the keys.
 * @param keys   In/out array of @p count keys; sorted ascending on return.
 * @param values In/out array of @p count values; permuted to match @p keys.
 * @param count  Number of elements in each array.
 * @note Neither @c std::sort nor @c tbb::parallel_sort is stable, so equal keys may have
 *       their relative order changed.
 * @warning @p keys and @p values must both point to at least @p count host elements; the
 *          pointers are read and written directly.
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
        tbb::parallel_sort(permutation.begin(), permutation.end(), by_key);

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count),
                          [&](const tbb::blocked_range<std::size_t>& range) {
                              for (std::size_t i = range.begin(); i != range.end(); ++i) {
                                  keys[i]   = keys_copy[permutation[i]];
                                  values[i] = values_copy[permutation[i]];
                              }
                          });
    } else {
        std::sort(permutation.begin(), permutation.end(), by_key);

        for (std::size_t i = 0; i < count; ++i) {
            keys[i]   = keys_copy[permutation[i]];
            values[i] = values_copy[permutation[i]];
        }
    }
}

namespace detail {

    /**
     * @brief Sorts a contiguous, host-accessible range through @ref raw_pointer_cast.
     *
     * Kept out of @ref parallel_sort so the CUDA backend's device branch never
     * instantiates it with a device iterator.
     *
     * @tparam Parallel Use @c tbb::parallel_sort when true, @c std::sort otherwise.
     * @tparam RandomIt Random-access iterator over contiguous host storage.
     * @param first Beginning of the range.
     * @param last One past the end.
     */
    template <bool Parallel, typename RandomIt>
    ATLAS_FORCE_INLINE void
    host_sort_range(RandomIt first, RandomIt last) {
        auto* raw_first = atlas::raw_pointer_cast(&*first);
        auto* raw_last  = raw_first + static_cast<std::ptrdiff_t>(last - first);

        if constexpr (Parallel) {
            tbb::parallel_sort(raw_first, raw_last);
        } else {
            std::sort(raw_first, raw_last);
        }
    }

    /**
     * @brief Lowers a key/value iterator pair to raw pointers and calls @ref host_sort_by_key.
     *
     * Same reason for existing as @ref host_sort_range: it must not be instantiated for a
     * device iterator.
     *
     * @tparam Parallel Forwarded to @ref host_sort_by_key.
     * @tparam KeyIt Random-access iterator over the keys.
     * @tparam ValueIt Random-access iterator over the values.
     * @param keys_first Beginning of the key range.
     * @param keys_last One past the end of the key range.
     * @param values_first Beginning of the value range.
     */
    template <bool Parallel, typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    host_sort_by_key_range(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        auto* keys              = atlas::raw_pointer_cast(&*keys_first);
        auto* values            = atlas::raw_pointer_cast(&*values_first);
        const std::size_t count = static_cast<std::size_t>(keys_last - keys_first);

        atlas::host_sort_by_key<Parallel>(keys, values, count);
    }

}

/**
 * @brief Sorts the range [first, last) into ascending order on the chosen backend.
 *
 * Dispatches at compile time. Only @c device under the CUDA backend takes the
 * @c thrust::sort GPU path; every other combination lowers the iterators to raw pointers
 * and sorts contiguous host memory. @c serial runs single-threaded; the parallel policies
 * use @c tbb::parallel_sort.
 *
 * @tparam P        Execution backend to run on.
 * @tparam RandomIt Random-access iterator over the range to sort.
 * @param first Beginning of the range.
 * @param last  One past the end of the range.
 * @note Returns immediately when the range is empty (`first == last`).
 * @note Under the host backend @c device sorts on the host, which is correct there because
 *       a @c DeviceBuffer *is* host memory.
 */
template <ExecutionPolicy P, typename RandomIt>
ATLAS_FORCE_INLINE void
parallel_sort(RandomIt first, RandomIt last) {
    if (first == last) return;

#if defined(ATLAS_BACKEND_CUDA)
    if constexpr (P == ExecutionPolicy::device) {
        thrust::sort(thrust::device, first, last);
    } else {
        detail::host_sort_range<P != ExecutionPolicy::serial>(first, last);
    }
#else
    detail::host_sort_range<P != ExecutionPolicy::serial>(first, last);
#endif
}

/**
 * @brief Sorts keys ascending and reorders a parallel value range to match.
 *
 * The keyed counterpart to @ref parallel_sort, used to bring a payload array (particle
 * indices) along with its sort keys (cell ids). Mirrors the same dispatch: @c device under
 * the CUDA backend goes straight to @c thrust::sort_by_key on the GPU; otherwise the
 * iterators are lowered to raw pointers and forwarded to @ref host_sort_by_key.
 *
 * @tparam P       Execution backend to run on.
 * @tparam KeyIt   Random-access iterator over the keys.
 * @tparam ValueIt Random-access iterator over the values, parallel to the keys.
 * @param keys_first   Beginning of the key range; sorted ascending on return.
 * @param keys_last    One past the end of the key range.
 * @param values_first Beginning of the value range; permuted to track the keys. Must be
 *                     at least as long as the key range.
 * @note Returns immediately when the key range is empty.
 */
template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
ATLAS_FORCE_INLINE void
parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
    if (keys_first == keys_last) return;

#if defined(ATLAS_BACKEND_CUDA)
    if constexpr (P == ExecutionPolicy::device) {
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    } else {
        detail::host_sort_by_key_range<P != ExecutionPolicy::serial>(keys_first, keys_last, values_first);
    }
#else
    detail::host_sort_by_key_range<P != ExecutionPolicy::serial>(keys_first, keys_last, values_first);
#endif
}

}
