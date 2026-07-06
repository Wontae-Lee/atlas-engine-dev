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

namespace detail {

    // sort_by_key on host memory: std has no sort_by_key, so sort an index
    // permutation by key and scatter both arrays through it. `Parallel` picks
    // the parallel STL (TBB-backed) vs the sequential algorithms.
    template <bool Parallel, typename Key, typename Value>
    ATLAS_FORCE_INLINE void
    host_sort_by_key(Key* keys, Value* values, const std::size_t count) {
        std::vector<std::size_t> permutation(count);
        std::iota(permutation.begin(), permutation.end(), std::size_t { 0 });

        const auto by_key = [keys](const std::size_t a, const std::size_t b) { return keys[a] < keys[b]; };

        const std::vector<Key> keys_copy(keys, keys + count);
        const std::vector<Value> values_copy(values, values + count);

        if constexpr (Parallel) {
            std::sort(std::execution::par, permutation.begin(), permutation.end(), by_key);
            std::vector<std::size_t> indices(count);
            std::iota(indices.begin(), indices.end(), std::size_t { 0 });
            std::for_each(std::execution::par, indices.begin(), indices.end(),
                          [&](const std::size_t i) {
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

}

// Only the CUDA device policy is safe with thrust here: thrust's temp-allocating
// algorithms route through a __host__ __device__ temporary_allocator whose
// bad_alloc throw is guarded ONLY for the CUDA device system. Under the TBB
// build every other thrust policy (host, seq, tbb-device) is still
// __host__ __device__, so nvcc device-compiles the illegal throw. Use the STL
// for those — parallel for host/device (libstdc++ backs par with TBB),
// sequential for the serial policy.

template <ExecutionPolicy P, typename RandomIt>
ATLAS_FORCE_INLINE void
parallel_sort(RandomIt first, RandomIt last) {
    if (first == last) return;

    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        thrust::sort(thrust::device, first, last);
    } else {
        auto* raw_first = atlas::raw_pointer_cast(&*first);
        auto* raw_last  = raw_first + static_cast<std::ptrdiff_t>(last - first);
        if constexpr (P == ExecutionPolicy::serial) {
            std::sort(raw_first, raw_last);
        } else {
            std::sort(std::execution::par, raw_first, raw_last);
        }
    }
}

template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
ATLAS_FORCE_INLINE void
parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
    if (keys_first == keys_last) return;

    if constexpr (P == ExecutionPolicy::device && THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA) {
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    } else {
        auto* keys              = atlas::raw_pointer_cast(&*keys_first);
        auto* values            = atlas::raw_pointer_cast(&*values_first);
        const std::size_t count = static_cast<std::size_t>(keys_last - keys_first);
        detail::host_sort_by_key<P != ExecutionPolicy::serial>(keys, values, count);
    }
}

}
