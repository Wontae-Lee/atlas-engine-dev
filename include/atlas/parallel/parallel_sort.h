#pragma once
#include <algorithm>
#include <atlas/parallel/parallel_for.h>
#include <numeric>
#include <vector>

/**
 * @file parallel_sort.h
 * @brief Backend-dispatched parallel sorting utilities (Thrust/TBB/serial).
 *
 * @details
 * This header provides two algorithms that mirror common Thrust/TBB utilities while keeping
 * call sites backend-agnostic:
 *
 * - `atlas::parallel_sort<P>(first, last)`
 * - `atlas::parallel_sort_by_key<P>(keys_first, keys_last, values_first)`
 *
 * Dispatch is controlled by:
 * - Build flag `ATLAS_TASKING_CUDA`
 * - Compile-time `atlas::ExecutionPolicy` template parameter `P`
 *
 * Backends:
 * - **CUDA build (`ATLAS_TASKING_CUDA`)**
 *   - `host`   -> Thrust host policy
 *   - `device` -> Thrust device policy
 *   - `serial` -> `std::sort` / CPU stable key-value reorder fallback
 *
 * - **Non-CUDA build**
 *   - `host`   -> `tbb::parallel_sort` for `parallel_sort`
 *   - `device` -> same as host (device policy maps to CPU fallback)
 *   - `serial` -> `std::sort` / CPU key-value reorder fallback
 *
 * @note
 * - `parallel_sort` expects **random-access iterators** (required by `std::sort`, Thrust sort,
 *   and TBB parallel_sort).
 * - `parallel_sort_by_key` expects key/value ranges of equal length and that `values_first`
 *   points to a range of at least `distance(keys_first, keys_last)` elements.
 * - The provided serial `sort_by_key` implementation uses temporary buffers and an index
 *   permutation; it is stable with respect to equal keys only if `std::sort` on indices is
 *   stable (it is not). If stable behavior is needed, consider `std::stable_sort` there.
 */

#if defined(ATLAS_TASKING_CUDA)
#include <thrust/sort.h>

namespace atlas {
namespace detail {

    /**
     * @brief Host sort implementation using Thrust host execution policy.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::host, first, last);
    }

    /**
     * @brief Device sort implementation using Thrust device execution policy.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::device, first, last);
    }

    /**
     * @brief Serial sort implementation using `std::sort`.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

    /**
     * @brief Serial `sort_by_key` implementation that reorders keys and values together.
     *
     * @details
     * Builds an index permutation based on the key order and then writes the reordered
     * key/value pairs back into the original ranges.
     *
     * @note
     * - Allocates O(n) temporary storage for indices, keys, and values.
     * - Not guaranteed stable for equal keys due to `std::sort`.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_serial_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        using diff_t = typename std::iterator_traits<KeyIt>::difference_type;

        diff_t n = std::distance(keys_first, keys_last);
        if (n <= 1) return;

        // Build index permutation [0..n-1].
        std::vector<diff_t> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), diff_t(0));

        // Copy input keys/values into temporaries so we can reorder safely.
        std::vector<typename std::iterator_traits<KeyIt>::value_type> key_tmp(static_cast<std::size_t>(n));
        std::vector<typename std::iterator_traits<ValueIt>::value_type> val_tmp(static_cast<std::size_t>(n));
        {
            KeyIt k_it   = keys_first;
            ValueIt v_it = values_first;
            for (diff_t i = 0; i < n; ++i, ++k_it, ++v_it) {
                key_tmp[static_cast<std::size_t>(i)] = *k_it;
                val_tmp[static_cast<std::size_t>(i)] = *v_it;
            }
        }

        // Sort indices by corresponding key.
        std::sort(indices.begin(),
                  indices.end(),
                  [&key_tmp](diff_t a, diff_t b) {
                      return key_tmp[static_cast<std::size_t>(a)] < key_tmp[static_cast<std::size_t>(b)];
                  });

        // Write back reordered keys/values.
        {
            KeyIt k_it   = keys_first;
            ValueIt v_it = values_first;
            for (diff_t i = 0; i < n; ++i, ++k_it, ++v_it) {
                const diff_t idx = indices[static_cast<std::size_t>(i)];
                *k_it            = key_tmp[static_cast<std::size_t>(idx)];
                *v_it            = val_tmp[static_cast<std::size_t>(idx)];
            }
        }
    }

    /**
     * @brief Host `sort_by_key` implementation using Thrust host execution policy.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::host, keys_first, keys_last, values_first);
    }

    /**
     * @brief Device `sort_by_key` implementation using Thrust device execution policy.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    }

} // namespace detail

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

/**
 * @brief Sorts the range `[first, last)` using the selected execution policy.
 *
 * @tparam P        Execution policy (host/device/serial).
 * @tparam RandomIt Random-access iterator type.
 *
 * @param first Range begin.
 * @param last  Range end.
 */
template <ExecutionPolicy P, typename RandomIt>
ATLAS_FORCE_INLINE void
parallel_sort(RandomIt first, RandomIt last) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_sort_host_impl(first, last);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_sort_device_impl(first, last);
    } else {
        detail::parallel_sort_serial_impl(first, last);
    }
}

/**
 * @brief Sorts keys in `[keys_first, keys_last)` and reorders values starting at `values_first` accordingly.
 *
 * @tparam P       Execution policy (host/device/serial).
 * @tparam KeyIt   Random-access iterator over keys.
 * @tparam ValueIt Iterator over values (must have at least the same length as keys).
 *
 * @param keys_first Begin iterator for keys.
 * @param keys_last  End iterator for keys.
 * @param values_first Begin iterator for associated values.
 *
 * @note
 * - In CUDA builds, this maps to `thrust::sort_by_key` for host/device policies.
 * - In serial policy, a CPU permutation-based reorder is used.
 */
template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
ATLAS_FORCE_INLINE void
parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_sort_by_key_host_impl(keys_first, keys_last, values_first);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_sort_by_key_device_impl(keys_first, keys_last, values_first);
    } else {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }
}

} // namespace atlas

#else // --------------------------------------------------------
// Non-CUDA build (TBB for sort; serial fallback for sort_by_key)
// --------------------------------------------------------

#include <tbb/tbb.h>

namespace atlas {
namespace detail {

    /**
     * @brief Host sort implementation using TBB parallel_sort.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        tbb::parallel_sort(first, last);
    }

    /**
     * @brief "Device" sort implementation in non-CUDA builds (maps to host).
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        detail::parallel_sort_host_impl(first, last);
    }

    /**
     * @brief Serial sort implementation using `std::sort`.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

    /**
     * @brief Serial `sort_by_key` implementation that reorders keys and values together.
     *
     * @details
     * Uses the same permutation strategy as the CUDA file's serial path.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_serial_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        using diff_t = typename std::iterator_traits<KeyIt>::difference_type;

        diff_t n = std::distance(keys_first, keys_last);
        if (n <= 1) return;

        std::vector<diff_t> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), diff_t(0));

        std::vector<typename std::iterator_traits<KeyIt>::value_type> key_tmp(static_cast<std::size_t>(n));
        std::vector<typename std::iterator_traits<ValueIt>::value_type> val_tmp(static_cast<std::size_t>(n));
        {
            KeyIt k_it   = keys_first;
            ValueIt v_it = values_first;
            for (diff_t i = 0; i < n; ++i, ++k_it, ++v_it) {
                key_tmp[static_cast<std::size_t>(i)] = *k_it;
                val_tmp[static_cast<std::size_t>(i)] = *v_it;
            }
        }

        std::sort(indices.begin(),
                  indices.end(),
                  [&key_tmp](diff_t a, diff_t b) {
                      return key_tmp[static_cast<std::size_t>(a)] < key_tmp[static_cast<std::size_t>(b)];
                  });

        {
            KeyIt k_it   = keys_first;
            ValueIt v_it = values_first;
            for (diff_t i = 0; i < n; ++i, ++k_it, ++v_it) {
                const diff_t idx = indices[static_cast<std::size_t>(i)];
                *k_it            = key_tmp[static_cast<std::size_t>(idx)];
                *v_it            = val_tmp[static_cast<std::size_t>(idx)];
            }
        }
    }

    /**
     * @brief Host `sort_by_key` implementation in non-CUDA builds (falls back to serial permutation).
     *
     * @note
     * If you later want a true parallel `sort_by_key` on CPU, you can implement it via
     * zipped iterators + `tbb::parallel_sort` or a parallel stable sort.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }

    /**
     * @brief "Device" `sort_by_key` implementation in non-CUDA builds (maps to host fallback).
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }

} // namespace detail

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

/**
 * @brief Sorts the range `[first, last)` using the selected execution policy.
 */
template <ExecutionPolicy P, typename RandomIt>
ATLAS_FORCE_INLINE void
parallel_sort(RandomIt first, RandomIt last) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_sort_host_impl(first, last);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_sort_device_impl(first, last);
    } else {
        detail::parallel_sort_serial_impl(first, last);
    }
}

/**
 * @brief Sorts keys and reorders values with them.
 *
 * @note
 * In non-CUDA builds, host/device policies currently use the same serial permutation fallback.
 */
template <ExecutionPolicy P, typename KeyIt, typename ValueIt>
ATLAS_FORCE_INLINE void
parallel_sort_by_key(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
    if constexpr (P == ExecutionPolicy::host) {
        detail::parallel_sort_by_key_host_impl(keys_first, keys_last, values_first);
    } else if constexpr (P == ExecutionPolicy::device) {
        detail::parallel_sort_by_key_device_impl(keys_first, keys_last, values_first);
    } else {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }
}

} // namespace atlas
#endif // ATLAS_TASKING_CUDA
