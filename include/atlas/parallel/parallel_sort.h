#pragma once

/**
 * @file parallel_sort.h
 * @brief Declares backend-portable sorting utilities parameterized by execution policy.
 *
 * @details
 * This header defines:
 * - @ref atlas::parallel_sort, a policy-driven range sorting utility,
 * - @ref atlas::parallel_sort_by_key, a policy-driven key-value paired sorting utility.
 *
 * The goal of these utilities is to provide a consistent Atlas-level interface
 * for sorting operations across multiple execution environments while allowing
 * compile-time dispatch to the most appropriate backend implementation.
 *
 * ## Supported execution styles
 * Sorting behavior is selected through the compile-time @ref ExecutionPolicy:
 * - `ExecutionPolicy::host`
 * - `ExecutionPolicy::device`
 * - a fallback serial path for all other policies
 *
 * ## CUDA-enabled builds
 * When `ATLAS_TASKING_CUDA` is defined:
 * - host sorting uses Thrust host execution,
 * - device sorting uses Thrust device execution,
 * - serial fallback uses the C++ standard library.
 *
 * In this configuration:
 * - @ref parallel_sort dispatches to `thrust::sort` or `std::sort`,
 * - @ref parallel_sort_by_key dispatches to `thrust::sort_by_key` or a serial
 *   index-based fallback implementation.
 *
 * ## Non-CUDA builds
 * When CUDA tasking is not enabled:
 * - host sorting uses `tbb::parallel_sort`,
 * - device sorting falls back to the host implementation,
 * - serial fallback uses `std::sort`.
 *
 * For key-value sorting in non-CUDA builds, both host and device paths currently
 * use a serial index-based reorder implementation.
 *
 * ## Key-value sorting semantics
 * @ref parallel_sort_by_key sorts the key range in ascending order and reorders
 * the associated value range so that each value remains paired with its original key.
 *
 * The fallback serial implementation works by:
 * 1. copying keys and values into temporary buffers,
 * 2. building an index permutation,
 * 3. sorting that permutation by comparing copied keys,
 * 4. writing the reordered keys and values back into the original ranges.
 *
 * ## Iterator expectations
 * - Plain range sorting expects iterators acceptable to the selected backend sort.
 * - Key-value sorting expects:
 *   - a valid key range `[keys_first, keys_last)`,
 *   - a value range beginning at `values_first` with matching logical length.
 *
 * ---
 */

#include <algorithm>
#include <atlas/parallel/parallel_for.h>
#include <numeric>
#include <vector>

#if defined(ATLAS_TASKING_CUDA)
#include <thrust/sort.h>

namespace atlas {
namespace detail {

    /**
     * @brief Sort a range using the host backend in CUDA-enabled builds.
     *
     * @details
     * Delegates to `thrust::sort` with Thrust's host execution policy.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::host, first, last);
    }

    /**
     * @brief Sort a range using the device backend in CUDA-enabled builds.
     *
     * @details
     * Delegates to `thrust::sort` with Thrust's device execution policy.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::device, first, last);
    }

    /**
     * @brief Sort a range using the serial fallback implementation.
     *
     * @details
     * Delegates to `std::sort`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

    /**
     * @brief Serial fallback implementation for sorting keys and associated values together.
     *
     * @details
     * This routine performs a stable pair-preserving reorder in ascending key order by:
     * - copying the key and value ranges into temporary buffers,
     * - constructing an index array `[0, 1, ..., n-1]`,
     * - sorting the indices by comparing copied keys,
     * - writing keys and values back according to the sorted permutation.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
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
     * @brief Sort keys and associated values using the host backend in CUDA-enabled builds.
     *
     * @details
     * Delegates to `thrust::sort_by_key` with Thrust's host execution policy.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::host, keys_first, keys_last, values_first);
    }

    /**
     * @brief Sort keys and associated values using the device backend in CUDA-enabled builds.
     *
     * @details
     * Delegates to `thrust::sort_by_key` with Thrust's device execution policy.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    }

} // namespace detail

/**
 * @brief Sort a range according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to `P`:
 * - `ExecutionPolicy::host`   -> host backend sort,
 * - `ExecutionPolicy::device` -> device backend sort,
 * - otherwise                 -> serial fallback sort.
 *
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam RandomIt Random-access iterator type.
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
 * @brief Sort keys and associated values according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to `P`:
 * - `ExecutionPolicy::host`   -> host backend key-value sort,
 * - `ExecutionPolicy::device` -> device backend key-value sort,
 * - otherwise                 -> serial fallback key-value sort.
 *
 * The key range is sorted in ascending order, and the value range is permuted
 * so that key-value associations are preserved.
 *
 * @param keys_first Iterator to the beginning of the key range.
 * @param keys_last Iterator to the end of the key range.
 * @param values_first Iterator to the beginning of the associated value range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam KeyIt Iterator type over keys.
 * @tparam ValueIt Iterator type over values.
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

#else

#include <tbb/tbb.h>

namespace atlas {
namespace detail {

    /**
     * @brief Sort a range using the host backend in non-CUDA builds.
     *
     * @details
     * Delegates to `tbb::parallel_sort`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        tbb::parallel_sort(first, last);
    }

    /**
     * @brief Sort a range using the device backend in non-CUDA builds.
     *
     * @details
     * Since no dedicated device backend is available, this path falls back to the
     * host parallel implementation.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        detail::parallel_sort_host_impl(first, last);
    }

    /**
     * @brief Sort a range using the serial fallback implementation.
     *
     * @details
     * Delegates to `std::sort`.
     *
     * @param first Iterator to the beginning of the range.
     * @param last Iterator to the end of the range.
     *
     * @tparam RandomIt Random-access iterator type.
     */
    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

    /**
     * @brief Serial fallback implementation for sorting keys and associated values together.
     *
     * @details
     * This routine performs a pair-preserving reorder in ascending key order by:
     * - copying keys and values into temporary buffers,
     * - constructing an index array,
     * - sorting indices by copied keys,
     * - writing keys and values back according to the sorted permutation.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
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
     * @brief Sort keys and associated values using the host backend in non-CUDA builds.
     *
     * @details
     * This currently falls back to the serial index-based implementation.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }

    /**
     * @brief Sort keys and associated values using the device backend in non-CUDA builds.
     *
     * @details
     * Since no dedicated device backend is available, this currently falls back to
     * the serial index-based implementation.
     *
     * @param keys_first Iterator to the beginning of the key range.
     * @param keys_last Iterator to the end of the key range.
     * @param values_first Iterator to the beginning of the associated value range.
     *
     * @tparam KeyIt Iterator type over keys.
     * @tparam ValueIt Iterator type over values.
     */
    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_serial_impl(keys_first, keys_last, values_first);
    }

} // namespace detail

/**
 * @brief Sort a range according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to `P`:
 * - `ExecutionPolicy::host`   -> host parallel sort,
 * - `ExecutionPolicy::device` -> device path, which falls back to host sort,
 * - otherwise                 -> serial fallback sort.
 *
 * @param first Iterator to the beginning of the range.
 * @param last Iterator to the end of the range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam RandomIt Random-access iterator type.
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
 * @brief Sort keys and associated values according to the selected execution policy.
 *
 * @details
 * Dispatches at compile time to the backend implementation corresponding to `P`:
 * - `ExecutionPolicy::host`   -> host key-value sort,
 * - `ExecutionPolicy::device` -> device path, currently falling back to the same implementation,
 * - otherwise                 -> serial fallback key-value sort.
 *
 * The key range is sorted in ascending order, and the value range is permuted
 * so that key-value associations are preserved.
 *
 * @param keys_first Iterator to the beginning of the key range.
 * @param keys_last Iterator to the end of the key range.
 * @param values_first Iterator to the beginning of the associated value range.
 *
 * @tparam P Compile-time execution policy.
 * @tparam KeyIt Iterator type over keys.
 * @tparam ValueIt Iterator type over values.
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
#endif