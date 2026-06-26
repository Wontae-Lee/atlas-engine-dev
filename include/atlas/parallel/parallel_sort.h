#pragma once

#include <algorithm>
#include <atlas/parallel/parallel_for.h>
#include <iterator>
#include <numeric>
#include <vector>

#if defined(ATLAS_TASKING_CUDA)
#include <thrust/sort.h>

namespace atlas {
namespace detail {

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::host, first, last);
    }

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        thrust::sort(thrust::device, first, last);
    }

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

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

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::host, keys_first, keys_last, values_first);
    }

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        if (keys_first == keys_last) return;
        thrust::sort_by_key(thrust::device, keys_first, keys_last, values_first);
    }

}

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

}

#else

#include <tbb/tbb.h>

namespace atlas {
namespace detail {

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_host_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        tbb::parallel_sort(first, last);
    }

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_device_impl(RandomIt first, RandomIt last) {
        detail::parallel_sort_host_impl(first, last);
    }

    template <typename RandomIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_serial_impl(RandomIt first, RandomIt last) {
        if (first == last) return;
        std::sort(first, last);
    }

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_index_impl(KeyIt keys_first,
                                    KeyIt keys_last,
                                    ValueIt values_first,
                                    const bool use_parallel_sort) {
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

        const auto compare = [&key_tmp](diff_t a, diff_t b) {
            return key_tmp[static_cast<std::size_t>(a)] < key_tmp[static_cast<std::size_t>(b)];
        };

        if (use_parallel_sort) {
            tbb::parallel_sort(indices.begin(), indices.end(), compare);
        } else {
            std::sort(indices.begin(), indices.end(), compare);
        }

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

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_serial_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_index_impl(keys_first, keys_last, values_first, false);
    }

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_host_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_index_impl(keys_first, keys_last, values_first, true);
    }

    template <typename KeyIt, typename ValueIt>
    ATLAS_FORCE_INLINE void
    parallel_sort_by_key_device_impl(KeyIt keys_first, KeyIt keys_last, ValueIt values_first) {
        detail::parallel_sort_by_key_index_impl(keys_first, keys_last, values_first, true);
    }

}

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

}
#endif