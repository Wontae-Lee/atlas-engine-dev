#pragma once
#include <atlas/parallel/parallel_for.h>

/**
 * @file transform.h
 * @brief Backend-dispatched `transform` algorithm (Thrust/TBB/serial), unary and binary overloads.
 *
 * @details
 * This header provides `atlas::transform<P>(...)` overloads analogous to `std::transform`,
 * but with an execution-policy template parameter to select the backend at compile time.
 *
 * Supported overloads:
 * 1) Unary transform:
 *    `transform<P>(first, last, d_first, op)`
 *    Writes: `d_first[i] = op(first[i])` for each element in `[first, last)`.
 *
 * 2) Binary transform:
 *    `transform<P>(first1, last1, first2, d_first, op)`
 *    Writes: `d_first[i] = op(first1[i], first2[i])` for each element in `[first1, last1)`,
 *    reading the second input starting at `first2`.
 *
 * Backend selection:
 * - **CUDA build (`ATLAS_TASKING_CUDA`)**
 *   - `host`   -> `thrust::transform(thrust::host, ...)`
 *   - `device` -> `thrust::transform(thrust::device, ...)`
 *   - `serial` -> `thrust::transform(thrust::seq, ...)`
 *
 * - **Non-CUDA build**
 *   - `host`   -> oneTBB `tbb::parallel_for` over blocked index ranges
 *   - `device` -> same as host (device policy maps to CPU fallback)
 *   - `serial` -> simple loops
 *
 * @tparam P        Execution policy (`ExecutionPolicy::host`, `device`, `serial`).
 * @tparam InputIt  Input iterator type (first range).
 * @tparam OutputIt Output iterator type.
 * @tparam UnaryOp  Unary callable type.
 * @tparam InputIt1 First input iterator type for binary overload.
 * @tparam InputIt2 Second input iterator type for binary overload.
 * @tparam BinaryOp Binary callable type.
 *
 * @param first   Begin of input range.
 * @param last    End of input range (exclusive).
 * @param d_first Begin of destination range.
 * @param op      Operation applied to inputs to produce each output element.
 *
 * @return Iterator to the end of the written output range (`d_first + n`), where
 *         `n = distance(first, last)` or `n = distance(first1, last1)`.
 *
 * @note
 * - Empty ranges return `d_first` unchanged.
 * - For the non-CUDA host backend, the implementation requires **random-access iterators**
 *   for the first input range because it indexes with `first[i]`/`first1[i]`.
 * - This header does not validate that the output range is large enough; the caller must
 *   ensure `d_first` can accommodate `n` elements.
 */

namespace atlas {

#if defined(ATLAS_TASKING_CUDA)

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace detail {

    /**
     * @brief Unary transform on host using Thrust host execution policy.
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt first, InputIt last,
                        OutputIt d_first,
                        UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::host, first, last, d_first, op);
    }

    /**
     * @brief Unary transform on device using Thrust device execution policy.
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::device, first, last, d_first, op);
    }

    /**
     * @brief Unary transform in serial using Thrust sequential execution policy.
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        if (first == last) return d_first;
        return thrust::transform(thrust::seq, first, last, d_first, op);
    }

    /**
     * @brief Binary transform on host using Thrust host execution policy.
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2,
                        OutputIt d_first,
                        BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::host, first1, last1, first2, d_first, op);
    }

    /**
     * @brief Binary transform on device using Thrust device execution policy.
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::device, first1, last1, first2, d_first, op);
    }

    /**
     * @brief Binary transform in serial using Thrust sequential execution policy.
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        if (first1 == last1) return d_first;
        return thrust::transform(thrust::seq, first1, last1, first2, d_first, op);
    }

} // namespace detail

// ------------------------------------------------------------
// Public API: unary transform
// ------------------------------------------------------------

/**
 * @brief Applies a unary operation to each element in `[first, last)` and writes to `d_first`.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt first, InputIt last,
          OutputIt d_first,
          UnaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first, last, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first, last, d_first, op);
    } else {
        return detail::transform_serial_impl(first, last, d_first, op);
    }
}

// ------------------------------------------------------------
// Public API: binary transform
// ------------------------------------------------------------

/**
 * @brief Applies a binary operation pairwise over `[first1, last1)` and `[first2, first2+n)`.
 */
template <ExecutionPolicy P, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt1 first1, InputIt1 last1,
          InputIt2 first2,
          OutputIt d_first,
          BinaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first1, last1, first2, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first1, last1, first2, d_first, op);
    } else {
        return detail::transform_serial_impl(first1, last1, first2, d_first, op);
    }
}

#else // --------------------------------------------------------
// Non-CUDA backend (oneTBB)
// --------------------------------------------------------

#include <iterator>
#include <tbb/tbb.h>
#include <type_traits>

namespace detail {

    /**
     * @brief Trait to check whether an iterator is random-access.
     */
    template <typename It>
    using is_random_access_iterator = std::is_base_of<std::random_access_iterator_tag,
                                                      typename std::iterator_traits<It>::iterator_category>;

    /**
     * @brief Unary transform on host using TBB blocked ranges.
     *
     * @note
     * Requires random-access iterators for the first input range (uses `first[i]`).
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt first, InputIt last,
                        OutputIt d_first,
                        UnaryOp op) {
        if (first == last) return d_first;

        static_assert(is_random_access_iterator<InputIt>::value,
                      "atlas::transform (TBB backend) requires a random-access iterator.");

        using diff_t   = typename std::iterator_traits<InputIt>::difference_type;
        const diff_t n = std::distance(first, last);

        tbb::parallel_for(
            tbb::blocked_range<diff_t>(0, n),
            [&](const tbb::blocked_range<diff_t>& r) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    d_first[i] = op(first[i]); // Apply unary op to element i.
                }
            });

        return d_first + n;
    }

    /**
     * @brief Unary "device" transform in non-CUDA builds (maps to host).
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        return transform_host_impl(first, last, d_first, op);
    }

    /**
     * @brief Unary serial transform implementation.
     */
    template <typename InputIt, typename OutputIt, typename UnaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt first, InputIt last,
                          OutputIt d_first,
                          UnaryOp op) {
        for (; first != last; ++first, ++d_first) {
            *d_first = op(*first);
        }
        return d_first;
    }

    /**
     * @brief Binary transform on host using TBB blocked ranges.
     *
     * @note
     * Requires random-access iterators for the first range (uses `first1[i]` and `first2[i]`).
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_host_impl(InputIt1 first1, InputIt1 last1,
                        InputIt2 first2,
                        OutputIt d_first,
                        BinaryOp op) {
        if (first1 == last1) return d_first;

        static_assert(is_random_access_iterator<InputIt1>::value,
                      "atlas::transform (TBB backend) requires a random-access iterator for first range.");

        using diff_t   = typename std::iterator_traits<InputIt1>::difference_type;
        const diff_t n = std::distance(first1, last1);

        tbb::parallel_for(
            tbb::blocked_range<diff_t>(0, n),
            [&](const tbb::blocked_range<diff_t>& r) {
                for (diff_t i = r.begin(); i != r.end(); ++i) {
                    d_first[i] = op(first1[i], first2[i]); // Pairwise apply op.
                }
            });

        return d_first + n;
    }

    /**
     * @brief Binary "device" transform in non-CUDA builds (maps to host).
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_device_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        return transform_host_impl(first1, last1, first2, d_first, op);
    }

    /**
     * @brief Binary serial transform implementation.
     */
    template <typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
    ATLAS_FORCE_INLINE OutputIt
    transform_serial_impl(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2,
                          OutputIt d_first,
                          BinaryOp op) {
        for (; first1 != last1; ++first1, ++first2, ++d_first) {
            *d_first = op(*first1, *first2);
        }
        return d_first;
    }

} // namespace detail

// ------------------------------------------------------------
// Public API: unary transform
// ------------------------------------------------------------

/**
 * @brief Applies a unary operation to each element in `[first, last)` and writes to `d_first`.
 */
template <ExecutionPolicy P, typename InputIt, typename OutputIt, typename UnaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt first, InputIt last,
          OutputIt d_first,
          UnaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first, last, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first, last, d_first, op);
    } else {
        return detail::transform_serial_impl(first, last, d_first, op);
    }
}

// ------------------------------------------------------------
// Public API: binary transform
// ------------------------------------------------------------

/**
 * @brief Applies a binary operation pairwise over `[first1, last1)` and `[first2, first2+n)`.
 */
template <ExecutionPolicy P, typename InputIt1, typename InputIt2, typename OutputIt, typename BinaryOp>
ATLAS_FORCE_INLINE OutputIt
transform(InputIt1 first1, InputIt1 last1,
          InputIt2 first2,
          OutputIt d_first,
          BinaryOp op) {
    if constexpr (P == ExecutionPolicy::host) {
        return detail::transform_host_impl(first1, last1, first2, d_first, op);
    } else if constexpr (P == ExecutionPolicy::device) {
        return detail::transform_device_impl(first1, last1, first2, d_first, op);
    } else {
        return detail::transform_serial_impl(first1, last1, first2, d_first, op);
    }
}

#endif // ATLAS_TASKING_CUDA

} // namespace atlas
