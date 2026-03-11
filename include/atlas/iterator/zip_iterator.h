#pragma once


#ifdef ATLAS_TASKING_CUDA
#include <iterator>
#include <thrust/iterator/zip_iterator.h>
#include <tuple>
#include <atlas/core/macros.h>
namespace atlas {

/**
 * @file zip_iterator.h
 * @brief Backend-selected zip-iterator abstraction (Thrust on CUDA, lightweight std::tuple-based on CPU).
 *
 * @details
 * This header provides an `atlas::zip_iterator` and `atlas::make_zip_iterator(...)` that behave
 * similarly across CUDA and non-CUDA builds:
 *
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - `atlas::zip_iterator<...>` is an alias to `thrust::zip_iterator<thrust::tuple<...>>`.
 *   - `make_zip_iterator` forwards to `thrust::make_zip_iterator`.
 *
 * - **CPU build**:
 *   - `atlas::zip_iterator<...>` is a small iterator adaptor that stores a `std::tuple` of iterators.
 *   - Dereferencing returns a tuple of references, one per underlying iterator.
 *   - Increment advances all underlying iterators in lock-step.
 *
 * Zip iterators are useful for algorithms that operate on multiple ranges in parallel
 * (e.g., sorting or transforming "struct-of-arrays" data as if it were a zipped "array-of-structs").
 *
 * @note
 * - The CPU implementation intentionally keeps the API surface minimal: `*`, `++`, `==/!=`,
 *   `operator+`, and `operator-`.
 * - In the CPU path, `operator-` computes distance using only the **first** iterator; therefore,
 *   all iterators are assumed to advance together and be the same length.
 * - The CPU implementation uses `std::next(...)`, so it requires at least forward iterators
 *   for `operator+` to be valid (random-access is preferable).
 */

template <typename... Iterators>
using zip_iterator = thrust::zip_iterator<thrust::tuple<Iterators...>>;

/**
 * @brief Creates a Thrust zip iterator from a `thrust::tuple` of iterators.
 *
 * @tparam Iterators Underlying iterator types.
 * @param t Thrust tuple of iterators.
 * @return A `thrust::zip_iterator` that zips the provided iterators.
 *
 * @note
 * This is a thin wrapper over `thrust::make_zip_iterator`.
 */
template <typename... Iterators>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE zip_iterator<Iterators...>
make_zip_iterator(thrust::tuple<Iterators...> t) {
    return thrust::make_zip_iterator(t);
}

} // namespace atlas

#else // --------------------------------------------------------
// CPU fallback
// --------------------------------------------------------

#include <iterator>
#include <tuple>

namespace atlas {

/**
 * @file zip_iterator.h
 * @brief Backend-selected zip-iterator abstraction (Thrust on CUDA, lightweight std::tuple-based on CPU).
 *
 * @details
 * CPU fallback zip iterator:
 * - Stores a tuple of iterators.
 * - `operator*()` returns a tuple of references `(*it0, *it1, ...)`.
 * - `operator++()` increments all iterators.
 */

template <typename... Iterators>
class zip_iterator {
public:
    /// @brief Tuple type holding the underlying iterators.
    using tuple_type = std::tuple<Iterators...>;

    /// @brief Value type (tuple of value types from each iterator).
    using value_type = std::tuple<typename std::iterator_traits<Iterators>::value_type...>;

    /// @brief Reference type (tuple of references from each iterator dereference).
    using reference = std::tuple<typename std::iterator_traits<Iterators>::reference...>;

    /// @brief Iterator category (kept conservative; operations assume forward traversal).
    using iterator_category = std::forward_iterator_tag;

    /// @brief Default constructor.
    zip_iterator() = default;

    /**
     * @brief Constructs a zip iterator from a tuple of iterators.
     *
     * @param iterators Tuple containing underlying iterators.
     */
    explicit zip_iterator(tuple_type iterators)
        : iters_(iterators) { }

    /**
     * @brief Dereferences all underlying iterators and returns a tuple of references.
     *
     * @return Tuple `(*it0, *it1, ...)` as references.
     *
     * @note
     * The returned tuple holds references (or proxy references) matching each iterator's
     * `reference` type.
     */
    reference
    operator*() const {
        return deref(std::index_sequence_for<Iterators...> {});
    }

    /**
     * @brief Pre-increment: advances all underlying iterators by one.
     *
     * @return Reference to `*this`.
     */
    zip_iterator&
    operator++() {
        increment(std::index_sequence_for<Iterators...> {});
        return *this;
    }

    /**
     * @brief Post-increment: advances all underlying iterators by one and returns the old value.
     *
     * @return Copy of iterator prior to increment.
     */
    zip_iterator
    operator++(int) {
        zip_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Equality comparison (compares the underlying iterator tuple).
     *
     * @param other Other zip iterator.
     * @return `true` if all underlying iterators compare equal.
     */
    bool
    operator==(const zip_iterator& other) const {
        return iters_ == other.iters_;
    }

    /**
     * @brief Inequality comparison.
     */
    bool
    operator!=(const zip_iterator& other) const {
        return !(*this == other);
    }

    /**
     * @brief Advances the iterator by `n` steps and returns a new zip iterator.
     *
     * @param n Number of steps to advance (can be negative if underlying iterators support it).
     * @return Advanced zip iterator.
     *
     * @note
     * Uses `std::next` for each underlying iterator.
     */
    zip_iterator
    operator+(std::ptrdiff_t n) const {
        return advance(n, std::index_sequence_for<Iterators...> {});
    }

    /**
     * @brief Computes distance between two zip iterators using the first iterator.
     *
     * @param other Other iterator (treated as begin).
     * @return `distance(other.first, this.first)`.
     *
     * @warning
     * This assumes all iterators move in lock-step and have the same extent.
     */
    std::ptrdiff_t
    operator-(const zip_iterator& other) const {
        return distance(other, std::index_sequence_for<Iterators...> {});
    }

private:
    tuple_type iters_ {};

    // ---- helpers ----

    template <std::size_t... I>
    reference
    deref(std::index_sequence<I...>) const {
        // Expand to (*it0, *it1, ...) and build a reference tuple.
        return reference(*std::get<I>(iters_)...);
    }

    template <std::size_t... I>
    void
    increment(std::index_sequence<I...>) {
        // Fold expression increments all iterators.
        ((++std::get<I>(iters_)), ...);
    }

    template <std::size_t... I>
    zip_iterator
    advance(std::ptrdiff_t n, std::index_sequence<I...>) const {
        // Advance each iterator by n and construct a new zip_iterator.
        return zip_iterator(tuple_type(std::next(std::get<I>(iters_), n)...));
    }

    template <std::size_t... I>
    std::ptrdiff_t
    distance(const zip_iterator& other, std::index_sequence<I...>) const {
        (void)sizeof...(I);
        // Compute distance from the first iterator only.
        return std::distance(std::get<0>(other.iters_), std::get<0>(iters_));
    }
};

/**
 * @brief Creates a CPU zip iterator from a `std::tuple` of iterators.
 *
 * @tparam Iterators Underlying iterator types.
 * @param t Tuple of iterators.
 * @return `atlas::zip_iterator` instance.
 */
template <typename... Iterators>
inline zip_iterator<Iterators...>
make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<Iterators...>(t);
}

} // namespace atlas
#endif // ATLAS_TASKING_CUDA
