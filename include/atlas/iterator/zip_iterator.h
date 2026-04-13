#pragma once

/**
 * @file zip_iterator.h
 * @brief Declares a backend-portable zip-iterator utility for grouped iteration over multiple sequences.
 *
 * @details
 * This header provides a unified zip-iterator abstraction for Atlas across
 * different execution backends.
 *
 * Depending on whether CUDA tasking is enabled:
 * - under `ATLAS_TASKING_CUDA`, the implementation is a thin alias/wrapper over
 *   Thrust zip iterators and tuples,
 * - otherwise, a host-side standard-library-based fallback implementation is used.
 *
 * ## Purpose
 * A zip iterator allows multiple independent iterator streams to be traversed
 * together as though they formed a single logical iterator over tuples of values
 * or references.
 *
 * This is useful when algorithms need to process several arrays in lockstep,
 * for example:
 * - positions and velocities,
 * - temperatures and species ids,
 * - multiple field buffers updated together.
 *
 * ## Cross-backend design
 * The goal of this header is to present a consistent small API:
 * - @ref zip_iterator
 * - @ref make_zip_iterator
 * - @ref make_zip_tuple
 * - @ref zip_get
 *
 * so that higher-level Atlas code can remain backend-agnostic.
 *
 * ## CUDA path
 * When `ATLAS_TASKING_CUDA` is defined, the implementation delegates to Thrust:
 * - `thrust::zip_iterator`
 * - `thrust::make_zip_iterator`
 * - `thrust::make_tuple`
 * - `thrust::get`
 *
 * ## Host fallback path
 * When CUDA tasking is not enabled, a lightweight zip iterator is implemented
 * using:
 * - `std::tuple`
 * - `std::iterator_traits`
 * - `std::next`
 * - `std::distance`
 *
 * The fallback iterator models a forward-style iterator interface and additionally
 * supports:
 * - postfix increment,
 * - tuple dereference,
 * - offset advance through `operator+`,
 * - distance computation through `operator-`.
 *
 * ---
 */

#ifdef ATLAS_TASKING_CUDA
#include <atlas/core/macros.h>
#include <iterator>
#include <thrust/iterator/zip_iterator.h>
#include <tuple>

namespace atlas {

/**
 * @brief Backend zip iterator alias based on Thrust.
 *
 * @details
 * In CUDA-enabled builds, Atlas reuses `thrust::zip_iterator` as the canonical
 * zip-iterator implementation.
 *
 * The iterator traverses multiple underlying iterator streams in lockstep and
 * dereferences to a tuple-like object of element references.
 *
 * @tparam Iterators Underlying iterator types to zip together.
 */
template <typename... Iterators>
using zip_iterator = thrust::zip_iterator<thrust::tuple<Iterators...>>;

/**
 * @brief Construct a Thrust zip iterator from a tuple of iterators.
 *
 * @param t Tuple containing the iterators to zip.
 * @return Zip iterator over the supplied iterator tuple.
 *
 * @tparam Iterators Underlying iterator types.
 */
template <typename... Iterators>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE zip_iterator<Iterators...>
make_zip_iterator(thrust::tuple<Iterators...> t) {
    return thrust::make_zip_iterator(t);
}

/**
 * @brief Construct a backend tuple suitable for zip-iterator creation and tuple access.
 *
 * @param args Elements to forward into the tuple.
 * @return Thrust tuple containing the forwarded arguments.
 *
 * @tparam Ts Element types.
 */
template <typename... Ts>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE auto
make_zip_tuple(Ts&&... args) {
    return thrust::make_tuple(std::forward<Ts>(args)...);
}

/**
 * @brief Return the `I`-th element of a backend tuple-like object.
 *
 * @param t Tuple-like object.
 * @return Reference or value corresponding to element `I`.
 *
 * @tparam I Compile-time tuple index.
 * @tparam Tuple Tuple-like type.
 */
template <std::size_t I, typename Tuple>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE decltype(auto)
zip_get(Tuple&& t) {
    return thrust::get<I>(std::forward<Tuple>(t));
}

} // namespace atlas

#else

#include <iterator>
#include <tuple>

namespace atlas {

/**
 * @brief Host-side zip iterator that traverses multiple iterators in lockstep.
 *
 * @details
 * This fallback implementation is used when CUDA tasking is not enabled.
 *
 * The iterator stores a tuple of underlying iterators and exposes a small API
 * compatible with Atlas usage patterns:
 * - dereference returns a tuple of iterator references,
 * - increment advances all iterators together,
 * - equality compares the full iterator tuple,
 * - `operator+` advances all iterators by a given offset,
 * - `operator-` computes the distance using the first iterator.
 *
 * ## Iterator model
 * The iterator declares `std::forward_iterator_tag` as its category. In practice,
 * some operations such as `operator+` assume that advancing the underlying
 * iterators by `n` is supported through `std::next`.
 *
 * ## Distance convention
 * The distance returned by `operator-` is derived from the first iterator stored
 * in the tuple. This assumes that all zipped iterators remain synchronized and
 * describe ranges of matching length.
 *
 * @tparam Iterators Underlying iterator types to zip together.
 */
template <typename... Iterators>
class zip_iterator {
public:
    /**
     * @brief Tuple type storing the underlying iterators.
     */
    using tuple_type = std::tuple<Iterators...>;

    /**
     * @brief Value type obtained by dereferencing all underlying iterators by value.
     */
    using value_type = std::tuple<typename std::iterator_traits<Iterators>::value_type...>;

    /**
     * @brief Reference type obtained by dereferencing all underlying iterators.
     *
     * @details
     * This is a tuple of the reference types produced by the underlying iterators.
     */
    using reference = std::tuple<typename std::iterator_traits<Iterators>::reference...>;

    /**
     * @brief Iterator category.
     *
     * @details
     * The fallback zip iterator models forward iteration semantics.
     */
    using iterator_category = std::forward_iterator_tag;

    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a default-initialized zip iterator with default-constructed
     * underlying iterators.
     */
    zip_iterator() = default;

    /**
     * @brief Construct a zip iterator from a tuple of iterators.
     *
     * @param iterators Tuple containing the iterators to zip together.
     */
    explicit zip_iterator(tuple_type iterators)
        : iters_(iterators) { }

    /**
     * @brief Dereference the zip iterator.
     *
     * @details
     * Returns a tuple of references corresponding to the dereferenced underlying
     * iterators at the current position.
     *
     * @return Tuple of references to the current elements.
     */
    reference
    operator*() const {
        return deref(std::index_sequence_for<Iterators...> {});
    }

    /**
     * @brief Pre-increment the zip iterator.
     *
     * @details
     * Advances all underlying iterators by one position.
     *
     * @return `*this`.
     */
    zip_iterator&
    operator++() {
        increment(std::index_sequence_for<Iterators...> {});
        return *this;
    }

    /**
     * @brief Post-increment the zip iterator.
     *
     * @details
     * Returns the previous iterator value and then advances all underlying
     * iterators by one position.
     *
     * @return Iterator state prior to increment.
     */
    zip_iterator
    operator++(int) {
        zip_iterator tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Compare two zip iterators for equality.
     *
     * @details
     * Equality is defined as equality of the full underlying iterator tuple.
     *
     * @param other Iterator to compare against.
     * @return `true` if all underlying iterators are equal; otherwise `false`.
     */
    bool
    operator==(const zip_iterator& other) const {
        return iters_ == other.iters_;
    }

    /**
     * @brief Compare two zip iterators for inequality.
     *
     * @param other Iterator to compare against.
     * @return `true` if the iterators are not equal; otherwise `false`.
     */
    bool
    operator!=(const zip_iterator& other) const {
        return !(*this == other);
    }

    /**
     * @brief Return a new zip iterator advanced by `n` positions.
     *
     * @details
     * Each underlying iterator is independently advanced by `n` using `std::next`.
     *
     * @param n Offset by which to advance.
     * @return Advanced zip iterator.
     */
    zip_iterator
    operator+(std::ptrdiff_t n) const {
        return advance(n, std::index_sequence_for<Iterators...> {});
    }

    /**
     * @brief Compute the distance between two zip iterators.
     *
     * @details
     * The distance is computed using only the first underlying iterator. This
     * assumes that all zipped iterators remain synchronized.
     *
     * @param other Iterator used as the reference point.
     * @return Distance from @p other to `*this`.
     */
    std::ptrdiff_t
    operator-(const zip_iterator& other) const {
        return distance(other, std::index_sequence_for<Iterators...> {});
    }

private:
    /**
     * @brief Tuple storing the active underlying iterators.
     */
    tuple_type iters_ {};

    /**
     * @brief Dereference all underlying iterators and return a tuple of references.
     *
     * @param std::index_sequence<I...> Compile-time index pack.
     * @return Tuple of dereferenced iterator references.
     */
    template <std::size_t... I>
    reference
    deref(std::index_sequence<I...>) const {

        return reference(*std::get<I>(iters_)...);
    }

    /**
     * @brief Increment all underlying iterators.
     *
     * @param std::index_sequence<I...> Compile-time index pack.
     */
    template <std::size_t... I>
    void
    increment(std::index_sequence<I...>) {

        ((++std::get<I>(iters_)), ...);
    }

    /**
     * @brief Advance all underlying iterators by `n` and return a new zip iterator.
     *
     * @param n Offset by which to advance.
     * @param std::index_sequence<I...> Compile-time index pack.
     * @return Advanced zip iterator.
     */
    template <std::size_t... I>
    zip_iterator
    advance(std::ptrdiff_t n, std::index_sequence<I...>) const {

        return zip_iterator(tuple_type(std::next(std::get<I>(iters_), n)...));
    }

    /**
     * @brief Compute iterator distance using the first underlying iterator.
     *
     * @param other Iterator used as the reference point.
     * @param std::index_sequence<I...> Compile-time index pack.
     * @return Distance from @p other to `*this`.
     */
    template <std::size_t... I>
    std::ptrdiff_t
    distance(const zip_iterator& other, std::index_sequence<I...>) const {
        (void)sizeof...(I);

        return std::distance(std::get<0>(other.iters_), std::get<0>(iters_));
    }
};

/**
 * @brief Construct a host-side zip iterator from a tuple of iterators.
 *
 * @param t Tuple containing the iterators to zip.
 * @return Zip iterator over the supplied iterator tuple.
 *
 * @tparam Iterators Underlying iterator types.
 */
template <typename... Iterators>
inline zip_iterator<Iterators...>
make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<Iterators...>(t);
}

/**
 * @brief Construct a tuple suitable for zip-iterator creation and tuple access.
 *
 * @param args Elements to forward into the tuple.
 * @return `std::tuple` containing the forwarded arguments.
 *
 * @tparam Ts Element types.
 */
template <typename... Ts>
inline auto
make_zip_tuple(Ts&&... args) {
    return std::make_tuple(std::forward<Ts>(args)...);
}

/**
 * @brief Return the `I`-th element of a tuple-like object.
 *
 * @param t Tuple-like object.
 * @return Reference or value corresponding to element `I`.
 *
 * @tparam I Compile-time tuple index.
 * @tparam Tuple Tuple-like type.
 */
template <std::size_t I, typename Tuple>
inline decltype(auto)
zip_get(Tuple&& t) {
    return std::get<I>(std::forward<Tuple>(t));
}

} // namespace atlas
#endif