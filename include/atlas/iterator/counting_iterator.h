#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

/**
 * @file counting_iterator.h
 * @brief Backend-selected counting iterator (Thrust on CUDA, lightweight CPU fallback otherwise).
 *
 * @details
 * This header provides `atlas::counting_iterator<T>` as a unified abstraction that matches
 * Thrust's counting iterator semantics where possible:
 *
 * - **CUDA build (`ATLAS_TASKING_CUDA` defined)**:
 *   - `atlas::counting_iterator<T>` aliases `thrust::counting_iterator<T>`.
 *
 * - **Non-CUDA build**:
 *   - `atlas::counting_iterator<T>` is a small random-access iterator that produces a
 *     monotonically increasing sequence of values starting from a given `start`.
 *
 * A counting iterator is useful for:
 * - Generating index ranges `[0, n)` without materializing an array
 * - Parallel algorithms that require an index as input
 * - Zip/transform pipelines (especially in Thrust-style workflows)
 *
 * @tparam T The produced value type. Typically an integral type.
 *
 * @note
 * - The CPU implementation returns values **by value** (not by reference), mirroring Thrust's
 *   behavior (a counting iterator does not point to actual stored elements).
 * - For unsigned `T`, subtraction casts through `difference_type` to reduce surprises.
 */

template <typename T>
using counting_iterator = thrust::counting_iterator<T>;

} // namespace atlas

#else

#include <cstddef>
#include <iterator>


namespace atlas {

/**
 * @file counting_iterator.h
 * @brief Backend-selected counting iterator (Thrust on CUDA, lightweight CPU fallback otherwise).
 *
 * @details
 * CPU fallback implementation:
 * - Stores the current counter value.
 * - Supports random-access operations (`+`, `-`, `[]`, comparisons).
 * - Dereference returns the current counter value **by value**.
 *
 * This is intended to be "good enough" for STL/TBB algorithms in non-CUDA builds,
 * while keeping call sites compatible with the Thrust backend.
 */

/**
 * @brief Simple random-access counting iterator producing `T` values.
 *
 * @tparam T The produced value type (typically integral).
 *
 * @details
 * The iterator conceptually represents an infinite (or externally bounded) sequence:
 * \f[
 *   value, value+1, value+2, \ldots
 * \f]
 *
 * Since no underlying storage exists, `reference` is defined as `value_type` and dereference
 * returns by value.
 */
template <typename T>
class counting_iterator {
public:
    /// @brief Value type produced by the iterator.
    using value_type = T;

    /// @brief Signed difference type for iterator arithmetic.
    using difference_type = std::ptrdiff_t;

    /// @brief Random access iterator category (supports `+/-` and indexing).
    using iterator_category = std::random_access_iterator_tag;

    /// @brief "Reference" type; returned by value (matches Thrust counting iterator behavior).
    using reference = value_type;

    /// @brief Pointer type is not meaningful because no storage exists.
    using pointer = void;

    /// @brief Constructs a counting iterator starting at `T{}`.
    constexpr counting_iterator() noexcept = default;

    /**
     * @brief Constructs a counting iterator starting at `start`.
     *
     * @param start Initial counter value.
     */
    constexpr explicit counting_iterator(T start) noexcept
        : _value(start) {}

    // ------------------------------------------------------------
    // Dereference / indexing
    // ------------------------------------------------------------

    /**
     * @brief Returns the current counter value.
     *
     * @return Current value (by value).
     */
    constexpr reference
    operator*() const noexcept { return _value; }

    /**
     * @brief Returns the value at an offset from the current iterator.
     *
     * @param n Offset (can be negative).
     * @return `value_ + n` converted to `T`.
     */
    constexpr reference
    operator[](difference_type n) const noexcept {
        return static_cast<T>(_value + static_cast<T>(n));
    }

    // ------------------------------------------------------------
    // Increment / decrement
    // ------------------------------------------------------------

    /// @brief Pre-increment (`++it`).
    constexpr counting_iterator&
    operator++() noexcept {
        ++_value;
        return *this;
    }

    /// @brief Post-increment (`it++`).
    constexpr counting_iterator
    operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    /// @brief Pre-decrement (`--it`).
    constexpr counting_iterator&
    operator--() noexcept {
        --_value;
        return *this;
    }

    /// @brief Post-decrement (`it--`).
    constexpr counting_iterator
    operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    // ------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------

    /**
     * @brief Advances the iterator by `n`.
     *
     * @param n Signed offset.
     * @return Reference to `*this`.
     */
    constexpr counting_iterator&
    operator+=(difference_type n) noexcept {
        _value = static_cast<T>(_value + static_cast<T>(n));
        return *this;
    }

    /**
     * @brief Moves the iterator backward by `n`.
     *
     * @param n Signed offset.
     * @return Reference to `*this`.
     */
    constexpr counting_iterator&
    operator-=(difference_type n) noexcept {
        _value = static_cast<T>(_value - static_cast<T>(n));
        return *this;
    }

    /// @brief Returns `it + n`.
    friend constexpr counting_iterator
    operator+(counting_iterator it, difference_type n) noexcept {
        it += n;
        return it;
    }

    /// @brief Returns `n + it`.
    friend constexpr counting_iterator
    operator+(difference_type n, counting_iterator it) noexcept {
        it += n;
        return it;
    }

    /// @brief Returns `it - n`.
    friend constexpr counting_iterator
    operator-(counting_iterator it, difference_type n) noexcept {
        it -= n;
        return it;
    }

    /**
     * @brief Computes the distance between two counting iterators.
     *
     * @param a Right-hand iterator.
     * @param b Left-hand iterator.
     * @return `a.base() - b.base()` as `difference_type`.
     *
     * @note
     * Casts through `difference_type` to reduce issues with unsigned `T`.
     */
    friend constexpr difference_type
    operator-(const counting_iterator& a, const counting_iterator& b) noexcept {
        return static_cast<difference_type>(a._value) - static_cast<difference_type>(b._value);
    }

    // ------------------------------------------------------------
    // Comparisons
    // ------------------------------------------------------------

    friend constexpr bool
    operator==(const counting_iterator& a, const counting_iterator& b) noexcept {
        return a._value == b._value;
    }

    friend constexpr bool
    operator!=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(a == b);
    }

    friend constexpr bool
    operator<(const counting_iterator& a, const counting_iterator& b) noexcept {
        return a._value < b._value;
    }

    friend constexpr bool
    operator>(const counting_iterator& a, const counting_iterator& b) noexcept {
        return b < a;
    }

    friend constexpr bool
    operator<=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(b < a);
    }

    friend constexpr bool
    operator>=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(a < b);
    }

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------

    /**
     * @brief Returns the current counter value (Thrust uses `base()` for similar access).
     *
     * @return Current value.
     */
    constexpr T
    base() const noexcept { return _value; }

private:
    /// @brief Current counter value.
    T _value = T{};
};

} // namespace atlas
#endif // ATLAS_TASKING_CUDA
