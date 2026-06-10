#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <cstddef>
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

/**
 * @brief Alias to Thrust's counting iterator when CUDA tasking is enabled.
 *
 * @tparam T Integral or incrementable value type used by the iterator.
 *
 * @details
 * Under CUDA builds, Atlas defers to `thrust::counting_iterator<T>` so the
 * iterator integrates naturally with Thrust- and device-oriented algorithms.
 *
 * The iterator conceptually represents a virtual sequence:
 * `start, start + 1, start + 2, ...`
 *
 * without storing an actual backing container.
 */
template <typename T>
using counting_iterator = thrust::counting_iterator<T, thrust::use_default, thrust::use_default, std::ptrdiff_t>;

} // namespace atlas

#else

#include <cstddef>
#include <iterator>

namespace atlas {

/**
 * @brief Lightweight random-access counting iterator for non-CUDA builds.
 *
 * @tparam T Value type produced by the iterator.
 *
 * @details
 * `counting_iterator` models a virtual random-access iterator over an arithmetic
 * sequence beginning at a stored value and increasing by `1` per step.
 *
 * This iterator does not reference external storage. Instead, dereferencing
 * yields the current counter value directly. It is useful for algorithms that
 * require iterator-based traversal over index-like values without constructing
 * an explicit container.
 *
 * ## Example
 * @code
 * atlas::counting_iterator<int> first(0);
 * atlas::counting_iterator<int> last(10);
 *
 * // Represents the sequence: 0, 1, 2, ..., 9
 * @endcode
 *
 * ## Iterator properties
 * - random-access iterator
 * - trivially small state (`_value`)
 * - no backing memory or ownership
 * - dereference returns by value, not by reference to stored sequence memory
 */
template <typename T>
class counting_iterator {
public:
    /**
     * @brief Value type produced by dereferencing the iterator.
     */
    using value_type = T;

    /**
     * @brief Difference type used for iterator arithmetic.
     */
    using difference_type = std::ptrdiff_t;

    /**
     * @brief Iterator category tag.
     *
     * @details
     * Declares this iterator as a random-access iterator.
     */
    using iterator_category = std::random_access_iterator_tag;

    /**
     * @brief Dereference result type.
     *
     * @details
     * Since the iterator generates values virtually, dereferencing returns
     * by value rather than by reference to an underlying container element.
     */
    using reference = value_type;

    /**
     * @brief Pointer type.
     *
     * @details
     * No actual addressable element exists behind this iterator, so `pointer`
     * is defined as `void`.
     */
    using pointer = void;

    /**
     * @brief Default-construct the iterator.
     *
     * @details
     * Initializes the iterator to the default-constructed value of `T`.
     */
    constexpr counting_iterator() noexcept = default;

    /**
     * @brief Construct the iterator from an initial counter value.
     *
     * @param start First logical value represented by the iterator.
     */
    constexpr explicit counting_iterator(T start) noexcept
        : _value(start) { }

    /**
     * @brief Dereference the iterator.
     *
     * @return Current counter value.
     *
     * @details
     * This does not access memory. It simply returns the iterator's current
     * logical value.
     */
    constexpr reference
    operator*() const noexcept { return _value; }

    /**
     * @brief Random-access dereference with offset.
     *
     * @param n Offset from the current iterator position.
     * @return Logical value at the offset position.
     *
     * @details
     * Equivalent to dereferencing `(*this + n)`, but computed directly.
     */
    constexpr reference
    operator[](difference_type n) const noexcept {
        return static_cast<T>(_value + static_cast<T>(n));
    }

    /**
     * @brief Pre-increment the iterator.
     *
     * @return Reference to the incremented iterator.
     */
    constexpr counting_iterator&
    operator++() noexcept {
        ++_value;
        return *this;
    }

    /**
     * @brief Post-increment the iterator.
     *
     * @return Copy of the iterator before increment.
     */
    constexpr counting_iterator
    operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    /**
     * @brief Pre-decrement the iterator.
     *
     * @return Reference to the decremented iterator.
     */
    constexpr counting_iterator&
    operator--() noexcept {
        --_value;
        return *this;
    }

    /**
     * @brief Post-decrement the iterator.
     *
     * @return Copy of the iterator before decrement.
     */
    constexpr counting_iterator
    operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    /**
     * @brief Advance the iterator by an offset.
     *
     * @param n Signed offset to add.
     * @return Reference to the updated iterator.
     */
    constexpr counting_iterator&
    operator+=(difference_type n) noexcept {
        _value = static_cast<T>(_value + static_cast<T>(n));
        return *this;
    }

    /**
     * @brief Retreat the iterator by an offset.
     *
     * @param n Signed offset to subtract.
     * @return Reference to the updated iterator.
     */
    constexpr counting_iterator&
    operator-=(difference_type n) noexcept {
        _value = static_cast<T>(_value - static_cast<T>(n));
        return *this;
    }

    /**
     * @brief Return an iterator advanced by @p n.
     *
     * @param it Base iterator.
     * @param n Signed offset.
     * @return Advanced iterator.
     */
    friend constexpr counting_iterator
    operator+(counting_iterator it, difference_type n) noexcept {
        it += n;
        return it;
    }

    /**
     * @brief Return an iterator advanced by @p n.
     *
     * @param n Signed offset.
     * @param it Base iterator.
     * @return Advanced iterator.
     */
    friend constexpr counting_iterator
    operator+(difference_type n, counting_iterator it) noexcept {
        it += n;
        return it;
    }

    /**
     * @brief Return an iterator moved backward by @p n.
     *
     * @param it Base iterator.
     * @param n Signed offset.
     * @return Retreated iterator.
     */
    friend constexpr counting_iterator
    operator-(counting_iterator it, difference_type n) noexcept {
        it -= n;
        return it;
    }

    /**
     * @brief Compute the distance between two counting iterators.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return Difference `a - b`.
     */
    friend constexpr difference_type
    operator-(const counting_iterator& a, const counting_iterator& b) noexcept {
        return static_cast<difference_type>(a._value) - static_cast<difference_type>(b._value);
    }

    /**
     * @brief Equality comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if both iterators represent the same logical position.
     */
    friend constexpr bool
    operator==(const counting_iterator& a, const counting_iterator& b) noexcept {
        return a._value == b._value;
    }

    /**
     * @brief Inequality comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if the iterators represent different logical positions.
     */
    friend constexpr bool
    operator!=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(a == b);
    }

    /**
     * @brief Strict less-than comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if `a` precedes `b`.
     */
    friend constexpr bool
    operator<(const counting_iterator& a, const counting_iterator& b) noexcept {
        return a._value < b._value;
    }

    /**
     * @brief Strict greater-than comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if `a` follows `b`.
     */
    friend constexpr bool
    operator>(const counting_iterator& a, const counting_iterator& b) noexcept {
        return b < a;
    }

    /**
     * @brief Less-than-or-equal comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if `a` does not follow `b`.
     */
    friend constexpr bool
    operator<=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(b < a);
    }

    /**
     * @brief Greater-than-or-equal comparison.
     *
     * @param a Left iterator.
     * @param b Right iterator.
     * @return `true` if `a` does not precede `b`.
     */
    friend constexpr bool
    operator>=(const counting_iterator& a, const counting_iterator& b) noexcept {
        return !(a < b);
    }

    /**
     * @brief Return the stored base counter value.
     *
     * @return Current logical iterator value.
     */
    constexpr T
    base() const noexcept { return _value; }

private:
    /**
     * @brief Current logical value represented by the iterator.
     */
    T _value = T {};
};

} // namespace atlas
#endif
