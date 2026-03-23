#pragma once

#ifdef ATLAS_TASKING_CUDA
#include <thrust/iterator/counting_iterator.h>

namespace atlas {

template <typename T>
using counting_iterator = thrust::counting_iterator<T>;

}

#else

#include <cstddef>
#include <iterator>

namespace atlas {

template <typename T>
class counting_iterator {
public:
    using value_type = T;

    using difference_type = std::ptrdiff_t;

    using iterator_category = std::random_access_iterator_tag;

    using reference = value_type;

    using pointer = void;

    constexpr counting_iterator() noexcept = default;

    constexpr explicit counting_iterator(T start) noexcept
        : _value(start) { }

    constexpr reference
    operator*() const noexcept { return _value; }

    constexpr reference
    operator[](difference_type n) const noexcept {
        return static_cast<T>(_value + static_cast<T>(n));
    }

    constexpr counting_iterator&
    operator++() noexcept {
        ++_value;
        return *this;
    }

    constexpr counting_iterator
    operator++(int) noexcept {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }

    constexpr counting_iterator&
    operator--() noexcept {
        --_value;
        return *this;
    }

    constexpr counting_iterator
    operator--(int) noexcept {
        auto tmp = *this;
        --(*this);
        return tmp;
    }

    constexpr counting_iterator&
    operator+=(difference_type n) noexcept {
        _value = static_cast<T>(_value + static_cast<T>(n));
        return *this;
    }

    constexpr counting_iterator&
    operator-=(difference_type n) noexcept {
        _value = static_cast<T>(_value - static_cast<T>(n));
        return *this;
    }

    friend constexpr counting_iterator
    operator+(counting_iterator it, difference_type n) noexcept {
        it += n;
        return it;
    }

    friend constexpr counting_iterator
    operator+(difference_type n, counting_iterator it) noexcept {
        it += n;
        return it;
    }

    friend constexpr counting_iterator
    operator-(counting_iterator it, difference_type n) noexcept {
        it -= n;
        return it;
    }

    friend constexpr difference_type
    operator-(const counting_iterator& a, const counting_iterator& b) noexcept {
        return static_cast<difference_type>(a._value) - static_cast<difference_type>(b._value);
    }

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

    constexpr T
    base() const noexcept { return _value; }

private:
    T _value = T {};
};

}
#endif