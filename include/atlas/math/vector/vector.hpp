#pragma once
#include <atlas/math/vector/reductions.h>
#include <cmath>

namespace atlas::math {
template <typename T, std::size_t N>
Vector<T, N>::Vector() noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = T(0);
}

template <typename T, std::size_t N>
Vector<T, N>::Vector(T s) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = s;
}

template <typename T, std::size_t N>
template <typename... Args, typename>
Vector<T, N>::Vector(Args... args) noexcept
    : _data{ static_cast<T>(args)... } {}

template <typename T, std::size_t N>
ATLAS_HOST ATLAS_FORCE_INLINE

Vector<T, N>::Vector(std::initializer_list<T> list) noexcept {
    std::size_t i = 0;
    for (auto it = list.begin(); it != list.end() && i < N; ++it, ++i) _data[i] = *it;
    for (; i < N; ++i) _data[i] = T(0);
}

template <typename T, std::size_t N>
template <typename Expression>
Vector<T, N>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = e[i];
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::size() noexcept {
    return N;
}

template <typename T, std::size_t N>
const T&
Vector<T, N>::operator[](std::size_t index) const noexcept {
    return _data[index];
}

template <typename T, std::size_t N>
T&
Vector<T, N>::operator[](std::size_t index) noexcept {
    return _data[index];
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator=(const Vector& rhs) noexcept {
    if (this != &rhs) {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) _data[i] = rhs._data[i];
    }
    return *this;
}

template <typename T, std::size_t N>
void
Vector<T, N>::set(T s) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = s;
}

template <typename T, std::size_t N>
template <typename... Args, typename>
void
Vector<T, N>::set_values(Args... args) noexcept {
    const T tmp[N] = { static_cast<T>(args)... };
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = tmp[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::set_zero() noexcept {
    set(T(0));
}

template <typename T, std::size_t N>
void
Vector<T, N>::add(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] += v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::sub(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] -= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::mul(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] *= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::div(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] /= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::add(const Vector& v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] += v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::sub(const Vector& v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] -= v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::mul(const Vector& v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] *= v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::div(const Vector& v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] /= v._data[i];
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator+=(T v) noexcept {
    add(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator-=(T v) noexcept {
    sub(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator*=(T v) noexcept {
    mul(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator/=(T v) noexcept {
    div(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator+=(const Vector& v) noexcept {
    add(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator-=(const Vector& v) noexcept {
    sub(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator*=(const Vector& v) noexcept {
    mul(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator/=(const Vector& v) noexcept {
    div(v);
    return *this;
}

template <typename T, std::size_t N>
bool
Vector<T, N>::operator==(const Vector& other) const noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) {
        if (_data[i] != other._data[i]) return false;
    }
    return true;
}

template <typename T, std::size_t N>
T
Vector<T, N>::dot(const Vector& v) const noexcept {
    T s = T(0);
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) { s = fma(_data[i], v._data[i], s); }
    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::length_squared() const noexcept {
    T s = T(0);
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) { s = fma(_data[i], _data[i], s); }
    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::length() const noexcept {
    return static_cast<T>(std::sqrt(static_cast<double>(length_squared())));
}

template <typename T, std::size_t N>
void
Vector<T, N>::normalize() noexcept {
    const T len = length();
    if (len != T(0)) {
        const T inv = T(1) / len;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) _data[i] *= inv;
    }
}

template <typename T, std::size_t N>
Vector<T, N>
Vector<T, N>::normalized() const noexcept {
    Vector out;
    const T len = length();
    if (len != T(0)) {
        const T inv = T(1) / len;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) out._data[i] = _data[i] * inv;
    }
    return out;
}

template <typename T, std::size_t N>
template <typename To>
Vector<To, N>
Vector<T, N>::cast_to() const noexcept {
    Vector<To, N> out;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) out[i] = static_cast<To>(_data[i]);
    return out;
}

template <typename T, std::size_t N>
const T*
Vector<T, N>::data() const noexcept {
    return _data;
}

template <typename T, std::size_t N>
T*
Vector<T, N>::data() noexcept {
    return _data;
}

template <typename T, std::size_t N>
const T&
Vector<T, N>::at(std::size_t index) const noexcept {
    return _data[index];
}

template <typename T, std::size_t N>
T&
Vector<T, N>::at(std::size_t index) noexcept {
    return _data[index];
}

template <typename T, std::size_t N>
T
Vector<T, N>::sum() const noexcept {
    T s = T(0);
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) { s = fma(_data[i], T(1), s); }
    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::avg() const noexcept {
    return sum() / static_cast<T>(N);
}

template <typename T, std::size_t N>
T
Vector<T, N>::min() const noexcept {
    T m = _data[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < N; ++i) m = (_data[i] < m) ? _data[i] : m;
    return m;
}

template <typename T, std::size_t N>
T
Vector<T, N>::max() const noexcept {
    T m = _data[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < N; ++i) m = (_data[i] > m) ? _data[i] : m;
    return m;
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::major_axis() const noexcept {
    return math::argabsmax(*this);
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::minor_axis() const noexcept {
    return math::argmin(*this);
}
}