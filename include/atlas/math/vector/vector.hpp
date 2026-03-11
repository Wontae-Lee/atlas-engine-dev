#pragma once

#include <atlas/math/vector/vector_reductions.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace atlas::math {

// ------------------------------------------------------------
// Vector<T,N>
// ------------------------------------------------------------

template <typename T, std::size_t N>
Vector<T, N>::Vector() noexcept {
    // Default construct as a zero vector.
    // _data = [0, 0, ..., 0]
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = T(0);
}

template <typename T, std::size_t N>
Vector<T, N>::Vector(T s) noexcept {
    // Fill constructor: every component = s.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = s;
}

template <typename T, std::size_t N>
template <typename... Args, typename>
Vector<T, N>::Vector(Args... args) noexcept
    : _data { static_cast<T>(args)... } {
    // Variadic constructor: initializes components from arguments in order.
    // Assumes caller provides exactly N values (enforced by the SFINAE guard).
}

template <typename T, std::size_t N>
ATLAS_HOST ATLAS_FORCE_INLINE
Vector<T, N>::Vector(std::initializer_list<T> list) noexcept {
    // Initialize from { ... }.
    // - Copies up to N values.
    // - Remaining entries default to 0.
    std::size_t i = 0;
    for (auto it = list.begin(); it != list.end() && i < N; ++it, ++i) _data[i] = *it;
    for (; i < N; ++i) _data[i] = T(0);
}

template <typename T, std::size_t N>
template <typename Expression>
Vector<T, N>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    // Expression-template materialization:
    // Evaluate a lazy expression into concrete storage.
    const Expression& e = expr();
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = static_cast<T>(e[i]);
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::size() noexcept {
    // Static dimension query.
    return N;
}

template <typename T, std::size_t N>
const T&
Vector<T, N>::operator[](std::size_t index) const noexcept {
    // Flat indexing (no bounds check).
    return _data[index];
}

template <typename T, std::size_t N>
T&
Vector<T, N>::operator[](std::size_t index) noexcept {
    // Mutable flat indexing.
    return _data[index];
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator=(const Vector& rhs) noexcept {
    // Copy assignment with self-assignment check.
    if (this != &rhs) {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) _data[i] = rhs._data[i];
    }
    return *this;
}

template <typename T, std::size_t N>
void
Vector<T, N>::set(T s) noexcept {
    // Fill all components with the same scalar.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = s;
}

template <typename T, std::size_t N>
template <typename... Args, typename>
void
Vector<T, N>::set_values(Args... args) noexcept {
    // Set from N scalars (enforced by SFINAE guard).
    // Uses a local array to avoid tricky pack expansion loops.
    const T tmp[N] = { static_cast<T>(args)... };
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] = tmp[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::set_zero() noexcept {
    // Set to the zero vector.
    set(T(0));
}

template <typename T, std::size_t N>
void
Vector<T, N>::add(T v) noexcept {
    // Component-wise add scalar: x_i += v.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] += v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::sub(T v) noexcept {
    // Component-wise subtract scalar: x_i -= v.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] -= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::mul(T v) noexcept {
    // Component-wise multiply by scalar: x_i *= v.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] *= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::div(T v) noexcept {
    // Component-wise divide by scalar: x_i /= v.
    // Caller must ensure v != 0.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] /= v;
}

template <typename T, std::size_t N>
void
Vector<T, N>::add(const Vector& v) noexcept {
    // Component-wise addition: x_i += v_i.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] += v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::sub(const Vector& v) noexcept {
    // Component-wise subtraction: x_i -= v_i.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] -= v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::mul(const Vector& v) noexcept {
    // Hadamard product: x_i *= v_i.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] *= v._data[i];
}

template <typename T, std::size_t N>
void
Vector<T, N>::div(const Vector& v) noexcept {
    // Component-wise division: x_i /= v_i.
    // Caller must ensure v_i != 0 for all i.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) _data[i] /= v._data[i];
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator+=(T v) noexcept {
    // In-place scalar add.
    add(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator-=(T v) noexcept {
    // In-place scalar subtract.
    sub(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator*=(T v) noexcept {
    // In-place scalar multiply.
    mul(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator/=(T v) noexcept {
    // In-place scalar divide.
    div(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator+=(const Vector& v) noexcept {
    // In-place component-wise add.
    add(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator-=(const Vector& v) noexcept {
    // In-place component-wise subtract.
    sub(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator*=(const Vector& v) noexcept {
    // In-place Hadamard multiply.
    mul(v);
    return *this;
}

template <typename T, std::size_t N>
Vector<T, N>&
Vector<T, N>::operator/=(const Vector& v) noexcept {
    // In-place component-wise divide.
    div(v);
    return *this;
}

template <typename T, std::size_t N>
bool
Vector<T, N>::operator==(const Vector& other) const noexcept {
    // Exact component-wise equality.
    // For floating point, approximate equality should be done at a higher level.
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) {
        if (_data[i] != other._data[i]) return false;
    }
    return true;
}

template <typename T, std::size_t N>
T
Vector<T, N>::dot(const Vector& v) const noexcept {
    // Dot product:
    //   sum_i (x_i * v_i)
    T s = T(0);

    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s = fma(_data[i], v._data[i], s);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s += _data[i] * v._data[i];
        }
    }

    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::length_squared() const noexcept {
    // Squared Euclidean norm:
    //   ||x||^2 = dot(x, x)
    T s = T(0);

    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s = fma(_data[i], _data[i], s);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s += _data[i] * _data[i];
        }
    }

    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::length() const noexcept {
    // Euclidean norm:
    //   ||x|| = sqrt(||x||^2)
    return static_cast<T>(std::sqrt(static_cast<double>(length_squared())));
}

template <typename T, std::size_t N>
void
Vector<T, N>::normalize() noexcept {
    // In-place normalization:
    //   x := x / ||x||   (if ||x|| != 0)
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
    // Return a normalized copy. Returns zero vector if length == 0.
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
    // Component-wise type cast.
    Vector<To, N> out;
    ATLAS_UNROLL
    for (std::size_t i = 0; i < N; ++i) out[i] = static_cast<To>(_data[i]);
    return out;
}

template <typename T, std::size_t N>
const T*
Vector<T, N>::data() const noexcept {
    // Pointer to contiguous storage.
    return _data;
}

template <typename T, std::size_t N>
T*
Vector<T, N>::data() noexcept {
    // Mutable pointer to contiguous storage.
    return _data;
}

template <typename T, std::size_t N>
const T&
Vector<T, N>::at(std::size_t index) const noexcept {
    // Bounds-unchecked access (matches your Matrix::at style).
    return _data[index];
}

template <typename T, std::size_t N>
T&
Vector<T, N>::at(std::size_t index) noexcept {
    // Mutable bounds-unchecked access.
    return _data[index];
}

template <typename T, std::size_t N>
T
Vector<T, N>::sum() const noexcept {
    // Sum reduction:
    //   sum_i x_i
    T s = T(0);

    if constexpr (std::is_floating_point_v<T>) {
        using std::fma;
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s = fma(_data[i], T(1), s);
        }
    } else {
        ATLAS_UNROLL
        for (std::size_t i = 0; i < N; ++i) {
            s += _data[i];
        }
    }

    return s;
}

template <typename T, std::size_t N>
T
Vector<T, N>::avg() const noexcept {
    // Average:
    //   sum(x) / N
    return sum() / static_cast<T>(N);
}

template <typename T, std::size_t N>
T
Vector<T, N>::min() const noexcept {
    // Minimum component by value.
    T m = _data[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < N; ++i) m = (_data[i] < m) ? _data[i] : m;
    return m;
}

template <typename T, std::size_t N>
T
Vector<T, N>::max() const noexcept {
    // Maximum component by value.
    T m = _data[0];
    ATLAS_UNROLL
    for (std::size_t i = 1; i < N; ++i) m = (_data[i] > m) ? _data[i] : m;
    return m;
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::major_axis() const noexcept {
    // Index of the component with maximum absolute value.
    // NOTE: This relies on vector_reductions.h contract for argabsmax().
    return math::argabsmax(*this);
}

template <typename T, std::size_t N>
std::size_t
Vector<T, N>::minor_axis() const noexcept {
    // Index of the component with minimum absolute value.
    // NOTE: If vector_reductions.h does not provide argabsmin(), fall back to local implementation.

    using std::abs;

    std::size_t idx = 0;
    auto best       = abs(_data[0]);

    ATLAS_UNROLL
    for (std::size_t i = 1; i < N; ++i) {
        const auto a = abs(_data[i]);
        if (a <= best) { // tie -> later index wins
            best = a;
            idx  = i;
        }
    }
    return idx;
}

} // namespace atlas::math
