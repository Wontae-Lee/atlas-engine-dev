#pragma once
namespace atlas::math {
template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix() noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = T(0);
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix(T s) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = s;
}

template <typename T, std::size_t R, std::size_t C>
template <typename... Args, typename>
Matrix<T, R, C>::Matrix(Args... args) noexcept
    : _data { static_cast<T>(args)... } {
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>::Matrix(std::initializer_list<T> list) noexcept {
    std::size_t i = 0;
    for (const auto& v : list) {
        if (i >= R * C) break;
        _data[i++] = v;
    }
    for (; i < R * C; ++i) _data[i] = T(0);
}

template <typename T, std::size_t R, std::size_t C>
ATLAS_NODISCARD const T&
Matrix<T, R, C>::operator[](std::size_t i) const noexcept {
    return _data[i];
}

template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::operator[](std::size_t i) noexcept {
    return _data[i];
}

template <typename T, std::size_t R, std::size_t C>
ATLAS_NODISCARD const T&
Matrix<T, R, C>::operator()(std::size_t r, std::size_t c) const noexcept {
    return _data[index(r, c)];
}

template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::operator()(std::size_t r, std::size_t c) noexcept {
    return _data[index(r, c)];
}

template <typename T, std::size_t R, std::size_t C>
ATLAS_NODISCARD const T&
Matrix<T, R, C>::at(std::size_t r, std::size_t c) const noexcept {
    ATLAS_ASSERT(r < R && c < C && "Matrix::at() index out of range");
    return _data[index(r, c)];
}

template <typename T, std::size_t R, std::size_t C>
T&
Matrix<T, R, C>::at(std::size_t r, std::size_t c) noexcept {
    ATLAS_ASSERT(r < R && c < C && "Matrix::at() index out of range");
    return _data[index(r, c)];
}

template <typename T, std::size_t R, std::size_t C>
template <typename E>
Matrix<T, R, C>&
Matrix<T, R, C>::operator=(const MatrixExpression<T, E>& expr) noexcept {
    const E& e = expr();
    ATLAS_ASSERT(e.rows() == R && e.cols() == C && "Matrix::operator=: shape mismatch");
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = e[i];
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::set(T s) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = s;
}

template <typename T, std::size_t R, std::size_t C>
template <typename... Args, typename>
void
Matrix<T, R, C>::set_values(Args... args) noexcept {
    const T tmp[R * C] { static_cast<T>(args)... };
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = tmp[i];
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::set_zero() noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] = T(0);
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::add(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] += v;
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::sub(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] -= v;
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::mul(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] *= v;
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::div(T v) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] /= v;
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::add(const Matrix& m) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] += m._data[i];
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::sub(const Matrix& m) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] -= m._data[i];
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::mul(const Matrix& m) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] *= m._data[i];
}

template <typename T, std::size_t R, std::size_t C>
void
Matrix<T, R, C>::div(const Matrix& m) noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i) _data[i] /= m._data[i];
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator+=(T v) noexcept {
    add(v);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator-=(T v) noexcept {
    sub(v);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator*=(T v) noexcept {
    mul(v);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator/=(T v) noexcept {
    div(v);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator+=(const Matrix& m) noexcept {
    add(m);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator-=(const Matrix& m) noexcept {
    sub(m);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator*=(const Matrix& m) noexcept {
    mul(m);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
Matrix<T, R, C>&
Matrix<T, R, C>::operator/=(const Matrix& m) noexcept {
    div(m);
    return *this;
}

template <typename T, std::size_t R, std::size_t C>
bool
Matrix<T, R, C>::operator==(const Matrix& other) const noexcept {
    ATLAS_UNROLL
    for (std::size_t i = 0; i < R * C; ++i)
        if (!(_data[i] == other._data[i])) return false;
    return true;
}

template <typename T, std::size_t R, std::size_t C, std::size_t K>
Matrix<T, R, K>
matmul(const Matrix<T, R, C>& a, const Matrix<T, C, K>& b) noexcept {
    Matrix<T, R, K> out;
    out.set_zero();
    for (std::size_t r = 0; r < R; ++r) {
        for (std::size_t k = 0; k < K; ++k) {
            T acc = T(0);
            for (std::size_t c = 0; c < C; ++c) { acc = static_cast<T>(acc + a(r, c) * b(c, k)); }
            out(r, k) = acc;
        }
    }
    return out;
}

template <typename T, std::size_t R, std::size_t C>
Vector<T, R>
matmul(const Matrix<T, R, C>& a, const Vector<T, C>& x) noexcept {
    Vector<T, R> y;
    y.set_zero();
    for (std::size_t r = 0; r < R; ++r) {
        T acc = T(0);
        for (std::size_t c = 0; c < C; ++c) { acc = static_cast<T>(acc + a(r, c) * x[c]); }
        y[r] = acc;
    }
    return y;
}
}