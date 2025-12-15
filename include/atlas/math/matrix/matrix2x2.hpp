#pragma once
namespace atlas::math {
template <typename T>
constexpr
Matrix<T, 2, 2>::Matrix() noexcept
    : m00(T(0))
      , m01(T(0))
      , m10(T(0))
      , m11(T(0)) {}

template <typename T>
constexpr
Matrix<T, 2, 2>::Matrix(T s) noexcept
    : m00(s)
      , m01(T(0))
      , m10(T(0))
      , m11(s) {}

template <typename T>
constexpr
Matrix<T, 2, 2>::Matrix(T a00, T a01, T a10, T a11) noexcept
    : m00(a00)
      , m01(a01)
      , m10(a10)
      , m11(a11) {}

template <typename T>
Matrix<T, 2, 2>::Matrix(std::initializer_list<T> list) noexcept {
    const T* it = list.begin();
    m00         = (it != list.end()) ? *it++ : T(0);
    m01         = (it != list.end()) ? *it++ : T(0);
    m10         = (it != list.end()) ? *it++ : T(0);
    m11         = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Matrix<T, 2, 2>::Matrix(const MatrixExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    m00                 = static_cast<T>(e(0, 0));
    m01                 = static_cast<T>(e(0, 1));
    m10                 = static_cast<T>(e(1, 0));
    m11                 = static_cast<T>(e(1, 1));
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 2, 2>::rows() noexcept {
    return 2;
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 2, 2>::cols() noexcept {
    return 2;
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 2, 2>::size() noexcept {
    return 4;
}

template <typename T>
ATLAS_NODISCARD const T*
Matrix<T, 2, 2>::data() const noexcept {
    return &m00;
}

template <typename T>
T*
Matrix<T, 2, 2>::data() noexcept {
    return &m00;
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 2, 2>::operator[](std::size_t i) const noexcept {
    return (&m00)[i];
}

template <typename T>
T&
Matrix<T, 2, 2>::operator[](std::size_t i) noexcept {
    return (&m00)[i];
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 2, 2>::at(std::size_t r, std::size_t c) const noexcept {
    return (&m00)[r * 2 + c];
}

template <typename T>
T&
Matrix<T, 2, 2>::at(std::size_t r, std::size_t c) noexcept {
    return (&m00)[r * 2 + c];
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 2, 2>::operator()(std::size_t r, std::size_t c) const noexcept {
    return (&m00)[r * 2 + c];
}

template <typename T>
T&
Matrix<T, 2, 2>::operator()(std::size_t r, std::size_t c) noexcept {
    return (&m00)[r * 2 + c];
}

template <typename T>
void
Matrix<T, 2, 2>::set_zero() noexcept {
    m00 = m01 = m10 = m11 = T(0);
}

template <typename T>
void
Matrix<T, 2, 2>::set_identity() noexcept {
    m00 = T(1);
    m01 = T(0);
    m10 = T(0);
    m11 = T(1);
}

template <typename T>
void
Matrix<T, 2, 2>::set(T a00, T a01, T a10, T a11) noexcept {
    m00 = a00;
    m01 = a01;
    m10 = a10;
    m11 = a11;
}

template <typename T>
void
Matrix<T, 2, 2>::add(T s) noexcept {
    m00 += s;
    m01 += s;
    m10 += s;
    m11 += s;
}

template <typename T>
void
Matrix<T, 2, 2>::sub(T s) noexcept {
    m00 -= s;
    m01 -= s;
    m10 -= s;
    m11 -= s;
}

template <typename T>
void
Matrix<T, 2, 2>::mul(T s) noexcept {
    m00 *= s;
    m01 *= s;
    m10 *= s;
    m11 *= s;
}

template <typename T>
void
Matrix<T, 2, 2>::div(T s) noexcept {
    const T inv = T(1) / s;
    m00         *= inv;
    m01         *= inv;
    m10         *= inv;
    m11         *= inv;
}

template <typename T>
void
Matrix<T, 2, 2>::add(const Matrix& m) noexcept {
    m00 += m.m00;
    m01 += m.m01;
    m10 += m.m10;
    m11 += m.m11;
}

template <typename T>
void
Matrix<T, 2, 2>::sub(const Matrix& m) noexcept {
    m00 -= m.m00;
    m01 -= m.m01;
    m10 -= m.m10;
    m11 -= m.m11;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator+=(T s) noexcept {
    add(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator-=(T s) noexcept {
    sub(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator*=(T s) noexcept {
    mul(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator/=(T s) noexcept {
    div(s);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator+=(const Matrix& m) noexcept {
    add(m);
    return *this;
}

template <typename T>
Matrix<T, 2, 2>&
Matrix<T, 2, 2>::operator-=(const Matrix& m) noexcept {
    sub(m);
    return *this;
}

template <typename T>
bool
Matrix<T, 2, 2>::operator==(const Matrix& other) const noexcept {
    return m00 == other.m00 && m01 == other.m01 && m10 == other.m10 && m11 == other.m11;
}

template <typename T>
bool
Matrix<T, 2, 2>::operator!=(const Matrix& other) const noexcept {
    return !(*this == other);
}

template <typename T>
ATLAS_NODISCARD
T

Matrix<T, 2, 2>::determinant() const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(m00, m11, -m01 * m10);
    } else {
        return m00 * m11 - m01 * m10;
    }
}

template <typename T>
ATLAS_NODISCARD
T

Matrix<T, 2, 2>::trace() const noexcept {
    return m00 + m11;
}

template <typename T>
void
Matrix<T, 2, 2>::transpose() noexcept {
    const T t = m01;
    m01       = m10;
    m10       = t;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

Matrix<T, 2, 2>::transposed() const noexcept {
    return Matrix(m00, m10, m01, m11);
}

template <typename T>
void
Matrix<T, 2, 2>::inverse() noexcept {
    const T det = determinant();
    const T inv = T(1) / det;
    const T a   = m00, b = m01, c = m10, d = m11;
    m00         = d * inv;
    m01         = -b * inv;
    m10         = -c * inv;
    m11         = a * inv;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

Matrix<T, 2, 2>::inversed() const noexcept {
    const T det = determinant();
    const T inv = T(1) / det;
    return Matrix(m11 * inv,
                  -m01 * inv,
                  -m10 * inv,
                  m00 * inv);
}

template <typename T>
bool
Matrix<T, 2, 2>::try_inverse(Matrix& out, T eps) const noexcept {
    const T det = determinant();
    if (std::abs(det) <= eps) return false;
    const T inv = T(1) / det;
    out.m00     = m11 * inv;
    out.m01     = -m01 * inv;
    out.m10     = -m10 * inv;
    out.m11     = m00 * inv;
    return true;
}

template <typename T>
ATLAS_NODISCARD bool
Matrix<T, 2, 2>::is_invertible(T eps) const noexcept {
    return std::abs(determinant()) > eps;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

Matrix<T, 2, 2>::mul(const Matrix& r) const noexcept {
    return Matrix(
        std::fma(m01, r.m10, m00 * r.m00),
        std::fma(m01, r.m11, m00 * r.m01),
        std::fma(m11, r.m10, m10 * r.m00),
        std::fma(m11, r.m11, m10 * r.m01));
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 2>

Matrix<T, 2, 2>::mul(const Vector<T, 2>& v) const noexcept {
    return Vector<T, 2>(
        std::fma(m01, v[1], m00 * v[0]),
        std::fma(m11, v[1], m10 * v[0]));
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

identity2x2() noexcept {
    return Matrix<T, 2, 2>(T(1));
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

zero2x2() noexcept {
    return Matrix<T, 2, 2>(T(0), T(0), T(0), T(0));
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

transpose(const Matrix<T, 2, 2>& m) noexcept {
    return m.transposed();
}

template <typename T>
ATLAS_NODISCARD
T

determinant(const Matrix<T, 2, 2>& m) noexcept {
    return m.determinant();
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

inverse(const Matrix<T, 2, 2>& m) noexcept {
    return m.inversed();
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator+(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return Matrix<T, 2, 2>(
        a.m00 + b.m00,
        a.m01 + b.m01,
        a.m10 + b.m10,
        a.m11 + b.m11);
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator-(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return Matrix<T, 2, 2>(
        a.m00 - b.m00,
        a.m01 - b.m01,
        a.m10 - b.m10,
        a.m11 - b.m11);
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator*(const Matrix<T, 2, 2>& a, T s) {
    return Matrix<T, 2, 2>(
        a.m00 * s,
        a.m01 * s,
        a.m10 * s,
        a.m11 * s);
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator*(T s, const Matrix<T, 2, 2>& a) {
    return a * s;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator/(const Matrix<T, 2, 2>& a, T s) {
    const T inv = T(1) / s;
    return Matrix<T, 2, 2>(
        a.m00 * inv,
        a.m01 * inv,
        a.m10 * inv,
        a.m11 * inv);
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 2, 2>

operator*(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b) {
    return a.mul(b);
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 2>

operator*(const Matrix<T, 2, 2>& a, const Vector<T, 2>& v) {
    return a.mul(v);
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 2>

Matrix<T, 2, 2>::solved(const Vector<T, 2>& b) const noexcept {
    const T det = determinant();
    const T inv = T(1) / det;
    const T a   = m00, b01 = m01, c = m10, d = m11;
    const T n0  = std::fma(b[0], d, -b01 * b[1]);
    const T n1  = std::fma(a, b[1], -b[0] * c);
    return Vector<T, 2>(n0 * inv, n1 * inv);
}

template <typename T>
bool
Matrix<T, 2, 2>::solve(const Vector<T, 2>& b, Vector<T, 2>& x, T eps) const noexcept {
    const T det = determinant();
    if (std::abs(det) <= eps) return false;
    const T a   = m00, b01 = m01, c = m10, d = m11;
    const T inv = T(1) / det;
    const T n0  = std::fma(b[0], d, -b01 * b[1]);
    const T n1  = std::fma(a, b[1], -b[0] * c);
    x           = Vector<T, 2>(n0 * inv, n1 * inv);
    return true;
}

template <typename T>
ATLAS_NODISCARD bool
solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b, Vector<T, 2>& x, T eps) noexcept {
    return A.solve(b, x, eps);
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 2>

solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b) noexcept {
    return A.solved(b);
}
}