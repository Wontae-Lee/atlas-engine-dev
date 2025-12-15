#pragma once
namespace atlas::math {
template <typename T>
constexpr
Matrix<T, 3, 3>::Matrix() noexcept
    : m00(T(0))
      , m01(T(0))
      , m02(T(0))
      , m10(T(0))
      , m11(T(0))
      , m12(T(0))
      , m20(T(0))
      , m21(T(0))
      , m22(T(0)) {}

template <typename T>
constexpr
Matrix<T, 3, 3>::Matrix(T s) noexcept
    : m00(s)
      , m01(T(0))
      , m02(T(0))
      , m10(T(0))
      , m11(s)
      , m12(T(0))
      , m20(T(0))
      , m21(T(0))
      , m22(s) {}

template <typename T>
constexpr
Matrix<T, 3, 3>::Matrix(
    T a00, T a01, T a02,
    T a10, T a11, T a12,
    T a20, T a21, T a22) noexcept
    : m00(a00)
      , m01(a01)
      , m02(a02)
      , m10(a10)
      , m11(a11)
      , m12(a12)
      , m20(a20)
      , m21(a21)
      , m22(a22) {}

template <typename T>
Matrix<T, 3, 3>::Matrix(std::initializer_list<T> list) noexcept {
    const T* it = list.begin();
    m00         = (it != list.end()) ? *it++ : T(0);
    m01         = (it != list.end()) ? *it++ : T(0);
    m02         = (it != list.end()) ? *it++ : T(0);
    m10         = (it != list.end()) ? *it++ : T(0);
    m11         = (it != list.end()) ? *it++ : T(0);
    m12         = (it != list.end()) ? *it++ : T(0);
    m20         = (it != list.end()) ? *it++ : T(0);
    m21         = (it != list.end()) ? *it++ : T(0);
    m22         = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Matrix<T, 3, 3>::Matrix(const MatrixExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    m00                 = static_cast<T>(e(0, 0));
    m01                 = static_cast<T>(e(0, 1));
    m02                 = static_cast<T>(e(0, 2));
    m10                 = static_cast<T>(e(1, 0));
    m11                 = static_cast<T>(e(1, 1));
    m12                 = static_cast<T>(e(1, 2));
    m20                 = static_cast<T>(e(2, 0));
    m21                 = static_cast<T>(e(2, 1));
    m22                 = static_cast<T>(e(2, 2));
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 3, 3>::rows() noexcept {
    return 3;
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 3, 3>::cols() noexcept {
    return 3;
}

template <typename T>
ATLAS_NODISCARD
std::size_t

Matrix<T, 3, 3>::size() noexcept {
    return 9;
}

template <typename T>
ATLAS_NODISCARD const T*
Matrix<T, 3, 3>::data() const noexcept {
    return &m00;
}

template <typename T>
T*
Matrix<T, 3, 3>::data() noexcept {
    return &m00;
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 3, 3>::operator[](std::size_t i) const noexcept {
    return (&m00)[i];
}

template <typename T>
T&
Matrix<T, 3, 3>::operator[](std::size_t i) noexcept {
    return (&m00)[i];
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 3, 3>::at(std::size_t r, std::size_t c) const noexcept {
    return (&m00)[r * 3 + c];
}

template <typename T>
T&
Matrix<T, 3, 3>::at(std::size_t r, std::size_t c) noexcept {
    return (&m00)[r * 3 + c];
}

template <typename T>
ATLAS_NODISCARD const T&
Matrix<T, 3, 3>::operator()(std::size_t r, std::size_t c) const noexcept {
    return (&m00)[r * 3 + c];
}

template <typename T>
T&
Matrix<T, 3, 3>::operator()(std::size_t r, std::size_t c) noexcept {
    return (&m00)[r * 3 + c];
}

template <typename T>
void
Matrix<T, 3, 3>::set_zero() noexcept {
    m00 = m01 = m02 = T(0);
    m10 = m11 = m12 = T(0);
    m20 = m21 = m22 = T(0);
}

template <typename T>
void
Matrix<T, 3, 3>::set_identity() noexcept {
    m00 = T(1);
    m01 = T(0);
    m02 = T(0);
    m10 = T(0);
    m11 = T(1);
    m12 = T(0);
    m20 = T(0);
    m21 = T(0);
    m22 = T(1);
}

template <typename T>
void
Matrix<T, 3, 3>::set(
    T a00, T a01, T a02,
    T a10, T a11, T a12,
    T a20, T a21, T a22) noexcept {
    m00 = a00;
    m01 = a01;
    m02 = a02;
    m10 = a10;
    m11 = a11;
    m12 = a12;
    m20 = a20;
    m21 = a21;
    m22 = a22;
}

template <typename T>
void
Matrix<T, 3, 3>::add(T s) noexcept {
    for (int i = 0; i < 9; ++i) (&m00)[i] += s;
}

template <typename T>
void
Matrix<T, 3, 3>::sub(T s) noexcept {
    for (int i = 0; i < 9; ++i) (&m00)[i] -= s;
}

template <typename T>
void
Matrix<T, 3, 3>::mul(T s) noexcept {
    for (int i = 0; i < 9; ++i) (&m00)[i] *= s;
}

template <typename T>
void
Matrix<T, 3, 3>::div(T s) noexcept {
    const T inv = T(1) / s;
    for (int i = 0; i < 9; ++i) (&m00)[i] *= inv;
}

template <typename T>
void
Matrix<T, 3, 3>::add(const Matrix& m) noexcept {
    for (int i = 0; i < 9; ++i) (&m00)[i] += (&m.m00)[i];
}

template <typename T>
void
Matrix<T, 3, 3>::sub(const Matrix& m) noexcept {
    for (int i = 0; i < 9; ++i) (&m00)[i] -= (&m.m00)[i];
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator+=(T s) noexcept {
    add(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator-=(T s) noexcept {
    sub(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator*=(T s) noexcept {
    mul(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator/=(T s) noexcept {
    div(s);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator+=(const Matrix& m) noexcept {
    add(m);
    return *this;
}

template <typename T>
Matrix<T, 3, 3>&
Matrix<T, 3, 3>::operator-=(const Matrix& m) noexcept {
    sub(m);
    return *this;
}

template <typename T>
bool
Matrix<T, 3, 3>::operator==(const Matrix& o) const noexcept {
    for (int i = 0; i < 9; ++i) if ((&m00)[i] != (&o.m00)[i]) return false;
    return true;
}

template <typename T>
bool
Matrix<T, 3, 3>::operator!=(const Matrix& o) const noexcept {
    return !(*this == o);
}

template <typename T>
ATLAS_NODISCARD
T

Matrix<T, 3, 3>::determinant() const noexcept {
    const T a     = m00, b = m01, c = m02;
    const T d     = m10, e = m11, f = m12;
    const T g     = m20, h = m21, i = m22;
    const T ei_fh = e * i - f * h;
    const T di_fg = d * i - f * g;
    const T dh_eg = d * h - e * g;
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(a, ei_fh, std::fma(-b, di_fg, c * dh_eg));
    } else {
        return a * ei_fh - b * di_fg + c * dh_eg;
    }
}

template <typename T>
ATLAS_NODISCARD
T

Matrix<T, 3, 3>::trace() const noexcept {
    return m00 + m11 + m22;
}

template <typename T>
void
Matrix<T, 3, 3>::transpose() noexcept {
    std::swap(m01, m10);
    std::swap(m02, m20);
    std::swap(m12, m21);
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

Matrix<T, 3, 3>::transposed() const noexcept {
    return Matrix(
        m00,
        m10,
        m20,
        m01,
        m11,
        m21,
        m02,
        m12,
        m22);
}

template <typename T>
void
Matrix<T, 3, 3>::inverse() noexcept {
    const T a      = m00, b = m01, c = m02;
    const T d      = m10, e = m11, f = m12;
    const T g      = m20, h = m21, i = m22;
    const T C00    = (e * i - f * h);
    const T C01    = -(d * i - f * g);
    const T C02    = (d * h - e * g);
    const T C10    = -(b * i - c * h);
    const T C11    = (a * i - c * g);
    const T C12    = -(a * h - b * g);
    const T C20    = (b * f - c * e);
    const T C21    = -(a * f - c * d);
    const T C22    = (a * e - b * d);
    const T det    = a * C00 + b * C01 + c * C02;
    const T invDet = T(1) / det;
    m00            = C00 * invDet;
    m01            = C10 * invDet;
    m02            = C20 * invDet;
    m10            = C01 * invDet;
    m11            = C11 * invDet;
    m12            = C21 * invDet;
    m20            = C02 * invDet;
    m21            = C12 * invDet;
    m22            = C22 * invDet;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

Matrix<T, 3, 3>::inversed() const noexcept {
    Matrix A = *this;
    A.inverse();
    return A;
}

template <typename T>
bool
Matrix<T, 3, 3>::try_inverse(Matrix& out, T eps) const noexcept {
    const T a   = m00, b = m01, c = m02;
    const T d   = m10, e = m11, f = m12;
    const T g   = m20, h = m21, i = m22;
    const T C00 = (e * i - f * h);
    const T C01 = -(d * i - f * g);
    const T C02 = (d * h - e * g);
    const T C10 = -(b * i - c * h);
    const T C11 = (a * i - c * g);
    const T C12 = -(a * h - b * g);
    const T C20 = (b * f - c * e);
    const T C21 = -(a * f - c * d);
    const T C22 = (a * e - b * d);
    const T det = a * C00 + b * C01 + c * C02;
    if (std::abs(det) <= eps) return false;
    const T invDet = T(1) / det;
    out.m00        = C00 * invDet;
    out.m01        = C10 * invDet;
    out.m02        = C20 * invDet;
    out.m10        = C01 * invDet;
    out.m11        = C11 * invDet;
    out.m12        = C21 * invDet;
    out.m20        = C02 * invDet;
    out.m21        = C12 * invDet;
    out.m22        = C22 * invDet;
    return true;
}

template <typename T>
ATLAS_NODISCARD bool
Matrix<T, 3, 3>::is_invertible(T eps) const noexcept {
    return std::abs(determinant()) > eps;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

Matrix<T, 3, 3>::mul(const Matrix& r) const noexcept {
    const Matrix& a = *this;
    Matrix out;
    if constexpr (std::is_floating_point_v<T>) {
        out.m00 = std::fma(a.m01, r.m10, std::fma(a.m02, r.m20, a.m00 * r.m00));
        out.m01 = std::fma(a.m01, r.m11, std::fma(a.m02, r.m21, a.m00 * r.m01));
        out.m02 = std::fma(a.m01, r.m12, std::fma(a.m02, r.m22, a.m00 * r.m02));
        out.m10 = std::fma(a.m11, r.m10, std::fma(a.m12, r.m20, a.m10 * r.m00));
        out.m11 = std::fma(a.m11, r.m11, std::fma(a.m12, r.m21, a.m10 * r.m01));
        out.m12 = std::fma(a.m11, r.m12, std::fma(a.m12, r.m22, a.m10 * r.m02));
        out.m20 = std::fma(a.m21, r.m10, std::fma(a.m22, r.m20, a.m20 * r.m00));
        out.m21 = std::fma(a.m21, r.m11, std::fma(a.m22, r.m21, a.m20 * r.m01));
        out.m22 = std::fma(a.m21, r.m12, std::fma(a.m22, r.m22, a.m20 * r.m02));
    } else {
        out.m00 = a.m00 * r.m00 + a.m01 * r.m10 + a.m02 * r.m20;
        out.m01 = a.m00 * r.m01 + a.m01 * r.m11 + a.m02 * r.m21;
        out.m02 = a.m00 * r.m02 + a.m01 * r.m12 + a.m02 * r.m22;
        out.m10 = a.m10 * r.m00 + a.m11 * r.m10 + a.m12 * r.m20;
        out.m11 = a.m10 * r.m01 + a.m11 * r.m11 + a.m12 * r.m21;
        out.m12 = a.m10 * r.m02 + a.m11 * r.m12 + a.m12 * r.m22;
        out.m20 = a.m20 * r.m00 + a.m21 * r.m10 + a.m22 * r.m20;
        out.m21 = a.m20 * r.m01 + a.m21 * r.m11 + a.m22 * r.m21;
        out.m22 = a.m20 * r.m02 + a.m21 * r.m12 + a.m22 * r.m22;
    }
    return out;
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 3>

Matrix<T, 3, 3>::mul(const Vector<T, 3>& v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return Vector<T, 3>(
            std::fma(m01, v[1], std::fma(m02, v[2], m00 * v[0])),
            std::fma(m11, v[1], std::fma(m12, v[2], m10 * v[0])),
            std::fma(m21, v[1], std::fma(m22, v[2], m20 * v[0])));
    } else {
        return Vector<T, 3>(
            m00 * v[0] + m01 * v[1] + m02 * v[2],
            m10 * v[0] + m11 * v[1] + m12 * v[2],
            m20 * v[0] + m21 * v[1] + m22 * v[2]);
    }
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 3>

Matrix<T, 3, 3>::solved(const Vector<T, 3>& b) const noexcept {
    Matrix inv = inversed();
    return inv.mul(b);
}

template <typename T>
bool
Matrix<T, 3, 3>::solve(const Vector<T, 3>& b, Vector<T, 3>& x, T eps) const noexcept {
    Matrix inv;
    if (!try_inverse(inv, eps)) return false;
    x = inv.mul(b);
    return true;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

identity3x3() noexcept {
    return Matrix<T, 3, 3>(T(1));
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

zero3x3() noexcept {
    return Matrix<T, 3, 3>(T(0),
                           T(0),
                           T(0),
                           T(0),
                           T(0),
                           T(0),
                           T(0),
                           T(0),
                           T(0));
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

transpose(const Matrix<T, 3, 3>& m) noexcept {
    return m.transposed();
}

template <typename T>
ATLAS_NODISCARD
T

determinant(const Matrix<T, 3, 3>& m) noexcept {
    return m.determinant();
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

inverse(const Matrix<T, 3, 3>& m) noexcept {
    return m.inversed();
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator+(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    Matrix<T, 3, 3> out;
    for (int i = 0; i < 9; ++i) (&out.m00)[i] = (&a.m00)[i] + (&b.m00)[i];
    return out;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator-(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    Matrix<T, 3, 3> out;
    for (int i = 0; i < 9; ++i) (&out.m00)[i] = (&a.m00)[i] - (&b.m00)[i];
    return out;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator*(const Matrix<T, 3, 3>& a, T s) {
    Matrix<T, 3, 3> out;
    for (int i = 0; i < 9; ++i) (&out.m00)[i] = (&a.m00)[i] * s;
    return out;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator*(T s, const Matrix<T, 3, 3>& a) {
    return a * s;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator/(const Matrix<T, 3, 3>& a, T s) {
    const T inv = T(1) / s;
    return a * inv;
}

template <typename T>
ATLAS_NODISCARD
Matrix<T, 3, 3>

operator*(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b) {
    return a.mul(b);
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 3>

operator*(const Matrix<T, 3, 3>& a, const Vector<T, 3>& v) {
    return a.mul(v);
}

template <typename T>
ATLAS_NODISCARD bool
solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b, Vector<T, 3>& x, T eps) noexcept {
    return A.solve(b, x, eps);
}

template <typename T>
ATLAS_NODISCARD
Vector<T, 3>

solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b) noexcept {
    return A.solved(b);
}
}