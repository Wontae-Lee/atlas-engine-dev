#pragma once
namespace atlas {
template <typename T>
constexpr Matrix<T, 4, 4>::Matrix() noexcept
    : m00(T(0))
    , m01(T(0))
    , m02(T(0))
    , m03(T(0))
    , m10(T(0))
    , m11(T(0))
    , m12(T(0))
    , m13(T(0))
    , m20(T(0))
    , m21(T(0))
    , m22(T(0))
    , m23(T(0))
    , m30(T(0))
    , m31(T(0))
    , m32(T(0))
    , m33(T(0)) {
}

template <typename T>
constexpr Matrix<T, 4, 4>::Matrix(T s) noexcept
    : m00(s)
    , m01(T(0))
    , m02(T(0))
    , m03(T(0))
    , m10(T(0))
    , m11(s)
    , m12(T(0))
    , m13(T(0))
    , m20(T(0))
    , m21(T(0))
    , m22(s)
    , m23(T(0))
    , m30(T(0))
    , m31(T(0))
    , m32(T(0))
    , m33(s) {
}

template <typename T>
constexpr Matrix<T, 4, 4>::Matrix(
    T a00, T a01, T a02, T a03,
    T a10, T a11, T a12, T a13,
    T a20, T a21, T a22, T a23,
    T a30, T a31, T a32, T a33) noexcept
    : m00(a00)
    , m01(a01)
    , m02(a02)
    , m03(a03)
    , m10(a10)
    , m11(a11)
    , m12(a12)
    , m13(a13)
    , m20(a20)
    , m21(a21)
    , m22(a22)
    , m23(a23)
    , m30(a30)
    , m31(a31)
    , m32(a32)
    , m33(a33) {
}

template <typename T>
Matrix<T, 4, 4>::Matrix(std::initializer_list<T> list) noexcept {
    const T* it = list.begin();
    m00         = (it != list.end()) ? *it++ : T(0);
    m01         = (it != list.end()) ? *it++ : T(0);
    m02         = (it != list.end()) ? *it++ : T(0);
    m03         = (it != list.end()) ? *it++ : T(0);
    m10         = (it != list.end()) ? *it++ : T(0);
    m11         = (it != list.end()) ? *it++ : T(0);
    m12         = (it != list.end()) ? *it++ : T(0);
    m13         = (it != list.end()) ? *it++ : T(0);
    m20         = (it != list.end()) ? *it++ : T(0);
    m21         = (it != list.end()) ? *it++ : T(0);
    m22         = (it != list.end()) ? *it++ : T(0);
    m23         = (it != list.end()) ? *it++ : T(0);
    m30         = (it != list.end()) ? *it++ : T(0);
    m31         = (it != list.end()) ? *it++ : T(0);
    m32         = (it != list.end()) ? *it++ : T(0);
    m33         = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Matrix<T, 4, 4>::Matrix(const MatrixExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    m00                 = static_cast<T>(e(0, 0));
    m01                 = static_cast<T>(e(0, 1));
    m02                 = static_cast<T>(e(0, 2));
    m03                 = static_cast<T>(e(0, 3));
    m10                 = static_cast<T>(e(1, 0));
    m11                 = static_cast<T>(e(1, 1));
    m12                 = static_cast<T>(e(1, 2));
    m13                 = static_cast<T>(e(1, 3));
    m20                 = static_cast<T>(e(2, 0));
    m21                 = static_cast<T>(e(2, 1));
    m22                 = static_cast<T>(e(2, 2));
    m23                 = static_cast<T>(e(2, 3));
    m30                 = static_cast<T>(e(3, 0));
    m31                 = static_cast<T>(e(3, 1));
    m32                 = static_cast<T>(e(3, 2));
    m33                 = static_cast<T>(e(3, 3));
}

template <typename T>
std::size_t
Matrix<T, 4, 4>::rows() noexcept {
    return 4;
}

template <typename T>
std::size_t
Matrix<T, 4, 4>::cols() noexcept {
    return 4;
}

template <typename T>
std::size_t
Matrix<T, 4, 4>::size() noexcept {
    return 16;
}

template <typename T>
const T*
Matrix<T, 4, 4>::data() const noexcept {
    return _data;
}

template <typename T>
T*
Matrix<T, 4, 4>::data() noexcept {
    return _data;
}

template <typename T>
const T&
Matrix<T, 4, 4>::operator[](std::size_t i) const noexcept {
    return _data[i];
}

template <typename T>
T&
Matrix<T, 4, 4>::operator[](std::size_t i) noexcept {
    return _data[i];
}

template <typename T>
const T&
Matrix<T, 4, 4>::at(std::size_t r, std::size_t c) const noexcept {
    return _data[r * 4 + c];
}

template <typename T>
T&
Matrix<T, 4, 4>::at(std::size_t r, std::size_t c) noexcept {
    return _data[r * 4 + c];
}

template <typename T>
const T&
Matrix<T, 4, 4>::operator()(std::size_t r, std::size_t c) const noexcept {
    return _data[r * 4 + c];
}

template <typename T>
T&
Matrix<T, 4, 4>::operator()(std::size_t r, std::size_t c) noexcept {
    return _data[r * 4 + c];
}

template <typename T>
void
Matrix<T, 4, 4>::set_zero() noexcept {
    for (int i = 0; i < 16; ++i) _data[i] = T(0);
}

template <typename T>
void
Matrix<T, 4, 4>::set_identity() noexcept {
    m00 = T(1);
    m01 = T(0);
    m02 = T(0);
    m03 = T(0);
    m10 = T(0);
    m11 = T(1);
    m12 = T(0);
    m13 = T(0);
    m20 = T(0);
    m21 = T(0);
    m22 = T(1);
    m23 = T(0);
    m30 = T(0);
    m31 = T(0);
    m32 = T(0);
    m33 = T(1);
}

template <typename T>
void
Matrix<T, 4, 4>::set(
    T a00, T a01, T a02, T a03,
    T a10, T a11, T a12, T a13,
    T a20, T a21, T a22, T a23,
    T a30, T a31, T a32, T a33) noexcept {
    m00 = a00;
    m01 = a01;
    m02 = a02;
    m03 = a03;
    m10 = a10;
    m11 = a11;
    m12 = a12;
    m13 = a13;
    m20 = a20;
    m21 = a21;
    m22 = a22;
    m23 = a23;
    m30 = a30;
    m31 = a31;
    m32 = a32;
    m33 = a33;
}

template <typename T>
void
Matrix<T, 4, 4>::add(T s) noexcept {
    for (int i = 0; i < 16; ++i) _data[i] += s;
}

template <typename T>
void
Matrix<T, 4, 4>::sub(T s) noexcept {
    for (int i = 0; i < 16; ++i) _data[i] -= s;
}

template <typename T>
void
Matrix<T, 4, 4>::mul(T s) noexcept {
    for (int i = 0; i < 16; ++i) _data[i] *= s;
}

template <typename T>
void
Matrix<T, 4, 4>::div(T s) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        const T inv = T(1) / s;
        for (int i = 0; i < 16; ++i) _data[i] *= inv;
    } else {
        for (int i = 0; i < 16; ++i) _data[i] /= s;
    }
}

template <typename T>
void
Matrix<T, 4, 4>::add(const Matrix& m) noexcept {
    for (int i = 0; i < 16; ++i) _data[i] += m._data[i];
}

template <typename T>
void
Matrix<T, 4, 4>::sub(const Matrix& m) noexcept {
    for (int i = 0; i < 16; ++i) _data[i] -= m._data[i];
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator+=(T s) noexcept {
    add(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator-=(T s) noexcept {
    sub(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator*=(T s) noexcept {
    mul(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator/=(T s) noexcept {
    div(s);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator+=(const Matrix& m) noexcept {
    add(m);
    return *this;
}

template <typename T>
Matrix<T, 4, 4>&
Matrix<T, 4, 4>::operator-=(const Matrix& m) noexcept {
    sub(m);
    return *this;
}

template <typename T>
bool
Matrix<T, 4, 4>::operator==(const Matrix& o) const noexcept {
    for (int i = 0; i < 16; ++i)
        if (_data[i] != o._data[i]) return false;
    return true;
}

template <typename T>
bool
Matrix<T, 4, 4>::operator!=(const Matrix& o) const noexcept {
    return !(*this == o);
}

template <typename T>
T
Matrix<T, 4, 4>::trace() const noexcept {
    return m00 + m11 + m22 + m33;
}

template <typename T>
T
Matrix<T, 4, 4>::determinant() const noexcept {
    auto det3 = [](T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) -> T {
        const T ei_fh = a11 * a22 - a12 * a21;
        const T di_fg = a10 * a22 - a12 * a20;
        const T dh_eg = a10 * a21 - a11 * a20;
        if constexpr (std::is_floating_point_v<T>) {
            return std::fma(a00, ei_fh, std::fma(-a01, di_fg, a02 * dh_eg));
        } else {
            return a00 * ei_fh - a01 * di_fg + a02 * dh_eg;
        }
    };
    const T a00 = m00, a01 = m01, a02 = m02, a03 = m03;
    const T a10 = m10, a11 = m11, a12 = m12, a13 = m13;
    const T a20 = m20, a21 = m21, a22 = m22, a23 = m23;
    const T a30 = m30, a31 = m31, a32 = m32, a33 = m33;
    const T M0 = det3(a11, a12, a13, a21, a22, a23, a31, a32, a33);
    const T M1 = det3(a10, a12, a13, a20, a22, a23, a30, a32, a33);
    const T M2 = det3(a10, a11, a13, a20, a21, a23, a30, a31, a33);
    const T M3 = det3(a10, a11, a12, a20, a21, a22, a30, a31, a32);
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(a00, M0, std::fma(-a01, M1, std::fma(a02, M2, -a03 * M3)));
    } else {
        return a00 * M0 - a01 * M1 + a02 * M2 - a03 * M3;
    }
}

template <typename T>
void
Matrix<T, 4, 4>::transpose() noexcept {
    std::swap(m01, m10);
    std::swap(m02, m20);
    std::swap(m03, m30);
    std::swap(m12, m21);
    std::swap(m13, m31);
    std::swap(m23, m32);
}

template <typename T>
Matrix<T, 4, 4>
Matrix<T, 4, 4>::transposed() const noexcept {
    return Matrix(
        m00,
        m10,
        m20,
        m30,
        m01,
        m11,
        m21,
        m31,
        m02,
        m12,
        m22,
        m32,
        m03,
        m13,
        m23,
        m33);
}

template <typename T>
void
Matrix<T, 4, 4>::inverse() noexcept {
    auto det3 = [](T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) -> T {
        return a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);
    };
    const T a00 = m00, a01 = m01, a02 = m02, a03 = m03;
    const T a10 = m10, a11 = m11, a12 = m12, a13 = m13;
    const T a20 = m20, a21 = m21, a22 = m22, a23 = m23;
    const T a30 = m30, a31 = m31, a32 = m32, a33 = m33;
    const T C00    = det3(a11, a12, a13, a21, a22, a23, a31, a32, a33);
    const T C01    = -det3(a10, a12, a13, a20, a22, a23, a30, a32, a33);
    const T C02    = det3(a10, a11, a13, a20, a21, a23, a30, a31, a33);
    const T C03    = -det3(a10, a11, a12, a20, a21, a22, a30, a31, a32);
    const T C10    = -det3(a01, a02, a03, a21, a22, a23, a31, a32, a33);
    const T C11    = det3(a00, a02, a03, a20, a22, a23, a30, a32, a33);
    const T C12    = -det3(a00, a01, a03, a20, a21, a23, a30, a31, a33);
    const T C13    = det3(a00, a01, a02, a20, a21, a22, a30, a31, a32);
    const T C20    = det3(a01, a02, a03, a11, a12, a13, a31, a32, a33);
    const T C21    = -det3(a00, a02, a03, a10, a12, a13, a30, a32, a33);
    const T C22    = det3(a00, a01, a03, a10, a11, a13, a30, a31, a33);
    const T C23    = -det3(a00, a01, a02, a10, a11, a12, a30, a31, a32);
    const T C30    = -det3(a01, a02, a03, a11, a12, a13, a21, a22, a23);
    const T C31    = det3(a00, a02, a03, a10, a12, a13, a20, a22, a23);
    const T C32    = -det3(a00, a01, a03, a10, a11, a13, a20, a21, a23);
    const T C33    = det3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    const T det    = a00 * C00 + a01 * C01 + a02 * C02 + a03 * C03;
    const T invDet = T(1) / det;
    m00            = C00 * invDet;
    m01            = C10 * invDet;
    m02            = C20 * invDet;
    m03            = C30 * invDet;
    m10            = C01 * invDet;
    m11            = C11 * invDet;
    m12            = C21 * invDet;
    m13            = C31 * invDet;
    m20            = C02 * invDet;
    m21            = C12 * invDet;
    m22            = C22 * invDet;
    m23            = C32 * invDet;
    m30            = C03 * invDet;
    m31            = C13 * invDet;
    m32            = C23 * invDet;
    m33            = C33 * invDet;
}

template <typename T>
Matrix<T, 4, 4>
Matrix<T, 4, 4>::inversed() const noexcept {
    Matrix A = *this;
    A.inverse();
    return A;
}

template <typename T>
bool
Matrix<T, 4, 4>::try_inverse(Matrix& out, T eps) const noexcept {
    auto det3 = [](T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) -> T {
        return a00 * (a11 * a22 - a12 * a21)
            - a01 * (a10 * a22 - a12 * a20)
            + a02 * (a10 * a21 - a11 * a20);
    };
    const T a00 = m00, a01 = m01, a02 = m02, a03 = m03;
    const T a10 = m10, a11 = m11, a12 = m12, a13 = m13;
    const T a20 = m20, a21 = m21, a22 = m22, a23 = m23;
    const T a30 = m30, a31 = m31, a32 = m32, a33 = m33;
    const T C00 = det3(a11, a12, a13, a21, a22, a23, a31, a32, a33);
    const T C01 = -det3(a10, a12, a13, a20, a22, a23, a30, a32, a33);
    const T C02 = det3(a10, a11, a13, a20, a21, a23, a30, a31, a33);
    const T C03 = -det3(a10, a11, a12, a20, a21, a22, a30, a31, a32);
    const T C10 = -det3(a01, a02, a03, a21, a22, a23, a31, a32, a33);
    const T C11 = det3(a00, a02, a03, a20, a22, a23, a30, a32, a33);
    const T C12 = -det3(a00, a01, a03, a20, a21, a23, a30, a31, a33);
    const T C13 = det3(a00, a01, a02, a20, a21, a22, a30, a31, a32);
    const T C20 = det3(a01, a02, a03, a11, a12, a13, a31, a32, a33);
    const T C21 = -det3(a00, a02, a03, a10, a12, a13, a30, a32, a33);
    const T C22 = det3(a00, a01, a03, a10, a11, a13, a30, a31, a33);
    const T C23 = -det3(a00, a01, a02, a10, a11, a12, a30, a31, a32);
    const T C30 = -det3(a01, a02, a03, a11, a12, a13, a21, a22, a23);
    const T C31 = det3(a00, a02, a03, a10, a12, a13, a20, a22, a23);
    const T C32 = -det3(a00, a01, a03, a10, a11, a13, a20, a21, a23);
    const T C33 = det3(a00, a01, a02, a10, a11, a12, a20, a21, a22);
    const T det = a00 * C00 + a01 * C01 + a02 * C02 + a03 * C03;
    if (std::abs(det) <= eps) return false;
    const T invDet = T(1) / det;
    out.m00        = C00 * invDet;
    out.m01        = C10 * invDet;
    out.m02        = C20 * invDet;
    out.m03        = C30 * invDet;
    out.m10        = C01 * invDet;
    out.m11        = C11 * invDet;
    out.m12        = C21 * invDet;
    out.m13        = C31 * invDet;
    out.m20        = C02 * invDet;
    out.m21        = C12 * invDet;
    out.m22        = C22 * invDet;
    out.m23        = C32 * invDet;
    out.m30        = C03 * invDet;
    out.m31        = C13 * invDet;
    out.m32        = C23 * invDet;
    out.m33        = C33 * invDet;
    return true;
}

template <typename T>
bool
Matrix<T, 4, 4>::is_invertible(T eps) const noexcept {
    return std::abs(determinant()) > eps;
}

template <typename T>
Matrix<T, 4, 4>
Matrix<T, 4, 4>::mul(const Matrix& r) const noexcept {
    const Matrix& a = *this;
    Matrix out;
    if constexpr (std::is_floating_point_v<T>) {
        out.m00 = std::fma(a.m01, r.m10, std::fma(a.m02, r.m20, std::fma(a.m03, r.m30, a.m00 * r.m00)));
        out.m01 = std::fma(a.m01, r.m11, std::fma(a.m02, r.m21, std::fma(a.m03, r.m31, a.m00 * r.m01)));
        out.m02 = std::fma(a.m01, r.m12, std::fma(a.m02, r.m22, std::fma(a.m03, r.m32, a.m00 * r.m02)));
        out.m03 = std::fma(a.m01, r.m13, std::fma(a.m02, r.m23, std::fma(a.m03, r.m33, a.m00 * r.m03)));
        out.m10 = std::fma(a.m11, r.m10, std::fma(a.m12, r.m20, std::fma(a.m13, r.m30, a.m10 * r.m00)));
        out.m11 = std::fma(a.m11, r.m11, std::fma(a.m12, r.m21, std::fma(a.m13, r.m31, a.m10 * r.m01)));
        out.m12 = std::fma(a.m11, r.m12, std::fma(a.m12, r.m22, std::fma(a.m13, r.m32, a.m10 * r.m02)));
        out.m13 = std::fma(a.m11, r.m13, std::fma(a.m12, r.m23, std::fma(a.m13, r.m33, a.m10 * r.m03)));
        out.m20 = std::fma(a.m21, r.m10, std::fma(a.m22, r.m20, std::fma(a.m23, r.m30, a.m20 * r.m00)));
        out.m21 = std::fma(a.m21, r.m11, std::fma(a.m22, r.m21, std::fma(a.m23, r.m31, a.m20 * r.m01)));
        out.m22 = std::fma(a.m21, r.m12, std::fma(a.m22, r.m22, std::fma(a.m23, r.m32, a.m20 * r.m02)));
        out.m23 = std::fma(a.m21, r.m13, std::fma(a.m22, r.m23, std::fma(a.m23, r.m33, a.m20 * r.m03)));
        out.m30 = std::fma(a.m31, r.m10, std::fma(a.m32, r.m20, std::fma(a.m33, r.m30, a.m30 * r.m00)));
        out.m31 = std::fma(a.m31, r.m11, std::fma(a.m32, r.m21, std::fma(a.m33, r.m31, a.m30 * r.m01)));
        out.m32 = std::fma(a.m31, r.m12, std::fma(a.m32, r.m22, std::fma(a.m33, r.m32, a.m30 * r.m02)));
        out.m33 = std::fma(a.m31, r.m13, std::fma(a.m32, r.m23, std::fma(a.m33, r.m33, a.m30 * r.m03)));
    } else {
        out.m00 = a.m00 * r.m00 + a.m01 * r.m10 + a.m02 * r.m20 + a.m03 * r.m30;
        out.m01 = a.m00 * r.m01 + a.m01 * r.m11 + a.m02 * r.m21 + a.m03 * r.m31;
        out.m02 = a.m00 * r.m02 + a.m01 * r.m12 + a.m02 * r.m22 + a.m03 * r.m32;
        out.m03 = a.m00 * r.m03 + a.m01 * r.m13 + a.m02 * r.m23 + a.m03 * r.m33;
        out.m10 = a.m10 * r.m00 + a.m11 * r.m10 + a.m12 * r.m20 + a.m13 * r.m30;
        out.m11 = a.m10 * r.m01 + a.m11 * r.m11 + a.m12 * r.m21 + a.m13 * r.m31;
        out.m12 = a.m10 * r.m02 + a.m11 * r.m12 + a.m12 * r.m22 + a.m13 * r.m32;
        out.m13 = a.m10 * r.m03 + a.m11 * r.m13 + a.m12 * r.m23 + a.m13 * r.m33;
        out.m20 = a.m20 * r.m00 + a.m21 * r.m10 + a.m22 * r.m20 + a.m23 * r.m30;
        out.m21 = a.m20 * r.m01 + a.m21 * r.m11 + a.m22 * r.m21 + a.m23 * r.m31;
        out.m22 = a.m20 * r.m02 + a.m21 * r.m12 + a.m22 * r.m22 + a.m23 * r.m32;
        out.m23 = a.m20 * r.m03 + a.m21 * r.m13 + a.m22 * r.m23 + a.m23 * r.m33;
        out.m30 = a.m30 * r.m00 + a.m31 * r.m10 + a.m32 * r.m20 + a.m33 * r.m30;
        out.m31 = a.m30 * r.m01 + a.m31 * r.m11 + a.m32 * r.m21 + a.m33 * r.m31;
        out.m32 = a.m30 * r.m02 + a.m31 * r.m12 + a.m32 * r.m22 + a.m33 * r.m32;
        out.m33 = a.m30 * r.m03 + a.m31 * r.m13 + a.m32 * r.m23 + a.m33 * r.m33;
    }
    return out;
}

template <typename T>
Vector<T, 4>
Matrix<T, 4, 4>::mul(const Vector<T, 4>& v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return Vector<T, 4>(
            std::fma(m01, v[1], std::fma(m02, v[2], std::fma(m03, v[3], m00 * v[0]))),
            std::fma(m11, v[1], std::fma(m12, v[2], std::fma(m13, v[3], m10 * v[0]))),
            std::fma(m21, v[1], std::fma(m22, v[2], std::fma(m23, v[3], m20 * v[0]))),
            std::fma(m31, v[1], std::fma(m32, v[2], std::fma(m33, v[3], m30 * v[0]))));
    } else {
        return Vector<T, 4>(
            m00 * v[0] + m01 * v[1] + m02 * v[2] + m03 * v[3],
            m10 * v[0] + m11 * v[1] + m12 * v[2] + m13 * v[3],
            m20 * v[0] + m21 * v[1] + m22 * v[2] + m23 * v[3],
            m30 * v[0] + m31 * v[1] + m32 * v[2] + m33 * v[3]);
    }
}

template <typename T>
Vector<T, 4>
Matrix<T, 4, 4>::solved(const Vector<T, 4>& b) const noexcept {
    Matrix inv = inversed();
    return inv.mul(b);
}

template <typename T>
bool
Matrix<T, 4, 4>::solve(const Vector<T, 4>& b, Vector<T, 4>& x, T eps) const noexcept {
    Matrix inv;
    if (!try_inverse(inv, eps)) return false;
    x = inv.mul(b);
    return true;
}

template <typename T>
Matrix<T, 4, 4>
identity4x4() noexcept {
    return Matrix<T, 4, 4>(T(1));
}

template <typename T>
Matrix<T, 4, 4>
zero4x4() noexcept {
    return Matrix<T, 4, 4>(
        T(0),
        T(0),
        T(0),
        T(0),
        T(0),
        T(0),
        T(0),
        T(0),
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
Matrix<T, 4, 4>
transpose(const Matrix<T, 4, 4>& m) noexcept {
    return m.transposed();
}

template <typename T>
T
determinant(const Matrix<T, 4, 4>& m) noexcept {
    return m.determinant();
}

template <typename T>
Matrix<T, 4, 4>
inverse(const Matrix<T, 4, 4>& m) noexcept {
    return m.inversed();
}

template <typename T>
Matrix<T, 4, 4>
operator+(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    Matrix<T, 4, 4> out;
    for (int i = 0; i < 16; ++i) (&out.m00)[i] = (&a.m00)[i] + (&b.m00)[i];
    return out;
}

template <typename T>
Matrix<T, 4, 4>
operator-(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    Matrix<T, 4, 4> out;
    for (int i = 0; i < 16; ++i) (&out.m00)[i] = (&a.m00)[i] - (&b.m00)[i];
    return out;
}

template <typename T>
Matrix<T, 4, 4>
operator*(const Matrix<T, 4, 4>& a, T s) {
    Matrix<T, 4, 4> out;
    for (int i = 0; i < 16; ++i) (&out.m00)[i] = (&a.m00)[i] * s;
    return out;
}

template <typename T>
Matrix<T, 4, 4>
operator*(T s, const Matrix<T, 4, 4>& a) {
    return a * s;
}

template <typename T>
Matrix<T, 4, 4>
operator/(const Matrix<T, 4, 4>& a, T s) {
    if constexpr (std::is_floating_point_v<T>) {
        const T inv = T(1) / s;
        return a * inv;
    } else {
        Matrix<T, 4, 4> out;
        for (int i = 0; i < 16; ++i) out._data[i] = a._data[i] / s;
        return out;
    }
}

template <typename T>
Matrix<T, 4, 4>
operator*(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b) {
    return a.mul(b);
}

template <typename T>
Vector<T, 4>
operator*(const Matrix<T, 4, 4>& a, const Vector<T, 4>& v) {
    return a.mul(v);
}

template <typename T>
bool
solve(const Matrix<T, 4, 4>& A, const Vector<T, 4>& b, Vector<T, 4>& x, T eps) noexcept {
    return A.solve(b, x, eps);
}

template <typename T>
Vector<T, 4>
solve(const Matrix<T, 4, 4>& A, const Vector<T, 4>& b) noexcept {
    return A.solved(b);
}

}