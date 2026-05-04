#pragma once
#include <cmath>
#include <limits>
namespace atlas::math {
template <typename T>
constexpr Quaternion<T>::Quaternion() noexcept
    : w(T(1))
    , x(T(0))
    , y(T(0))
    , z(T(0)) {
}

template <typename T>
constexpr Quaternion<T>::Quaternion(T w_, T x_, T y_, T z_) noexcept
    : w(w_)
    , x(x_)
    , y(y_)
    , z(z_) {
}

template <typename T>
Quaternion<T>::Quaternion(std::initializer_list<T> list) noexcept {
    auto it = list.begin();
    w       = (it != list.end()) ? *it++ : T(1);
    x       = (it != list.end()) ? *it++ : T(0);
    y       = (it != list.end()) ? *it++ : T(0);
    z       = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
Quaternion<T>::Quaternion(const Vector3<T>& axis, T radians) noexcept {
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    w            = std::cos(half);
    x            = axis.x * s;
    y            = axis.y * s;
    z            = axis.z * s;
}

template <typename T>
Quaternion<T>::Quaternion(T rx, T ry, T rz) noexcept {
    *this = from_euler_xyz(rx, ry, rz);
}

template <typename T>
Quaternion<T>::Quaternion(const Matrix<T, 3, 3>& m) noexcept {
    *this = from_matrix3x3(m);
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_axis_angle(const Vector3<T>& axis, T radians) noexcept {
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    return Quaternion(
        std::cos(half),
        axis.x * s,
        axis.y * s,
        axis.z * s);
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_euler_xyz(T rx, T ry, T rz) noexcept {
    const T hx = rx * T(0.5);
    const T hy = ry * T(0.5);
    const T hz = rz * T(0.5);
    const T cx = std::cos(hx), sx = std::sin(hx);
    const T cy = std::cos(hy), sy = std::sin(hy);
    const T cz = std::cos(hz), sz = std::sin(hz);
    Quaternion qx(cx, sx, 0, 0);
    Quaternion qy(cy, 0, sy, 0);
    Quaternion qz(cz, 0, 0, sz);
    return qz * qy * qx;
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_matrix3x3(const Matrix<T, 3, 3>& m) noexcept {
    const T tr = m.m00 + m.m11 + m.m22;
    if (tr > T(0)) {
        const T s = std::sqrt(tr + T(1)) * T(2);
        return Quaternion(
            s * T(0.25),
            (m.m21 - m.m12) / s,
            (m.m02 - m.m20) / s,
            (m.m10 - m.m01) / s);
    } else if (m.m00 > m.m11 && m.m00 > m.m22) {
        const T s = std::sqrt(T(1) + m.m00 - m.m11 - m.m22) * T(2);
        return Quaternion(
            (m.m21 - m.m12) / s,
            s * T(0.25),
            (m.m01 + m.m10) / s,
            (m.m02 + m.m20) / s);
    } else if (m.m11 > m.m22) {
        const T s = std::sqrt(T(1) + m.m11 - m.m00 - m.m22) * T(2);
        return Quaternion(
            (m.m02 - m.m20) / s,
            (m.m01 + m.m10) / s,
            s * T(0.25),
            (m.m12 + m.m21) / s);
    } else {
        const T s = std::sqrt(T(1) + m.m22 - m.m00 - m.m11) * T(2);
        return Quaternion(
            (m.m10 - m.m01) / s,
            (m.m02 + m.m20) / s,
            (m.m12 + m.m21) / s,
            s * T(0.25));
    }
}

template <typename T>
const T*
Quaternion<T>::data() const noexcept {
    return &w;
}

template <typename T>
T*
Quaternion<T>::data() noexcept {
    return &w;
}

template <typename T>
T
Quaternion<T>::dot(const Quaternion& q) const noexcept {
    return w * q.w + x * q.x + y * q.y + z * q.z;
}

template <typename T>
T
Quaternion<T>::length_squared() const noexcept {
    return w * w + x * x + y * y + z * z;
}

template <typename T>
T
Quaternion<T>::length() const noexcept {
    return std::sqrt(length_squared());
}

template <typename T>
void
Quaternion<T>::normalize() noexcept {
    const T len = length();
    if (len > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / len;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
    }
}

template <typename T>
Quaternion<T>
Quaternion<T>::normalized() const noexcept {
    Quaternion q = *this;
    q.normalize();
    return q;
}

template <typename T>
Quaternion<T>
Quaternion<T>::conjugate() const noexcept {
    return Quaternion(w, -x, -y, -z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::inverse() const noexcept {
    const T ls = length_squared();
    if (ls > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / ls;
        return Quaternion(w * inv, -x * inv, -y * inv, -z * inv);
    }
    return Quaternion();
}

template <typename T>
bool
Quaternion<T>::is_identity(T eps) const noexcept {
    return std::abs(w - T(1)) < eps && std::abs(x) < eps && std::abs(y) < eps && std::abs(z) < eps;
}

template <typename T>
Vector3<T>
Quaternion<T>::rotate(const Vector3<T>& v) const noexcept {
    Quaternion<T> qv(T(0), v.x, v.y, v.z);
    Quaternion<T> inv = conjugate();
    Quaternion<T> res = (*this) * qv * inv;
    return Vector3<T>(res.x, res.y, res.z);
}

template <typename T>
Matrix<T, 3, 3>
Quaternion<T>::to_matrix3x3() const noexcept {
    const T xx = x * x, yy = y * y, zz = z * z;
    const T xy = x * y, xz = x * z, yz = y * z;
    const T wx = w * x, wy = w * y, wz = w * z;
    Matrix<T, 3, 3> m;
    m.set(
        T(1) - T(2) * (yy + zz),
        T(2) * (xy - wz),
        T(2) * (xz + wy),
        T(2) * (xy + wz),
        T(1) - T(2) * (xx + zz),
        T(2) * (yz - wx),
        T(2) * (xz - wy),
        T(2) * (yz + wx),
        T(1) - T(2) * (xx + yy));
    return m;
}

template <typename T>
Matrix<T, 4, 4>
Quaternion<T>::to_matrix4x4() const noexcept {
    Matrix<T, 4, 4> m;
    m.set_identity();
    Matrix<T, 3, 3> r = to_matrix3x3();
    m.m00             = r.m00;
    m.m01             = r.m01;
    m.m02             = r.m02;
    m.m10             = r.m10;
    m.m11             = r.m11;
    m.m12             = r.m12;
    m.m20             = r.m20;
    m.m21             = r.m21;
    m.m22             = r.m22;
    return m;
}

template <typename T>
Quaternion<T>
Quaternion<T>::lerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    return a * (T(1) - t) + b * t;
}

template <typename T>
Quaternion<T>
Quaternion<T>::nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    Quaternion r = lerp(a, b, t);
    r.normalize();
    return r;
}

template <typename T>
Quaternion<T>
Quaternion<T>::slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    T cos_theta   = a.dot(b);
    Quaternion bb = b;
    if (cos_theta < T(0)) {
        bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
        cos_theta = -cos_theta;
    }
    if (cos_theta > T(1) - T(1e-6)) {
        return nlerp(a, bb, t);
    }
    const T theta     = std::acos(cos_theta);
    const T sin_theta = std::sin(theta);
    const T w1        = std::sin((T(1) - t) * theta) / sin_theta;
    const T w2        = std::sin(t * theta) / sin_theta;
    return a * w1 + bb * w2;
}

template <typename T>
template <typename To>
Quaternion<To>
Quaternion<T>::cast_to() const noexcept {
    return Quaternion<To>(
        static_cast<To>(w),
        static_cast<To>(x),
        static_cast<To>(y),
        static_cast<To>(z));
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator+(const Quaternion& q) const noexcept {
    return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator-(const Quaternion& q) const noexcept {
    return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator*(const Quaternion& q) const noexcept {
    return Quaternion(
        w * q.w - x * q.x - y * q.y - z * q.z,
        w * q.x + x * q.w + y * q.z - z * q.y,
        w * q.y - x * q.z + y * q.w + z * q.x,
        w * q.z + x * q.y - y * q.x + z * q.w);
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator*=(const Quaternion& q) noexcept {
    *this = (*this) * q;
    return *this;
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator*(T s) const noexcept {
    return Quaternion(w * s, x * s, y * s, z * s);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator/(T s) const noexcept {
    const T inv = T(1) / s;
    return Quaternion(w * inv, x * inv, y * inv, z * inv);
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator+=(const Quaternion& q) noexcept {
    w += q.w;
    x += q.x;
    y += q.y;
    z += q.z;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator-=(const Quaternion& q) noexcept {
    w -= q.w;
    x -= q.x;
    y -= q.y;
    z -= q.z;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator*=(T s) noexcept {
    w *= s;
    x *= s;
    y *= s;
    z *= s;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator/=(T s) noexcept {
    const T inv = T(1) / s;
    w *= inv;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
}

template <typename T>
bool
Quaternion<T>::operator==(const Quaternion& q) const noexcept {
    return std::abs(w - q.w) < eps && std::abs(x - q.x) < eps && std::abs(y - q.y) < eps && std::abs(z - q.z) < eps;
}

template <typename T>
bool
Quaternion<T>::operator!=(const Quaternion& q) const noexcept {
    return !(*this == q);
}

}