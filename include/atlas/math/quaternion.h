#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/constants.h>
#include <atlas/math/matrix/float3x3.h>
#include <atlas/math/vector/float3.h>

#include <cmath>
#include <initializer_list>
#include <limits>

namespace atlas {

class Quaternion {
public:
    float w;

    float x;

    float y;

    float z;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion() noexcept
        : w(1.0f)
        , x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion(const float w_, const float x_, const float y_, const float z_) noexcept
        : w(w_)
        , x(x_)
        , y(y_)
        , z(z_) { }

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Quaternion(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        w               = (it != list.end()) ? *it++ : 1.0f;
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const Float3& axis, const float radians) noexcept {
        *this = from_axis_angle(axis, radians);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const float rx, const float ry, const float rz) noexcept {
        *this = from_euler_xyz(rx, ry, rz);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(const Float3x3& m) noexcept {
        *this = from_matrix3x3(m);
    }

    Quaternion(const Quaternion&) noexcept = default;
    ~Quaternion() noexcept                 = default;
    Quaternion&
    operator=(const Quaternion&) noexcept = default;
    Quaternion&
    operator=(Quaternion&&) noexcept = default;

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_axis_angle(const Float3& axis, const float radians) noexcept {
        const float half = radians * 0.5f;
        const float s    = std::sin(half);
        return Quaternion(std::cos(half), axis.x * s, axis.y * s, axis.z * s);
    }

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_euler_xyz(const float rx, const float ry, const float rz) noexcept {
        const float hx = rx * 0.5f;
        const float hy = ry * 0.5f;
        const float hz = rz * 0.5f;
        const float cx = std::cos(hx), sx = std::sin(hx);
        const float cy = std::cos(hy), sy = std::sin(hy);
        const float cz = std::cos(hz), sz = std::sin(hz);
        const Quaternion qx(cx, sx, 0.0f, 0.0f);
        const Quaternion qy(cy, 0.0f, sy, 0.0f);
        const Quaternion qz(cz, 0.0f, 0.0f, sz);
        return qz * qy * qx;
    }

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_matrix3x3(const Float3x3& m) noexcept {
        const float tr = m.m00 + m.m11 + m.m22;
        if (tr > 0.0f) {
            const float s = std::sqrt(tr + 1.0f) * 2.0f;
            return Quaternion(s * 0.25f,
                              (m.m21 - m.m12) / s,
                              (m.m02 - m.m20) / s,
                              (m.m10 - m.m01) / s);
        } else if (m.m00 > m.m11 && m.m00 > m.m22) {
            const float s = std::sqrt(1.0f + m.m00 - m.m11 - m.m22) * 2.0f;
            return Quaternion((m.m21 - m.m12) / s,
                              s * 0.25f,
                              (m.m01 + m.m10) / s,
                              (m.m02 + m.m20) / s);
        } else if (m.m11 > m.m22) {
            const float s = std::sqrt(1.0f + m.m11 - m.m00 - m.m22) * 2.0f;
            return Quaternion((m.m02 - m.m20) / s,
                              (m.m01 + m.m10) / s,
                              s * 0.25f,
                              (m.m12 + m.m21) / s);
        } else {
            const float s = std::sqrt(1.0f + m.m22 - m.m00 - m.m11) * 2.0f;
            return Quaternion((m.m10 - m.m01) / s,
                              (m.m02 + m.m20) / s,
                              (m.m12 + m.m21) / s,
                              s * 0.25f);
        }
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &w;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &w;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Quaternion& q) const noexcept {
        return std::fma(z, q.z, std::fma(y, q.y, std::fma(x, q.x, w * q.w)));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, std::fma(x, x, w * w)));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept {
        const float len = length();
        if (len > std::numeric_limits<float>::epsilon()) {
            const float inv = 1.0f / len;
            w *= inv;
            x *= inv;
            y *= inv;
            z *= inv;
        }
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    normalized() const noexcept {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    conjugate() const noexcept {
        return Quaternion(w, -x, -y, -z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    inverse() const noexcept {
        const float ls = length_squared();
        if (ls > std::numeric_limits<float>::epsilon()) {
            const float inv = 1.0f / ls;
            return Quaternion(w * inv, -x * inv, -y * inv, -z * inv);
        }
        return Quaternion();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_identity(const float tolerance = atlas::eps) const noexcept {
        return std::abs(w - 1.0f) < tolerance
            && std::abs(x) < tolerance
            && std::abs(y) < tolerance
            && std::abs(z) < tolerance;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    rotate(const Float3& v) const noexcept {
        const float uxv_x  = y * v.z - z * v.y;
        const float uxv_y  = z * v.x - x * v.z;
        const float uxv_z  = x * v.y - y * v.x;
        const float uv     = std::fma(z, v.z, std::fma(y, v.y, x * v.x));
        const float uu     = std::fma(z, z, std::fma(y, y, x * x));
        const float scale  = std::fma(w, w, -uu);
        const float two_uv = 2.0f * uv;
        const float two_w  = 2.0f * w;
        return Float3(std::fma(two_w, uxv_x, std::fma(two_uv, x, scale * v.x)),
                      std::fma(two_w, uxv_y, std::fma(two_uv, y, scale * v.y)),
                      std::fma(two_w, uxv_z, std::fma(two_uv, z, scale * v.z)));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    to_matrix3x3() const noexcept {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;
        return Float3x3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy), 2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx), 2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));
    }

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    lerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        return a * (1.0f - t) + b * t;
    }

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    nlerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        Quaternion r = lerp(a, b, t);
        r.normalize();
        return r;
    }

    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    slerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        float cos_theta = a.dot(b);
        Quaternion bb   = b;
        if (cos_theta < 0.0f) {
            bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
            cos_theta = -cos_theta;
        }
        if (cos_theta > 1.0f - 1e-6f) {
            return nlerp(a, bb, t);
        }
        const float theta     = std::acos(cos_theta);
        const float sin_theta = std::sin(theta);
        const float w1        = std::sin((1.0f - t) * theta) / sin_theta;
        const float w2        = std::sin(t * theta) / sin_theta;
        return a * w1 + bb * w2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator+(const Quaternion& q) const noexcept {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator-(const Quaternion& q) const noexcept {
        return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const Quaternion& q) const noexcept {
        return Quaternion(w * q.w - x * q.x - y * q.y - z * q.z,
                          w * q.x + x * q.w + y * q.z - z * q.y,
                          w * q.y - x * q.z + y * q.w + z * q.x,
                          w * q.z + x * q.y - y * q.x + z * q.w);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const float s) const noexcept {
        return Quaternion(w * s, x * s, y * s, z * s);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator/(const float s) const noexcept {
        const float inv = 1.0f / s;
        return Quaternion(w * inv, x * inv, y * inv, z * inv);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator+=(const Quaternion& q) noexcept {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator-=(const Quaternion& q) noexcept {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const Quaternion& q) noexcept {
        *this = (*this) * q;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const float s) noexcept {
        w *= s;
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator/=(const float s) noexcept {
        const float inv = 1.0f / s;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Quaternion& q) const noexcept {
        return std::abs(w - q.w) < eps
            && std::abs(x - q.x) < eps
            && std::abs(y - q.y) < eps
            && std::abs(z - q.z) < eps;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Quaternion& q) const noexcept {
        return !(*this == q);
    }
};

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Quaternion& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

}
