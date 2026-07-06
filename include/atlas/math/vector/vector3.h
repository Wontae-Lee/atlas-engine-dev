#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/constants.h>
#include <atlas/math/vector/bool3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <tuple>

namespace atlas {

class Vector3 {
public:
    float x;

    float y;

    float z;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector3() noexcept
        : x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    constexpr Vector3(const Vector3&) noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector3(const float s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector3(const float x_, const float y_, const float z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Vector3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    ~Vector3() noexcept = default;

    Vector3&
    operator=(const Vector3&) noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &x;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &x;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t i) const noexcept {
        return (&x)[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t i) noexcept {
        return (&x)[i];
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(const float s) noexcept {
        x = y = z = s;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_values(const float x_, const float y_, const float z_) noexcept {
        x = x_;
        y = y_;
        z = z_;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0.0f;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float v) noexcept {
        x += v;
        y += v;
        z += v;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float v) noexcept {
        x -= v;
        y -= v;
        z -= v;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float v) noexcept {
        x *= v;
        y *= v;
        z *= v;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float v) noexcept {
        const float inv = 1.0f / v;
        x *= inv;
        y *= inv;
        z *= inv;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Vector3& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Vector3& v) noexcept {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const Vector3& v) noexcept {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const Vector3& v) noexcept {
        x /= v.x;
        y /= v.y;
        z /= v.z;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    min() const noexcept {
        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator+=(const float v) noexcept {
        add(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator-=(const float v) noexcept {
        sub(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator*=(const float v) noexcept {
        mul(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator/=(const float v) noexcept {
        div(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator+=(const Vector3& v) noexcept {
        add(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator-=(const Vector3& v) noexcept {
        sub(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator*=(const Vector3& v) noexcept {
        mul(v);
        return *this;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3&
    operator/=(const Vector3& v) noexcept {
        div(v);
        return *this;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Vector3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Vector3& other) const noexcept {
        return !(*this == other);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Vector3& v) const noexcept {
        return std::fma(z, v.z, std::fma(y, v.y, x * v.x));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    cross(const Vector3& v) const noexcept {
        return Vector3(y * v.z - z * v.y,
                       z * v.x - x * v.z,
                       x * v.y - y * v.x);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, x * x));
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    major_axis() const noexcept {
        if (x >= y && x >= z) return 0;
        if (y >= x && y >= z) return 1;
        return 2;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    minor_axis() const noexcept {
        if (x <= y && x <= z) return 0;
        if (y <= x && y <= z) return 1;
        return 2;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return;
        const float inv = 1.0f / std::sqrt(ls);
        x *= inv;
        y *= inv;
        z *= inv;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    normalized() const noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return *this;
        const float inv = 1.0f / std::sqrt(ls);
        return Vector3(x * inv, y * inv, z * inv);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    reflected(const Vector3& n) const noexcept {
        const float d = dot(n);
        return Vector3(x - 2.0f * d * n.x,
                       y - 2.0f * d * n.y,
                       z - 2.0f * d * n.z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    projected(const Vector3& n) const noexcept {
        const float d = dot(n);
        return Vector3(x - d * n.x,
                       y - d * n.y,
                       z - d * n.z);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector3, Vector3>
    tangential() const noexcept {
        const float ax = std::abs(x);
        const float ay = std::abs(y);
        Vector3 t1;
        if (ax > ay) {
            const float d2  = x * x + z * z;
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Vector3(-z * inv, 0.0f, x * inv);
        } else {
            const float d2  = y * y + z * z;
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Vector3(0.0f, z * inv, -y * inv);
        }
        const Vector3 t2 = cross(t1).normalized();
        return { t1, t2 };
    }
};

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
cross(const Vector3& a, const Vector3& b) noexcept {
    return Vector3(a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
dot(const Vector3& a, const Vector3& b) noexcept {
    return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
reflected(const Vector3& v, const Vector3& normal) noexcept {
    const float d = dot(v, normal);
    return Vector3(v.x - 2.0f * d * normal.x,
                   v.y - 2.0f * d * normal.y,
                   v.z - 2.0f * d * normal.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
projected(const Vector3& v, const Vector3& normal) noexcept {
    const float d = dot(v, normal);
    return Vector3(v.x - d * normal.x,
                   v.y - d * normal.y,
                   v.z - d * normal.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector3, Vector3>
tangential(const Vector3& normal) noexcept {
    return normal.tangential();
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator+(const Vector3& a) noexcept {
    return a;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator+(const float a, const Vector3& b) noexcept {
    return Vector3(a + b.x, a + b.y, a + b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator+(const Vector3& a, const float b) noexcept {
    return Vector3(a.x + b, a.y + b, a.z + b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator+(const Vector3& a, const Vector3& b) noexcept {
    return Vector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator-(const Vector3& a) noexcept {
    return Vector3(-a.x, -a.y, -a.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator-(const Vector3& a, const float b) noexcept {
    return Vector3(a.x - b, a.y - b, a.z - b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator-(const float a, const Vector3& b) noexcept {
    return Vector3(a - b.x, a - b.y, a - b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator-(const Vector3& a, const Vector3& b) noexcept {
    return Vector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator*(const Vector3& a, const float b) noexcept {
    return Vector3(a.x * b, a.y * b, a.z * b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator*(const float a, const Vector3& b) noexcept {
    return Vector3(a * b.x, a * b.y, a * b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator*(const Vector3& a, const Vector3& b) noexcept {
    return Vector3(a.x * b.x, a.y * b.y, a.z * b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator/(const Vector3& a, const float b) noexcept {
    const float inv = 1.0f / b;
    return Vector3(a.x * inv, a.y * inv, a.z * inv);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator/(const float a, const Vector3& b) noexcept {
    return Vector3(a / b.x, a / b.y, a / b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
operator/(const Vector3& a, const Vector3& b) noexcept {
    return Vector3(a.x / b.x, a.y / b.y, a.z / b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Vector3& a, const Vector3& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Vector3& a, const Vector3& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Vector3& a, const Vector3& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Vector3& a, const Vector3& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
min(const Vector3& a, const Vector3& b) noexcept {
    return Vector3((a.x < b.x) ? a.x : b.x,
                   (a.y < b.y) ? a.y : b.y,
                   (a.z < b.z) ? a.z : b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
max(const Vector3& a, const Vector3& b) noexcept {
    return Vector3((a.x > b.x) ? a.x : b.x,
                   (a.y > b.y) ? a.y : b.y,
                   (a.z > b.z) ? a.z : b.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
cmin(const Vector3& a, const Vector3& b) noexcept {
    return min(a, b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
cmax(const Vector3& a, const Vector3& b) noexcept {
    return max(a, b);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
clamp(const Vector3& v, const Vector3& low, const Vector3& high) noexcept {
    return min(max(v, low), high);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
ceil(const Vector3& a) noexcept {
    return Vector3(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
floor(const Vector3& a) noexcept {
    return Vector3(std::floor(a.x), std::floor(a.y), std::floor(a.z));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
abs(const Vector3& v) noexcept {
    return Vector3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Vector3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_dot(const Vector3& a, const Vector3& b) noexcept {
    return std::fma(a.y, b.y, a.x * b.x);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length_squared(const Vector3& v) noexcept {
    return xy_dot(v, v);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length(const Vector3& v) noexcept {
    return std::sqrt(xy_length_squared(v));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
normalized_or(const Vector3& v,
              const Vector3& fallback,
              const float min_length_squared = 0.0f) noexcept {
    const float len2 = v.length_squared();
    if (!(len2 > min_length_squared)) {
        return fallback;
    }

    return v * (1.0f / std::sqrt(len2));
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
xy_normalized_or(const Vector3& v,
                 const Vector3& fallback,
                 const float min_length_squared = 0.0f) noexcept {
    const float len2 = xy_length_squared(v);
    if (!(len2 > min_length_squared)) {
        return fallback;
    }

    const float inv = 1.0f / std::sqrt(len2);
    return Vector3(v.x * inv, v.y * inv, 0.0f);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
reject(const Vector3& v, const Vector3& normal) noexcept {
    return v - normal * v.dot(normal);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Vector3& normal,
                  Vector3& unit_normal,
                  Vector3& tangent,
                  Vector3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {
    const float normal_length_squared = normal.length_squared();
    if (!(normal_length_squared > min_length_squared)) {
        return false;
    }

    unit_normal = normal * (1.0f / std::sqrt(normal_length_squared));

    const Vector3 axis = std::abs(unit_normal.z) < 0.9f
        ? Vector3(0.0f, 0.0f, 1.0f)
        : Vector3(0.0f, 1.0f, 0.0f);

    tangent                            = axis.cross(unit_normal);
    const float tangent_length_squared = tangent.length_squared();
    if (!(tangent_length_squared > min_length_squared)) {
        return false;
    }

    tangent *= 1.0f / std::sqrt(tangent_length_squared);

    bitangent                            = unit_normal.cross(tangent);
    const float bitangent_length_squared = bitangent.length_squared();
    if (!(bitangent_length_squared > min_length_squared)) {
        return false;
    }

    bitangent *= 1.0f / std::sqrt(bitangent_length_squared);
    return true;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Vector3& normal,
                  Vector3& tangent,
                  Vector3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {
    Vector3 unit_normal;
    return orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
orthogonal_unit_vector(const Vector3& normal,
                       const Vector3& seed,
                       const float min_length_squared = 0.0f) noexcept {
    const Vector3 fallback(1.0f, 0.0f, 0.0f);
    Vector3 tangent = normal.cross(seed);
    if (tangent.length_squared() > min_length_squared) {
        return normalized_or(tangent, fallback, min_length_squared);
    }

    Vector3 unit_normal;
    Vector3 bitangent;
    if (orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared)) {
        return tangent;
    }

    return fallback;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
spherical_direction(const Vector3& unit_axis, const float cos_theta, const float phi) noexcept {
    const auto tangents      = unit_axis.tangential();
    const Vector3 tangent    = std::get<0>(tangents);
    const Vector3 bitangent  = std::get<1>(tangents);

    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    const float cos_phi   = std::cos(phi);
    const float sin_phi   = std::sin(phi);

    return unit_axis * cos_theta + (tangent * cos_phi + bitangent * sin_phi) * sin_theta;
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
spherical_direction(const float cos_theta, const float phi) noexcept {
    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    return Vector3(sin_theta * std::cos(phi),
                   sin_theta * std::sin(phi),
                   cos_theta);
}

}
