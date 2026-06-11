#pragma once
#include <algorithm>
namespace atlas::math {
template <typename T>
constexpr Vector<T, 3>::Vector() noexcept
    : x(T(0))
    , y(T(0))
    , z(T(0)) {
}

template <typename T>
constexpr Vector<T, 3>::Vector(T s) noexcept
    : x(s)
    , y(s)
    , z(s) {
}

template <typename T>
constexpr Vector<T, 3>::Vector(T x_, T y_, T z_) noexcept
    : x(x_)
    , y(y_)
    , z(z_) {
}

template <typename T>
Vector<T, 3>::Vector(std::initializer_list<T> list) noexcept {
    const T* it = list.begin();
    x           = (it != list.end()) ? *it++ : T(0);
    y           = (it != list.end()) ? *it++ : T(0);
    z           = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Vector<T, 3>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    x                   = static_cast<T>(e[0]);
    y                   = static_cast<T>(e[1]);
    z                   = static_cast<T>(e[2]);
}

template <typename T>
std::size_t
Vector<T, 3>::size() noexcept {
    return 3;
}

template <typename T>
const T*
Vector<T, 3>::data() const noexcept {
    return &x;
}

template <typename T>
T*
Vector<T, 3>::data() noexcept {
    return &x;
}

template <typename T>
const T&
Vector<T, 3>::operator[](std::size_t i) const noexcept {
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 3>::operator[](std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
const T&
Vector<T, 3>::at(std::size_t i) const noexcept {
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 3>::at(std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
void
Vector<T, 3>::set(T s) noexcept {
    x = y = z = s;
}

template <typename T>
template <typename... Args, typename>
void
Vector<T, 3>::set_values(Args... args) noexcept {
    T tmp[3] = { static_cast<T>(args)... };
    x        = tmp[0];
    y        = tmp[1];
    z        = tmp[2];
}

template <typename T>
void
Vector<T, 3>::set_zero() noexcept {
    x = y = z = T(0);
}

template <typename T>
void
Vector<T, 3>::add(T v) noexcept {
    x += v;
    y += v;
    z += v;
}

template <typename T>
void
Vector<T, 3>::sub(T v) noexcept {
    x -= v;
    y -= v;
    z -= v;
}

template <typename T>
void
Vector<T, 3>::mul(T v) noexcept {
    x *= v;
    y *= v;
    z *= v;
}

template <typename T>
void
Vector<T, 3>::div(T v) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        const T inv = T(1) / v;
        x *= inv;
        y *= inv;
        z *= inv;
    } else {
        x /= v;
        y /= v;
        z /= v;
    }
}

template <typename T>
void
Vector<T, 3>::add(const Vector& v) noexcept {
    x += v.x;
    y += v.y;
    z += v.z;
}

template <typename T>
void
Vector<T, 3>::sub(const Vector& v) noexcept {
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

template <typename T>
void
Vector<T, 3>::mul(const Vector& v) noexcept {
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

template <typename T>
void
Vector<T, 3>::div(const Vector& v) noexcept {
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

template <typename T>
T
Vector<T, 3>::min() const noexcept {
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

template <typename T>
T
Vector<T, 3>::max() const noexcept {
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator+=(T v) noexcept {
    add(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator-=(T v) noexcept {
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator*=(T v) noexcept {
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator/=(T v) noexcept {
    div(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator+=(const Vector& v) noexcept {
    add(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator-=(const Vector& v) noexcept {
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator*=(const Vector& v) noexcept {
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator/=(const Vector& v) noexcept {
    div(v);
    return *this;
}

template <typename T>
bool
Vector<T, 3>::operator==(const Vector& other) const noexcept {
    return x == other.x && y == other.y && z == other.z;
}

template <typename T>
bool
Vector<T, 3>::operator!=(const Vector& other) const noexcept {
    return !(*this == other);
}

template <typename T>
T
Vector<T, 3>::dot(const Vector& v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(z, v.z, std::fma(y, v.y, x * v.x));
    } else {
        return x * v.x + y * v.y + z * v.z;
    }
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::cross(const Vector& v) const noexcept {
    return Vector3<T>(y * v.z - z * v.y,
                      z * v.x - x * v.z,
                      x * v.y - y * v.x);
}

template <typename T>
T
Vector<T, 3>::length_squared() const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(z, z, std::fma(y, y, x * x));
    } else {
        return x * x + y * y + z * z;
    }
}

template <typename T>
T
Vector<T, 3>::length() const noexcept {
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared()));
}

template <typename T>
std::size_t
Vector<T, 3>::major_axis() const noexcept {
    if (x >= y && x >= z) return 0;
    if (y >= x && y >= z) return 1;
    return 2;
}

template <typename T>
std::size_t
Vector<T, 3>::minor_axis() const noexcept {
    if (x <= y && x <= z) return 0;
    if (y <= x && y <= z) return 1;
    return 2;
}

template <typename T>
void
Vector<T, 3>::normalize() noexcept {
    const T ls = length_squared();
    if (ls == T(0)) return;
    using std::sqrt;
    const T inv = T(1) / static_cast<T>(sqrt(ls));
    x *= inv;
    y *= inv;
    z *= inv;
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::normalized() const noexcept {
    const T ls = length_squared();
    if (ls == T(0)) return *this;
    using std::sqrt;
    const T inv = T(1) / static_cast<T>(sqrt(ls));
    return Vector3<T>(x * inv, y * inv, z * inv);
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::reflected(const Vector& n) const noexcept {
    const T d = dot(n);
    return Vector3<T>(x - T(2) * d * n.x,
                      y - T(2) * d * n.y,
                      z - T(2) * d * n.z);
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::projected(const Vector& n) const noexcept {
    const T d = dot(n);
    return Vector3<T>(x - d * n.x,
                      y - d * n.y,
                      z - d * n.z);
}

template <typename T>
std::tuple<Vector<T, 3>, Vector<T, 3>>
Vector<T, 3>::tangential() const noexcept {
    const T ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
    Vector t1;
    if (ax > ay) {
        const T d2  = x * x + z * z;
        using std::sqrt;
        const T inv = T(1) / static_cast<T>(sqrt(d2 + (d2 == T(0) ? T(1) : T(0))));
        t1          = Vector(-z * inv, T(0), x * inv);
    } else {
        const T d2  = y * y + z * z;
        using std::sqrt;
        const T inv = T(1) / static_cast<T>(sqrt(d2 + (d2 == T(0) ? T(1) : T(0))));
        t1          = Vector(T(0), z * inv, -y * inv);
    }
    const Vector t2 = cross(t1).normalized();
    return { t1, t2 };
}

template <typename T>
template <typename To>
Vector<To, 3>
Vector<T, 3>::cast_to() const noexcept {
    return Vector<To, 3>(static_cast<To>(x),
                         static_cast<To>(y),
                         static_cast<To>(z));
}

template <typename T>
Vector<T, 3>
cross(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept {
    return Vector<T, 3>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template <typename T>
T
dot(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x));
    } else {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
}

template <typename T>
Vector<T, 3>
reflected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept {
    const T d = dot(v, normal);
    return Vector<T, 3>(
        v.x - T(2) * d * normal.x,
        v.y - T(2) * d * normal.y,
        v.z - T(2) * d * normal.z);
}

template <typename T>
Vector<T, 3>
projected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept {
    const T d = dot(v, normal);
    return Vector<T, 3>(
        v.x - d * normal.x,
        v.y - d * normal.y,
        v.z - d * normal.z);
}

template <typename T>
std::tuple<Vector<T, 3>, Vector<T, 3>>
tangential(const Vector<T, 3>& normal) noexcept {
    const T nx = normal.x, ny = normal.y, nz = normal.z;
    const T ax = std::abs(nx), ay = std::abs(ny), az = std::abs(nz);
    Vector<T, 3> t1;
    if (ax > ay) {
        const T d2  = nx * nx + nz * nz;
        using std::sqrt;
        const T inv = T(1) / static_cast<T>(sqrt(d2 + (d2 == T(0) ? T(1) : T(0))));
        t1          = Vector<T, 3>(-nz * inv, T(0), nx * inv);
    } else {
        const T d2  = ny * ny + nz * nz;
        using std::sqrt;
        const T inv = T(1) / static_cast<T>(sqrt(d2 + (d2 == T(0) ? T(1) : T(0))));
        t1          = Vector<T, 3>(T(0), nz * inv, -ny * inv);
    }
    const Vector<T, 3> t2 = cross(normal, t1).normalized();
    return { t1, t2 };
}

template <typename T>
Vector3<T>
operator+(const Vector3<T>& a) {
    return a;
}

template <typename T>
Vector3<T>
operator+(T a, const Vector3<T>& b) {
    return Vector3<T>(a + b.x, a + b.y, a + b.z);
}

template <typename T>
Vector3<T>
operator+(const Vector3<T>& a, T b) {
    return Vector3<T>(a.x + b, a.y + b, a.z + b);
}

template <typename T>
Vector3<T>
operator+(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename T>
Vector3<T>
operator-(const Vector3<T>& a) {
    return Vector3<T>(-a.x, -a.y, -a.z);
}

template <typename T>
Vector3<T>
operator-(const Vector3<T>& a, T b) {
    return Vector3<T>(a.x - b, a.y - b, a.z - b);
}

template <typename T>
Vector3<T>
operator-(T a, const Vector3<T>& b) {
    return Vector3<T>(a - b.x, a - b.y, a - b.z);
}

template <typename T>
Vector3<T>
operator-(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template <typename T>
Vector3<T>
operator*(const Vector3<T>& a, T b) {
    return Vector3<T>(a.x * b, a.y * b, a.z * b);
}

template <typename T>
Vector3<T>
operator*(T a, const Vector3<T>& b) {
    return Vector3<T>(a * b.x, a * b.y, a * b.z);
}

template <typename T>
Vector3<T>
operator*(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}

template <typename T>
Vector3<T>
operator/(const Vector3<T>& a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        const T inv = T(1) / b;
        return Vector3<T>(a.x * inv, a.y * inv, a.z * inv);
    } else {
        return Vector3<T>(a.x / b, a.y / b, a.z / b);
    }
}

template <typename T>
Vector3<T>
operator/(T a, const Vector3<T>& b) {
    return Vector3<T>(a / b.x, a / b.y, a / b.z);
}

template <typename T>
Vector3<T>
operator/(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

template <typename T>
Vector3<T>
min(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(std::min(a.x, b.x),
                      std::min(a.y, b.y),
                      std::min(a.z, b.z));
}

template <typename T>
Vector3<T>
max(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>(std::max(a.x, b.x),
                      std::max(a.y, b.y),
                      std::max(a.z, b.z));
}

template <typename T>
Vector3<T>
clamp(const Vector3<T>& v, const Vector3<T>& low, const Vector3<T>& high) {
    return Vector3<T>(std::clamp(v.x, low.x, high.x),
                      std::clamp(v.y, low.y, high.y),
                      std::clamp(v.z, low.z, high.z));
}

template <typename T>
Vector3<T>
ceil(const Vector3<T>& a) {
    return Vector3<T>(std::ceil(a.x),
                      std::ceil(a.y),
                      std::ceil(a.z));
}

template <typename T>
Vector3<T>
floor(const Vector3<T>& a) {
    return Vector3<T>(std::floor(a.x),
                      std::floor(a.y),
                      std::floor(a.z));
}

template <typename T>
Vector3<T>
abs(const Vector3<T>& v) {
    return Vector3<T>(std::abs(v.x),
                      std::abs(v.y),
                      std::abs(v.z));
}

template <typename T>
bool
isfinite(const Vector3<T>& v) noexcept {
    return std::isfinite(static_cast<double>(v.x))
        && std::isfinite(static_cast<double>(v.y))
        && std::isfinite(static_cast<double>(v.z));
}

template <typename T>
T
xy_dot(const Vector3<T>& a, const Vector3<T>& b) noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(a.y, b.y, a.x * b.x);
    } else {
        return a.x * b.x + a.y * b.y;
    }
}

template <typename T>
T
xy_length_squared(const Vector3<T>& v) noexcept {
    return xy_dot(v, v);
}

template <typename T>
T
xy_length(const Vector3<T>& v) noexcept {
    using std::sqrt;
    return static_cast<T>(sqrt(xy_length_squared(v)));
}

template <typename T>
Vector3<T>
normalized_or(const Vector3<T>& v,
              const Vector3<T>& fallback,
              const T min_length_squared) noexcept {
    const T len2 = v.length_squared();
    if (!(len2 > min_length_squared)) {
        return fallback;
    }

    using std::sqrt;
    return v * (T(1) / static_cast<T>(sqrt(len2)));
}

template <typename T>
Vector3<T>
xy_normalized_or(const Vector3<T>& v,
                 const Vector3<T>& fallback,
                 const T min_length_squared) noexcept {
    const T len2 = xy_length_squared(v);
    if (!(len2 > min_length_squared)) {
        return fallback;
    }

    using std::sqrt;
    const T inv = T(1) / static_cast<T>(sqrt(len2));
    return Vector3<T>(v.x * inv, v.y * inv, T(0));
}

template <typename T>
Vector3<T>
reject(const Vector3<T>& v, const Vector3<T>& normal) noexcept {
    return v - normal * v.dot(normal);
}

template <typename T>
bool
orthonormal_basis(const Vector3<T>& normal,
                  Vector3<T>& unit_normal,
                  Vector3<T>& tangent,
                  Vector3<T>& bitangent,
                  const T min_length_squared) noexcept {
    const T normal_length_squared = normal.length_squared();
    if (!(normal_length_squared > min_length_squared)) {
        return false;
    }

    using std::sqrt;
    unit_normal = normal * (T(1) / static_cast<T>(sqrt(normal_length_squared)));

    const Vector3<T> axis = std::abs(unit_normal.z) < T(0.9)
        ? Vector3<T>(T(0), T(0), T(1))
        : Vector3<T>(T(0), T(1), T(0));

    tangent = axis.cross(unit_normal);
    const T tangent_length_squared = tangent.length_squared();
    if (!(tangent_length_squared > min_length_squared)) {
        return false;
    }

    tangent *= T(1) / static_cast<T>(sqrt(tangent_length_squared));

    bitangent = unit_normal.cross(tangent);
    const T bitangent_length_squared = bitangent.length_squared();
    if (!(bitangent_length_squared > min_length_squared)) {
        return false;
    }

    bitangent *= T(1) / static_cast<T>(sqrt(bitangent_length_squared));
    return true;
}

template <typename T>
bool
orthonormal_basis(const Vector3<T>& normal,
                  Vector3<T>& tangent,
                  Vector3<T>& bitangent,
                  const T min_length_squared) noexcept {
    Vector3<T> unit_normal;
    return orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared);
}

template <typename T>
Vector3<T>
orthogonal_unit_vector(const Vector3<T>& normal,
                       const Vector3<T>& seed,
                       const T min_length_squared) noexcept {
    const Vector3<T> fallback(T(1), T(0), T(0));
    Vector3<T> tangent = normal.cross(seed);
    if (tangent.length_squared() > min_length_squared) {
        return normalized_or(tangent, fallback, min_length_squared);
    }

    Vector3<T> unit_normal;
    Vector3<T> bitangent;
    if (orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared)) {
        return tangent;
    }

    return fallback;
}

template <typename T>
Vector3<T>
cmin(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>((a.x < b.x) ? a.x : b.x,
                      (a.y < b.y) ? a.y : b.y,
                      (a.z < b.z) ? a.z : b.z);
}

template <typename T>
Vector3<T>
cmax(const Vector3<T>& a, const Vector3<T>& b) {
    return Vector3<T>((a.x > b.x) ? a.x : b.x,
                      (a.y > b.y) ? a.y : b.y,
                      (a.z > b.z) ? a.z : b.z);
}

template <typename To, typename From>
Vector3<To>
cast_to(const Vector3<From>& v) noexcept {
    return Vector3<To>(static_cast<To>(v.x),
                       static_cast<To>(v.y),
                       static_cast<To>(v.z));
}

}
