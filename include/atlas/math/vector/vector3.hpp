#pragma once
#include <algorithm>

namespace atlas::math {

// ------------------------------------------------------------
// Vector<T,3>
// ------------------------------------------------------------
template <typename T>
constexpr Vector<T, 3>::Vector() noexcept
    : x(T(0))
    , y(T(0))
    , z(T(0)) {
    // Default: zero vector (0,0,0)
}

template <typename T>
constexpr Vector<T, 3>::Vector(T s) noexcept
    : x(s)
    , y(s)
    , z(s) {
    // Fill: (s,s,s)
}

template <typename T>
constexpr Vector<T, 3>::Vector(T x_, T y_, T z_) noexcept
    : x(x_)
    , y(y_)
    , z(z_) {
    // Component constructor: (x_, y_, z_)
}

template <typename T>
Vector<T, 3>::Vector(std::initializer_list<T> list) noexcept {
    // Init-list: {x, y, z}, missing values default to 0.
    const T* it = list.begin();
    x           = (it != list.end()) ? *it++ : T(0);
    y           = (it != list.end()) ? *it++ : T(0);
    z           = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Vector<T, 3>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    // Expression-template materialization.
    const Expression& e = expr();
    x                   = static_cast<T>(e[0]);
    y                   = static_cast<T>(e[1]);
    z                   = static_cast<T>(e[2]);
}

template <typename T>
std::size_t
Vector<T, 3>::size() noexcept {
    // Static dimension query.
    return 3;
}

template <typename T>
const T*
Vector<T, 3>::data() const noexcept {
    // Contiguous pointer to x,y,z.
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
    // Flat indexing: 0->x, 1->y, 2->z. No bounds check.
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
    // Accessor matching Matrix::at style (no bounds check here).
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
    // Fill: (s,s,s)
    x = y = z = s;
}

template <typename T>
template <typename... Args, typename>
void
Vector<T, 3>::set_values(Args... args) noexcept {
    // Set from 3 scalars (SFINAE guard enforces arity).
    T tmp[3] = { static_cast<T>(args)... };
    x        = tmp[0];
    y        = tmp[1];
    z        = tmp[2];
}

template <typename T>
void
Vector<T, 3>::set_zero() noexcept {
    // Set to (0,0,0)
    x = y = z = T(0);
}

template <typename T>
void
Vector<T, 3>::add(T v) noexcept {
    // Component-wise add scalar.
    x += v;
    y += v;
    z += v;
}

template <typename T>
void
Vector<T, 3>::sub(T v) noexcept {
    // Component-wise subtract scalar.
    x -= v;
    y -= v;
    z -= v;
}

template <typename T>
void
Vector<T, 3>::mul(T v) noexcept {
    // Component-wise multiply by scalar.
    x *= v;
    y *= v;
    z *= v;
}

template <typename T>
void
Vector<T, 3>::div(T v) noexcept {
    // Component-wise divide by scalar (via reciprocal).
    const T inv = T(1) / v;
    x *= inv;
    y *= inv;
    z *= inv;
}

template <typename T>
void
Vector<T, 3>::add(const Vector& v) noexcept {
    // Component-wise add vector.
    x += v.x;
    y += v.y;
    z += v.z;
}

template <typename T>
void
Vector<T, 3>::sub(const Vector& v) noexcept {
    // Component-wise subtract vector.
    x -= v.x;
    y -= v.y;
    z -= v.z;
}

template <typename T>
void
Vector<T, 3>::mul(const Vector& v) noexcept {
    // Hadamard product.
    x *= v.x;
    y *= v.y;
    z *= v.z;
}

template <typename T>
void
Vector<T, 3>::div(const Vector& v) noexcept {
    // Component-wise division.
    x /= v.x;
    y /= v.y;
    z /= v.z;
}

template <typename T>
T
Vector<T, 3>::min() const noexcept {
    // Minimum component (by value).
    return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
}

template <typename T>
T
Vector<T, 3>::max() const noexcept {
    // Maximum component (by value).
    return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator+=(T v) noexcept {
    // In-place scalar add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator-=(T v) noexcept {
    // In-place scalar subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator*=(T v) noexcept {
    // In-place scalar multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator/=(T v) noexcept {
    // In-place scalar divide.
    div(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator+=(const Vector& v) noexcept {
    // In-place vector add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator-=(const Vector& v) noexcept {
    // In-place vector subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator*=(const Vector& v) noexcept {
    // In-place Hadamard multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 3>&
Vector<T, 3>::operator/=(const Vector& v) noexcept {
    // In-place component-wise divide.
    div(v);
    return *this;
}

template <typename T>
bool
Vector<T, 3>::operator==(const Vector& other) const noexcept {
    // Exact equality.
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
    // Dot product: x*vx + y*vy + z*vz
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(z, v.z, std::fma(y, v.y, x * v.x));
    } else {
        return x * v.x + y * v.y + z * v.z;
    }
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::cross(const Vector& v) const noexcept {
    // Cross product.
    return Vector3<T>(y * v.z - z * v.y,
                      z * v.x - x * v.z,
                      x * v.y - y * v.x);
}

template <typename T>
T
Vector<T, 3>::length_squared() const noexcept {
    // Squared length: x^2 + y^2 + z^2
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(z, z, std::fma(y, y, x * x));
    } else {
        return x * x + y * y + z * z;
    }
}

template <typename T>
T
Vector<T, 3>::length() const noexcept {
    // Euclidean norm.
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared()));
}

template <typename T>
std::size_t
Vector<T, 3>::major_axis() const noexcept {
    // Index of the largest component by value (NOT by magnitude).
    // Note: if you wanted abs-based axis, use abs comparisons.
    if (x >= y && x >= z) return 0;
    if (y >= x && y >= z) return 1;
    return 2;
}

template <typename T>
std::size_t
Vector<T, 3>::minor_axis() const noexcept {
    // Index of the smallest component by value (NOT by magnitude).
    if (x <= y && x <= z) return 0;
    if (y <= x && y <= z) return 1;
    return 2;
}

template <typename T>
void
Vector<T, 3>::normalize() noexcept {
    // In-place normalization. Leaves zero vector unchanged.
    const T ls = length_squared();
    if (ls == T(0)) return;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    x *= inv;
    y *= inv;
    z *= inv;
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::normalized() const noexcept {
    // Return normalized copy (or self if zero).
    const T ls = length_squared();
    if (ls == T(0)) return *this;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    return Vector3<T>(x * inv, y * inv, z * inv);
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::reflected(const Vector& n) const noexcept {
    // Reflection about a (typically unit) normal n:
    //   r = v - 2*(v·n)*n
    const T d = dot(n);
    return Vector3<T>(x - T(2) * d * n.x,
                      y - T(2) * d * n.y,
                      z - T(2) * d * n.z);
}

template <typename T>
Vector<T, 3>
Vector<T, 3>::projected(const Vector& n) const noexcept {
    // Reject normal component:
    //   v_proj = v - (v·n)*n
    const T d = dot(n);
    return Vector3<T>(x - d * n.x,
                      y - d * n.y,
                      z - d * n.z);
}

template <typename T>
std::tuple<Vector<T, 3>, Vector<T, 3>>
Vector<T, 3>::tangential() const noexcept {
    // Build two orthonormal tangents (t1,t2) perpendicular to this vector.
    //
    // Strategy:
    //  - pick a stable perpendicular direction t1 using the dominant axis heuristic,
    //    then t2 = (this x t1) normalized.
    const T ax = std::abs(x), ay = std::abs(y), az = std::abs(z);

    Vector t1;
    if (ax > ay) {
        // Use (-z, 0, x) normalized (avoid degeneracy when x,z ~ 0).
        const T d2  = x * x + z * z;
        const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(d2 + (d2 == T(0) ? T(1) : T(0)))));
        t1          = Vector(-z * inv, T(0), x * inv);
    } else {
        // Use (0, z, -y) normalized (avoid degeneracy when y,z ~ 0).
        const T d2  = y * y + z * z;
        const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(d2 + (d2 == T(0) ? T(1) : T(0)))));
        t1          = Vector(T(0), z * inv, -y * inv);
    }

    const Vector t2 = cross(t1).normalized();
    return { t1, t2 };
}

template <typename T>
template <typename To>
Vector<To, 3>
Vector<T, 3>::cast_to() const noexcept {
    // Component-wise cast to another scalar type.
    return Vector<To, 3>(static_cast<To>(x),
                         static_cast<To>(y),
                         static_cast<To>(z));
}

// ------------------------------------------------------------
// Free functions
// ------------------------------------------------------------
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
    // r = v - 2*(v·n)*n
    const T d = dot(v, normal);
    return Vector<T, 3>(
        v.x - T(2) * d * normal.x,
        v.y - T(2) * d * normal.y,
        v.z - T(2) * d * normal.z);
}

template <typename T>
Vector<T, 3>
projected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept {
    // v_proj = v - (v·n)*n  (reject normal component)
    const T d = dot(v, normal);
    return Vector<T, 3>(
        v.x - d * normal.x,
        v.y - d * normal.y,
        v.z - d * normal.z);
}

template <typename T>
std::tuple<Vector<T, 3>, Vector<T, 3>>
tangential(const Vector<T, 3>& normal) noexcept {
    // Standalone tangent-frame builder for a normal.
    const T nx = normal.x, ny = normal.y, nz = normal.z;
    const T ax = std::abs(nx), ay = std::abs(ny), az = std::abs(nz);

    Vector<T, 3> t1;
    if (ax > ay) {
        const T d2  = nx * nx + nz * nz;
        const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(d2 + (d2 == T(0) ? T(1) : T(0)))));
        t1          = Vector<T, 3>(-nz * inv, T(0), nx * inv);
    } else {
        const T d2  = ny * ny + nz * nz;
        const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(d2 + (d2 == T(0) ? T(1) : T(0)))));
        t1          = Vector<T, 3>(T(0), nz * inv, -ny * inv);
    }

    const Vector<T, 3> t2 = cross(normal, t1).normalized();
    return { t1, t2 };
}

// ------------------------------------------------------------
// Operators / utilities (Vector3 alias assumed)
// ------------------------------------------------------------
template <typename T>
Vector3<T>
operator+(const Vector3<T>& a) {
    // Unary plus (no-op).
    return a;
}

template <typename T>
Vector3<T>
operator+(T a, const Vector3<T>& b) {
    // Scalar + vector (broadcast).
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
    // Unary minus.
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
    // Hadamard product.
    return Vector3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}

template <typename T>
Vector3<T>
operator/(const Vector3<T>& a, T b) {
    const T inv = T(1) / b;
    return Vector3<T>(a.x * inv, a.y * inv, a.z * inv);
}

template <typename T>
Vector3<T>
operator/(T a, const Vector3<T>& b) {
    // Scalar / vector (component-wise).
    return Vector3<T>(a / b.x, a / b.y, a / b.z);
}

template <typename T>
Vector3<T>
operator/(const Vector3<T>& a, const Vector3<T>& b) {
    // Component-wise division.
    return Vector3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

template <typename T>
Vector3<T>
min(const Vector3<T>& a, const Vector3<T>& b) {
    // Component-wise min.
    return Vector3<T>(std::min(a.x, b.x),
                      std::min(a.y, b.y),
                      std::min(a.z, b.z));
}

template <typename T>
Vector3<T>
max(const Vector3<T>& a, const Vector3<T>& b) {
    // Component-wise max.
    return Vector3<T>(std::max(a.x, b.x),
                      std::max(a.y, b.y),
                      std::max(a.z, b.z));
}

template <typename T>
Vector3<T>
clamp(const Vector3<T>& v, const Vector3<T>& low, const Vector3<T>& high) {
    // Component-wise clamp.
    return Vector3<T>(std::clamp(v.x, low.x, high.x),
                      std::clamp(v.y, low.y, high.y),
                      std::clamp(v.z, low.z, high.z));
}

template <typename T>
Vector3<T>
ceil(const Vector3<T>& a) {
    // Component-wise ceil.
    return Vector3<T>(std::ceil(a.x),
                      std::ceil(a.y),
                      std::ceil(a.z));
}

template <typename T>
Vector3<T>
floor(const Vector3<T>& a) {
    // Component-wise floor.
    return Vector3<T>(std::floor(a.x),
                      std::floor(a.y),
                      std::floor(a.z));
}

template <typename T>
Vector3<T>
abs(const Vector3<T>& v) {
    // Component-wise abs.
    return Vector3<T>(std::abs(v.x),
                      std::abs(v.y),
                      std::abs(v.z));
}

template <typename T>
Vector3<T>
cmin(const Vector3<T>& a, const Vector3<T>& b) {
    // Branch-based component min.
    return Vector3<T>((a.x < b.x) ? a.x : b.x,
                      (a.y < b.y) ? a.y : b.y,
                      (a.z < b.z) ? a.z : b.z);
}

template <typename T>
Vector3<T>
cmax(const Vector3<T>& a, const Vector3<T>& b) {
    // Branch-based component max.
    return Vector3<T>((a.x > b.x) ? a.x : b.x,
                      (a.y > b.y) ? a.y : b.y,
                      (a.z > b.z) ? a.z : b.z);
}

template <typename To, typename From>
Vector3<To>
cast_to(const Vector3<From>& v) noexcept {
    // Free-function cast to match other math APIs.
    return Vector3<To>(static_cast<To>(v.x),
                       static_cast<To>(v.y),
                       static_cast<To>(v.z));
}

} // namespace atlas::math
