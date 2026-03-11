#pragma once
namespace atlas::math {

// ------------------------------------------------------------
// Vector<T,2>
// ------------------------------------------------------------
template <typename T>
constexpr Vector<T, 2>::Vector() noexcept
    : x(T(0))
    , y(T(0)) {
    // Default: zero vector (0,0)
}

template <typename T>
constexpr Vector<T, 2>::Vector(T s) noexcept
    : x(s)
    , y(s) {
    // Fill: (s,s)
}

template <typename T>
constexpr Vector<T, 2>::Vector(T x_, T y_) noexcept
    : x(x_)
    , y(y_) {
    // Component constructor: (x_, y_)
}

template <typename T>
Vector<T, 2>::Vector(std::initializer_list<T> list) noexcept {
    // Init-list: {x, y}
    // Missing values default to 0.
    const T* it = list.begin();
    x           = (it != list.end()) ? *it++ : T(0);
    y           = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Vector<T, 2>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    // Expression-template materialization into concrete storage.
    const Expression& e = expr();
    x                   = static_cast<T>(e[0]);
    y                   = static_cast<T>(e[1]);
}

template <typename T>
std::size_t
Vector<T, 2>::size() noexcept {
    // Static dimension query.
    return 2;
}

template <typename T>
const T*
Vector<T, 2>::data() const noexcept {
    // Contiguous pointer to x,y.
    return &x;
}

template <typename T>
T*
Vector<T, 2>::data() noexcept {
    // Mutable pointer to x,y.
    return &x;
}

template <typename T>
const T&
Vector<T, 2>::operator[](std::size_t i) const noexcept {
    // Flat indexing: 0 -> x, 1 -> y. No bounds check.
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 2>::operator[](std::size_t i) noexcept {
    // Mutable flat indexing.
    return (&x)[i];
}

template <typename T>
const T&
Vector<T, 2>::at(std::size_t i) const noexcept {
    // Accessor matching Matrix::at style (no bounds check in this implementation).
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 2>::at(std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
void
Vector<T, 2>::set(T s) noexcept {
    // Fill: (s,s)
    x = y = s;
}

template <typename T>
template <typename... Args, typename>
void
Vector<T, 2>::set_values(Args... args) noexcept {
    // Set from 2 scalars (SFINAE guard enforces arity).
    T tmp[2] = { static_cast<T>(args)... };
    x        = tmp[0];
    y        = tmp[1];
}

template <typename T>
void
Vector<T, 2>::set_zero() noexcept {
    // Set to (0,0)
    x = y = T(0);
}

template <typename T>
void
Vector<T, 2>::add(T v) noexcept {
    // Component-wise add scalar.
    x += v;
    y += v;
}

template <typename T>
void
Vector<T, 2>::sub(T v) noexcept {
    // Component-wise subtract scalar.
    x -= v;
    y -= v;
}

template <typename T>
void
Vector<T, 2>::mul(T v) noexcept {
    // Component-wise multiply by scalar.
    x *= v;
    y *= v;
}

template <typename T>
void
Vector<T, 2>::div(T v) noexcept {
    // Component-wise divide by scalar (via reciprocal).
    const T inv = T(1) / v;
    x *= inv;
    y *= inv;
}

template <typename T>
void
Vector<T, 2>::add(const Vector& v) noexcept {
    // Component-wise add vector.
    x += v.x;
    y += v.y;
}

template <typename T>
void
Vector<T, 2>::sub(const Vector& v) noexcept {
    // Component-wise subtract vector.
    x -= v.x;
    y -= v.y;
}

template <typename T>
void
Vector<T, 2>::mul(const Vector& v) noexcept {
    // Hadamard product.
    x *= v.x;
    y *= v.y;
}

template <typename T>
void
Vector<T, 2>::div(const Vector& v) noexcept {
    // Component-wise division.
    x /= v.x;
    y /= v.y;
}

template <typename T>
T
Vector<T, 2>::min() const noexcept {
    // Minimum component (by value).
    return (x < y) ? x : y;
}

template <typename T>
T
Vector<T, 2>::max() const noexcept {
    // Maximum component (by value).
    return (x > y) ? x : y;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator+=(T v) noexcept {
    // In-place scalar add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator-=(T v) noexcept {
    // In-place scalar subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator*=(T v) noexcept {
    // In-place scalar multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator/=(T v) noexcept {
    // In-place scalar divide.
    div(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator+=(const Vector& v) noexcept {
    // In-place vector add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator-=(const Vector& v) noexcept {
    // In-place vector subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator*=(const Vector& v) noexcept {
    // In-place Hadamard multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator/=(const Vector& v) noexcept {
    // In-place component-wise divide.
    div(v);
    return *this;
}

template <typename T>
bool
Vector<T, 2>::operator==(const Vector& other) const noexcept {
    // Exact equality.
    return x == other.x && y == other.y;
}

template <typename T>
bool
Vector<T, 2>::operator!=(const Vector& other) const noexcept {
    return !(*this == other);
}

template <typename T>
T
Vector<T, 2>::dot(const Vector& v) const noexcept {
    // Dot product: x*vx + y*vy
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(y, v.y, x * v.x);
    } else {
        return x * v.x + y * v.y;
    }
}

template <typename T>
T
Vector<T, 2>::cross(const Vector& v) const noexcept {
    // 2D "cross" (scalar z-component of 3D cross):
    //   cross(a,b) = ax*by - ay*bx
    return x * v.y - y * v.x;
}

template <typename T>
T
Vector<T, 2>::length_squared() const noexcept {
    // Squared length: x^2 + y^2
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(y, y, x * x);
    } else {
        return x * x + y * y;
    }
}

template <typename T>
T
Vector<T, 2>::length() const noexcept {
    // Euclidean norm.
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared()));
}

template <typename T>
std::size_t
Vector<T, 2>::major_axis() const noexcept {
    // Index of the component with larger magnitude (abs).
    return (std::abs(x) >= std::abs(y)) ? 0u : 1u;
}

template <typename T>
std::size_t
Vector<T, 2>::minor_axis() const noexcept {
    // Index of the component with smaller magnitude (abs).
    return (std::abs(x) <= std::abs(y)) ? 0u : 1u;
}

template <typename T>
void
Vector<T, 2>::normalize() noexcept {
    // In-place normalization. Leaves zero vector unchanged.
    const T ls = length_squared();
    if (ls == T(0)) return;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    x *= inv;
    y *= inv;
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::normalized() const noexcept {
    // Return normalized copy (or self if zero).
    const T ls = length_squared();
    if (ls == T(0)) return *this;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    return Vector<T, 2>(x * inv, y * inv);
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::reflected(const Vector& n) const noexcept {
    // Reflection about a (typically unit) normal n:
    //   r = v - 2*(v·n)*n
    const T d = dot(n);
    return Vector<T, 2>(x - T(2) * d * n.x,
                        y - T(2) * d * n.y);
}

template <typename T>
Vector<T, 2> Vector<T, 2>::projected(const Vector& n) const noexcept {
    // NOTE: In this codebase, "projected" returns the component orthogonal to n (rejection).
    // rejection_n(v) = v - proj_n(v), where proj_n(v) = (dot(v,n)/dot(n,n)) * n
    const T nn = n.dot(n);
    if (nn == T(0)) {
        // If direction is zero, treat as "no constraint": return v (already orthogonal to nothing).
        return *this;
    }
    const T s = this->dot(n) / nn;
    const Vector proj = n * s;
    return (*this) - proj;
}
template <typename T>
Vector<T, 2>
Vector<T, 2>::tangential() const noexcept {
    // Return a unit tangent direction perpendicular to this vector.
    // For v=(x,y), one perpendicular is (-y, x).
    // If v is zero, return (0,0).
    const T ls = length_squared();
    if (ls == T(0)) return Vector<T, 2>(T(0), T(0));

    Vector<T, 2> t(-y, x);
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(t.x * t.x + t.y * t.y)));
    return Vector<T, 2>(t.x * inv, t.y * inv);
}

template <typename T>
template <typename To>
Vector<To, 2>
Vector<T, 2>::cast_to() const noexcept {
    // Component-wise cast to another scalar type.
    return Vector<To, 2>(static_cast<To>(x), static_cast<To>(y));
}

// ------------------------------------------------------------
// Free functions / operators
// ------------------------------------------------------------
template <typename T>
T
dot(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept {
    return a.dot(b);
}

template <typename T>
T
cross(const Vector<T, 2>& a, const Vector<T, 2>& b) noexcept {
    return a.cross(b);
}

template <typename T>
Vector<T, 2>
reflected(const Vector<T, 2>& v, const Vector<T, 2>& normal) noexcept {
    // r = v - 2*(v·n)*n
    const T d = dot(v, normal);
    return Vector<T, 2>(v.x - T(2) * d * normal.x,
                        v.y - T(2) * d * normal.y);
}




template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
Vector<T, 2> projected(const Vector<T, 2>& v, const Vector<T, 2>& n) noexcept {
    const T nn = dot(n, n);
    if (nn == T(0)) {
        return v;
    }
    const T s = dot(v, n) / nn;
    const Vector<T, 2> proj = n * s;
    return v - proj;
}

template <typename T>
Vector<T, 2>
operator+(const Vector<T, 2>& a) {
    // Unary plus (no-op).
    return a;
}

template <typename T>
Vector<T, 2>
operator-(const Vector<T, 2>& a) {
    // Unary minus.
    return Vector<T, 2>(-a.x, -a.y);
}

template <typename T>
Vector<T, 2>
operator+(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>(a.x + b.x, a.y + b.y);
}

template <typename T>
Vector<T, 2>
operator-(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>(a.x - b.x, a.y - b.y);
}

template <typename T>
Vector<T, 2>
operator+(T a, const Vector<T, 2>& b) {
    // Scalar + vector (broadcast).
    return Vector<T, 2>(a + b.x, a + b.y);
}

template <typename T>
Vector<T, 2>
operator+(const Vector<T, 2>& a, T b) {
    return Vector<T, 2>(a.x + b, a.y + b);
}

template <typename T>
Vector<T, 2>
operator-(T a, const Vector<T, 2>& b) {
    // Scalar - vector (broadcast).
    return Vector<T, 2>(a - b.x, a - b.y);
}

template <typename T>
Vector<T, 2>
operator-(const Vector<T, 2>& a, T b) {
    return Vector<T, 2>(a.x - b, a.y - b);
}

template <typename T>
Vector<T, 2>
operator*(const Vector<T, 2>& a, T b) {
    return Vector<T, 2>(a.x * b, a.y * b);
}

template <typename T>
Vector<T, 2>
operator*(T a, const Vector<T, 2>& b) {
    return Vector<T, 2>(a * b.x, a * b.y);
}

template <typename T>
Vector<T, 2>
operator*(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Hadamard product.
    return Vector<T, 2>(a.x * b.x, a.y * b.y);
}

template <typename T>
Vector<T, 2>
operator/(const Vector<T, 2>& a, T b) {
    const T inv = T(1) / b;
    return Vector<T, 2>(a.x * inv, a.y * inv);
}

template <typename T>
Vector<T, 2>
operator/(T a, const Vector<T, 2>& b) {
    // Scalar / vector (component-wise).
    return Vector<T, 2>(a / b.x, a / b.y);
}

template <typename T>
Vector<T, 2>
operator/(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Component-wise division.
    return Vector<T, 2>(a.x / b.x, a.y / b.y);
}

template <typename T>
Vector<T, 2>
min(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Component-wise min using std::min.
    return Vector<T, 2>(std::min(a.x, b.x), std::min(a.y, b.y));
}

template <typename T>
Vector<T, 2>
max(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Component-wise max using std::max.
    return Vector<T, 2>(std::max(a.x, b.x), std::max(a.y, b.y));
}

template <typename T>
Vector<T, 2>
clamp(const Vector<T, 2>& v, const Vector<T, 2>& low, const Vector<T, 2>& high) {
    // Component-wise clamp.
    return Vector<T, 2>(std::clamp(v.x, low.x, high.x),
                        std::clamp(v.y, low.y, high.y));
}

template <typename T>
Vector<T, 2>
ceil(const Vector<T, 2>& a) {
    // Component-wise ceil.
    using std::ceil;
    return Vector<T, 2>(static_cast<T>(ceil(a.x)),
                        static_cast<T>(ceil(a.y)));
}

template <typename T>
Vector<T, 2>
floor(const Vector<T, 2>& a) {
    // Component-wise floor.
    using std::floor;
    return Vector<T, 2>(static_cast<T>(floor(a.x)),
                        static_cast<T>(floor(a.y)));
}

template <typename T>
Vector<T, 2>
abs(const Vector<T, 2>& v) {
    // Component-wise abs.
    using std::abs;
    return Vector<T, 2>(static_cast<T>(abs(v.x)),
                        static_cast<T>(abs(v.y)));
}

template <typename T>
Vector<T, 2>
cmin(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Branch-based component min (avoids std::min overhead / includes).
    return Vector<T, 2>((a.x < b.x) ? a.x : b.x,
                        (a.y < b.y) ? a.y : b.y);
}

template <typename T>
Vector<T, 2>
cmax(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    // Branch-based component max.
    return Vector<T, 2>((a.x > b.x) ? a.x : b.x,
                        (a.y > b.y) ? a.y : b.y);
}

} // namespace atlas::math
