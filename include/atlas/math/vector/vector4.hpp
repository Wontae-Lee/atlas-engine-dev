#pragma once
namespace atlas::math {

// ------------------------------------------------------------
// Vector<T,4>
// ------------------------------------------------------------
template <typename T>
constexpr Vector<T, 4>::Vector() noexcept
    : x(T(0))
    , y(T(0))
    , z(T(0))
    , w(T(0)) {
    // Default: zero vector (0,0,0,0)
}

template <typename T>
constexpr Vector<T, 4>::Vector(T s) noexcept
    : x(s)
    , y(s)
    , z(s)
    , w(s) {
    // Fill: (s,s,s,s)
}

template <typename T>
constexpr Vector<T, 4>::Vector(T x_, T y_, T z_, T w_) noexcept
    : x(x_)
    , y(y_)
    , z(z_)
    , w(w_) {
    // Component constructor: (x_, y_, z_, w_)
}

template <typename T>
Vector<T, 4>::Vector(std::initializer_list<T> list) noexcept {
    // Init-list: {x, y, z, w}, missing values default to 0.
    const T* it = list.begin();
    x           = (it != list.end()) ? *it++ : T(0);
    y           = (it != list.end()) ? *it++ : T(0);
    z           = (it != list.end()) ? *it++ : T(0);
    w           = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Vector<T, 4>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    // Expression-template materialization.
    const Expression& e = expr();
    x                   = static_cast<T>(e[0]);
    y                   = static_cast<T>(e[1]);
    z                   = static_cast<T>(e[2]);
    w                   = static_cast<T>(e[3]);
}

template <typename T>
std::size_t
Vector<T, 4>::size() noexcept {
    // Static dimension query.
    return 4;
}

template <typename T>
const T*
Vector<T, 4>::data() const noexcept {
    // Contiguous pointer to x,y,z,w.
    return &x;
}

template <typename T>
T*
Vector<T, 4>::data() noexcept {
    return &x;
}

template <typename T>
const T&
Vector<T, 4>::operator[](std::size_t i) const noexcept {
    // Flat indexing: 0->x, 1->y, 2->z, 3->w. No bounds check.
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 4>::operator[](std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
const T&
Vector<T, 4>::at(std::size_t i) const noexcept {
    // Accessor matching Matrix::at style (no bounds check here).
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 4>::at(std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
void
Vector<T, 4>::set(T s) noexcept {
    // Fill: (s,s,s,s)
    x = y = z = w = s;
}

template <typename T>
template <typename... Args, typename>
void
Vector<T, 4>::set_values(Args... args) noexcept {
    // Set from 4 scalars (SFINAE guard enforces arity).
    T tmp[4] = { static_cast<T>(args)... };
    x        = tmp[0];
    y        = tmp[1];
    z        = tmp[2];
    w        = tmp[3];
}

template <typename T>
void
Vector<T, 4>::set_zero() noexcept {
    // Set to (0,0,0,0)
    x = y = z = w = T(0);
}

template <typename T>
void
Vector<T, 4>::add(T v) noexcept {
    // Component-wise add scalar.
    x += v;
    y += v;
    z += v;
    w += v;
}

template <typename T>
void
Vector<T, 4>::sub(T v) noexcept {
    // Component-wise subtract scalar.
    x -= v;
    y -= v;
    z -= v;
    w -= v;
}

template <typename T>
void
Vector<T, 4>::mul(T v) noexcept {
    // Component-wise multiply by scalar.
    x *= v;
    y *= v;
    z *= v;
    w *= v;
}

template <typename T>
void
Vector<T, 4>::div(T v) noexcept {
    // Component-wise divide by scalar (via reciprocal).
    const T inv = T(1) / v;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
}

template <typename T>
void
Vector<T, 4>::add(const Vector& v) noexcept {
    // Component-wise add vector.
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
}

template <typename T>
void
Vector<T, 4>::sub(const Vector& v) noexcept {
    // Component-wise subtract vector.
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
}

template <typename T>
void
Vector<T, 4>::mul(const Vector& v) noexcept {
    // Hadamard product.
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
}

template <typename T>
void
Vector<T, 4>::div(const Vector& v) noexcept {
    // Component-wise division.
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
}

template <typename T>
T
Vector<T, 4>::min() const noexcept {
    // Minimum component (by value).
    const T m1 = (x < y) ? x : y;
    const T m2 = (z < w) ? z : w;
    return (m1 < m2) ? m1 : m2;
}

template <typename T>
T
Vector<T, 4>::max() const noexcept {
    // Maximum component (by value).
    const T m1 = (x > y) ? x : y;
    const T m2 = (z > w) ? z : w;
    return (m1 > m2) ? m1 : m2;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator+=(T v) noexcept {
    // In-place scalar add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator-=(T v) noexcept {
    // In-place scalar subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator*=(T v) noexcept {
    // In-place scalar multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator/=(T v) noexcept {
    // In-place scalar divide.
    div(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator+=(const Vector& v) noexcept {
    // In-place vector add.
    add(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator-=(const Vector& v) noexcept {
    // In-place vector subtract.
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator*=(const Vector& v) noexcept {
    // In-place Hadamard multiply.
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 4>&
Vector<T, 4>::operator/=(const Vector& v) noexcept {
    // In-place component-wise divide.
    div(v);
    return *this;
}

template <typename T>
bool
Vector<T, 4>::operator==(const Vector& other) const noexcept {
    // Exact equality.
    return x == other.x && y == other.y && z == other.z && w == other.w;
}

template <typename T>
bool
Vector<T, 4>::operator!=(const Vector& other) const noexcept {
    return !(*this == other);
}

template <typename T>
T
Vector<T, 4>::dot(const Vector& v) const noexcept {
    // Dot product: x*vx + y*vy + z*vz + w*vw
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(w, v.w, std::fma(z, v.z, std::fma(y, v.y, x * v.x)));
    } else {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }
}

template <typename T>
T
Vector<T, 4>::length_squared() const noexcept {
    // Squared length: x^2 + y^2 + z^2 + w^2
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(w, w, std::fma(z, z, std::fma(y, y, x * x)));
    } else {
        return x * x + y * y + z * z + w * w;
    }
}

template <typename T>
T
Vector<T, 4>::length() const noexcept {
    // Euclidean norm.
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared()));
}

template <typename T>
std::size_t
Vector<T, 4>::major_axis() const noexcept {
    // Index of the component with largest absolute value.
    const T ax = std::abs(x), ay = std::abs(y), az = std::abs(z), aw = std::abs(w);
    std::size_t idx = 0;
    T best          = ax;

    if (ay >= best) {
        best = ay;
        idx  = 1;
    }
    if (az >= best) {
        best = az;
        idx  = 2;
    }
    if (aw >= best) { idx = 3; }

    return idx;
}

template <typename T>
std::size_t
Vector<T, 4>::minor_axis() const noexcept {
    // Index of the component with smallest absolute value.
    const T ax = std::abs(x), ay = std::abs(y), az = std::abs(z), aw = std::abs(w);
    std::size_t idx = 0;
    T best          = ax;

    if (ay <= best) {
        best = ay;
        idx  = 1;
    }
    if (az <= best) {
        best = az;
        idx  = 2;
    }
    if (aw <= best) { idx = 3; }

    return idx;
}

template <typename T>
void
Vector<T, 4>::normalize() noexcept {
    // In-place normalization. Leaves zero vector unchanged.
    const T ls = length_squared();
    if (ls == T(0)) return;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
}

template <typename T>
Vector<T, 4>
Vector<T, 4>::normalized() const noexcept {
    // Return normalized copy (or self if zero).
    const T ls = length_squared();
    if (ls == T(0)) return *this;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    return Vector<T, 4>(x * inv, y * inv, z * inv, w * inv);
}

template <typename T>
Vector<T, 4>
Vector<T, 4>::reflected(const Vector& n) const noexcept {
    // Reflection about a (typically unit) normal n:
    //   r = v - 2*(v·n)*n
    const T d = dot(n);
    return Vector<T, 4>(x - T(2) * d * n.x,
                        y - T(2) * d * n.y,
                        z - T(2) * d * n.z,
                        w - T(2) * d * n.w);
}

template <typename T>
Vector<T, 4>
Vector<T, 4>::projected(const Vector& n) const noexcept {
    const T nn = n.dot(n);
    if (nn == T(0)) return *this;

    const T s         = this->dot(n) / nn;
    const Vector proj = n * s;
    return (*this) - proj;
}

template <typename T>
template <typename To>
Vector<To, 4>
Vector<T, 4>::cast_to() const noexcept {
    // Component-wise cast to another scalar type.
    return Vector<To, 4>(static_cast<To>(x),
                         static_cast<To>(y),
                         static_cast<To>(z),
                         static_cast<To>(w));
}

// ------------------------------------------------------------
// Free functions
// ------------------------------------------------------------
template <typename T>
T
dot(const Vector<T, 4>& a, const Vector<T, 4>& b) noexcept {
    return a.dot(b);
}

template <typename T>
Vector<T, 4>
reflected(const Vector<T, 4>& v, const Vector<T, 4>& normal) noexcept {
    // r = v - 2*(v·n)*n
    const T d = dot(v, normal);
    return Vector<T, 4>(v.x - T(2) * d * normal.x,
                        v.y - T(2) * d * normal.y,
                        v.z - T(2) * d * normal.z,
                        v.w - T(2) * d * normal.w);
}

template <typename T>
Vector<T, 4>
projected(const Vector<T, 4>& v, const Vector<T, 4>& normal) noexcept {
    const T nn = dot(normal, normal);
    if (nn == T(0)) return v;

    const T s               = dot(v, normal) / nn;
    const Vector<T, 4> proj = normal * s;
    return v - proj;
}

// ------------------------------------------------------------
// Operators / component-wise utilities
// ------------------------------------------------------------
template <typename T>
Vector<T, 4>
operator+(const Vector<T, 4>& a) {
    // Unary plus (no-op).
    return a;
}

template <typename T>
Vector<T, 4>
operator-(const Vector<T, 4>& a) {
    // Unary minus.
    return Vector<T, 4>(-a.x, -a.y, -a.z, -a.w);
}

template <typename T>
Vector<T, 4>
operator+(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return Vector<T, 4>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <typename T>
Vector<T, 4>
operator-(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    return Vector<T, 4>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

template <typename T>
Vector<T, 4>
operator+(T a, const Vector<T, 4>& b) {
    // Scalar + vector (broadcast).
    return Vector<T, 4>(a + b.x, a + b.y, a + b.z, a + b.w);
}

template <typename T>
Vector<T, 4>
operator+(const Vector<T, 4>& a, T b) {
    return Vector<T, 4>(a.x + b, a.y + b, a.z + b, a.w + b);
}

template <typename T>
Vector<T, 4>
operator-(T a, const Vector<T, 4>& b) {
    // Scalar - vector (broadcast).
    return Vector<T, 4>(a - b.x, a - b.y, a - b.z, a - b.w);
}

template <typename T>
Vector<T, 4>
operator-(const Vector<T, 4>& a, T b) {
    return Vector<T, 4>(a.x - b, a.y - b, a.z - b, a.w - b);
}

template <typename T>
Vector<T, 4>
operator*(const Vector<T, 4>& a, T b) {
    return Vector<T, 4>(a.x * b, a.y * b, a.z * b, a.w * b);
}

template <typename T>
Vector<T, 4>
operator*(T a, const Vector<T, 4>& b) {
    return Vector<T, 4>(a * b.x, a * b.y, a * b.z, a * b.w);
}

template <typename T>
Vector<T, 4>
operator*(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Hadamard product.
    return Vector<T, 4>(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template <typename T>
Vector<T, 4>
operator/(const Vector<T, 4>& a, T b) {
    const T inv = T(1) / b;
    return Vector<T, 4>(a.x * inv, a.y * inv, a.z * inv, a.w * inv);
}

template <typename T>
Vector<T, 4>
operator/(T a, const Vector<T, 4>& b) {
    // Scalar / vector (component-wise).
    return Vector<T, 4>(a / b.x, a / b.y, a / b.z, a / b.w);
}

template <typename T>
Vector<T, 4>
operator/(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Component-wise division.
    return Vector<T, 4>(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

template <typename T>
Vector<T, 4>
min(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Component-wise min.
    return Vector<T, 4>(std::min(a.x, b.x),
                        std::min(a.y, b.y),
                        std::min(a.z, b.z),
                        std::min(a.w, b.w));
}

template <typename T>
Vector<T, 4>
max(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Component-wise max.
    return Vector<T, 4>(std::max(a.x, b.x),
                        std::max(a.y, b.y),
                        std::max(a.z, b.z),
                        std::max(a.w, b.w));
}

template <typename T>
Vector<T, 4>
clamp(const Vector<T, 4>& v, const Vector<T, 4>& low, const Vector<T, 4>& high) {
    // Component-wise clamp.
    return Vector<T, 4>(std::clamp(v.x, low.x, high.x),
                        std::clamp(v.y, low.y, high.y),
                        std::clamp(v.z, low.z, high.z),
                        std::clamp(v.w, low.w, high.w));
}

template <typename T>
Vector<T, 4>
ceil(const Vector<T, 4>& a) {
    // Component-wise ceil.
    using std::ceil;
    return Vector<T, 4>(static_cast<T>(ceil(a.x)),
                        static_cast<T>(ceil(a.y)),
                        static_cast<T>(ceil(a.z)),
                        static_cast<T>(ceil(a.w)));
}

template <typename T>
Vector<T, 4>
floor(const Vector<T, 4>& a) {
    // Component-wise floor.
    using std::floor;
    return Vector<T, 4>(static_cast<T>(floor(a.x)),
                        static_cast<T>(floor(a.y)),
                        static_cast<T>(floor(a.z)),
                        static_cast<T>(floor(a.w)));
}

template <typename T>
Vector<T, 4>
abs(const Vector<T, 4>& v) {
    // Component-wise abs.
    using std::abs;
    return Vector<T, 4>(static_cast<T>(abs(v.x)),
                        static_cast<T>(abs(v.y)),
                        static_cast<T>(abs(v.z)),
                        static_cast<T>(abs(v.w)));
}

template <typename T>
Vector<T, 4>
cmin(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Branch-based component min.
    return Vector<T, 4>((a.x < b.x) ? a.x : b.x,
                        (a.y < b.y) ? a.y : b.y,
                        (a.z < b.z) ? a.z : b.z,
                        (a.w < b.w) ? a.w : b.w);
}

template <typename T>
Vector<T, 4>
cmax(const Vector<T, 4>& a, const Vector<T, 4>& b) {
    // Branch-based component max.
    return Vector<T, 4>((a.x > b.x) ? a.x : b.x,
                        (a.y > b.y) ? a.y : b.y,
                        (a.z > b.z) ? a.z : b.z,
                        (a.w > b.w) ? a.w : b.w);
}

} // namespace atlas::math
