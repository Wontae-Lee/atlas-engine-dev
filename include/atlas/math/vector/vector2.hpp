#pragma once
namespace atlas::math {
template <typename T>
constexpr
Vector<T, 2>::Vector() noexcept
    : x(T(0))
      , y(T(0)) {}

template <typename T>
constexpr
Vector<T, 2>::Vector(T s) noexcept
    : x(s)
      , y(s) {}

template <typename T>
constexpr
Vector<T, 2>::Vector(T x_, T y_) noexcept
    : x(x_)
      , y(y_) {}

template <typename T>
Vector<T, 2>::Vector(std::initializer_list<T> list) noexcept {
    const T* it = list.begin();
    x           = (it != list.end()) ? *it++ : T(0);
    y           = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
template <typename Expression>
Vector<T, 2>::Vector(const VectorExpression<T, Expression>& expr) noexcept {
    const Expression& e = expr();
    x                   = static_cast<T>(e[0]);
    y                   = static_cast<T>(e[1]);
}

template <typename T>
std::size_t
Vector<T, 2>::size() noexcept {
    return 2;
}

template <typename T>
const T*
Vector<T, 2>::data() const noexcept {
    return &x;
}

template <typename T>
T*
Vector<T, 2>::data() noexcept {
    return &x;
}

template <typename T>
const T&
Vector<T, 2>::operator[](std::size_t i) const noexcept {
    return (&x)[i];
}

template <typename T>
T&
Vector<T, 2>::operator[](std::size_t i) noexcept {
    return (&x)[i];
}

template <typename T>
const T&
Vector<T, 2>::at(std::size_t i) const noexcept {
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
    x = y = s;
}

template <typename T>
template <typename... Args, typename>
void
Vector<T, 2>::set_values(Args... args) noexcept {
    T tmp[2] = { static_cast<T>(args)... };
    x        = tmp[0];
    y        = tmp[1];
}

template <typename T>
void
Vector<T, 2>::set_zero() noexcept {
    x = y = T(0);
}

template <typename T>
void
Vector<T, 2>::add(T v) noexcept {
    x += v;
    y += v;
}

template <typename T>
void
Vector<T, 2>::sub(T v) noexcept {
    x -= v;
    y -= v;
}

template <typename T>
void
Vector<T, 2>::mul(T v) noexcept {
    x *= v;
    y *= v;
}

template <typename T>
void
Vector<T, 2>::div(T v) noexcept {
    const T inv = T(1) / v;
    x           *= inv;
    y           *= inv;
}

template <typename T>
void
Vector<T, 2>::add(const Vector& v) noexcept {
    x += v.x;
    y += v.y;
}

template <typename T>
void
Vector<T, 2>::sub(const Vector& v) noexcept {
    x -= v.x;
    y -= v.y;
}

template <typename T>
void
Vector<T, 2>::mul(const Vector& v) noexcept {
    x *= v.x;
    y *= v.y;
}

template <typename T>
void
Vector<T, 2>::div(const Vector& v) noexcept {
    x /= v.x;
    y /= v.y;
}

template <typename T>
T
Vector<T, 2>::min() const noexcept {
    return (x < y) ? x : y;
}

template <typename T>
T
Vector<T, 2>::max() const noexcept {
    return (x > y) ? x : y;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator+=(T v) noexcept {
    add(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator-=(T v) noexcept {
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator*=(T v) noexcept {
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator/=(T v) noexcept {
    div(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator+=(const Vector& v) noexcept {
    add(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator-=(const Vector& v) noexcept {
    sub(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator*=(const Vector& v) noexcept {
    mul(v);
    return *this;
}

template <typename T>
Vector<T, 2>&
Vector<T, 2>::operator/=(const Vector& v) noexcept {
    div(v);
    return *this;
}

template <typename T>
bool
Vector<T, 2>::operator==(const Vector& other) const noexcept {
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
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(y, v.y, x * v.x);
    } else {
        return x * v.x + y * v.y;
    }
}

template <typename T>
T
Vector<T, 2>::cross(const Vector& v) const noexcept {
    return x * v.y - y * v.x;
}

template <typename T>
T
Vector<T, 2>::length_squared() const noexcept {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fma(y, y, x * x);
    } else {
        return x * x + y * y;
    }
}

template <typename T>
T
Vector<T, 2>::length() const noexcept {
    using std::sqrt;
    return static_cast<T>(sqrt(length_squared()));
}

template <typename T>
std::size_t
Vector<T, 2>::major_axis() const noexcept {
    return (std::abs(x) >= std::abs(y)) ? 0u : 1u;
}

template <typename T>
std::size_t
Vector<T, 2>::minor_axis() const noexcept {
    return (std::abs(x) <= std::abs(y)) ? 0u : 1u;
}

template <typename T>
void
Vector<T, 2>::normalize() noexcept {
    const T ls = length_squared();
    if (ls == T(0)) return;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    x           *= inv;
    y           *= inv;
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::normalized() const noexcept {
    const T ls = length_squared();
    if (ls == T(0)) return *this;
    const T inv = T(1) / static_cast<T>(std::sqrt(static_cast<double>(ls)));
    return Vector<T, 2>(x * inv, y * inv);
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::reflected(const Vector& n) const noexcept {
    const T d = dot(n);
    return Vector<T, 2>(x - T(2) * d * n.x,
                        y - T(2) * d * n.y);
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::projected(const Vector& n) const noexcept {
    const T d = dot(n);
    return Vector<T, 2>(x - d * n.x,
                        y - d * n.y);
}

template <typename T>
Vector<T, 2>
Vector<T, 2>::tangential() const noexcept {
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
    return Vector<To, 2>(static_cast<To>(x), static_cast<To>(y));
}

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
    const T d = dot(v, normal);
    return Vector<T, 2>(v.x - T(2) * d * normal.x,
                        v.y - T(2) * d * normal.y);
}

template <typename T>
Vector<T, 2>
projected(const Vector<T, 2>& v, const Vector<T, 2>& normal) noexcept {
    const T d = dot(v, normal);
    return Vector<T, 2>(v.x - d * normal.x,
                        v.y - d * normal.y);
}

template <typename T>
Vector<T, 2>
operator+(const Vector<T, 2>& a) {
    return a;
}

template <typename T>
Vector<T, 2>
operator-(const Vector<T, 2>& a) {
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
    return Vector<T, 2>(a / b.x, a / b.y);
}

template <typename T>
Vector<T, 2>
operator/(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>(a.x / b.x, a.y / b.y);
}

template <typename T>
Vector<T, 2>
min(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>(std::min(a.x, b.x), std::min(a.y, b.y));
}

template <typename T>
Vector<T, 2>
max(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>(std::max(a.x, b.x), std::max(a.y, b.y));
}

template <typename T>
Vector<T, 2>
clamp(const Vector<T, 2>& v, const Vector<T, 2>& low, const Vector<T, 2>& high) {
    return Vector<T, 2>(std::clamp(v.x, low.x, high.x),
                        std::clamp(v.y, low.y, high.y));
}

template <typename T>
Vector<T, 2>
ceil(const Vector<T, 2>& a) {
    using std::ceil;
    return Vector<T, 2>(static_cast<T>(ceil(a.x)),
                        static_cast<T>(ceil(a.y)));
}

template <typename T>
Vector<T, 2>
floor(const Vector<T, 2>& a) {
    using std::floor;
    return Vector<T, 2>(static_cast<T>(floor(a.x)),
                        static_cast<T>(floor(a.y)));
}

template <typename T>
Vector<T, 2>
abs(const Vector<T, 2>& v) {
    using std::abs;
    return Vector<T, 2>(static_cast<T>(abs(v.x)),
                        static_cast<T>(abs(v.y)));
}

template <typename T>
Vector<T, 2>
cmin(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>((a.x < b.x) ? a.x : b.x,
                        (a.y < b.y) ? a.y : b.y);
}

template <typename T>
Vector<T, 2>
cmax(const Vector<T, 2>& a, const Vector<T, 2>& b) {
    return Vector<T, 2>((a.x > b.x) ? a.x : b.x,
                        (a.y > b.y) ? a.y : b.y);
}
}