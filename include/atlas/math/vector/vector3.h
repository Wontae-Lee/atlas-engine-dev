#ifndef INCLUDE_ATLAS_MATH_VECTOR_VECTOR3_H
#define INCLUDE_ATLAS_MATH_VECTOR_VECTOR3_H
#include <atlas/math/vector/vector.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>
namespace atlas {
namespace math {
    template <typename T>
    class Vector<T, 3> : public VectorExpression<T, Vector<T, 3>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        T x, y, z;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector() noexcept;

        constexpr Vector(const Vector& v) noexcept = default;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Vector(T s) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Vector(T x_, T y_, T z_) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(std::initializer_list<T> list) noexcept;
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(const VectorExpression<T, Expression>& expr) noexcept;
        ~Vector() noexcept = default;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t i) noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t i) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t i) noexcept;
        Vector&
        operator=(const Vector& rhs) noexcept = default;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;
        template <typename... Args, typename = std::enable_if_t<(sizeof...(Args) == 3) && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_values(Args... args) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(const Vector& v) noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        max() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator+=(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator-=(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator*=(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator/=(T v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator+=(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator-=(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator*=(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator/=(const Vector& v) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Vector& other) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Vector& other) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        cross(const Vector& v) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length_squared() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        major_axis() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        minor_axis() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        normalize() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        normalized() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        reflected(const Vector& n) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        projected(const Vector& n) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector, Vector>
        tangential() const noexcept;
        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, 3>
        cast_to() const noexcept;
    };
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cross(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    dot(const Vector<T, 3>& a, const Vector<T, 3>& b) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    reflected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    projected(const Vector<T, 3>& v, const Vector<T, 3>& normal) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Vector<T, 3>, Vector<T, 3>>
    tangential(const Vector<T, 3>& normal) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(T a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a, T b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator+(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a, T b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(T a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator-(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Vector<T, 3>& a, T b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(T a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(const Vector<T, 3>& a, T b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(T a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator/(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    min(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    max(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    clamp(const Vector<T, 3>& v, const Vector<T, 3>& low, const Vector<T, 3>& high);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    ceil(const Vector<T, 3>& a);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    floor(const Vector<T, 3>& a);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    abs(const Vector<T, 3>& v);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cmin(const Vector<T, 3>& a, const Vector<T, 3>& b);

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    cmax(const Vector<T, 3>& a, const Vector<T, 3>& b);
}
template <typename T>
using Vector3  = math::Vector<T, 3>;
using Vector3F = Vector3<float>;
using Vector3D = Vector3<double>;
using Point3UI = Vector3<std::uint32_t>;
}
#include <atlas/math/vector/vector3.hpp>
#endif