#ifndef INCLUDE_ATLAS_MATH_QUATERNION_H
#define INCLUDE_ATLAS_MATH_QUATERNION_H

#include <atlas/math/matrix/matrix.h>
#include <atlas/math/vector/vector3.h>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace atlas {
namespace math {

    template <typename T>
    class Quaternion {
        static_assert(std::is_floating_point_v<T>, "Quaternion requires a floating-point type");

    public:
        T w, x, y, z;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion() noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion(T w_, T x_, T y_, T z_) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(std::initializer_list<T> list) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Quaternion(const Vector3<T>& axis, T radians) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Quaternion(T rx, T ry, T rz) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(const Matrix<T, 3, 3>& m) noexcept;

        Quaternion(const Quaternion&) noexcept = default;

        ~Quaternion() noexcept = default;

        Quaternion&
        operator=(const Quaternion&) noexcept = default;

        Quaternion&
        operator=(Quaternion&&) noexcept = default;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        from_axis_angle(const Vector3<T>& axis, T radians) noexcept;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        from_euler_xyz(T rx, T ry, T rz) noexcept;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        from_matrix3x3(const Matrix<T, 3, 3>& m) noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Quaternion& q) const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length_squared() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length() const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        normalize() noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        normalized() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        conjugate() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        inverse() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        is_identity(T eps = std::numeric_limits<T>::epsilon()) const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
        rotate(const Vector3<T>& v) const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
        to_matrix3x3() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
        to_matrix4x4() const noexcept;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        lerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

        ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
        slerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion<To>
        cast_to() const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        operator+(const Quaternion& q) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        operator-(const Quaternion& q) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        operator*(const Quaternion& q) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        operator*(T s) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
        operator/(T s) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
        operator+=(const Quaternion& q) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
        operator-=(const Quaternion& q) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
        operator*=(const Quaternion& q) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
        operator*=(T s) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
        operator/=(T s) noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Quaternion& q) const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Quaternion& q) const noexcept;
    };

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion<T>
    operator*(T s, const Quaternion<T>& q) noexcept;

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    operator*(const Quaternion<T>& q, const Vector3<T>& v) noexcept;

}

template <typename T>
using Quaternion = math::Quaternion<T>;

using QuaternionF = math::Quaternion<float>;

using QuaternionD = math::Quaternion<double>;

}

#include <atlas/math/quaternion.hpp>
#endif