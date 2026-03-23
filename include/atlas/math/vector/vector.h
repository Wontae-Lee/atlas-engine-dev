#pragma once
#include <atlas/math/vector/vector_expression.h>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace atlas {
namespace math {

    template <typename T, std::size_t N>
    class Vector : public VectorExpression<T, Vector<T, N>> {
        static_assert(N >= 1, "Vector dimension must be >= 1");
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Vector() noexcept;

        Vector(const Vector& other) noexcept = default;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(T s) noexcept;

        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == N)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Vector(Args... args) noexcept;

        ATLAS_HOST ATLAS_FORCE_INLINE
        Vector(std::initializer_list<T> list) noexcept;

        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
        Vector(const VectorExpression<T, Expression>& expr) noexcept;

        ~Vector() noexcept = default;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t index) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t index) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector&
        operator=(const Vector& rhs) noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T s) noexcept;

        template <typename... Args,
                  typename = std::enable_if_t<(sizeof...(Args) == N)
                                              && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
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

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        dot(const Vector& v) const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length_squared() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        length() const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        normalize() noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector
        normalized() const noexcept;

        template <typename To>
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<To, N>
        cast_to() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t index) const noexcept;

        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t index) noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        sum() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        avg() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        min() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        max() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        major_axis() const noexcept;

        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
        minor_axis() const noexcept;

    private:
        alignas(32) T _data[N];
    };

}

template <typename T, std::size_t N>
using Vector = atlas::math::Vector<T, N>;

}

#include <atlas/math/vector/vector.hpp>