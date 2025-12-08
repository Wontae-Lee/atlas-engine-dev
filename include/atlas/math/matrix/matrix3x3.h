#pragma once
#include <atlas/math/matrix/matrix.h>
#include <atlas/math/vector/vector.h>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace atlas {
namespace math {
    template <typename T>
    class Matrix<T, 3, 3> : public MatrixExpression<T, Matrix<T, 3, 3>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        T m00, m01, m02;
        T m10, m11, m12;
        T m20, m21, m22;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix() noexcept;
        constexpr Matrix(const Matrix&) noexcept = default;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Matrix(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix(
            T a00, T a01, T a02,
            T a10, T a11, T a12,
            T a20, T a21, T a22) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(const MatrixExpression<T, Expression>& expr) noexcept;
        ~Matrix() noexcept = default;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        rows() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        cols() noexcept;
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
        at(std::size_t r, std::size_t c) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t r, std::size_t c) noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator()(std::size_t r, std::size_t c) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator()(std::size_t r, std::size_t c) noexcept;
        Matrix&
        operator=(const Matrix&) noexcept = default;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_identity() noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(
            T a00, T a01, T a02,
            T a10, T a11, T a12,
            T a20, T a21, T a22) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Matrix& m) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(const Matrix& m) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator+=(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator-=(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator*=(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator/=(T s) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator+=(const Matrix& m) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator-=(const Matrix& m) noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Matrix& other) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Matrix& other) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        determinant() const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        trace() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        transpose() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        transposed() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        inverse() noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        inversed() const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        try_inverse(Matrix& out, T eps = std::numeric_limits<T>::epsilon()) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        is_invertible(T eps = std::numeric_limits<T>::epsilon()) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        mul(const Matrix& rhs) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
        mul(const Vector<T, 3>& v) const noexcept;
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
        solved(const Vector<T, 3>& b) const noexcept;
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        solve(const Vector<T, 3>& b, Vector<T, 3>& x,
              T eps = std::numeric_limits<T>::epsilon()) const noexcept;
    };

    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    identity3x3() noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    zero3x3() noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    transpose(const Matrix<T, 3, 3>& m) noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    determinant(const Matrix<T, 3, 3>& m) noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    inverse(const Matrix<T, 3, 3>& m) noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator+(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator-(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(const Matrix<T, 3, 3>& a, T s);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(T s, const Matrix<T, 3, 3>& a);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator/(const Matrix<T, 3, 3>& a, T s);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Matrix<T, 3, 3>& a, const Vector<T, 3>& v);
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b, Vector<T, 3>& x,
          T eps = std::numeric_limits<T>::epsilon()) noexcept;
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE
        ATLAS_FORCE_INLINE
            Vector<T, 3>
            solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b) noexcept;
}

template <typename T>
using Matrix3x3  = math::Matrix<T, 3, 3>;
using Matrix3x3F = Matrix3x3<float>;
using Matrix3x3D = Matrix3x3<double>;
}

#include <atlas/math/matrix/matrix3x3.hpp>