#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <atlas/math/vector/vector.h>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
namespace atlas ::math {

template <typename T, std::size_t R, std::size_t C>
class Matrix : public MatrixExpression<T, Matrix<T, R, C>> {
    static_assert(R >= 1 && C >= 1, "Matrix dimensions must be >= 1");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Matrix() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(T s) noexcept;

    template <typename... Args,
              typename = std::enable_if_t<(sizeof...(Args) == R * C)
                                          && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(Args... args) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
    Matrix(const Matrix&) noexcept = default;
    Matrix&
    operator=(const Matrix&) noexcept
        = default;
    ~Matrix() noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows_static() noexcept { return R; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols_static() noexcept { return C; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size_static() noexcept { return R * C; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows() noexcept { return R; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols() noexcept { return C; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept { return R * C; }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    operator[](std::size_t i) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    operator[](std::size_t i) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    operator()(std::size_t r, std::size_t c) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    operator()(std::size_t r, std::size_t c) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    at(std::size_t r, std::size_t c) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    at(std::size_t r, std::size_t c) noexcept;

    template <typename E>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator=(const MatrixExpression<T, E>& expr) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(T s) noexcept;

    template <typename... Args,
              typename = std::enable_if_t<(sizeof...(Args) == R * C)
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
    add(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator+=(T v) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator-=(T v) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator*=(T v) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator/=(T v) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator+=(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator-=(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator*=(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator/=(const Matrix& m) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Matrix& other) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
    data_ptr() const noexcept { return _data; }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
    data_ptr() noexcept { return _data; }

private:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::size_t
    index(std::size_t r, std::size_t c) noexcept { return r * C + c; }

private:
    alignas(32) T _data[R * C];
};

template <typename T, std::size_t R, std::size_t C, std::size_t K>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, R, K>
matmul(const Matrix<T, R, C>& a, const Matrix<T, C, K>& b) noexcept;

template <typename T, std::size_t R, std::size_t C>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, R>
matmul(const Matrix<T, R, C>& a, const Vector<T, C>& x) noexcept;
}
namespace atlas {
template <typename T, std::size_t R, std::size_t C>
using Matrix = math::Matrix<T, R, C>;
}
#include <atlas/math/matrix/matrix.hpp>