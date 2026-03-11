#pragma once
#include <atlas/math/matrix/matrix_expression.h>
#include <atlas/math/vector/vector.h>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
namespace atlas ::math {
/**
 * @brief Fixed-size dense matrix with expression-template assignment support.
 *
 * @details
 * `Matrix<T, R, C>` is a statically sized dense matrix storing `R*C` elements in a contiguous
 * row-major array. It also participates in the matrix expression system by inheriting from
 * `MatrixExpression<T, Matrix<T, R, C>>`, enabling assignment from compatible expressions
 * without virtual dispatch.
 *
 * Storage layout is row-major:
 * \f[
 *   \text{index}(r,c) = r \cdot C + c
 * \f]
 * where `r ∈ [0, R)` and `c ∈ [0, C)`.
 *
 * ---
 *
 * @tparam T Scalar element type. Must be trivially copyable.
 * @tparam R Number of rows (compile-time constant, must be >= 1).
 * @tparam C Number of columns (compile-time constant, must be >= 1).
 *
 * @note
 * - The internal buffer is aligned to 32 bytes (`alignas(32)`) for SIMD-friendly access.
 * - Bounds checking is not implied for `operator[]` and `operator()`. Use `at()` if it performs checks
 *   in the implementation (header shows the same signature; behavior is implementation-defined).
 * - `operator==` is exact element-wise equality; for floating-point matrices prefer an epsilon test
 *   at a higher level.
 */
template <typename T, std::size_t R, std::size_t C>
class Matrix : public MatrixExpression<T, Matrix<T, R, C>> {
    static_assert(R >= 1 && C >= 1, "Matrix dimensions must be >= 1");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the matrix to an implementation-defined state (commonly zero-initialized).
     * Check `matrix.hpp` for the exact behavior.
     *
     * @note
     * Even if it is zero-initialized, prefer `set_zero()` for clarity in performance-critical code.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Matrix() noexcept;
    /**
     * @brief Constructs a matrix with all entries set to the same scalar.
     *
     * @param s Scalar value assigned to all `R*C` elements.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(T s) noexcept;
    /**
     * @brief Constructs a matrix from exactly `R*C` values (row-major order).
     *
     * @details
     * Accepts a variadic list of values convertible to `T`. The number of arguments must be
     * exactly `R*C`. Values are interpreted in row-major order:
     * \f[
     *   (0,0),(0,1),\dots,(0,C-1),(1,0),\dots,(R-1,C-1).
     * \f]
     *
     * ---
     *
     * @tparam Args Argument pack types (must be convertible to `T`).
     * @param args  Exactly `R*C` values.
     */
    template <typename... Args,
              typename = std::enable_if_t<(sizeof...(Args) == R * C)
                                          && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(Args... args) noexcept;
    /**
     * @brief Constructs a matrix from an initializer list (host only).
     *
     * @details
     * The list is interpreted in row-major order. If the list provides fewer than `R*C` values,
     * the remaining elements are initialized in an implementation-defined way (commonly zero).
     * If more values are provided, extras are typically ignored (implementation-defined).
     *
     * @param list Initializer list of values.
     *
     * @note
     * This constructor is marked `ATLAS_HOST`, so it is not available in device code.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
    Matrix(const Matrix&) noexcept = default;
    Matrix&
    operator=(const Matrix&) noexcept = default;
    ~Matrix() noexcept                = default;
    /// @brief Compile-time number of rows.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows_static() noexcept { return R; }
    /// @brief Compile-time number of columns.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols_static() noexcept { return C; }
    /// @brief Compile-time total number of elements.
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size_static() noexcept { return R * C; }
    /// @brief Runtime number of rows (returns `R`).
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows() noexcept { return R; }
    /// @brief Runtime number of columns (returns `C`).
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols() noexcept { return C; }
    /// @brief Runtime total number of elements (returns `R*C`).
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept { return R * C; }
    /**
     * @brief Flat element access (const) by linear index.
     *
     * @param i Linear index in `[0, R*C)`.
     * @return Reference to element `i`.
     *
     * @note
     * No bounds checking is implied.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    operator[](std::size_t i) const noexcept;
    /**
     * @brief Flat element access (mutable) by linear index.
     *
     * @param i Linear index in `[0, R*C)`.
     * @return Reference to element `i`.
     *
     * @note
     * No bounds checking is implied.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    operator[](std::size_t i) noexcept;
    /**
     * @brief 2D element access (const) by row and column.
     *
     * @param r Row index in `[0, R)`.
     * @param c Column index in `[0, C)`.
     * @return Reference to element at `(r,c)`.
     *
     * @note
     * No bounds checking is implied.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    operator()(std::size_t r, std::size_t c) const noexcept;
    /**
     * @brief 2D element access (mutable) by row and column.
     *
     * @param r Row index in `[0, R)`.
     * @param c Column index in `[0, C)`.
     * @return Reference to element at `(r,c)`.
     *
     * @note
     * No bounds checking is implied.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    operator()(std::size_t r, std::size_t c) noexcept;
    /**
     * @brief Bounds-checked 2D element access (const), if implemented as such.
     *
     * @details
     * Semantically equivalent to `operator()(r,c)` but intended for safer access.
     * Check the implementation for whether it actually performs checks.
     *
     * @param r Row index.
     * @param c Column index.
     * @return Reference to element at `(r,c)`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
    at(std::size_t r, std::size_t c) const noexcept;
    /**
     * @brief Bounds-checked 2D element access (mutable), if implemented as such.
     *
     * @param r Row index.
     * @param c Column index.
     * @return Reference to element at `(r,c)`.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
    at(std::size_t r, std::size_t c) noexcept;
    /**
     * @brief Assigns from a matrix expression.
     *
     * @details
     * Evaluates `expr` element-wise into this matrix. Expression templates allow composing
     * operations without creating intermediate temporaries (depending on the expression system).
     *
     * ---
     *
     * @tparam E Expression type.
     * @param expr Expression to evaluate.
     * @return Reference to `*this`.
     *
     * @note
     * Dimensions must be compatible; mismatches are typically compile-time errors in the expression system.
     */
    template <typename E>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator=(const MatrixExpression<T, E>& expr) noexcept;
    /**
     * @brief Sets all elements to a scalar value.
     *
     * @param s Scalar value assigned to all entries.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(T s) noexcept;
    /**
     * @brief Sets matrix entries from exactly `R*C` values (row-major order).
     *
     * @tparam Args Argument pack types convertible to `T`.
     * @param args Exactly `R*C` values in row-major order.
     */
    template <typename... Args,
              typename = std::enable_if_t<(sizeof...(Args) == R * C)
                                          && (std::conjunction_v<std::is_convertible<Args, T>...>)>>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_values(Args... args) noexcept;
    /**
     * @brief Sets all elements to zero.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept;
    /**
     * @brief Adds a scalar to all elements in-place.
     *
     * @param v Scalar value added to every element.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(T v) noexcept;
    /**
     * @brief Subtracts a scalar from all elements in-place.
     *
     * @param v Scalar value subtracted from every element.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(T v) noexcept;
    /**
     * @brief Multiplies all elements by a scalar in-place.
     *
     * @param v Scalar multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(T v) noexcept;
    /**
     * @brief Divides all elements by a scalar in-place.
     *
     * @param v Scalar divisor.
     *
     * @note
     * Division by zero behavior is implementation-defined.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(T v) noexcept;
    /**
     * @brief Adds another matrix element-wise in-place.
     *
     * @param m Matrix to add.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Matrix& m) noexcept;
    /**
     * @brief Subtracts another matrix element-wise in-place.
     *
     * @param m Matrix to subtract.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Matrix& m) noexcept;
    /**
     * @brief Multiplies by another matrix element-wise in-place (Hadamard product).
     *
     * @details
     * This is **not** matrix multiplication. Each element is multiplied by the corresponding
     * element in `m`.
     *
     * @param m Matrix of the same shape.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const Matrix& m) noexcept;
    /**
     * @brief Divides by another matrix element-wise in-place.
     *
     * @details
     * This is **not** solving a linear system. Each element is divided by the corresponding
     * element in `m`.
     *
     * @param m Matrix of the same shape.
     *
     * @note
     * Division by zero behavior is implementation-defined.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const Matrix& m) noexcept;
    /// @brief In-place scalar add.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator+=(T v) noexcept;
    /// @brief In-place scalar subtract.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator-=(T v) noexcept;
    /// @brief In-place scalar multiply.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator*=(T v) noexcept;
    /// @brief In-place scalar divide.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator/=(T v) noexcept;
    /// @brief In-place matrix add (element-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator+=(const Matrix& m) noexcept;
    /// @brief In-place matrix subtract (element-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator-=(const Matrix& m) noexcept;
    /// @brief In-place matrix multiply (element-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator*=(const Matrix& m) noexcept;
    /// @brief In-place matrix divide (element-wise).
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
    operator/=(const Matrix& m) noexcept;
    /**
     * @brief Exact element-wise equality.
     *
     * @param other Matrix to compare.
     * @return `true` if all elements are exactly equal; otherwise `false`.
     *
     * @note
     * For floating-point types, exact equality is usually not appropriate.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Matrix& other) const noexcept;
    /**
     * @brief Returns a pointer to the underlying contiguous storage (const).
     *
     * @return Pointer to the first element (row-major).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
    data_ptr() const noexcept { return _data; }
    /**
     * @brief Returns a pointer to the underlying contiguous storage (mutable).
     *
     * @return Pointer to the first element (row-major).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
    data_ptr() noexcept { return _data; }

private:
    /**
     * @brief Computes the row-major linear index for `(r,c)`.
     *
     * @param r Row index.
     * @param c Column index.
     * @return Linear index `r*C + c`.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::size_t
    index(std::size_t r, std::size_t c) noexcept { return r * C + c; }

private:
    /// @brief Row-major storage buffer (aligned for SIMD-friendly access).
    alignas(32) T _data[R * C];
};
/**
 * @brief Matrix-matrix multiplication.
 *
 * @details
 * Computes \f$C = A \cdot B\f$ where:
 * - `A` is `R×C`
 * - `B` is `C×K`
 * - result is `R×K`
 *
 * ---
 *
 * @tparam T Scalar type.
 * @tparam R Rows of A and result.
 * @tparam C Columns of A / rows of B.
 * @tparam K Columns of B and result.
 * @param a Left operand matrix.
 * @param b Right operand matrix.
 * @return Product matrix `R×K`.
 *
 * @note
 * This is a separate free function and is distinct from element-wise `mul(const Matrix&)`.
 */
template <typename T, std::size_t R, std::size_t C, std::size_t K>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, R, K>
matmul(const Matrix<T, R, C>& a, const Matrix<T, C, K>& b) noexcept;
/**
 * @brief Matrix-vector multiplication.
 *
 * @details
 * Computes \f$y = A \cdot x\f$ where:
 * - `A` is `R×C`
 * - `x` is length `C`
 * - result is length `R`
 *
 * ---
 *
 * @tparam T Scalar type.
 * @tparam R Rows of A / length of result.
 * @tparam C Columns of A / length of input vector.
 * @param a Matrix operand.
 * @param x Vector operand.
 * @return Result vector of length `R`.
 */
template <typename T, std::size_t R, std::size_t C>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, R>
matmul(const Matrix<T, R, C>& a, const Vector<T, C>& x) noexcept;
} // namespace math
namespace atlas {
template <typename T, std::size_t R, std::size_t C>
using Matrix = math::Matrix<T, R, C>;
} // namespace atlas
#include <atlas/math/matrix/matrix.hpp>
