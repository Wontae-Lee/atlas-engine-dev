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
    /**
     * @brief Specialized 4×4 matrix with named fields and small-matrix utilities.
     *
     * @details
     * This is a partial specialization of `Matrix<T, R, C>` for `R = 4` and `C = 4`.
     * It stores elements as named scalars (row-major):
     * - row 0: `m00 m01 m02 m03`
     * - row 1: `m10 m11 m12 m13`
     * - row 2: `m20 m21 m22 m23`
     * - row 3: `m30 m31 m32 m33`
     *
     * The 4×4 specialization is commonly used for affine and projective transforms in 3D
     * (homogeneous coordinates). It provides efficient implementations for trace,
     * determinant, transpose, inverse, matrix multiplication, and solving 4×4 linear systems,
     * while remaining compatible with the expression-template system via
     * `MatrixExpression<T, Matrix<T,4,4>>`.
     *
     * Element layout:
     * \f[
     * A =
     * \begin{bmatrix}
     * m_{00} & m_{01} & m_{02} & m_{03}\\
     * m_{10} & m_{11} & m_{12} & m_{13}\\
     * m_{20} & m_{21} & m_{22} & m_{23}\\
     * m_{30} & m_{31} & m_{32} & m_{33}
     * \end{bmatrix}
     * \f]
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality operators are exact element-wise comparisons.
     * - Inversion/solving use an `eps` threshold to determine invertibility.
     * - Many common 3D transforms are not full-rank; use `try_inverse()` if failure must be handled.
     */
    template <typename T>
    class Matrix<T, 4, 4> : public MatrixExpression<T, Matrix<T, 4, 4>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        T m00, m01, m02, m03;
        T m10, m11, m12, m13;
        T m20, m21, m22, m23;
        T m30, m31, m32, m33;
        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the matrix to an implementation-defined state (commonly zero).
         * See `matrix4x4.hpp` for exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix() noexcept;
        /// @brief Copy constructor.
        constexpr Matrix(const Matrix&) noexcept = default;
        /**
         * @brief Constructs a matrix with all elements set to the same scalar.
         *
         * @param s Scalar assigned to all 16 elements.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Matrix(T s) noexcept;
        /**
         * @brief Constructs a matrix from explicit elements in row-major order.
         *
         * @param a00 Element (0,0).
         * @param a01 Element (0,1).
         * @param a02 Element (0,2).
         * @param a03 Element (0,3).
         * @param a10 Element (1,0).
         * @param a11 Element (1,1).
         * @param a12 Element (1,2).
         * @param a13 Element (1,3).
         * @param a20 Element (2,0).
         * @param a21 Element (2,1).
         * @param a22 Element (2,2).
         * @param a23 Element (2,3).
         * @param a30 Element (3,0).
         * @param a31 Element (3,1).
         * @param a32 Element (3,2).
         * @param a33 Element (3,3).
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix(T a00, T a01, T a02, T a03,
                                                             T a10, T a11, T a12, T a13,
                                                             T a20, T a21, T a22, T a23,
                                                             T a30, T a31, T a32, T a33) noexcept;
        /**
         * @brief Constructs from an initializer list in row-major order.
         *
         * @details
         * Expected order:
         * `{m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33}`.
         * Missing elements (if fewer than 16 values are provided) are filled in an
         * implementation-defined way (commonly zero). Extra values are typically ignored.
         *
         * @param list Initializer list of values.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
        /**
         * @brief Constructs from a matrix expression.
         *
         * @details
         * Evaluates the provided expression into this 4×4 matrix.
         *
         * @tparam Expression Expression type.
         * @param expr Expression to evaluate.
         */
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(const MatrixExpression<T, Expression>& expr) noexcept;
        ~Matrix() noexcept = default;
        /// @brief Number of rows (returns 4).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        rows() noexcept;
        /// @brief Number of columns (returns 4).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        cols() noexcept;
        /// @brief Number of elements (returns 16).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;
        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to the first element (`m00`) followed by row-major order.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;
        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to the first element (`m00`) followed by row-major order.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;
        /**
         * @brief Flat element access (const) by linear index.
         *
         * @details
         * Row-major mapping:
         * - 0..3   → row 0 (m00..m03)
         * - 4..7   → row 1 (m10..m13)
         * - 8..11  → row 2 (m20..m23)
         * - 12..15 → row 3 (m30..m33)
         *
         * @param i Linear index in `[0,16)`.
         * @return Reference to the selected element.
         *
         * @note
         * No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;
        /**
         * @brief Flat element access (mutable) by linear index.
         *
         * @param i Linear index in `[0,16)`.
         * @return Reference to the selected element.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator[](std::size_t i) noexcept;
        /**
         * @brief 2D element access (const), intended as a safer accessor.
         *
         * @details
         * Semantically returns element `(r,c)`. Whether it checks bounds depends on the implementation.
         *
         * @param r Row index.
         * @param c Column index.
         * @return Reference to element `(r,c)`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        at(std::size_t r, std::size_t c) const noexcept;
        /**
         * @brief 2D element access (mutable), intended as a safer accessor.
         *
         * @param r Row index.
         * @param c Column index.
         * @return Reference to element `(r,c)`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        at(std::size_t r, std::size_t c) noexcept;
        /**
         * @brief 2D element access (const) by row and column.
         *
         * @param r Row index in `[0,4)`.
         * @param c Column index in `[0,4)`.
         * @return Reference to element `(r,c)`.
         *
         * @note
         * No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator()(std::size_t r, std::size_t c) const noexcept;
        /**
         * @brief 2D element access (mutable) by row and column.
         *
         * @param r Row index in `[0,4)`.
         * @param c Column index in `[0,4)`.
         * @return Reference to element `(r,c)`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T&
        operator()(std::size_t r, std::size_t c) noexcept;
        Matrix&
        operator=(const Matrix&) noexcept = default;
        /**
         * @brief Sets all elements to zero.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_zero() noexcept;
        /**
         * @brief Sets this matrix to the 4×4 identity.
         *
         * @details
         * \f[
         * I =
         * \begin{bmatrix}
         * 1&0&0&0\\
         * 0&1&0&0\\
         * 0&0&1&0\\
         * 0&0&0&1
         * \end{bmatrix}
         * \f]
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_identity() noexcept;
        /**
         * @brief Sets all elements explicitly in row-major order.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T a00, T a01, T a02, T a03,
            T a10, T a11, T a12, T a13,
            T a20, T a21, T a22, T a23,
            T a30, T a31, T a32, T a33) noexcept;
        /**
         * @brief Adds a scalar to all elements in-place.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T s) noexcept;
        /**
         * @brief Subtracts a scalar from all elements in-place.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T s) noexcept;
        /**
         * @brief Multiplies all elements by a scalar in-place.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T s) noexcept;
        /**
         * @brief Divides all elements by a scalar in-place.
         *
         * @note
         * Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T s) noexcept;
        /**
         * @brief Adds another matrix element-wise in-place.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Matrix& m) noexcept;
        /**
         * @brief Subtracts another matrix element-wise in-place.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(const Matrix& m) noexcept;
        /// @brief In-place scalar add.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator+=(T s) noexcept;
        /// @brief In-place scalar subtract.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator-=(T s) noexcept;
        /// @brief In-place scalar multiply.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator*=(T s) noexcept;
        /// @brief In-place scalar divide.
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator/=(T s) noexcept;
        /// @brief In-place matrix add (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator+=(const Matrix& m) noexcept;
        /// @brief In-place matrix subtract (element-wise).
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix&
        operator-=(const Matrix& m) noexcept;
        /**
         * @brief Exact element-wise equality.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Matrix& other) const noexcept;
        /**
         * @brief Exact element-wise inequality.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Matrix& other) const noexcept;
        /**
         * @brief Computes the trace of the 4×4 matrix.
         *
         * @details
         * \f[
         * \mathrm{tr}(A)=m_{00}+m_{11}+m_{22}+m_{33}.
         * \f]
         *
         * @return Trace value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        trace() const noexcept;
        /**
         * @brief Computes the determinant of the 4×4 matrix.
         *
         * @details
         * The implementation typically uses cofactor expansion or an equivalent optimized scheme.
         * Determinant is used for invertibility checks and inversion.
         *
         * @return Determinant value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        determinant() const noexcept;
        /**
         * @brief Transposes this matrix in-place.
         *
         * @details
         * Swaps the off-diagonal elements:
         * - `m01` ↔ `m10`, `m02` ↔ `m20`, `m03` ↔ `m30`
         * - `m12` ↔ `m21`, `m13` ↔ `m31`
         * - `m23` ↔ `m32`
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        transpose() noexcept;
        /**
         * @brief Returns a transposed copy of this matrix.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        transposed() const noexcept;
        /**
         * @brief Inverts this matrix in-place (requires invertibility).
         *
         * @details
         * Computes \f$A^{-1}\f$ using an implementation-defined method (commonly adjugate/determinant
         * or Gauss-Jordan elimination).
         *
         * @note
         * If non-invertible, behavior is implementation-defined. Prefer `try_inverse()`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        inverse() noexcept;
        /**
         * @brief Returns an inverted copy of this matrix.
         *
         * @note
         * Behavior is implementation-defined if the matrix is non-invertible.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        inversed() const noexcept;
        /**
         * @brief Attempts to invert this matrix into `out`.
         *
         * @details
         * Computes the inverse if the matrix is invertible under `eps` and writes it to `out`.
         *
         * @param out Output matrix receiving the inverse.
         * @param eps Invertibility threshold (defaults to machine epsilon).
         * @return `true` if inversion succeeded; otherwise `false`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        try_inverse(Matrix& out, T eps = std::numeric_limits<T>::epsilon()) const noexcept;
        /**
         * @brief Checks whether the matrix is invertible under an epsilon threshold.
         *
         * @param eps Threshold used to test invertibility (typically `|det| > eps`).
         * @return `true` if invertible; otherwise `false`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        is_invertible(T eps = std::numeric_limits<T>::epsilon()) const noexcept;
        /**
         * @brief Matrix-matrix multiplication (4×4 · 4×4).
         *
         * @details
         * Computes the standard product:
         * \f[
         *   C = A \cdot \text{rhs}.
         * \f]
         *
         * @param rhs Right-hand matrix.
         * @return Product matrix.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        mul(const Matrix& rhs) const noexcept;
        /**
         * @brief Matrix-vector multiplication (4×4 · 4).
         *
         * @param v Vector of length 4.
         * @return Result vector of length 4.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
        mul(const Vector<T, 4>& v) const noexcept;
        /**
         * @brief Solves the linear system \f$Ax=b\f$ and returns the solution.
         *
         * @note
         * Behavior is implementation-defined if the system is singular or ill-conditioned.
         * Prefer `solve(b, x, eps)` for a success flag.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
        solved(const Vector<T, 4>& b) const noexcept;
        /**
         * @brief Solves the linear system \f$Ax=b\f$ and writes the result to `x`.
         *
         * @param b Right-hand side vector.
         * @param x Output solution vector.
         * @param eps Invertibility threshold.
         * @return `true` if solved successfully; otherwise `false`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        solve(const Vector<T, 4>& b, Vector<T, 4>& x,
              T eps = std::numeric_limits<T>::epsilon()) const noexcept;
    };
    /** @brief Returns the 4×4 identity matrix. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    identity4x4() noexcept;
    /** @brief Returns the 4×4 zero matrix. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    zero4x4() noexcept;
    /** @brief Returns the transpose of a 4×4 matrix. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    transpose(const Matrix<T, 4, 4>& m) noexcept;
    /** @brief Returns the determinant of a 4×4 matrix. */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    determinant(const Matrix<T, 4, 4>& m) noexcept;
    /**
     * @brief Returns the inverse of a 4×4 matrix.
     *
     * @note
     * Behavior is implementation-defined if the matrix is non-invertible.
     * Prefer `try_inverse()` for failure handling.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    inverse(const Matrix<T, 4, 4>& m) noexcept;
    /** @brief Matrix addition (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator+(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b);
    /** @brief Matrix subtraction (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator-(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b);
    /** @brief Scalar multiplication (matrix * scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator*(const Matrix<T, 4, 4>& a, T s);
    /** @brief Scalar multiplication (scalar * matrix). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator*(T s, const Matrix<T, 4, 4>& a);
    /** @brief Scalar division (matrix / scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator/(const Matrix<T, 4, 4>& a, T s);
    /** @brief Matrix multiplication (4×4 · 4×4). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    operator*(const Matrix<T, 4, 4>& a, const Matrix<T, 4, 4>& b);
    /** @brief Matrix-vector multiplication (4×4 · 4). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    operator*(const Matrix<T, 4, 4>& a, const Vector<T, 4>& v);
    /**
     * @brief Solves \f$Ax=b\f$ for a 4×4 system and writes result to `x`.
     *
     * @return `true` if solved; otherwise `false`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Matrix<T, 4, 4>& A, const Vector<T, 4>& b, Vector<T, 4>& x,
          T eps = std::numeric_limits<T>::epsilon()) noexcept;
    /**
     * @brief Solves \f$Ax=b\f$ for a 4×4 system and returns the solution vector.
     *
     * @note
     * Behavior is implementation-defined if the matrix is non-invertible.
     * Prefer the overload returning `bool` for failure handling.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 4>
    solve(const Matrix<T, 4, 4>& A, const Vector<T, 4>& b) noexcept;
} // namespace math
template <typename T>
using Matrix4x4  = math::Matrix<T, 4, 4>;
using Matrix4x4F = Matrix4x4<float>;
using Matrix4x4D = Matrix4x4<double>;
} // namespace atlas
#include <atlas/math/matrix/matrix4x4.hpp>
