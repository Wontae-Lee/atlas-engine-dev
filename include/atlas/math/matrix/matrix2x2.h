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
     * @brief Specialized 2×2 matrix with named fields and small-matrix utilities.
     *
     * @details
     * This is a partial specialization of `Matrix<T, R, C>` for `R = 2` and `C = 2`.
     * It stores elements as named scalars:
     * - `m00 m01`
     * - `m10 m11`
     *
     * This layout avoids dynamic indexing overhead and enables highly optimized small-matrix
     * operations such as determinant, transpose, inverse, and solving 2×2 linear systems.
     *
     * Element access follows row-major semantics:
     * \f[
     * \begin{bmatrix}
     * m_{00} & m_{01}\\
     * m_{10} & m_{11}
     * \end{bmatrix}
     * \f]
     *
     * The type participates in the matrix expression system by inheriting from
     * `MatrixExpression<T, Matrix<T,2,2>>`.
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality operators are exact (bitwise/element-wise). For floating-point comparisons,
     *   consider an epsilon-based check at a higher level.
     * - Inversion/solving routines use an `eps` threshold to decide invertibility.
     */
    template <typename T>
    class Matrix<T, 2, 2> : public MatrixExpression<T, Matrix<T, 2, 2>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        T m00, m01;
        T m10, m11;
        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the matrix to an implementation-defined state (commonly zero).
         * See `matrix2x2.hpp` for the exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix() noexcept;
        /// @brief Copy constructor.
        constexpr Matrix(const Matrix&) noexcept = default;
        /**
         * @brief Constructs a matrix with all elements set to the same scalar.
         *
         * @param s Scalar assigned to every element.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Matrix(T s) noexcept;
        /**
         * @brief Constructs a matrix from explicit elements in row-major order.
         *
         * @param a00 Element (0,0).
         * @param a01 Element (0,1).
         * @param a10 Element (1,0).
         * @param a11 Element (1,1).
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix(T a00, T a01, T a10, T a11) noexcept;
        /**
         * @brief Constructs from an initializer list in row-major order.
         *
         * @details
         * Expected order: `{m00, m01, m10, m11}`.
         * If fewer values are provided, missing entries are filled in an implementation-defined way
         * (commonly zero). Extra values are typically ignored (implementation-defined).
         *
         * @param list Initializer list.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
        /**
         * @brief Constructs from a matrix expression.
         *
         * @details
         * Evaluates the provided expression into this 2×2 matrix.
         *
         * @tparam Expression Expression type.
         * @param expr Expression to evaluate.
         */
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(const MatrixExpression<T, Expression>& expr) noexcept;
        ~Matrix() noexcept = default;
        /// @brief Number of rows (returns 2).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        rows() noexcept;
        /// @brief Number of columns (returns 2).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        cols() noexcept;
        /// @brief Number of elements (returns 4).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;
        /**
         * @brief Returns a pointer to the contiguous storage (const).
         *
         * @return Pointer to `m00` followed by `m01`, `m10`, `m11`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;
        /**
         * @brief Returns a pointer to the contiguous storage (mutable).
         *
         * @return Pointer to `m00` followed by `m01`, `m10`, `m11`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;
        /**
         * @brief Flat element access (const) by index.
         *
         * @details
         * Index mapping (row-major):
         * - 0 → m00
         * - 1 → m01
         * - 2 → m10
         * - 3 → m11
         *
         * @param i Linear index in `[0,4)`.
         * @return Reference to the selected element.
         *
         * @note
         * No bounds checking is implied.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T&
        operator[](std::size_t i) const noexcept;
        /**
         * @brief Flat element access (mutable) by index.
         *
         * @param i Linear index in `[0,4)`.
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
         * @param r Row index in `[0,2)`.
         * @param c Column index in `[0,2)`.
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
         * @param r Row index in `[0,2)`.
         * @param c Column index in `[0,2)`.
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
         * @brief Sets this matrix to the 2×2 identity.
         *
         * @details
         * \f[
         * \begin{bmatrix}
         * 1 & 0 \\
         * 0 & 1
         * \end{bmatrix}
         * \f]
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set_identity() noexcept;
        /**
         * @brief Sets all elements explicitly in row-major order.
         *
         * @param a00 Element (0,0).
         * @param a01 Element (0,1).
         * @param a10 Element (1,0).
         * @param a11 Element (1,1).
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T a00, T a01, T a10, T a11) noexcept;
        /**
         * @brief Adds a scalar to all elements in-place.
         *
         * @param s Scalar to add.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(T s) noexcept;
        /**
         * @brief Subtracts a scalar from all elements in-place.
         *
         * @param s Scalar to subtract.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        sub(T s) noexcept;
        /**
         * @brief Multiplies all elements by a scalar in-place.
         *
         * @param s Scalar multiplier.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        mul(T s) noexcept;
        /**
         * @brief Divides all elements by a scalar in-place.
         *
         * @param s Scalar divisor.
         *
         * @note
         * Division by zero behavior is implementation-defined.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        div(T s) noexcept;
        /**
         * @brief Adds another 2×2 matrix element-wise in-place.
         *
         * @param m Matrix to add.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        add(const Matrix& m) noexcept;
        /**
         * @brief Subtracts another 2×2 matrix element-wise in-place.
         *
         * @param m Matrix to subtract.
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
         *
         * @param other Matrix to compare.
         * @return `true` if all entries are exactly equal.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator==(const Matrix& other) const noexcept;
        /**
         * @brief Exact element-wise inequality.
         *
         * @param other Matrix to compare.
         * @return `true` if any entry differs.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        operator!=(const Matrix& other) const noexcept;
        /**
         * @brief Computes the determinant of the 2×2 matrix.
         *
         * @details
         * \f[
         *   \det(A) = m_{00}m_{11} - m_{01}m_{10}.
         * \f]
         *
         * @return Determinant value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        determinant() const noexcept;
        /**
         * @brief Computes the trace of the 2×2 matrix.
         *
         * @details
         * \f[
         *   \mathrm{tr}(A) = m_{00} + m_{11}.
         * \f]
         *
         * @return Trace value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        trace() const noexcept;
        /**
         * @brief Transposes this matrix in-place.
         *
         * @details
         * Swaps `m01` and `m10`.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        transpose() noexcept;
        /**
         * @brief Returns a transposed copy of this matrix.
         *
         * @return Transposed matrix.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        transposed() const noexcept;
        /**
         * @brief Inverts this matrix in-place (requires invertibility).
         *
         * @details
         * For \f$A=\begin{bmatrix}a&b\\c&d\end{bmatrix}\f$:
         * \f[
         *   A^{-1} = \frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}.
         * \f]
         *
         * @note
         * If the matrix is not invertible, behavior is implementation-defined.
         * Prefer `try_inverse()` when robustness is required.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        inverse() noexcept;
        /**
         * @brief Returns an inverted copy of this matrix.
         *
         * @return Inverse matrix (if invertible; otherwise implementation-defined).
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        inversed() const noexcept;
        /**
         * @brief Attempts to invert this matrix into `out`.
         *
         * @details
         * Computes the inverse if `|determinant| > eps` and writes it to `out`.
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
         * @param eps Threshold used to test `|det| > eps`.
         * @return `true` if invertible; otherwise `false`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        is_invertible(T eps = std::numeric_limits<T>::epsilon()) const noexcept;
        /**
         * @brief Matrix-matrix multiplication (2×2 · 2×2).
         *
         * @details
         * Computes the standard product:
         * \f[
         *   C = A \cdot R.
         * \f]
         *
         * @param r Right-hand matrix.
         * @return Product matrix.
         *
         * @note
         * This is distinct from element-wise multiplication.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix
        mul(const Matrix& r) const noexcept;
        /**
         * @brief Matrix-vector multiplication (2×2 · 2).
         *
         * @param v Vector of length 2.
         * @return Result vector of length 2.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
        mul(const Vector<T, 2>& v) const noexcept;
        /**
         * @brief Solves the linear system \f$Ax=b\f$ and returns the solution.
         *
         * @details
         * Returns the computed solution vector. If the system is ill-conditioned or non-invertible,
         * the returned value is implementation-defined. Prefer `solve(b,x,eps)` for a success flag.
         *
         * @param b Right-hand side vector.
         * @return Solution vector `x`.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
        solved(const Vector<T, 2>& b) const noexcept;
        /**
         * @brief Solves the linear system \f$Ax=b\f$ and writes the result to `x`.
         *
         * @details
         * Computes a solution if the matrix is invertible under `eps` and returns `true`.
         *
         * @param b Right-hand side vector.
         * @param x Output vector receiving the solution.
         * @param eps Invertibility threshold.
         * @return `true` if solved successfully; otherwise `false`.
         *
         * @note
         * Uses an invertibility test based on determinant magnitude.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        solve(const Vector<T, 2>& b, Vector<T, 2>& x,
              T eps = std::numeric_limits<T>::epsilon()) const noexcept;
    };
    /**
     * @brief Returns the 2×2 identity matrix.
     *
     * @tparam T Scalar type.
     * @return Identity matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    identity2x2() noexcept;
    /**
     * @brief Returns the 2×2 zero matrix.
     *
     * @tparam T Scalar type.
     * @return Zero matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    zero2x2() noexcept;
    /**
     * @brief Returns the transpose of a 2×2 matrix.
     *
     * @tparam T Scalar type.
     * @param m Input matrix.
     * @return Transposed matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    transpose(const Matrix<T, 2, 2>& m) noexcept;
    /**
     * @brief Returns the determinant of a 2×2 matrix.
     *
     * @tparam T Scalar type.
     * @param m Input matrix.
     * @return Determinant value.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    determinant(const Matrix<T, 2, 2>& m) noexcept;
    /**
     * @brief Returns the inverse of a 2×2 matrix.
     *
     * @tparam T Scalar type.
     * @param m Input matrix.
     * @return Inverse matrix (behavior is implementation-defined if non-invertible).
     *
     * @note
     * Prefer `try_inverse()` / `solve(..., eps)` for robustness.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    inverse(const Matrix<T, 2, 2>& m) noexcept;
    /**
     * @brief Matrix addition (element-wise).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator+(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b);
    /**
     * @brief Matrix subtraction (element-wise).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator-(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b);
    /**
     * @brief Scalar multiplication (matrix * scalar).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator*(const Matrix<T, 2, 2>& a, T s);
    /**
     * @brief Scalar multiplication (scalar * matrix).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator*(T s, const Matrix<T, 2, 2>& a);
    /**
     * @brief Scalar division (matrix / scalar).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator/(const Matrix<T, 2, 2>& a, T s);
    /**
     * @brief Matrix multiplication (2×2 · 2×2).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 2, 2>
    operator*(const Matrix<T, 2, 2>& a, const Matrix<T, 2, 2>& b);
    /**
     * @brief Matrix-vector multiplication (2×2 · 2).
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    operator*(const Matrix<T, 2, 2>& a, const Vector<T, 2>& v);
    /**
     * @brief Solves \f$Ax=b\f$ for a 2×2 system and writes result to `x`.
     *
     * @tparam T Scalar type.
     * @param A Coefficient matrix.
     * @param b Right-hand side.
     * @param x Output solution.
     * @param eps Invertibility threshold.
     * @return `true` if solved; otherwise `false`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b, Vector<T, 2>& x,
          T eps = std::numeric_limits<T>::epsilon()) noexcept;
    /**
     * @brief Solves \f$Ax=b\f$ for a 2×2 system and returns the solution vector.
     *
     * @tparam T Scalar type.
     * @param A Coefficient matrix.
     * @param b Right-hand side.
     * @return Solution vector (behavior is implementation-defined if non-invertible).
     *
     * @note
     * Prefer the overload returning `bool` for failure handling.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 2>
    solve(const Matrix<T, 2, 2>& A, const Vector<T, 2>& b) noexcept;
} // namespace math
template <typename T>
using Matrix2x2  = math::Matrix<T, 2, 2>;
using Matrix2x2F = Matrix2x2<float>;
using Matrix2x2D = Matrix2x2<double>;
} // namespace atlas
#include <atlas/math/matrix/matrix2x2.hpp>
