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
     * @brief Specialized 3×3 matrix with named fields and small-matrix utilities.
     *
     * @details
     * This is a partial specialization of `Matrix<T, R, C>` for `R = 3` and `C = 3`.
     * It exposes elements through named scalar fields while also keeping a
     * contiguous row-major storage view:
     * - row 0: `m00 m01 m02`
     * - row 1: `m10 m11 m12`
     * - row 2: `m20 m21 m22`
     *
     * The named-field view and contiguous storage alias the same underlying
     * memory, so indexed access and field access stay consistent.
     *
     * The specialization provides efficient implementations for common 3×3 operations
     * (determinant, transpose, inverse, and solving 3×3 linear systems) while remaining
     * compatible with the expression-template system via:
     * `MatrixExpression<T, Matrix<T,3,3>>`.
     *
     * Element layout (row-major):
     * \f[
     * A =
     * \begin{bmatrix}
     * m_{00} & m_{01} & m_{02}\\
     * m_{10} & m_{11} & m_{12}\\
     * m_{20} & m_{21} & m_{22}
     * \end{bmatrix}
     * \f]
     *
     * ---
     *
     * @tparam T Scalar type (must be trivially copyable).
     *
     * @note
     * - Equality operators are exact element-wise comparisons.
     * - Inversion/solving routines use an `eps` threshold to determine invertibility.
     * - As with other small-matrix helpers, this is intended for tight math loops and GPU code.
     */
    template <typename T>
    class Matrix<T, 3, 3> : public MatrixExpression<T, Matrix<T, 3, 3>> {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    public:
        union {
            struct {
                T m00, m01, m02;
                T m10, m11, m12;
                T m20, m21, m22;
            };
            T _data[9];
        };
        /**
         * @brief Default constructor.
         *
         * @details
         * Initializes the matrix to an implementation-defined state (commonly zero).
         * See `matrix3x3.hpp` for exact behavior.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix() noexcept;
        /// @brief Copy constructor.
        constexpr Matrix(const Matrix&) noexcept = default;
        /**
         * @brief Constructs a matrix with all elements set to the same scalar.
         *
         * @param s Scalar assigned to all 9 elements.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Matrix(T s) noexcept;
        /**
         * @brief Constructs a matrix from explicit elements in row-major order.
         *
         * @param a00 Element (0,0).
         * @param a01 Element (0,1).
         * @param a02 Element (0,2).
         * @param a10 Element (1,0).
         * @param a11 Element (1,1).
         * @param a12 Element (1,2).
         * @param a20 Element (2,0).
         * @param a21 Element (2,1).
         * @param a22 Element (2,2).
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Matrix(T a00, T a01, T a02,
                                                             T a10, T a11, T a12,
                                                             T a20, T a21, T a22) noexcept;
        /**
         * @brief Constructs from an initializer list in row-major order.
         *
         * @details
         * Expected order:
         * `{m00, m01, m02, m10, m11, m12, m20, m21, m22}`.
         * Missing elements (if fewer than 9 values are provided) are filled in an
         * implementation-defined way (commonly zero). Extra values are typically ignored.
         *
         * @param list Initializer list of values.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(std::initializer_list<T> list) noexcept;
        /**
         * @brief Constructs from a matrix expression.
         *
         * @details
         * Evaluates the provided expression into this 3×3 matrix.
         *
         * @tparam Expression Expression type.
         * @param expr Expression to evaluate.
         */
        template <typename Expression>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Matrix(const MatrixExpression<T, Expression>& expr) noexcept;
        ~Matrix() noexcept = default;
        /// @brief Number of rows (returns 3).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        rows() noexcept;
        /// @brief Number of columns (returns 3).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        cols() noexcept;
        /// @brief Number of elements (returns 9).
        ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
        size() noexcept;
        /**
         * @brief Returns a pointer to contiguous storage (const).
         *
         * @return Pointer to the row-major contiguous storage view.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
        data() const noexcept;
        /**
         * @brief Returns a pointer to contiguous storage (mutable).
         *
         * @return Pointer to the row-major contiguous storage view.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
        data() noexcept;
        /**
         * @brief Flat element access (const) by linear index.
         *
         * @details
         * Row-major mapping:
         * - 0..2   → row 0 (m00,m01,m02)
         * - 3..5   → row 1 (m10,m11,m12)
         * - 6..8   → row 2 (m20,m21,m22)
         *
         * @param i Linear index in `[0,9)`.
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
         * @param i Linear index in `[0,9)`.
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
         * @param r Row index in `[0,3)`.
         * @param c Column index in `[0,3)`.
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
         * @param r Row index in `[0,3)`.
         * @param c Column index in `[0,3)`.
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
         * @brief Sets this matrix to the 3×3 identity.
         *
         * @details
         * \f[
         * I =
         * \begin{bmatrix}
         * 1&0&0\\
         * 0&1&0\\
         * 0&0&1
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
         * @param a02 Element (0,2).
         * @param a10 Element (1,0).
         * @param a11 Element (1,1).
         * @param a12 Element (1,2).
         * @param a20 Element (2,0).
         * @param a21 Element (2,1).
         * @param a22 Element (2,2).
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        set(T a00, T a01, T a02,
            T a10, T a11, T a12,
            T a20, T a21, T a22) noexcept;
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
         * @brief Computes the determinant of the 3×3 matrix.
         *
         * @details
         * Using cofactor expansion:
         * \f[
         * \det(A) =
         * m_{00}(m_{11}m_{22} - m_{12}m_{21})
         * - m_{01}(m_{10}m_{22} - m_{12}m_{20})
         * + m_{02}(m_{10}m_{21} - m_{11}m_{20}).
         * \f]
         *
         * @return Determinant value.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
        determinant() const noexcept;
        /**
         * @brief Computes the trace of the 3×3 matrix.
         *
         * @details
         * \f[
         * \mathrm{tr}(A) = m_{00} + m_{11} + m_{22}.
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
         * Swaps off-diagonal elements:
         * - `m01` ↔ `m10`
         * - `m02` ↔ `m20`
         * - `m12` ↔ `m21`
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
         * Computes the inverse via adjugate / determinant (or an equivalent method in the implementation):
         * \f[
         * A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A).
         * \f]
         *
         * @note
         * If the matrix is not invertible, behavior is implementation-defined.
         * Prefer `try_inverse()` when failure must be handled.
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        inverse() noexcept;
        /**
         * @brief Returns an inverted copy of this matrix.
         *
         * @return Inverse matrix (behavior is implementation-defined if non-invertible).
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
         * @brief Matrix-matrix multiplication (3×3 · 3×3).
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
         * @brief Matrix-vector multiplication (3×3 · 3).
         *
         * @param v Vector of length 3.
         * @return Result vector of length 3.
         */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
        mul(const Vector<T, 3>& v) const noexcept;
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
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
        solved(const Vector<T, 3>& b) const noexcept;
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
         */
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        solve(const Vector<T, 3>& b, Vector<T, 3>& x,
              T eps = std::numeric_limits<T>::epsilon()) const noexcept;
    };
    /**
     * @brief Returns the 3×3 identity matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    identity3x3() noexcept;
    /**
     * @brief Returns the 3×3 zero matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    zero3x3() noexcept;
    /**
     * @brief Returns the transpose of a 3×3 matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    transpose(const Matrix<T, 3, 3>& m) noexcept;
    /**
     * @brief Returns the determinant of a 3×3 matrix.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    determinant(const Matrix<T, 3, 3>& m) noexcept;
    /**
     * @brief Returns the inverse of a 3×3 matrix.
     *
     * @note
     * Behavior is implementation-defined if the matrix is non-invertible.
     * Prefer `try_inverse()` for failure handling.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    inverse(const Matrix<T, 3, 3>& m) noexcept;
    /** @brief Matrix addition (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator+(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    /** @brief Matrix subtraction (element-wise). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator-(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    /** @brief Scalar multiplication (matrix * scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(const Matrix<T, 3, 3>& a, T s);
    /** @brief Scalar multiplication (scalar * matrix). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(T s, const Matrix<T, 3, 3>& a);
    /** @brief Scalar division (matrix / scalar). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator/(const Matrix<T, 3, 3>& a, T s);
    /** @brief Matrix multiplication (3×3 · 3×3). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    operator*(const Matrix<T, 3, 3>& a, const Matrix<T, 3, 3>& b);
    /** @brief Matrix-vector multiplication (3×3 · 3). */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    operator*(const Matrix<T, 3, 3>& a, const Vector<T, 3>& v);
    /**
     * @brief Writes the matrix-vector product into an existing vector.
     */
    template <typename T>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Matrix<T, 3, 3>& matrix, const Vector<T, 3>& input, Vector<T, 3>& output) noexcept;
    /**
     * @brief Writes the matrix-vector product plus an offset into an existing vector.
     */
    template <typename T>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate_translate(const Matrix<T, 3, 3>& matrix,
                     const Vector<T, 3>& input,
                     const Vector<T, 3>& offset,
                     Vector<T, 3>& output) noexcept;
    /**
     * @brief Writes the matrix-vector product of an offset-subtracted vector.
     */
    template <typename T>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate_subtract(const Matrix<T, 3, 3>& matrix,
                    const Vector<T, 3>& input,
                    const Vector<T, 3>& offset,
                    Vector<T, 3>& output) noexcept;
    /**
     * @brief Solves \f$Ax=b\f$ for a 3×3 system and writes result to `x`.
     *
     * @return `true` if solved; otherwise `false`.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b, Vector<T, 3>& x,
          T eps = std::numeric_limits<T>::epsilon()) noexcept;
    /**
     * @brief Solves \f$Ax=b\f$ for a 3×3 system and returns the solution vector.
     *
     * @note
     * Behavior is implementation-defined if the matrix is non-invertible.
     * Prefer the overload returning `bool` for failure handling.
     */
    template <typename T>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector<T, 3>
    solve(const Matrix<T, 3, 3>& A, const Vector<T, 3>& b) noexcept;
} // namespace math
template <typename T>
using Matrix3x3  = math::Matrix<T, 3, 3>;
using Matrix3x3F = Matrix3x3<float>;
using Matrix3x3D = Matrix3x3<double>;
} // namespace atlas
#include <atlas/math/matrix/matrix3x3.hpp>
