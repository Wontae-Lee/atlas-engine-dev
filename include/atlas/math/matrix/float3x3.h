/**
 * @file   float3x3.h
 * @brief  3x3 single-precision floating point matrix type Float3x3 and related free functions.
 *
 * Float3x3 is a SIMD-independent, row-major 3x3 matrix type designed to
 * work on both CPU (Host) and CUDA GPU (Device). It is commonly used to
 * represent linear transforms (rotation, scale, shear) applied to Float3
 * vectors, and provides determinant/trace/transpose/inverse and a small
 * linear-solve utility built on the cofactor (adjugate) method.
 * All member and free functions explicitly declare their compilation
 * target (host/device) via ATLAS_ALL_DEVICE / ATLAS_HOST macros, and
 * force inlining via ATLAS_FORCE_INLINE to eliminate call overhead.
 *
 * ### Design principles
 * - **Value type**: holds nine floats directly (via an anonymous union
 *   with a flat array), not pointers/references.
 * - **noexcept guarantee**: no operation throws exceptions.
 * - **FMA usage**: determinant / mul and other accumulation operations
 *   use `std::fma` to minimize rounding error.
 */

#pragma once

#include <atlas/core/macros.h>       /**< ATLAS_ALL_DEVICE, ATLAS_HOST, ATLAS_FORCE_INLINE, ATLAS_NODISCARD, etc. */
#include <atlas/math/vector/float3.h> /**< Float3 type used for matrix-vector products and rotation helpers */

#include <cmath>           // std::fma, std::abs
#include <cstddef>         // std::size_t
#include <initializer_list>
#include <limits>          // std::numeric_limits
#include <utility>         // std::swap

namespace atlas {

/**
 * @class Float3x3
 * @brief 3x3 single-precision floating point matrix, stored row-major.
 *
 * A general-purpose math type representing a 3x3 linear transform.
 * The nine components are laid out contiguously via an anonymous union
 * (`m00..m22` alongside a flat `_data[9]` array), so a pointer obtained
 * via `data()` can be passed directly to GPU kernels or C libraries.
 *
 * @code{.cpp}
 * atlas::Float3x3 m = atlas::identity3x3();
 * atlas::Float3   v(1.f, 2.f, 3.f);
 * atlas::Float3   r = m * v;               // matrix-vector product
 * float           d = m.determinant();
 * @endcode
 */
class Float3x3 {
public:
    union {
        struct {
            float m00, m01, m02; /**< First row */
            float m10, m11, m12; /**< Second row */
            float m20, m21, m22; /**< Third row */
        };
        float _data[9]; /**< Flat row-major view of the same nine components */
    };

    /** @brief Default constructor. Initializes all components to 0. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3x3() noexcept
        : m00(0.0f)
        , m01(0.0f)
        , m02(0.0f)
        , m10(0.0f)
        , m11(0.0f)
        , m12(0.0f)
        , m20(0.0f)
        , m21(0.0f)
        , m22(0.0f) { }

    /** @brief Copy constructor (compiler-generated default). */
    constexpr Float3x3(const Float3x3&) noexcept = default;

    /**
     * @brief Scalar-diagonal constructor. `Float3x3(2.f)` -> `diag(2, 2, 2)`.
     * @param s  Value assigned to the diagonal; off-diagonal entries are 0
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Float3x3(const float s) noexcept
        : m00(s)
        , m01(0.0f)
        , m02(0.0f)
        , m10(0.0f)
        , m11(s)
        , m12(0.0f)
        , m20(0.0f)
        , m21(0.0f)
        , m22(s) { }

    /**
     * @brief Constructor specifying each of the nine components individually (row-major).
     * @param a00 Row 0, column 0
     * @param a01 Row 0, column 1
     * @param a02 Row 0, column 2
     * @param a10 Row 1, column 0
     * @param a11 Row 1, column 1
     * @param a12 Row 1, column 2
     * @param a20 Row 2, column 0
     * @param a21 Row 2, column 1
     * @param a22 Row 2, column 2
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3x3(const float a00, const float a01, const float a02,
                                                           const float a10, const float a11, const float a12,
                                                           const float a20, const float a21, const float a22) noexcept
        : m00(a00)
        , m01(a01)
        , m02(a02)
        , m10(a10)
        , m11(a11)
        , m12(a12)
        , m20(a20)
        , m21(a21)
        , m22(a22) { }

    /**
     * @brief Initializer-list constructor (host only).
     *
     * Fills the nine components in row-major order. Missing elements
     * are filled with 0.
     *
     * @param list  Initializer list of float values (up to 9 elements)
     * @note  Not available in CUDA device code (ATLAS_HOST only).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Float3x3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        for (int i = 0; i < 9; ++i) {
            _data[i] = (it != list.end()) ? *it++ : 0.0f;
        }
    }

    /** @brief Destructor (trivial). */
    ~Float3x3() noexcept = default;

    /** @brief Copy assignment operator (compiler-generated default). */
    Float3x3&
    operator=(const Float3x3&) noexcept = default;

    /**
     * @brief Returns the number of rows.
     * @return Always 3
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows() noexcept {
        return 3;
    }

    /**
     * @brief Returns the number of columns.
     * @return Always 3
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols() noexcept {
        return 3;
    }

    /**
     * @brief Returns the total number of components.
     * @return Always 9
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 9;
    }

    /**
     * @brief Returns a const pointer to the first component (m00).
     *
     * Since the nine components are contiguous in memory (row-major),
     * the resulting pointer can be used as a flat array.
     * @return &_data[0] (const)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return _data;
    }

    /**
     * @brief Returns a mutable pointer to the first component (m00).
     * @return &_data[0] (mutable)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return _data;
    }

    /**
     * @brief Accesses a component by flat row-major index (read-only).
     * @param i  Flat index in [0, 9)
     * @return  Const reference to the i-th component
     * @warning No bounds checking. Undefined behavior if i >= 9.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return _data[i];
    }

    /**
     * @brief Accesses a component by flat row-major index (mutable).
     * @param i  Flat index in [0, 9)
     * @return  Reference to the i-th component
     * @warning No bounds checking. Undefined behavior if i >= 9.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return _data[i];
    }

    /**
     * @brief Accesses a component by (row, column) index (read-only).
     * @param r  Row index (0-2)
     * @param c  Column index (0-2)
     * @return   Const reference to m[r][c]
     * @warning  No bounds checking. Undefined behavior if r >= 3 or c >= 3.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Accesses a component by (row, column) index (mutable).
     * @param r  Row index (0-2)
     * @param c  Column index (0-2)
     * @return   Reference to m[r][c]
     * @warning  No bounds checking. Undefined behavior if r >= 3 or c >= 3.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Same unchecked accessor as at() (const), via function-call syntax.
     * @param r  Row index (0-2)
     * @param c  Column index (0-2)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator()(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Same unchecked accessor as at() (mutable), via function-call syntax.
     * @param r  Row index (0-2)
     * @param c  Column index (0-2)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator()(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    /** @brief Sets all nine components to 0. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        for (int i = 0; i < 9; ++i) _data[i] = 0.0f;
    }

    /** @brief Resets this matrix to the 3x3 identity matrix. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_identity() noexcept {
        set(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    /**
     * @brief Sets all nine components individually (row-major).
     * @param a00 Row 0, column 0
     * @param a01 Row 0, column 1
     * @param a02 Row 0, column 2
     * @param a10 Row 1, column 0
     * @param a11 Row 1, column 1
     * @param a12 Row 1, column 2
     * @param a20 Row 2, column 0
     * @param a21 Row 2, column 1
     * @param a22 Row 2, column 2
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(const float a00, const float a01, const float a02,
        const float a10, const float a11, const float a12,
        const float a20, const float a21, const float a22) noexcept {
        m00 = a00;
        m01 = a01;
        m02 = a02;
        m10 = a10;
        m11 = a11;
        m12 = a12;
        m20 = a20;
        m21 = a21;
        m22 = a22;
    }

    /** @brief Adds a scalar to all nine components. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += s;
    }

    /** @brief Subtracts a scalar from all nine components. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= s;
    }

    /** @brief Multiplies all nine components by a scalar. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] *= s;
    }

    /**
     * @brief Divides all nine components by a scalar.
     *
     * Pre-computes 1/s and applies it as a multiplication, reducing the
     * number of divisions.
     * @param s  If 0, the result becomes infinity (inf); callers must check.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float s) noexcept {
        const float inv = 1.0f / s;
        for (int i = 0; i < 9; ++i) _data[i] *= inv;
    }

    /** @brief Component-wise matrix addition. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += m._data[i];
    }

    /** @brief Component-wise matrix subtraction. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= m._data[i];
    }

    /** @brief Scalar addition assignment. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const float s) noexcept {
        add(s);
        return *this;
    }

    /** @brief Scalar subtraction assignment. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const float s) noexcept {
        sub(s);
        return *this;
    }

    /** @brief Scalar multiplication assignment. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator*=(const float s) noexcept {
        mul(s);
        return *this;
    }

    /** @brief Scalar division assignment. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator/=(const float s) noexcept {
        div(s);
        return *this;
    }

    /** @brief Matrix addition assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const Float3x3& m) noexcept {
        add(m);
        return *this;
    }

    /** @brief Matrix subtraction assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const Float3x3& m) noexcept {
        sub(m);
        return *this;
    }

    /** @brief Checks whether two matrices are exactly equal (bit-exact), component-wise. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Float3x3& o) const noexcept {
        for (int i = 0; i < 9; ++i) {
            if (_data[i] != o._data[i]) return false;
        }
        return true;
    }

    /** @brief Checks whether two matrices are different. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Float3x3& o) const noexcept {
        return !(*this == o);
    }

    /**
     * @brief Computes the determinant via cofactor expansion along the first row.
     *
     * Uses two FMA (Fused Multiply-Add) calls to reduce rounding error.
     *
     * @return det(*this)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    determinant() const noexcept {
        const float ei_fh = m11 * m22 - m12 * m21;
        const float di_fg = m10 * m22 - m12 * m20;
        const float dh_eg = m10 * m21 - m11 * m20;
        return std::fma(m00, ei_fh, std::fma(-m01, di_fg, m02 * dh_eg));
    }

    /**
     * @brief Returns the trace (sum of diagonal components).
     * @return m00 + m11 + m22
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    trace() const noexcept {
        return m00 + m11 + m22;
    }

    /** @brief Transposes the matrix in place by swapping off-diagonal pairs. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    transpose() noexcept {
        std::swap(m01, m10);
        std::swap(m02, m20);
        std::swap(m12, m21);
    }

    /**
     * @brief Returns a new transposed matrix without modifying *this.
     * @return Transpose of *this
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    transposed() const noexcept {
        return Float3x3(m00, m10, m20, m01, m11, m21, m02, m12, m22);
    }

    /**
     * @brief Inverts the matrix in place via the cofactor (adjugate) method.
     *
     * @warning Undefined behavior if the matrix is singular (determinant
     *          is 0); use try_inverse() when singularity is possible.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    inverse() noexcept {
        Float3x3 out;
        cofactor_inverse(out, determinant());
        *this = out;
    }

    /**
     * @brief Returns a new inverted matrix without modifying *this.
     * @return Inverse of *this
     * @warning Undefined behavior if the matrix is singular; use
     *          try_inverse() when singularity is possible.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    inversed() const noexcept {
        Float3x3 out = *this;
        out.inverse();
        return out;
    }

    /**
     * @brief Attempts to invert the matrix, failing gracefully if singular.
     *
     * @param[out] out  Receives the inverse on success (untouched on failure)
     * @param[in]  eps  Determinant magnitude below which the matrix is
     *                  treated as singular (default: float epsilon)
     * @return     true on success, false if |determinant()| <= eps
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    try_inverse(Float3x3& out, const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        const float det = determinant();
        if (std::abs(det) <= eps) return false;
        cofactor_inverse(out, det);
        return true;
    }

    /**
     * @brief Checks whether the matrix is invertible (non-singular).
     * @param eps  Determinant magnitude threshold (default: float epsilon)
     * @return     true if |determinant()| > eps
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_invertible(const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        return std::abs(determinant()) > eps;
    }

    /**
     * @brief Computes the matrix product *this * r.
     *
     * Each output component uses two FMA calls to reduce rounding error.
     *
     * @param r  Right-hand-side matrix
     * @return   *this * r
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    mul(const Float3x3& r) const noexcept {
        const Float3x3& a = *this;
        Float3x3 out;
        out.m00 = std::fma(a.m01, r.m10, std::fma(a.m02, r.m20, a.m00 * r.m00));
        out.m01 = std::fma(a.m01, r.m11, std::fma(a.m02, r.m21, a.m00 * r.m01));
        out.m02 = std::fma(a.m01, r.m12, std::fma(a.m02, r.m22, a.m00 * r.m02));
        out.m10 = std::fma(a.m11, r.m10, std::fma(a.m12, r.m20, a.m10 * r.m00));
        out.m11 = std::fma(a.m11, r.m11, std::fma(a.m12, r.m21, a.m10 * r.m01));
        out.m12 = std::fma(a.m11, r.m12, std::fma(a.m12, r.m22, a.m10 * r.m02));
        out.m20 = std::fma(a.m21, r.m10, std::fma(a.m22, r.m20, a.m20 * r.m00));
        out.m21 = std::fma(a.m21, r.m11, std::fma(a.m22, r.m21, a.m20 * r.m01));
        out.m22 = std::fma(a.m21, r.m12, std::fma(a.m22, r.m22, a.m20 * r.m02));
        return out;
    }

    /**
     * @brief Computes the matrix-vector product *this * v.
     *
     * Each output component uses two FMA calls to reduce rounding error.
     *
     * @param v  Column vector to transform
     * @return   *this * v
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    mul(const Float3& v) const noexcept {
        return Float3(std::fma(m01, v.y, std::fma(m02, v.z, m00 * v.x)),
                      std::fma(m11, v.y, std::fma(m12, v.z, m10 * v.x)),
                      std::fma(m21, v.y, std::fma(m22, v.z, m20 * v.x)));
    }

    /**
     * @brief Solves the linear system *this * x = b via explicit inversion.
     *
     * @param b  Right-hand-side vector
     * @return   x = inverse(*this) * b
     * @warning  Undefined behavior if the matrix is singular; use solve()
     *           when singularity is possible.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    solved(const Float3& b) const noexcept {
        return inversed().mul(b);
    }

    /**
     * @brief Attempts to solve the linear system *this * x = b, failing
     *        gracefully if the matrix is singular.
     *
     * @param[in]  b    Right-hand-side vector
     * @param[out] x    Receives the solution on success (untouched on failure)
     * @param[in]  eps  Determinant magnitude threshold (default: float epsilon)
     * @return     true on success, false if the matrix is singular
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    solve(const Float3& b, Float3& x,
          const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        Float3x3 inv;
        if (!try_inverse(inv, eps)) return false;
        x = inv.mul(b);
        return true;
    }

private:
    /**
     * @brief Computes the cofactor (adjugate) inverse given a precomputed determinant.
     *
     * @param[out] out  Receives the inverse matrix
     * @param[in]  det  Precomputed determinant of *this (must be non-zero)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    cofactor_inverse(Float3x3& out, const float det) const noexcept {
        const float c00 = (m11 * m22 - m12 * m21);
        const float c01 = -(m10 * m22 - m12 * m20);
        const float c02 = (m10 * m21 - m11 * m20);
        const float c10 = -(m01 * m22 - m02 * m21);
        const float c11 = (m00 * m22 - m02 * m20);
        const float c12 = -(m00 * m21 - m01 * m20);
        const float c20 = (m01 * m12 - m02 * m11);
        const float c21 = -(m00 * m12 - m02 * m10);
        const float c22 = (m00 * m11 - m01 * m10);

        const float inv_det = 1.0f / det;
        out.set(c00 * inv_det, c10 * inv_det, c20 * inv_det, c01 * inv_det, c11 * inv_det, c21 * inv_det, c02 * inv_det, c12 * inv_det, c22 * inv_det);
    }
};

/**
 * @brief Returns the 3x3 identity matrix.
 * @return diag(1, 1, 1)
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
identity3x3() noexcept {
    return Float3x3(1.0f);
}

/**
 * @brief Returns the 3x3 zero matrix.
 * @return All components 0
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
zero3x3() noexcept {
    return Float3x3();
}

/**
 * @brief Returns the transpose of a matrix.
 * @param m  Input matrix
 * @return   m.transposed()
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
transpose(const Float3x3& m) noexcept {
    return m.transposed();
}

/**
 * @brief Returns the determinant of a matrix.
 * @param m  Input matrix
 * @return   m.determinant()
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
determinant(const Float3x3& m) noexcept {
    return m.determinant();
}

/**
 * @brief Returns the inverse of a matrix.
 * @param m  Input matrix
 * @return   m.inversed()
 * @warning  Undefined behavior if m is singular; use m.try_inverse() when
 *           singularity is possible.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
inverse(const Float3x3& m) noexcept {
    return m.inversed();
}

/** @brief Matrix + matrix (component-wise). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator+(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] + b._data[i];
    return out;
}

/** @brief Matrix - matrix (component-wise). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator-(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] - b._data[i];
    return out;
}

/** @brief Matrix * scalar. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const float s) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] * s;
    return out;
}

/** @brief Scalar * matrix. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const float s, const Float3x3& a) noexcept {
    return a * s;
}

/** @brief Matrix / scalar. */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator/(const Float3x3& a, const float s) noexcept {
    return a * (1.0f / s);
}

/** @brief Matrix * matrix product. Delegates to Float3x3::mul(). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const Float3x3& b) noexcept {
    return a.mul(b);
}

/** @brief Matrix * vector product. Delegates to Float3x3::mul(). */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3x3& a, const Float3& v) noexcept {
    return a.mul(v);
}

/**
 * @brief Applies a rotation (linear transform) to a vector, writing the result to an output parameter.
 * @param[in]  matrix  Transform matrix
 * @param[in]  input   Vector to transform
 * @param[out] output  Receives matrix * input
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate(const Float3x3& matrix, const Float3& input, Float3& output) noexcept {
    const float x = input.x;
    const float y = input.y;
    const float z = input.z;

    output.x = std::fma(matrix.m01, y, std::fma(matrix.m02, z, matrix.m00 * x));
    output.y = std::fma(matrix.m11, y, std::fma(matrix.m12, z, matrix.m10 * x));
    output.z = std::fma(matrix.m21, y, std::fma(matrix.m22, z, matrix.m20 * x));
}

/**
 * @brief Applies a rotation then a translation: output = matrix * input + offset.
 * @param[in]  matrix  Transform matrix
 * @param[in]  input   Vector to transform
 * @param[in]  offset  Translation applied after rotation
 * @param[out] output  Receives matrix * input + offset
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate_translate(const Float3x3& matrix,
                 const Float3& input,
                 const Float3& offset,
                 Float3& output) noexcept {
    rotate(matrix, input, output);
    output += offset;
}

/**
 * @brief Subtracts an offset then applies a rotation: output = matrix * (input - offset).
 * @param[in]  matrix  Transform matrix
 * @param[in]  input   Vector to transform
 * @param[in]  offset  Offset subtracted before rotation
 * @param[out] output  Receives matrix * (input - offset)
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate_subtract(const Float3x3& matrix,
                const Float3& input,
                const Float3& offset,
                Float3& output) noexcept {
    rotate(matrix, input - offset, output);
}

/**
 * @brief Attempts to solve the linear system A * x = b, failing gracefully if A is singular.
 * @param[in]  A    Coefficient matrix
 * @param[in]  b    Right-hand-side vector
 * @param[out] x    Receives the solution on success (untouched on failure)
 * @param[in]  eps  Determinant magnitude threshold (default: float epsilon)
 * @return     true on success, false if A is singular
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
solve(const Float3x3& A, const Float3& b, Float3& x,
      const float eps = std::numeric_limits<float>::epsilon()) noexcept {
    return A.solve(b, x, eps);
}

/**
 * @brief Solves the linear system A * x = b via explicit inversion.
 * @param A  Coefficient matrix
 * @param b  Right-hand-side vector
 * @return   x = inverse(A) * b
 * @warning  Undefined behavior if A is singular; use the try_inverse-based
 *           overload when singularity is possible.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
solve(const Float3x3& A, const Float3& b) noexcept {
    return A.solved(b);
}

} // namespace atlas
