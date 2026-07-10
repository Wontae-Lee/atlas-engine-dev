#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/vector/float3.h>

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <utility>

namespace atlas {

/**
 * @brief A 3x3 single-precision matrix in row-major storage.
 *
 * Stores nine floats that alias two ways through an anonymous union: named
 * elements m<row><col> and the flat array _data[9]. Row-major means element
 * (r, c) lives at _data[r*3 + c], so vector transforms treat rows as the
 * matrix's action on a column vector (see mul(const Float3&)). Trivially
 * copyable and constexpr-constructible; usable on host and device and
 * capturable by a device lambda. Used for rotation / linear transforms and for
 * small 3x3 linear solves.
 */
class Float3x3 {
public:
    /**
     * @brief The nine elements, addressable either by name or as a flat array.
     *
     * The named members m<row><col> and the _data[9] array occupy the same
     * storage. _data[r*3 + c] equals m<r><c>; use whichever access is clearer.
     */
    union {
        struct {
            float m00, m01, m02; ///< Row 0 elements (columns 0, 1, 2).
            float m10, m11, m12; ///< Row 1 elements (columns 0, 1, 2).
            float m20, m21, m22; ///< Row 2 elements (columns 0, 1, 2).
        };
        float _data[9]; ///< Flat row-major view of the same nine floats.
    };

    /**
     * @brief Construct the zero matrix (all nine elements 0).
     */
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

    /**
     * @brief Copy constructor; a plain element-wise copy.
     */
    constexpr Float3x3(const Float3x3&) noexcept = default;

    /**
     * @brief Construct a scaled identity: @p s on the diagonal, 0 elsewhere.
     *
     * Explicit so a bare float never implicitly becomes a matrix. Passing 1.0f
     * yields the identity (see identity3x3()).
     *
     * @param s The diagonal value.
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
     * @brief Construct from nine explicit elements in row-major order.
     *
     * @param a00 Row 0 col 0. @param a01 Row 0 col 1. @param a02 Row 0 col 2.
     * @param a10 Row 1 col 0. @param a11 Row 1 col 1. @param a12 Row 1 col 2.
     * @param a20 Row 2 col 0. @param a21 Row 2 col 1. @param a22 Row 2 col 2.
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
     * @brief Host-only construction from a brace list in row-major order.
     *
     * Host only because std::initializer_list is unavailable in device code.
     * Fewer than nine values leaves the trailing elements at 0; extra values
     * are ignored.
     *
     * @param list Up to nine values, filling _data[0..8] in order.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Float3x3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        for (int i = 0; i < 9; ++i) {
            _data[i] = (it != list.end()) ? *it++ : 0.0f;
        }
    }

    /**
     * @brief Trivial destructor.
     */
    ~Float3x3() noexcept = default;

    /**
     * @brief Copy assignment; a plain element-wise copy.
     */
    Float3x3&
    operator=(const Float3x3&) noexcept = default;

    /**
     * @brief The row count.
     * @return Always 3.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    rows() noexcept {
        return 3;
    }

    /**
     * @brief The column count.
     * @return Always 3.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    cols() noexcept {
        return 3;
    }

    /**
     * @brief The total element count.
     * @return Always 9.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 9;
    }

    /**
     * @brief Pointer to the flat row-major storage for read access.
     * @return _data, a length-9 float array.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return _data;
    }

    /**
     * @brief Pointer to the flat row-major storage for write access.
     * @return _data, a length-9 float array.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return _data;
    }

    /**
     * @brief Read a single element by flat index, unchecked.
     * @param i Flat index in [0, 9); equals r*3 + c. No bounds check.
     * @return Const reference to _data[i].
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return _data[i];
    }

    /**
     * @brief Access a single element by flat index, unchecked.
     * @param i Flat index in [0, 9); equals r*3 + c. No bounds check.
     * @return Mutable reference to _data[i].
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return _data[i];
    }

    /**
     * @brief Read element (r, c) by row/column, unchecked.
     * @param r Row index in [0, 3).
     * @param c Column index in [0, 3). No bounds check.
     * @return Const reference to _data[r*3 + c].
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Access element (r, c) by row/column, unchecked.
     * @param r Row index in [0, 3).
     * @param c Column index in [0, 3). No bounds check.
     * @return Mutable reference to _data[r*3 + c].
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Read element (r, c) via call syntax; identical to at().
     * @param r Row index in [0, 3).
     * @param c Column index in [0, 3). No bounds check.
     * @return Const reference to _data[r*3 + c].
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator()(const std::size_t r, const std::size_t c) const noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Access element (r, c) via call syntax; identical to at().
     * @param r Row index in [0, 3).
     * @param c Column index in [0, 3). No bounds check.
     * @return Mutable reference to _data[r*3 + c].
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator()(const std::size_t r, const std::size_t c) noexcept {
        return _data[r * 3 + c];
    }

    /**
     * @brief Set every element to zero.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        for (int i = 0; i < 9; ++i) _data[i] = 0.0f;
    }

    /**
     * @brief Overwrite with the identity matrix.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_identity() noexcept {
        set(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    /**
     * @brief Overwrite all nine elements in row-major order.
     *
     * @param a00 Row 0 col 0. @param a01 Row 0 col 1. @param a02 Row 0 col 2.
     * @param a10 Row 1 col 0. @param a11 Row 1 col 1. @param a12 Row 1 col 2.
     * @param a20 Row 2 col 0. @param a21 Row 2 col 1. @param a22 Row 2 col 2.
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

    /**
     * @brief Add a scalar to every element in place.
     * @param s The scalar addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += s;
    }

    /**
     * @brief Subtract a scalar from every element in place.
     * @param s The scalar subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= s;
    }

    /**
     * @brief Scale every element by a scalar in place.
     * @param s The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float s) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] *= s;
    }

    /**
     * @brief Divide every element by a scalar in place.
     *
     * Computes the reciprocal once so all elements share the same rounding.
     *
     * @param s The divisor; must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float s) noexcept {
        const float inv = 1.0f / s;
        for (int i = 0; i < 9; ++i) _data[i] *= inv;
    }

    /**
     * @brief Element-wise add another matrix in place.
     * @param m The matrix added to this one.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] += m._data[i];
    }

    /**
     * @brief Element-wise subtract another matrix in place.
     * @param m The matrix subtracted from this one.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Float3x3& m) noexcept {
        for (int i = 0; i < 9; ++i) _data[i] -= m._data[i];
    }

    /**
     * @brief Add a scalar to every element and return *this.
     * @param s The scalar addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const float s) noexcept {
        add(s);
        return *this;
    }

    /**
     * @brief Subtract a scalar from every element and return *this.
     * @param s The scalar subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const float s) noexcept {
        sub(s);
        return *this;
    }

    /**
     * @brief Scale every element by a scalar and return *this.
     * @param s The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator*=(const float s) noexcept {
        mul(s);
        return *this;
    }

    /**
     * @brief Divide every element by a scalar and return *this.
     * @param s The divisor; must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator/=(const float s) noexcept {
        div(s);
        return *this;
    }

    /**
     * @brief Element-wise add another matrix and return *this.
     * @param m The matrix addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator+=(const Float3x3& m) noexcept {
        add(m);
        return *this;
    }

    /**
     * @brief Element-wise subtract another matrix and return *this.
     * @param m The matrix subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3&
    operator-=(const Float3x3& m) noexcept {
        sub(m);
        return *this;
    }

    /**
     * @brief Exact element-wise equality.
     *
     * @param o The matrix to compare against.
     * @return True only when all nine elements compare bit-for-bit equal under
     * `!=` failing. This is exact float comparison, not an epsilon test.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Float3x3& o) const noexcept {
        for (int i = 0; i < 9; ++i) {
            if (_data[i] != o._data[i]) return false;
        }
        return true;
    }

    /**
     * @brief Negation of operator==.
     * @param o The matrix to compare against.
     * @return True when any element differs.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Float3x3& o) const noexcept {
        return !(*this == o);
    }

    /**
     * @brief The determinant, via cofactor expansion along the first row.
     *
     * The three 2x2 minors are reused by cofactor_inverse(); the final
     * combination is fused with std::fma to limit rounding error.
     *
     * @return det(*this); zero (or near-zero) signals a singular matrix.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    determinant() const noexcept {
        const float ei_fh = m11 * m22 - m12 * m21;
        const float di_fg = m10 * m22 - m12 * m20;
        const float dh_eg = m10 * m21 - m11 * m20;
        return std::fma(m00, ei_fh, std::fma(-m01, di_fg, m02 * dh_eg));
    }

    /**
     * @brief The trace (sum of the diagonal).
     * @return m00 + m11 + m22.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    trace() const noexcept {
        return m00 + m11 + m22;
    }

    /**
     * @brief Transpose the matrix in place by swapping the off-diagonal pairs.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    transpose() noexcept {
        std::swap(m01, m10);
        std::swap(m02, m20);
        std::swap(m12, m21);
    }

    /**
     * @brief A transposed copy, leaving this matrix unchanged.
     * @return The transpose of *this.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    transposed() const noexcept {
        return Float3x3(m00, m10, m20, m01, m11, m21, m02, m12, m22);
    }

    /**
     * @brief Invert the matrix in place.
     *
     * @warning No singularity check: if the determinant is zero this divides by
     * zero and fills the matrix with inf/NaN. Use try_inverse() when the input
     * may be singular.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    inverse() noexcept {
        Float3x3 out;
        cofactor_inverse(out, determinant());
        *this = out;
    }

    /**
     * @brief An inverted copy, leaving this matrix unchanged.
     *
     * @warning Same unchecked-singularity caveat as inverse().
     * @return The inverse of *this.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    inversed() const noexcept {
        Float3x3 out = *this;
        out.inverse();
        return out;
    }

    /**
     * @brief Invert only if non-singular, reporting success.
     *
     * The safe counterpart to inverse(): computes the determinant, and inverts
     * into @p out only when its magnitude exceeds @p eps.
     *
     * @param out Written with the inverse on success; untouched on failure.
     * @param eps Singularity threshold on |det|; defaults to float machine
     * epsilon.
     * @return True when the matrix was invertible and @p out was written.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    try_inverse(Float3x3& out, const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        const float det = determinant();
        if (std::abs(det) <= eps) return false;
        cofactor_inverse(out, det);
        return true;
    }

    /**
     * @brief Whether the matrix is numerically invertible.
     * @param eps Singularity threshold on |det|; defaults to float machine
     * epsilon.
     * @return True when |determinant()| > @p eps.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_invertible(const float eps = std::numeric_limits<float>::epsilon()) const noexcept {
        return std::abs(determinant()) > eps;
    }

    /**
     * @brief Matrix product this * r.
     *
     * Standard row-by-column multiply; each entry is accumulated with std::fma
     * for reduced rounding error. Not commutative.
     *
     * @param r The right-hand matrix.
     * @return The product this * r.
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
     * @brief Matrix-vector product this * v (v treated as a column vector).
     *
     * Row-major action: each output component is the dot of a matrix row with
     * @p v, fused with std::fma.
     *
     * @param v The column vector.
     * @return The transformed vector.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    mul(const Float3& v) const noexcept {
        return Float3(std::fma(m01, v.y, std::fma(m02, v.z, m00 * v.x)),
                      std::fma(m11, v.y, std::fma(m12, v.z, m10 * v.x)),
                      std::fma(m21, v.y, std::fma(m22, v.z, m20 * v.x)));
    }

    /**
     * @brief Solve A x = b for x by inverting A (this matrix).
     *
     * @warning Uses the unchecked inversed(), so a singular matrix yields
     * inf/NaN. Prefer solve() when A may be singular.
     *
     * @param b The right-hand side.
     * @return The solution inversed() * b.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    solved(const Float3& b) const noexcept {
        return inversed().mul(b);
    }

    /**
     * @brief Solve A x = b, reporting whether A (this matrix) was invertible.
     *
     * The safe counterpart to solved(): inverts only when non-singular and
     * writes the solution to @p x on success.
     *
     * @param b The right-hand side.
     * @param x Written with the solution on success; untouched on failure.
     * @param eps Singularity threshold on |det|; defaults to float machine
     * epsilon.
     * @return True when solved; false when A is singular.
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
     * @brief Fill @p out with the inverse, given a precomputed determinant.
     *
     * The shared implementation behind inverse(), inversed() and try_inverse():
     * builds the adjugate (the transposed cofactor matrix, note the c<row><col>
     * are laid out transposed into @p out) and scales by 1/det. Callers are
     * responsible for having checked @p det; this routine does not.
     *
     * @param out Written with the inverse.
     * @param det The determinant of *this, supplied so it is computed once.
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
 * @brief The 3x3 identity matrix.
 * @return A matrix with 1 on the diagonal and 0 elsewhere.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
identity3x3() noexcept {
    return Float3x3(1.0f);
}

/**
 * @brief The 3x3 zero matrix.
 * @return A matrix with every element 0.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
zero3x3() noexcept {
    return Float3x3();
}

/**
 * @brief Free-function transpose.
 * @param m The matrix to transpose.
 * @return A transposed copy of @p m.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
transpose(const Float3x3& m) noexcept {
    return m.transposed();
}

/**
 * @brief Free-function determinant.
 * @param m The matrix.
 * @return det(m).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
determinant(const Float3x3& m) noexcept {
    return m.determinant();
}

/**
 * @brief Free-function inverse (unchecked singularity).
 *
 * @warning Yields inf/NaN for a singular @p m; see Float3x3::try_inverse for a
 * checked variant.
 * @param m The matrix to invert.
 * @return The inverse of @p m.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
inverse(const Float3x3& m) noexcept {
    return m.inversed();
}

/**
 * @brief Element-wise matrix addition.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator+(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] + b._data[i];
    return out;
}

/**
 * @brief Element-wise matrix subtraction.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator-(const Float3x3& a, const Float3x3& b) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] - b._data[i];
    return out;
}

/**
 * @brief Matrix times scalar (every element scaled).
 * @param a The matrix.
 * @param s The multiplier.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const float s) noexcept {
    Float3x3 out;
    for (int i = 0; i < 9; ++i) out._data[i] = a._data[i] * s;
    return out;
}

/**
 * @brief Scalar times matrix (commutes with the overload above).
 * @param s The multiplier.
 * @param a The matrix.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const float s, const Float3x3& a) noexcept {
    return a * s;
}

/**
 * @brief Matrix divided by scalar.
 * @param a The matrix.
 * @param s The divisor; must be non-zero (no guard).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator/(const Float3x3& a, const float s) noexcept {
    return a * (1.0f / s);
}

/**
 * @brief Matrix product a * b (not commutative).
 * @param a Left matrix.
 * @param b Right matrix.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
operator*(const Float3x3& a, const Float3x3& b) noexcept {
    return a.mul(b);
}

/**
 * @brief Matrix-vector product a * v (v as a column vector).
 * @param a The matrix.
 * @param v The column vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3x3& a, const Float3& v) noexcept {
    return a.mul(v);
}

/**
 * @brief Apply a linear transform, writing the result through an out-parameter.
 *
 * Equivalent to output = matrix * input but caches the input components first,
 * so it is safe to pass the same Float3 as both @p input and @p output. Each
 * output component is fused with std::fma.
 *
 * @param matrix The transform.
 * @param input The vector to transform.
 * @param output Out: the transformed vector; may alias @p input.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate(const Float3x3& matrix, const Float3& input, Float3& output) noexcept {
    // Snapshot the inputs so aliasing input == output is safe before any write.
    const float x = input.x;
    const float y = input.y;
    const float z = input.z;

    output.x = std::fma(matrix.m01, y, std::fma(matrix.m02, z, matrix.m00 * x));
    output.y = std::fma(matrix.m11, y, std::fma(matrix.m12, z, matrix.m10 * x));
    output.z = std::fma(matrix.m21, y, std::fma(matrix.m22, z, matrix.m20 * x));
}

/**
 * @brief Rotate then translate: output = matrix * input + offset.
 *
 * Transforms a point from local into world space. Safe to alias @p input and
 * @p output (rotate() snapshots the input first).
 *
 * @param matrix The rotation/linear part.
 * @param input The local-space vector.
 * @param offset The translation added after rotation.
 * @param output Out: the world-space result.
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
 * @brief Translate then rotate: output = matrix * (input - offset).
 *
 * The inverse-order counterpart of rotate_translate(): subtracts the origin
 * offset before applying the linear part, mapping a world-space point into a
 * frame centered at @p offset.
 *
 * @param matrix The rotation/linear part.
 * @param input The world-space vector.
 * @param offset The origin subtracted before rotation.
 * @param output Out: the result in the shifted frame.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
rotate_subtract(const Float3x3& matrix,
                const Float3& input,
                const Float3& offset,
                Float3& output) noexcept {
    rotate(matrix, input - offset, output);
}

/**
 * @brief Free-function solve of A x = b, reporting invertibility.
 *
 * @param A The system matrix.
 * @param b The right-hand side.
 * @param x Written with the solution on success; untouched on failure.
 * @param eps Singularity threshold on |det|; defaults to float machine epsilon.
 * @return True when solved; false when @p A is singular.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
solve(const Float3x3& A, const Float3& b, Float3& x,
      const float eps = std::numeric_limits<float>::epsilon()) noexcept {
    return A.solve(b, x, eps);
}

/**
 * @brief Free-function solve of A x = b, unchecked.
 *
 * @warning Uses the unchecked inverse, so a singular @p A yields inf/NaN.
 * @param A The system matrix.
 * @param b The right-hand side.
 * @return The solution A^-1 * b.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
solve(const Float3x3& A, const Float3& b) noexcept {
    return A.solved(b);
}

}