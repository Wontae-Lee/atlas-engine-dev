#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/constants.h>
#include <atlas/math/vector/bool3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <tuple>

namespace atlas {

/**
 * @brief A three-component single-precision vector, the engine's core 3-D value.
 *
 * Stores x, y, z as public fields with standard-layout ordering so &x doubles
 * as a pointer to a length-3 array (see data(), operator[]). Trivially copyable
 * and constexpr-constructible, it works on both host and device and can be
 * captured by value inside a device lambda. Used for positions, velocities,
 * directions, and axis-aligned bounds throughout the engine.
 *
 * Member methods are the mutating / in-place forms; the free functions below
 * the class provide the non-mutating arithmetic, comparison, and geometry
 * helpers.
 */
class Float3 {
public:
    float x; ///< First component.
    float y; ///< Second component.
    float z; ///< Third component.

    /**
     * @brief Construct the zero vector (0, 0, 0).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3() noexcept
        : x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    /**
     * @brief Copy constructor; a plain component-wise copy.
     */
    constexpr Float3(const Float3&) noexcept = default;

    /**
     * @brief Construct a uniform vector with every component equal to @p s.
     *
     * Explicit so a bare float never implicitly converts to a Float3.
     *
     * @param s The value assigned to x, y and z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit constexpr Float3(const float s) noexcept
        : x(s)
        , y(s)
        , z(s) { }

    /**
     * @brief Construct from explicit components.
     *
     * @param x_ The x component.
     * @param y_ The y component.
     * @param z_ The z component.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Float3(const float x_, const float y_, const float z_) noexcept
        : x(x_)
        , y(y_)
        , z(z_) { }

    /**
     * @brief Host-only construction from a brace list, padding missing axes with 0.
     *
     * Host only because std::initializer_list is not usable in device code.
     * Extra elements beyond the third are ignored; fewer than three leaves the
     * remaining components at 0.
     *
     * @param list Up to three values in x, y, z order.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Float3(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    /**
     * @brief Trivial destructor.
     */
    ~Float3() noexcept = default;

    /**
     * @brief Copy assignment; a plain component-wise copy.
     */
    Float3&
    operator=(const Float3&) noexcept = default;

    /**
     * @brief The fixed component count of the vector.
     *
     * @return Always 3. Static so it can be queried without an instance.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE std::size_t
    size() noexcept {
        return 3;
    }

    /**
     * @brief Pointer to the first component for contiguous read access.
     *
     * @return &x. The three components are contiguous, so this addresses a
     * length-3 float array.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &x;
    }

    /**
     * @brief Pointer to the first component for contiguous write access.
     *
     * @return &x, addressing the length-3 float array.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &x;
    }

    /**
     * @brief Read a component by index, unchecked.
     *
     * @param i Axis index; only 0, 1, 2 are valid. No bounds check is performed.
     * @return Const reference to component @p i.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    operator[](const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /**
     * @brief Access a component by index, unchecked.
     *
     * @param i Axis index; only 0, 1, 2 are valid. No bounds check is performed.
     * @return Mutable reference to component @p i.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    operator[](const std::size_t i) noexcept {
        return (&x)[i];
    }

    /**
     * @brief Alias of operator[] for read access; performs no bounds check.
     *
     * @param i Axis index; only 0, 1, 2 are valid.
     * @return Const reference to component @p i.
     * @note Despite the name this does not throw or clamp; it matches operator[].
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float&
    at(const std::size_t i) const noexcept {
        return (&x)[i];
    }

    /**
     * @brief Alias of operator[] for write access; performs no bounds check.
     *
     * @param i Axis index; only 0, 1, 2 are valid.
     * @return Mutable reference to component @p i.
     * @note Despite the name this does not throw or clamp; it matches operator[].
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float&
    at(const std::size_t i) noexcept {
        return (&x)[i];
    }

    /**
     * @brief Set every component to the same scalar.
     *
     * @param s The value assigned to x, y and z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set(const float s) noexcept {
        x = y = z = s;
    }

    /**
     * @brief Overwrite all three components.
     *
     * @param x_ New x component.
     * @param y_ New y component.
     * @param z_ New z component.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_values(const float x_, const float y_, const float z_) noexcept {
        x = x_;
        y = y_;
        z = z_;
    }

    /**
     * @brief Reset the vector to (0, 0, 0).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_zero() noexcept {
        x = y = z = 0.0f;
    }

    /**
     * @brief Add a scalar to every component in place.
     *
     * @param v The scalar added to x, y and z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const float v) noexcept {
        x += v;
        y += v;
        z += v;
    }

    /**
     * @brief Subtract a scalar from every component in place.
     *
     * @param v The scalar subtracted from x, y and z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const float v) noexcept {
        x -= v;
        y -= v;
        z -= v;
    }

    /**
     * @brief Scale every component by a scalar in place.
     *
     * @param v The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const float v) noexcept {
        x *= v;
        y *= v;
        z *= v;
    }

    /**
     * @brief Divide every component by a scalar in place.
     *
     * Computes the reciprocal once and multiplies, so all three axes share the
     * same rounding.
     *
     * @param v The divisor; must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const float v) noexcept {
        const float inv = 1.0f / v;
        x *= inv;
        y *= inv;
        z *= inv;
    }

    /**
     * @brief Component-wise add another vector in place.
     *
     * @param v The vector added to this one.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    add(const Float3& v) noexcept {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    /**
     * @brief Component-wise subtract another vector in place.
     *
     * @param v The vector subtracted from this one.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    sub(const Float3& v) noexcept {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    /**
     * @brief Component-wise (Hadamard) multiply by another vector in place.
     *
     * @param v The per-axis multipliers.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    mul(const Float3& v) noexcept {
        x *= v.x;
        y *= v.y;
        z *= v.z;
    }

    /**
     * @brief Component-wise divide by another vector in place.
     *
     * @param v The per-axis divisors; each must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    div(const Float3& v) noexcept {
        x /= v.x;
        y /= v.y;
        z /= v.z;
    }

    /**
     * @brief The smallest of the three components.
     *
     * @return min(x, y, z). NaN handling follows the raw `<` comparisons.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    min() const noexcept {

        return (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    }

    /**
     * @brief The largest of the three components.
     *
     * @return max(x, y, z). NaN handling follows the raw `>` comparisons.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    max() const noexcept {
        return (x > y) ? ((x > z) ? x : z) : ((y > z) ? y : z);
    }

    /**
     * @brief Add a scalar to every component and return *this.
     * @param v The scalar addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator+=(const float v) noexcept {
        add(v);
        return *this;
    }

    /**
     * @brief Subtract a scalar from every component and return *this.
     * @param v The scalar subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator-=(const float v) noexcept {
        sub(v);
        return *this;
    }

    /**
     * @brief Scale every component by a scalar and return *this.
     * @param v The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator*=(const float v) noexcept {
        mul(v);
        return *this;
    }

    /**
     * @brief Divide every component by a scalar and return *this.
     * @param v The divisor; must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator/=(const float v) noexcept {
        div(v);
        return *this;
    }

    /**
     * @brief Component-wise add a vector and return *this.
     * @param v The vector addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator+=(const Float3& v) noexcept {
        add(v);
        return *this;
    }

    /**
     * @brief Component-wise subtract a vector and return *this.
     * @param v The vector subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator-=(const Float3& v) noexcept {
        sub(v);
        return *this;
    }

    /**
     * @brief Component-wise (Hadamard) multiply by a vector and return *this.
     * @param v The per-axis multipliers.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator*=(const Float3& v) noexcept {
        mul(v);
        return *this;
    }

    /**
     * @brief Component-wise divide by a vector and return *this.
     * @param v The per-axis divisors; each must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3&
    operator/=(const Float3& v) noexcept {
        div(v);
        return *this;
    }

    /**
     * @brief Exact component-wise equality.
     *
     * @param other The vector to compare against.
     * @return True only when all three components compare bit-for-bit equal
     * under `==`. This is exact float comparison, not an epsilon test.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Float3& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    /**
     * @brief Negation of operator==.
     *
     * @param other The vector to compare against.
     * @return True when any component differs.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Float3& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Dot (inner) product with another vector.
     *
     * Accumulated with std::fma so the two multiply-adds are fused, reducing
     * rounding error versus separate operations.
     *
     * @param v The other operand.
     * @return x*v.x + y*v.y + z*v.z.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Float3& v) const noexcept {
        return std::fma(z, v.z, std::fma(y, v.y, x * v.x));
    }

    /**
     * @brief Cross product this x v (right-handed).
     *
     * @param v The right-hand operand.
     * @return The vector orthogonal to both, with magnitude |this||v|sin(angle).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    cross(const Float3& v) const noexcept {
        return Float3(y * v.z - z * v.y,
                      z * v.x - x * v.z,
                      x * v.y - y * v.x);
    }

    /**
     * @brief Squared Euclidean length.
     *
     * Preferred over length() when only relative magnitude is needed, as it
     * skips the square root. Fused with std::fma.
     *
     * @return x*x + y*y + z*z.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, x * x));
    }

    /**
     * @brief Euclidean length (magnitude) of the vector.
     *
     * @return sqrt(length_squared()).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    /**
     * @brief Index of the component with the largest signed value.
     *
     * Compares signed values, not magnitudes. Ties resolve toward the lower
     * index (x, then y).
     *
     * @return 0 for x, 1 for y, 2 for z.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    major_axis() const noexcept {
        if (x >= y && x >= z) return 0;
        if (y >= x && y >= z) return 1;
        return 2;
    }

    /**
     * @brief Index of the component with the smallest signed value.
     *
     * Compares signed values, not magnitudes. Ties resolve toward the lower
     * index (x, then y).
     *
     * @return 0 for x, 1 for y, 2 for z.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::size_t
    minor_axis() const noexcept {
        if (x <= y && x <= z) return 0;
        if (y <= x && y <= z) return 1;
        return 2;
    }

    /**
     * @brief Scale to unit length in place, leaving a zero vector untouched.
     *
     * Guards against division by zero: a vector of exactly zero length is left
     * unchanged rather than becoming NaN.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return;
        const float inv = 1.0f / std::sqrt(ls);
        x *= inv;
        y *= inv;
        z *= inv;
    }

    /**
     * @brief A unit-length copy of this vector, or the vector itself if zero.
     *
     * @return The normalized vector, or an exact copy when length is zero (the
     * zero-length guard avoids NaN).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    normalized() const noexcept {
        const float ls = length_squared();
        if (ls == 0.0f) return *this;
        const float inv = 1.0f / std::sqrt(ls);
        return Float3(x * inv, y * inv, z * inv);
    }

    /**
     * @brief Reflect this vector about a plane with the given normal.
     *
     * Computes v - 2(v.n)n. For a physically correct mirror reflection @p n
     * should be unit length; the formula is not renormalized here.
     *
     * @param n The surface normal (expected unit length).
     * @return The reflected vector.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    reflected(const Float3& n) const noexcept {
        const float d = dot(n);
        return Float3(x - 2.0f * d * n.x,
                      y - 2.0f * d * n.y,
                      z - 2.0f * d * n.z);
    }

    /**
     * @brief Remove the component of this vector along @p n (project onto the plane).
     *
     * Computes v - (v.n)n, the component of v orthogonal to @p n. Assumes @p n
     * is unit length; see reject() for the un-normalized-normal equivalent free
     * function that divides out |n|^2 implicitly via dot.
     *
     * @param n The direction to remove (expected unit length).
     * @return The in-plane (tangential) part of the vector.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    projected(const Float3& n) const noexcept {
        const float d = dot(n);
        return Float3(x - d * n.x,
                      y - d * n.y,
                      z - d * n.z);
    }

    /**
     * @brief Build two unit tangents spanning the plane orthogonal to this vector.
     *
     * Treats *this as the plane normal (assumed roughly unit length) and returns
     * an orthonormal pair (t1, t2) perpendicular to it. To avoid a degenerate
     * (near-zero) tangent it selects the construction axis by whichever of |x|,
     * |y| dominates, and when the chosen in-plane length is exactly zero it adds
     * 1 to the radicand so the reciprocal-sqrt stays finite rather than dividing
     * by zero. The second tangent is the cross product, renormalized.
     *
     * @return A tuple {t1, t2} of orthonormal tangent vectors.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Float3, Float3>
    tangential() const noexcept {
        const float ax = std::abs(x);
        const float ay = std::abs(y);
        Float3 t1;

        if (ax > ay) {

            const float d2 = x * x + z * z;

            // Bias the radicand by 1 only when it is exactly zero so the inverse
            // sqrt never divides by zero for an axis-aligned degenerate normal.
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Float3(-z * inv, 0.0f, x * inv);
        } else {

            const float d2  = y * y + z * z;
            const float inv = 1.0f / std::sqrt(d2 + (d2 == 0.0f ? 1.0f : 0.0f));
            t1              = Float3(0.0f, z * inv, -y * inv);
        }

        const Float3 t2 = cross(t1).normalized();
        return { t1, t2 };
    }
};

/**
 * @brief Free-function cross product a x b (right-handed).
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return The vector orthogonal to both @p a and @p b.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cross(const Float3& a, const Float3& b) noexcept {
    return Float3(a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x);
}

/**
 * @brief Free-function dot product, fused with std::fma.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return a.x*b.x + a.y*b.y + a.z*b.z.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
dot(const Float3& a, const Float3& b) noexcept {
    return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x));
}

/**
 * @brief Reflect @p v about a plane with the given @p normal.
 *
 * Computes v - 2(v.n)n; @p normal is assumed unit length (not renormalized).
 *
 * @param v The incident vector.
 * @param normal The surface normal (expected unit length).
 * @return The reflected vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
reflected(const Float3& v, const Float3& normal) noexcept {
    const float d = dot(v, normal);
    return Float3(v.x - 2.0f * d * normal.x,
                  v.y - 2.0f * d * normal.y,
                  v.z - 2.0f * d * normal.z);
}

/**
 * @brief Remove the @p normal-parallel component of @p v (project onto the plane).
 *
 * Computes v - (v.n)n; @p normal is assumed unit length.
 *
 * @param v The vector to project.
 * @param normal The direction removed (expected unit length).
 * @return The in-plane part of @p v.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
projected(const Float3& v, const Float3& normal) noexcept {
    const float d = dot(v, normal);
    return Float3(v.x - d * normal.x,
                  v.y - d * normal.y,
                  v.z - d * normal.z);
}

/**
 * @brief Two orthonormal tangents spanning the plane orthogonal to @p normal.
 *
 * Forwards to Float3::tangential(); see that method for the degenerate-axis
 * handling.
 *
 * @param normal The plane normal (assumed roughly unit length).
 * @return A tuple {t1, t2} of orthonormal tangents.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::tuple<Float3, Float3>
tangential(const Float3& normal) noexcept {
    return normal.tangential();
}

/**
 * @brief Unary plus; returns the vector unchanged.
 *
 * @param a The operand.
 * @return A copy of @p a.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a) noexcept {
    return a;
}

/**
 * @brief Unary negation.
 *
 * @param a The operand.
 * @return (-a.x, -a.y, -a.z).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a) noexcept {
    return Float3(-a.x, -a.y, -a.z);
}

/**
 * @brief Scalar-plus-vector: add @p a to every component of @p b.
 * @param a The scalar addend.
 * @param b The vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const float a, const Float3& b) noexcept {
    return Float3(a + b.x, a + b.y, a + b.z);
}

/**
 * @brief Vector-plus-scalar: add @p b to every component of @p a.
 * @param a The vector.
 * @param b The scalar addend.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a, const float b) noexcept {
    return Float3(a.x + b, a.y + b, a.z + b);
}

/**
 * @brief Component-wise vector addition.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator+(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * @brief Vector-minus-scalar: subtract @p b from every component of @p a.
 * @param a The vector.
 * @param b The scalar subtrahend.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a, const float b) noexcept {
    return Float3(a.x - b, a.y - b, a.z - b);
}

/**
 * @brief Scalar-minus-vector: subtract each component of @p b from @p a.
 * @param a The scalar minuend.
 * @param b The vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const float a, const Float3& b) noexcept {
    return Float3(a - b.x, a - b.y, a - b.z);
}

/**
 * @brief Component-wise vector subtraction.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator-(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * @brief Vector times scalar.
 * @param a The vector.
 * @param b The multiplier.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3& a, const float b) noexcept {
    return Float3(a.x * b, a.y * b, a.z * b);
}

/**
 * @brief Scalar times vector.
 * @param a The multiplier.
 * @param b The vector.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const float a, const Float3& b) noexcept {
    return Float3(a * b.x, a * b.y, a * b.z);
}

/**
 * @brief Component-wise (Hadamard) vector product.
 * @param a Left operand.
 * @param b Right operand.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator*(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

/**
 * @brief Vector divided by scalar.
 *
 * Multiplies by the reciprocal so all axes share the same rounding.
 *
 * @param a The vector.
 * @param b The divisor; must be non-zero (no guard).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const Float3& a, const float b) noexcept {
    const float inv = 1.0f / b;
    return Float3(a.x * inv, a.y * inv, a.z * inv);
}

/**
 * @brief Scalar divided component-wise by a vector.
 * @param a The numerator scalar.
 * @param b The vector of divisors; each must be non-zero (no guard).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const float a, const Float3& b) noexcept {
    return Float3(a / b.x, a / b.y, a / b.z);
}

/**
 * @brief Component-wise vector division.
 * @param a The numerator vector.
 * @param b The denominator vector; each component must be non-zero (no guard).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
operator/(const Float3& a, const Float3& b) noexcept {
    return Float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

/**
 * @brief Per-axis "less than", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 * @return Bool3 with each axis set where a < b. Reduce with all()/any().
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x < b.x, a.y < b.y, a.z < b.z };
}

/**
 * @brief Per-axis "less than or equal", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 * @return Bool3 with each axis set where a <= b.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator<=(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x <= b.x, a.y <= b.y, a.z <= b.z };
}

/**
 * @brief Per-axis "greater than", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 * @return Bool3 with each axis set where a > b.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x > b.x, a.y > b.y, a.z > b.z };
}

/**
 * @brief Per-axis "greater than or equal", returning a boolean triple.
 * @param a Left operand.
 * @param b Right operand.
 * @return Bool3 with each axis set where a >= b.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Bool3
operator>=(const Float3& a, const Float3& b) noexcept {
    return Bool3 { a.x >= b.x, a.y >= b.y, a.z >= b.z };
}

/**
 * @brief Component-wise minimum of two vectors.
 * @param a First operand.
 * @param b Second operand.
 * @return A vector taking the smaller value on each axis.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
min(const Float3& a, const Float3& b) noexcept {
    return Float3((a.x < b.x) ? a.x : b.x,
                  (a.y < b.y) ? a.y : b.y,
                  (a.z < b.z) ? a.z : b.z);
}

/**
 * @brief Component-wise maximum of two vectors.
 * @param a First operand.
 * @param b Second operand.
 * @return A vector taking the larger value on each axis.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
max(const Float3& a, const Float3& b) noexcept {
    return Float3((a.x > b.x) ? a.x : b.x,
                  (a.y > b.y) ? a.y : b.y,
                  (a.z > b.z) ? a.z : b.z);
}

/**
 * @brief Component-wise minimum; a named alias of min() for AABB call sites.
 * @param a First operand.
 * @param b Second operand.
 * @return min(a, b).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cmin(const Float3& a, const Float3& b) noexcept {
    return min(a, b);
}

/**
 * @brief Component-wise maximum; a named alias of max() for AABB call sites.
 * @param a First operand.
 * @param b Second operand.
 * @return max(a, b).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
cmax(const Float3& a, const Float3& b) noexcept {
    return max(a, b);
}

/**
 * @brief Clamp each component of @p v into the box [low, high].
 *
 * @param v The vector to clamp.
 * @param low Per-axis lower bounds.
 * @param high Per-axis upper bounds.
 * @return min(max(v, low), high). Assumes low <= high on every axis.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
clamp(const Float3& v, const Float3& low, const Float3& high) noexcept {
    return min(max(v, low), high);
}

/**
 * @brief Component-wise ceiling.
 * @param a The operand.
 * @return Each component rounded up to the nearest integer value.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
ceil(const Float3& a) noexcept {
    return Float3(std::ceil(a.x), std::ceil(a.y), std::ceil(a.z));
}

/**
 * @brief Component-wise floor.
 * @param a The operand.
 * @return Each component rounded down to the nearest integer value.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
floor(const Float3& a) noexcept {
    return Float3(std::floor(a.x), std::floor(a.y), std::floor(a.z));
}

/**
 * @brief Component-wise absolute value.
 * @param v The operand.
 * @return (|v.x|, |v.y|, |v.z|).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
abs(const Float3& v) noexcept {
    return Float3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

/**
 * @brief True when all three components are finite (no inf, no NaN).
 * @param v The vector to test.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Float3& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

/**
 * @brief Dot product restricted to the x and y components (z ignored).
 *
 * Used where the domain is effectively planar and the z axis carries a separate
 * quantity that must not enter the horizontal magnitude.
 *
 * @param a Left operand.
 * @param b Right operand.
 * @return a.x*b.x + a.y*b.y.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_dot(const Float3& a, const Float3& b) noexcept {
    return std::fma(a.y, b.y, a.x * b.x);
}

/**
 * @brief Squared length of the xy projection (z ignored).
 * @param v The vector.
 * @return v.x*v.x + v.y*v.y.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length_squared(const Float3& v) noexcept {
    return xy_dot(v, v);
}

/**
 * @brief Length of the xy projection (z ignored).
 * @param v The vector.
 * @return sqrt(v.x^2 + v.y^2).
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
xy_length(const Float3& v) noexcept {
    return std::sqrt(xy_length_squared(v));
}

/**
 * @brief Normalize @p v, or return @p fallback when it is too short.
 *
 * A safe normalize for directions that may collapse: when the squared length
 * does not strictly exceed @p min_length_squared the given fallback is
 * returned instead of a NaN or garbage direction. The `!(len2 > threshold)`
 * form also rejects a NaN length (since any comparison with NaN is false).
 *
 * @param v The vector to normalize.
 * @param fallback The direction returned when @p v is degenerate.
 * @param min_length_squared Squared-length floor; defaults to 0, meaning only
 * an exactly-zero (or NaN) vector triggers the fallback.
 * @return The unit vector, or @p fallback.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
normalized_or(const Float3& v,
              const Float3& fallback,
              const float min_length_squared = 0.0f) noexcept {
    const float len2 = v.length_squared();
    if (!(len2 > min_length_squared)) {
        return fallback;
    }
    return v * (1.0f / std::sqrt(len2));
}

/**
 * @brief Normalize the xy projection of @p v into the plane, or return @p fallback.
 *
 * Like normalized_or() but ignores z: the result always has z == 0 and unit
 * length in xy. Used to derive a purely horizontal direction from a 3-D vector.
 *
 * @param v The vector whose xy part is normalized.
 * @param fallback The vector returned when the xy part is too short.
 * @param min_length_squared Squared-length floor on the xy magnitude; default 0.
 * @return A unit horizontal vector (z = 0), or @p fallback.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
xy_normalized_or(const Float3& v,
                 const Float3& fallback,
                 const float min_length_squared = 0.0f) noexcept {
    const float len2 = xy_length_squared(v);
    if (!(len2 > min_length_squared)) {
        return fallback;
    }
    const float inv = 1.0f / std::sqrt(len2);
    return Float3(v.x * inv, v.y * inv, 0.0f);
}

/**
 * @brief Vector rejection: the part of @p v orthogonal to @p normal.
 *
 * Computes v - (v.n)n. Unlike projected(), the name emphasizes "reject the
 * normal component"; @p normal is still assumed unit length for a true
 * orthogonal decomposition.
 *
 * @param v The vector.
 * @param normal The direction to remove (expected unit length).
 * @return The component of @p v perpendicular to @p normal.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
reject(const Float3& v, const Float3& normal) noexcept {
    return v - normal * v.dot(normal);
}

/**
 * @brief Build a right-handed orthonormal frame from an arbitrary @p normal.
 *
 * Normalizes @p normal into @p unit_normal, then constructs a tangent by
 * crossing it with a helper axis chosen to stay well away from parallel: the z
 * axis unless the normal is nearly vertical (|z| >= 0.9), in which case the y
 * axis is used. The bitangent completes the frame. Every intermediate length is
 * checked against @p min_length_squared so a degenerate input fails cleanly.
 *
 * @param normal The input direction (any non-zero length).
 * @param unit_normal Out: @p normal scaled to unit length.
 * @param tangent Out: a unit vector orthogonal to @p unit_normal.
 * @param bitangent Out: unit_normal x tangent, completing a right-handed basis.
 * @param min_length_squared Squared-length floor guarding each normalization.
 * @return True on success; false if @p normal, the tangent, or the bitangent is
 * shorter than the threshold, in which case the out-parameters are left partly
 * written and must not be relied upon.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Float3& normal,
                  Float3& unit_normal,
                  Float3& tangent,
                  Float3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {

    const float normal_length_squared = normal.length_squared();
    if (!(normal_length_squared > min_length_squared)) {
        return false;
    }
    unit_normal = normal * (1.0f / std::sqrt(normal_length_squared));

    // Pick a helper axis that is not near-parallel to the normal so the cross
    // product is well conditioned; switch off z once the normal is near-vertical.
    const Float3 axis = std::abs(unit_normal.z) < 0.9f
        ? Float3(0.0f, 0.0f, 1.0f)
        : Float3(0.0f, 1.0f, 0.0f);

    tangent                            = axis.cross(unit_normal);
    const float tangent_length_squared = tangent.length_squared();
    if (!(tangent_length_squared > min_length_squared)) {
        return false;
    }
    tangent *= 1.0f / std::sqrt(tangent_length_squared);

    bitangent                            = unit_normal.cross(tangent);
    const float bitangent_length_squared = bitangent.length_squared();
    if (!(bitangent_length_squared > min_length_squared)) {
        return false;
    }
    bitangent *= 1.0f / std::sqrt(bitangent_length_squared);
    return true;
}

/**
 * @brief Convenience overload of orthonormal_basis that discards the unit normal.
 *
 * Same behavior as the four-output overload but only the @p tangent and
 * @p bitangent are returned; the normalized normal is computed into a local and
 * dropped.
 *
 * @param normal The input direction.
 * @param tangent Out: a unit tangent orthogonal to @p normal.
 * @param bitangent Out: the completing bitangent.
 * @param min_length_squared Squared-length floor guarding each normalization.
 * @return True on success; false on a degenerate input.
 */
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
orthonormal_basis(const Float3& normal,
                  Float3& tangent,
                  Float3& bitangent,
                  const float min_length_squared = 0.0f) noexcept {
    Float3 unit_normal;
    return orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared);
}

/**
 * @brief A unit vector orthogonal to @p normal, biased toward @p seed's direction.
 *
 * First tries normal x seed: when @p seed is not parallel to @p normal this
 * gives a tangent aligned with the caller's preferred orientation, normalized
 * with the (1,0,0) fallback. If that cross product is degenerate (seed parallel
 * to normal) it falls back to an arbitrary orthonormal-basis tangent, and if
 * even the normal is degenerate it returns (1,0,0).
 *
 * @param normal The vector to be orthogonal to.
 * @param seed A preferred in-plane direction; may be parallel to @p normal.
 * @param min_length_squared Squared-length floor for the degeneracy tests.
 * @return A unit vector orthogonal to @p normal, or the (1,0,0) fallback.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
orthogonal_unit_vector(const Float3& normal,
                       const Float3& seed,
                       const float min_length_squared = 0.0f) noexcept {
    const Float3 fallback(1.0f, 0.0f, 0.0f);

    Float3 tangent = normal.cross(seed);
    if (tangent.length_squared() > min_length_squared) {
        return normalized_or(tangent, fallback, min_length_squared);
    }

    Float3 unit_normal;
    Float3 bitangent;
    if (orthonormal_basis(normal, unit_normal, tangent, bitangent, min_length_squared)) {
        return tangent;
    }

    return fallback;
}

/**
 * @brief Direction at polar angle theta from @p unit_axis and azimuth @p phi.
 *
 * Builds a tangent frame around @p unit_axis (via tangential()) and returns the
 * direction whose angle from the axis has cosine @p cos_theta and whose azimuth
 * about the axis is @p phi. The sine is taken with sqrt_nonnegative so a
 * @p cos_theta at or slightly past +/-1 stays finite. Used for cone / scattering
 * sampling about an arbitrary axis.
 *
 * @param unit_axis The reference axis (must be unit length).
 * @param cos_theta Cosine of the polar angle in [-1, 1].
 * @param phi Azimuthal angle in radians.
 * @return The resulting unit direction.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
spherical_direction(const Float3& unit_axis, const float cos_theta, const float phi) noexcept {
    const auto tangents    = unit_axis.tangential();
    const Float3 tangent   = std::get<0>(tangents);
    const Float3 bitangent = std::get<1>(tangents);

    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    const float cos_phi   = std::cos(phi);
    const float sin_phi   = std::sin(phi);

    return unit_axis * cos_theta + (tangent * cos_phi + bitangent * sin_phi) * sin_theta;
}

/**
 * @brief Direction in the canonical frame where the z axis is the pole.
 *
 * The @p unit_axis == +z specialization of the overload above, without building
 * a tangent frame: returns (sin_theta*cos(phi), sin_theta*sin(phi), cos_theta).
 * The sine uses sqrt_nonnegative to stay finite at |cos_theta| >= 1.
 *
 * @param cos_theta Cosine of the polar angle from +z, in [-1, 1].
 * @param phi Azimuthal angle in radians about +z.
 * @return The resulting unit direction.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
spherical_direction(const float cos_theta, const float phi) noexcept {

    const float sin_theta = sqrt_nonnegative(1.0f - cos_theta * cos_theta);
    return Float3(sin_theta * std::cos(phi),
                  sin_theta * std::sin(phi),
                  cos_theta);
}

}