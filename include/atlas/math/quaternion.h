/**
 * @file   quaternion.h
 * @brief  Single-precision floating point quaternion type Quaternion for 3D rotations.
 *
 * Quaternion is a SIMD-independent (w, x, y, z) quaternion designed to
 * work on both CPU (Host) and CUDA GPU (Device). It represents a 3D
 * rotation more compactly than a Float3x3 and avoids gimbal lock, and
 * interoperates with Float3 (rotate()) and Float3x3
 * (to_matrix3x3() / from_matrix3x3()).
 * All member and free functions explicitly declare their compilation
 * target (host/device) via ATLAS_ALL_DEVICE / ATLAS_HOST macros, and
 * force inlining via ATLAS_FORCE_INLINE to eliminate call overhead.
 *
 * @note This class does not enforce unit length on construction; call
 *       normalize() / normalized() after operations (e.g. lerp) that
 *       may leave the quaternion non-unit before using it to rotate.
 */

#pragma once

#include <atlas/core/macros.h>        /**< ATLAS_ALL_DEVICE, ATLAS_HOST, ATLAS_FORCE_INLINE, ATLAS_NODISCARD, etc. */
#include <atlas/math/constants.h>     /**< atlas::eps, used as the default is_identity()/operator== tolerance */
#include <atlas/math/matrix/float3x3.h> /**< Float3x3 conversion target/source for to_matrix3x3() / from_matrix3x3() */
#include <atlas/math/vector/float3.h>   /**< Float3 type used for axis-angle construction and rotate() */

#include <cmath>           // std::sin, std::cos, std::acos, std::sqrt, std::fma, std::abs
#include <initializer_list>
#include <limits>          // std::numeric_limits

namespace atlas {

/**
 * @class Quaternion
 * @brief Single-precision floating point quaternion (w, x, y, z) representing a 3D rotation.
 *
 * The scalar (real) component is w, and (x, y, z) form the vector
 * (imaginary) part. A unit quaternion `w + xi + yj + zk` rotates a
 * vector v by `q * v * q^-1`; see rotate().
 *
 * @code{.cpp}
 * atlas::Quaternion q(atlas::Float3(0.f, 1.f, 0.f), atlas::pi * 0.5f); // 90 deg about Y
 * atlas::Float3     v = q.rotate(atlas::Float3(1.f, 0.f, 0.f));
 * @endcode
 */
class Quaternion {
public:
    float w; /**< Scalar (real) component */

    float x; /**< First (X) component of the vector (imaginary) part */

    float y; /**< Second (Y) component of the vector (imaginary) part */

    float z; /**< Third (Z) component of the vector (imaginary) part */

    /** @brief Default constructor. Initializes to the identity quaternion (1, 0, 0, 0). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion() noexcept
        : w(1.0f)
        , x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    /**
     * @brief Constructor specifying each component individually.
     * @param w_  Scalar (real) component
     * @param x_  X component of the vector part
     * @param y_  Y component of the vector part
     * @param z_  Z component of the vector part
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion(const float w_, const float x_, const float y_, const float z_) noexcept
        : w(w_)
        , x(x_)
        , y(y_)
        , z(z_) { }

    /**
     * @brief Initializer-list constructor (host only).
     *
     * Fills (w, x, y, z) in order. If fewer than 4 elements are given,
     * w defaults to 1 and x/y/z default to 0 (i.e. the identity
     * quaternion is the fallback for missing trailing components).
     *
     * @param list  Initializer list of float values (up to 4 elements)
     * @note  Not available in CUDA device code (ATLAS_HOST only).
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Quaternion(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        w               = (it != list.end()) ? *it++ : 1.0f;
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    /**
     * @brief Axis-angle constructor. Delegates to from_axis_angle().
     * @param axis     Rotation axis (need not be pre-normalized by the caller, but is assumed unit-length by the formula)
     * @param radians  Rotation angle in radians
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const Float3& axis, const float radians) noexcept {
        *this = from_axis_angle(axis, radians);
    }

    /**
     * @brief Euler-angle (XYZ order) constructor. Delegates to from_euler_xyz().
     * @param rx  Rotation about X, in radians
     * @param ry  Rotation about Y, in radians
     * @param rz  Rotation about Z, in radians
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const float rx, const float ry, const float rz) noexcept {
        *this = from_euler_xyz(rx, ry, rz);
    }

    /**
     * @brief Rotation-matrix constructor. Delegates to from_matrix3x3().
     * @param m  Rotation matrix to convert
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(const Float3x3& m) noexcept {
        *this = from_matrix3x3(m);
    }

    /** @brief Copy constructor (compiler-generated default). */
    Quaternion(const Quaternion&) noexcept = default;
    /** @brief Destructor (trivial). */
    ~Quaternion() noexcept                 = default;
    /** @brief Copy assignment operator (compiler-generated default). */
    Quaternion&
    operator=(const Quaternion&) noexcept = default;
    /** @brief Move assignment operator (compiler-generated default). */
    Quaternion&
    operator=(Quaternion&&) noexcept = default;

    /**
     * @brief Builds a unit quaternion representing a rotation of `radians` about `axis`.
     * @param axis     Unit rotation axis
     * @param radians  Rotation angle in radians
     * @return         (cos(radians/2), axis * sin(radians/2))
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_axis_angle(const Float3& axis, const float radians) noexcept {
        const float half = radians * 0.5f;
        const float s    = std::sin(half);
        return Quaternion(std::cos(half), axis.x * s, axis.y * s, axis.z * s);
    }

    /**
     * @brief Builds a quaternion from Euler angles applied in X, then Y, then Z order.
     *
     * Composes as `qz * qy * qx`, i.e. the X rotation is applied first
     * to a vector, then Y, then Z.
     *
     * @param rx  Rotation about X, in radians
     * @param ry  Rotation about Y, in radians
     * @param rz  Rotation about Z, in radians
     * @return    Combined rotation quaternion
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_euler_xyz(const float rx, const float ry, const float rz) noexcept {
        const float hx = rx * 0.5f;
        const float hy = ry * 0.5f;
        const float hz = rz * 0.5f;
        const float cx = std::cos(hx), sx = std::sin(hx);
        const float cy = std::cos(hy), sy = std::sin(hy);
        const float cz = std::cos(hz), sz = std::sin(hz);
        const Quaternion qx(cx, sx, 0.0f, 0.0f);
        const Quaternion qy(cy, 0.0f, sy, 0.0f);
        const Quaternion qz(cz, 0.0f, 0.0f, sz);
        return qz * qy * qx;
    }

    /**
     * @brief Builds a quaternion from a (assumed orthonormal) rotation matrix.
     *
     * Uses the standard branch-on-trace method, selecting whichever of
     * the trace or the largest diagonal entry gives the best-conditioned
     * square root to avoid division by a near-zero value.
     *
     * @param m  Rotation matrix to convert
     * @return   Equivalent unit quaternion
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_matrix3x3(const Float3x3& m) noexcept {
        const float tr = m.m00 + m.m11 + m.m22;
        if (tr > 0.0f) {
            const float s = std::sqrt(tr + 1.0f) * 2.0f;
            return Quaternion(s * 0.25f,
                              (m.m21 - m.m12) / s,
                              (m.m02 - m.m20) / s,
                              (m.m10 - m.m01) / s);
        } else if (m.m00 > m.m11 && m.m00 > m.m22) {
            const float s = std::sqrt(1.0f + m.m00 - m.m11 - m.m22) * 2.0f;
            return Quaternion((m.m21 - m.m12) / s,
                              s * 0.25f,
                              (m.m01 + m.m10) / s,
                              (m.m02 + m.m20) / s);
        } else if (m.m11 > m.m22) {
            const float s = std::sqrt(1.0f + m.m11 - m.m00 - m.m22) * 2.0f;
            return Quaternion((m.m02 - m.m20) / s,
                              (m.m01 + m.m10) / s,
                              s * 0.25f,
                              (m.m12 + m.m21) / s);
        } else {
            const float s = std::sqrt(1.0f + m.m22 - m.m00 - m.m11) * 2.0f;
            return Quaternion((m.m10 - m.m01) / s,
                              (m.m02 + m.m20) / s,
                              (m.m12 + m.m21) / s,
                              s * 0.25f);
        }
    }

    /**
     * @brief Returns a const pointer to the first component (w).
     *
     * Since the four components are contiguous in memory, the
     * resulting pointer can be used as an array.
     * @return &w (const)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &w;
    }

    /**
     * @brief Returns a mutable pointer to the first component (w).
     * @return &w (mutable)
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &w;
    }

    /**
     * @brief Computes the dot product (4D component-wise) with another quaternion.
     *
     * Implemented with three FMA (Fused Multiply-Add) calls to reduce
     * rounding error.
     *
     * @param q  Quaternion to take the dot product with
     * @return   w*q.w + x*q.x + y*q.y + z*q.z
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Quaternion& q) const noexcept {
        return std::fma(z, q.z, std::fma(y, q.y, std::fma(x, q.x, w * q.w)));
    }

    /**
     * @brief Returns the squared length (norm) of the quaternion.
     * @return |*this|^2 = w^2 + x^2 + y^2 + z^2
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, std::fma(x, x, w * w)));
    }

    /**
     * @brief Returns the Euclidean length (norm) of the quaternion.
     * @return |*this| = sqrt(length_squared())
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    /**
     * @brief Normalizes the quaternion to unit length in place.
     *
     * If the length is smaller than the float epsilon, does nothing
     * (quaternion is preserved) to avoid dividing by a near-zero value.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept {
        const float len = length();
        if (len > std::numeric_limits<float>::epsilon()) {
            const float inv = 1.0f / len;
            w *= inv;
            x *= inv;
            y *= inv;
            z *= inv;
        }
    }

    /**
     * @brief Returns a new normalized quaternion without modifying *this.
     * @return Unit-length copy of *this (unchanged if length is near-zero)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    normalized() const noexcept {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    /**
     * @brief Returns the conjugate. For a unit quaternion, equals the inverse.
     * @return (w, -x, -y, -z)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    conjugate() const noexcept {
        return Quaternion(w, -x, -y, -z);
    }

    /**
     * @brief Returns the multiplicative inverse (conjugate / length_squared).
     *
     * If length_squared() is smaller than the float epsilon, returns the
     * identity quaternion to avoid dividing by a near-zero value.
     *
     * @return *this^-1
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    inverse() const noexcept {
        const float ls = length_squared();
        if (ls > std::numeric_limits<float>::epsilon()) {
            const float inv = 1.0f / ls;
            return Quaternion(w * inv, -x * inv, -y * inv, -z * inv);
        }
        return Quaternion();
    }

    /**
     * @brief Checks whether the quaternion is approximately the identity rotation.
     * @param tolerance  Per-component tolerance (default: atlas::eps)
     * @return           true if all components are within tolerance of (1, 0, 0, 0)
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_identity(const float tolerance = atlas::eps) const noexcept {
        return std::abs(w - 1.0f) < tolerance
            && std::abs(x) < tolerance
            && std::abs(y) < tolerance
            && std::abs(z) < tolerance;
    }

    /**
     * @brief Rotates a vector by this quaternion.
     *
     * Computes `q * v * q^-1` using the optimized cross-product form
     * (avoids constructing an intermediate pure quaternion), with FMA
     * calls used throughout to reduce rounding error.
     *
     * @param v  Vector to rotate
     * @return   Rotated vector
     * @note     Assumes *this is unit length; call normalize() first if unsure.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    rotate(const Float3& v) const noexcept {
        const float uxv_x  = y * v.z - z * v.y;
        const float uxv_y  = z * v.x - x * v.z;
        const float uxv_z  = x * v.y - y * v.x;
        const float uv     = std::fma(z, v.z, std::fma(y, v.y, x * v.x));
        const float uu     = std::fma(z, z, std::fma(y, y, x * x));
        const float scale  = std::fma(w, w, -uu);
        const float two_uv = 2.0f * uv;
        const float two_w  = 2.0f * w;
        return Float3(std::fma(two_w, uxv_x, std::fma(two_uv, x, scale * v.x)),
                      std::fma(two_w, uxv_y, std::fma(two_uv, y, scale * v.y)),
                      std::fma(two_w, uxv_z, std::fma(two_uv, z, scale * v.z)));
    }

    /**
     * @brief Converts this quaternion to an equivalent 3x3 rotation matrix.
     * @return Rotation matrix corresponding to *this
     * @note   Assumes *this is unit length; call normalize() first if unsure.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    to_matrix3x3() const noexcept {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;
        return Float3x3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy), 2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx), 2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));
    }

    /**
     * @brief Linearly interpolates the four components between two quaternions.
     *
     * @warning The result is generally not unit length; prefer nlerp()
     *          or slerp() when a valid rotation is required.
     *
     * @param a  Start quaternion
     * @param b  End quaternion
     * @param t  Interpolation parameter (0 -> a, 1 -> b)
     * @return   a*(1-t) + b*t
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    lerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        return a * (1.0f - t) + b * t;
    }

    /**
     * @brief Normalized linear interpolation between two quaternions.
     *
     * Cheaper than slerp() and a good approximation for small angles
     * between a and b, at the cost of non-constant angular velocity.
     *
     * @param a  Start quaternion
     * @param b  End quaternion
     * @param t  Interpolation parameter (0 -> a, 1 -> b)
     * @return   normalize(lerp(a, b, t))
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    nlerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        Quaternion r = lerp(a, b, t);
        r.normalize();
        return r;
    }

    /**
     * @brief Spherical linear interpolation between two quaternions.
     *
     * Negates b (and its dot product) when the two quaternions are more
     * than 90 degrees apart, so interpolation always takes the shorter
     * arc. Falls back to nlerp() when a and b are nearly parallel, since
     * the slerp formula becomes numerically unstable (division by a
     * near-zero sin_theta) in that regime.
     *
     * @param a  Start quaternion (assumed unit length)
     * @param b  End quaternion (assumed unit length)
     * @param t  Interpolation parameter (0 -> a, 1 -> b)
     * @return   Interpolated unit quaternion
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    slerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        float cos_theta = a.dot(b);
        Quaternion bb   = b;
        if (cos_theta < 0.0f) {
            bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
            cos_theta = -cos_theta;
        }
        if (cos_theta > 1.0f - 1e-6f) {
            return nlerp(a, bb, t);
        }
        const float theta     = std::acos(cos_theta);
        const float sin_theta = std::sin(theta);
        const float w1        = std::sin((1.0f - t) * theta) / sin_theta;
        const float w2        = std::sin(t * theta) / sin_theta;
        return a * w1 + bb * w2;
    }

    /** @brief Component-wise quaternion addition. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator+(const Quaternion& q) const noexcept {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }

    /** @brief Component-wise quaternion subtraction. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator-(const Quaternion& q) const noexcept {
        return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
    }

    /**
     * @brief Hamilton product (quaternion multiplication).
     *
     * Composes rotations: `(*this * q).rotate(v) == this->rotate(q.rotate(v))`.
     * @param q  Right-hand-side quaternion
     * @return   *this * q
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const Quaternion& q) const noexcept {
        return Quaternion(w * q.w - x * q.x - y * q.y - z * q.z,
                          w * q.x + x * q.w + y * q.z - z * q.y,
                          w * q.y - x * q.z + y * q.w + z * q.x,
                          w * q.z + x * q.y - y * q.x + z * q.w);
    }

    /** @brief Scalar multiplication (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const float s) const noexcept {
        return Quaternion(w * s, x * s, y * s, z * s);
    }

    /** @brief Scalar division (component-wise). Pre-computes the reciprocal. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator/(const float s) const noexcept {
        const float inv = 1.0f / s;
        return Quaternion(w * inv, x * inv, y * inv, z * inv);
    }

    /** @brief Quaternion addition assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator+=(const Quaternion& q) noexcept {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    /** @brief Quaternion subtraction assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator-=(const Quaternion& q) noexcept {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    /** @brief Hamilton product assignment. Equivalent to `*this = *this * q`. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const Quaternion& q) noexcept {
        *this = (*this) * q;
        return *this;
    }

    /** @brief Scalar multiplication assignment (component-wise). */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const float s) noexcept {
        w *= s;
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /** @brief Scalar division assignment (component-wise). Pre-computes the reciprocal. */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator/=(const float s) noexcept {
        const float inv = 1.0f / s;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
        return *this;
    }

    /**
     * @brief Checks whether two quaternions are approximately equal.
     * @param q  Quaternion to compare against
     * @return   true if every component differs by less than atlas::eps
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Quaternion& q) const noexcept {
        return std::abs(w - q.w) < eps
            && std::abs(x - q.x) < eps
            && std::abs(y - q.y) < eps
            && std::abs(z - q.z) < eps;
    }

    /** @brief Checks whether two quaternions are different. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Quaternion& q) const noexcept {
        return !(*this == q);
    }
};

/**
 * @brief Checks whether all four components are finite (excludes inf, NaN).
 * @param q  Quaternion to check
 * @return   true if w, x, y, z all pass isfinite
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Quaternion& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

} // namespace atlas
