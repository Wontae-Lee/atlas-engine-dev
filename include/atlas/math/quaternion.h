#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/constants.h>
#include <atlas/math/matrix/float3x3.h>
#include <atlas/math/vector/float3.h>

#include <cmath>
#include <initializer_list>
#include <limits>

namespace atlas {

/**
 * @brief A single-precision quaternion (w, x, y, z) for 3-D rotations.
 *
 * Stores the real (scalar) part w first, then the vector part (x, y, z), so
 * &w addresses a length-4 float array (see data()). The default value is the
 * identity rotation. A quaternion represents a rotation only when unit length;
 * the rotation and matrix-conversion methods assume that, while the arithmetic
 * operators treat it as a plain 4-component value. Trivially copyable and usable
 * on host and device.
 */
class Quaternion {
public:
    float w; ///< Real (scalar) part.

    float x; ///< Vector part, x component.

    float y; ///< Vector part, y component.

    float z; ///< Vector part, z component.

    /**
     * @brief Construct the identity rotation (1, 0, 0, 0).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion() noexcept
        : w(1.0f)
        , x(0.0f)
        , y(0.0f)
        , z(0.0f) { }

    /**
     * @brief Construct from explicit components.
     * @param w_ Real part. @param x_ Vector x. @param y_ Vector y. @param z_ Vector z.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion(const float w_, const float x_, const float y_, const float z_) noexcept
        : w(w_)
        , x(x_)
        , y(y_)
        , z(z_) { }

    /**
     * @brief Host-only construction from a brace list in (w, x, y, z) order.
     *
     * Host only because std::initializer_list is unavailable in device code.
     * A missing w defaults to 1 and missing vector components to 0, so an empty
     * list gives the identity.
     *
     * @param list Up to four values in w, x, y, z order.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Quaternion(std::initializer_list<float> list) noexcept {
        const float* it = list.begin();
        w               = (it != list.end()) ? *it++ : 1.0f;
        x               = (it != list.end()) ? *it++ : 0.0f;
        y               = (it != list.end()) ? *it++ : 0.0f;
        z               = (it != list.end()) ? *it++ : 0.0f;
    }

    /**
     * @brief Construct a rotation of @p radians about @p axis.
     *
     * @param axis The rotation axis; expected unit length (not renormalized).
     * @param radians The rotation angle in radians.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const Float3& axis, const float radians) noexcept {
        *this = from_axis_angle(axis, radians);
    }

    /**
     * @brief Construct from intrinsic X-then-Y-then-Z Euler angles.
     *
     * @param rx Rotation about the x axis, radians.
     * @param ry Rotation about the y axis, radians.
     * @param rz Rotation about the z axis, radians.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const float rx, const float ry, const float rz) noexcept {
        *this = from_euler_xyz(rx, ry, rz);
    }

    /**
     * @brief Construct from a rotation matrix.
     *
     * Explicit so a matrix never implicitly becomes a quaternion.
     *
     * @param m A rotation matrix (assumed orthonormal).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(const Float3x3& m) noexcept {
        *this = from_matrix3x3(m);
    }

    /**
     * @brief Copy constructor; a plain component-wise copy.
     */
    Quaternion(const Quaternion&) noexcept = default;

    /**
     * @brief Trivial destructor.
     */
    ~Quaternion() noexcept = default;

    /**
     * @brief Copy assignment.
     */
    Quaternion&
    operator=(const Quaternion&) noexcept = default;

    /**
     * @brief Move assignment (identical to copy for this trivial type).
     */
    Quaternion&
    operator=(Quaternion&&) noexcept = default;

    /**
     * @brief Build a rotation quaternion from an axis and angle.
     *
     * q = (cos(theta/2), sin(theta/2) * axis). The result is unit length only
     * when @p axis is unit length.
     *
     * @param axis The rotation axis (expected unit length).
     * @param radians The rotation angle in radians.
     * @return The corresponding quaternion.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_axis_angle(const Float3& axis, const float radians) noexcept {
        const float half = radians * 0.5f;
        const float s    = std::sin(half);
        return Quaternion(std::cos(half), axis.x * s, axis.y * s, axis.z * s);
    }

    /**
     * @brief Build a quaternion from intrinsic X-Y-Z Euler angles.
     *
     * Composes per-axis rotations as qz * qy * qx, i.e. x is applied first, then
     * y, then z (intrinsic X-Y-Z / roll-pitch-yaw order).
     *
     * @param rx Rotation about x, radians.
     * @param ry Rotation about y, radians.
     * @param rz Rotation about z, radians.
     * @return The composed rotation.
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
     * @brief Build a quaternion from a rotation matrix (Shepperd's method).
     *
     * Selects among four branches by the trace and the largest diagonal element
     * to keep the divisor s well away from zero, avoiding cancellation that a
     * single-formula conversion would suffer near 180-degree rotations. Assumes
     * @p m is a proper orthonormal rotation.
     *
     * @param m The rotation matrix.
     * @return The equivalent quaternion.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_matrix3x3(const Float3x3& m) noexcept {
        // Pick the branch with the largest pivot so s is never near zero.
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
     * @brief Pointer to the four components for contiguous read access.
     * @return &w, addressing the length-4 array (w, x, y, z).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const float*
    data() const noexcept {
        return &w;
    }

    /**
     * @brief Pointer to the four components for contiguous write access.
     * @return &w, addressing the length-4 array (w, x, y, z).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float*
    data() noexcept {
        return &w;
    }

    /**
     * @brief Four-component dot product, fused with std::fma.
     *
     * For unit quaternions this equals the cosine of half the angle between the
     * two rotations; slerp() uses it as such.
     *
     * @param q The other quaternion.
     * @return w*q.w + x*q.x + y*q.y + z*q.z.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    dot(const Quaternion& q) const noexcept {
        return std::fma(z, q.z, std::fma(y, q.y, std::fma(x, q.x, w * q.w)));
    }

    /**
     * @brief Squared norm (w^2 + x^2 + y^2 + z^2).
     * @return The squared length; 1 for a unit quaternion.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length_squared() const noexcept {
        return std::fma(z, z, std::fma(y, y, std::fma(x, x, w * w)));
    }

    /**
     * @brief Norm (magnitude) of the quaternion.
     * @return sqrt(length_squared()).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    length() const noexcept {
        return std::sqrt(length_squared());
    }

    /**
     * @brief Scale to unit length in place, leaving a near-zero quaternion alone.
     *
     * Guards against division by a tiny norm: if the length is at or below float
     * machine epsilon the quaternion is left unchanged rather than blown up to
     * inf/NaN.
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
     * @brief A unit-length copy; a near-zero quaternion is returned unchanged.
     * @return The normalized quaternion.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    normalized() const noexcept {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    /**
     * @brief The conjugate (negated vector part).
     *
     * For a unit quaternion the conjugate is the inverse rotation; see inverse()
     * for the general (non-unit) case.
     *
     * @return (w, -x, -y, -z).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    conjugate() const noexcept {
        return Quaternion(w, -x, -y, -z);
    }

    /**
     * @brief The multiplicative inverse, conjugate divided by squared norm.
     *
     * Correct for non-unit quaternions. If the squared norm is at or below float
     * machine epsilon the quaternion is treated as non-invertible and the
     * identity is returned.
     *
     * @return conjugate / length_squared(), or the identity for a near-zero input.
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
     * @brief Whether this is within @p tolerance of the identity rotation.
     *
     * Tests (w, x, y, z) against (1, 0, 0, 0) component-wise. Note it does not
     * account for the double cover: the equivalent (-1, 0, 0, 0) is not treated
     * as identity here.
     *
     * @param tolerance Per-component absolute threshold; defaults to atlas::eps.
     * @return True when every component is within @p tolerance of the identity.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_identity(const float tolerance = atlas::eps) const noexcept {
        return std::abs(w - 1.0f) < tolerance
            && std::abs(x) < tolerance
            && std::abs(y) < tolerance
            && std::abs(z) < tolerance;
    }

    /**
     * @brief Rotate a vector by this quaternion.
     *
     * Applies the rotation directly with the expanded q*v*q^-1 formula (avoiding
     * an intermediate matrix): v' = (w^2 - |u|^2) v + 2(u.v) u + 2w (u x v),
     * where u is the vector part. Assumes this quaternion is unit length.
     *
     * @param v The vector to rotate.
     * @return The rotated vector.
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
     * @brief Convert to the equivalent 3x3 rotation matrix.
     *
     * Uses the standard quaternion-to-matrix formula; the result is a proper
     * rotation only when this quaternion is unit length.
     *
     * @return The rotation matrix.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3x3
    to_matrix3x3() const noexcept {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;
        return Float3x3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy), 2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx), 2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy));
    }

    /**
     * @brief Component-wise linear interpolation (unnormalized).
     *
     * @param a Start quaternion (t = 0).
     * @param b End quaternion (t = 1).
     * @param t Interpolation parameter, typically in [0, 1].
     * @return (1 - t) a + t b. The result is generally not unit length; use
     * nlerp() or slerp() for a valid interpolated rotation.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    lerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        return a * (1.0f - t) + b * t;
    }

    /**
     * @brief Normalized linear interpolation.
     *
     * lerp() followed by a renormalize: cheap and a good approximation of slerp
     * for nearby rotations, though its angular speed is not constant. Does not
     * itself handle the double cover; see slerp() for the sign fix.
     *
     * @param a Start quaternion.
     * @param b End quaternion.
     * @param t Interpolation parameter.
     * @return The unit-length interpolated rotation.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    nlerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        Quaternion r = lerp(a, b, t);
        r.normalize();
        return r;
    }

    /**
     * @brief Spherical linear interpolation with constant angular velocity.
     *
     * Negates @p b when the dot is negative so interpolation takes the shorter
     * arc across the double cover, and falls back to nlerp() once the two
     * rotations are within ~1e-6 of parallel to avoid dividing by a vanishing
     * sin(theta). Assumes @p a and @p b are unit length.
     *
     * @param a Start rotation (t = 0).
     * @param b End rotation (t = 1).
     * @param t Interpolation parameter, typically in [0, 1].
     * @return The interpolated unit rotation.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    slerp(const Quaternion& a, const Quaternion& b, const float t) noexcept {
        float cos_theta = a.dot(b);
        Quaternion bb   = b;
        // Flip b onto the same hemisphere as a so we rotate along the short arc.
        if (cos_theta < 0.0f) {
            bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
            cos_theta = -cos_theta;
        }
        // Near-parallel: sin(theta) -> 0 makes the slerp weights blow up, so
        // switch to the numerically safe nlerp.
        if (cos_theta > 1.0f - 1e-6f) {
            return nlerp(a, bb, t);
        }
        const float theta     = std::acos(cos_theta);
        const float sin_theta = std::sin(theta);
        const float w1        = std::sin((1.0f - t) * theta) / sin_theta;
        const float w2        = std::sin(t * theta) / sin_theta;
        return a * w1 + bb * w2;
    }

    /**
     * @brief Component-wise quaternion addition.
     * @param q The addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator+(const Quaternion& q) const noexcept {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }

    /**
     * @brief Component-wise quaternion subtraction.
     * @param q The subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator-(const Quaternion& q) const noexcept {
        return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
    }

    /**
     * @brief Hamilton product this * q (composition of rotations).
     *
     * Not commutative; the result applies @p q first, then this rotation. For
     * unit operands the product is again unit length.
     *
     * @param q The right-hand rotation.
     * @return The composed rotation.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const Quaternion& q) const noexcept {
        return Quaternion(w * q.w - x * q.x - y * q.y - z * q.z,
                          w * q.x + x * q.w + y * q.z - z * q.y,
                          w * q.y - x * q.z + y * q.w + z * q.x,
                          w * q.z + x * q.y - y * q.x + z * q.w);
    }

    /**
     * @brief Scale every component by a scalar.
     * @param s The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const float s) const noexcept {
        return Quaternion(w * s, x * s, y * s, z * s);
    }

    /**
     * @brief Divide every component by a scalar.
     * @param s The divisor; must be non-zero (no guard).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator/(const float s) const noexcept {
        const float inv = 1.0f / s;
        return Quaternion(w * inv, x * inv, y * inv, z * inv);
    }

    /**
     * @brief Component-wise add a quaternion and return *this.
     * @param q The addend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator+=(const Quaternion& q) noexcept {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    /**
     * @brief Component-wise subtract a quaternion and return *this.
     * @param q The subtrahend.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator-=(const Quaternion& q) noexcept {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    /**
     * @brief Compose with @p q on the right (this = this * q) and return *this.
     * @param q The right-hand rotation.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const Quaternion& q) noexcept {
        *this = (*this) * q;
        return *this;
    }

    /**
     * @brief Scale every component by a scalar and return *this.
     * @param s The multiplier.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const float s) noexcept {
        w *= s;
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    /**
     * @brief Divide every component by a scalar and return *this.
     * @param s The divisor; must be non-zero (no guard).
     */
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
     * @brief Approximate equality within atlas::eps on every component.
     *
     * An epsilon comparison rather than exact float equality. It does not treat
     * a quaternion and its negation (the same rotation under the double cover)
     * as equal.
     *
     * @param q The quaternion to compare against.
     * @return True when all four components are within atlas::eps.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Quaternion& q) const noexcept {
        return std::abs(w - q.w) < eps
            && std::abs(x - q.x) < eps
            && std::abs(y - q.y) < eps
            && std::abs(z - q.z) < eps;
    }

    /**
     * @brief Negation of operator==.
     * @param q The quaternion to compare against.
     * @return True when any component differs by at least atlas::eps.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Quaternion& q) const noexcept {
        return !(*this == q);
    }
};

/**
 * @brief True when all four components are finite (no inf, no NaN).
 * @param q The quaternion to test.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const Quaternion& q) noexcept {
    return std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
}

}