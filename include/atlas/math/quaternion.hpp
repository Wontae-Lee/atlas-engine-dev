#pragma once

#include <cmath>
#include <limits>

namespace atlas::math {

template <typename T>
constexpr Quaternion<T>::Quaternion() noexcept
    : w(T(1))
    , x(T(0))
    , y(T(0))
    , z(T(0)) {
    // Identity quaternion (no rotation):
    //   q = (1, 0, 0, 0)
}

template <typename T>
constexpr Quaternion<T>::Quaternion(T w_, T x_, T y_, T z_) noexcept
    : w(w_)
    , x(x_)
    , y(y_)
    , z(z_) {
    // Direct component constructor:
    //   q = (w, x, y, z)
}

template <typename T>
Quaternion<T>::Quaternion(std::initializer_list<T> list) noexcept {
    // Initialize from {w, x, y, z} with defaults if fewer values are provided.
    // This is handy for tests and constant-ish initializations.
    auto it = list.begin();
    w       = (it != list.end()) ? *it++ : T(1);
    x       = (it != list.end()) ? *it++ : T(0);
    y       = (it != list.end()) ? *it++ : T(0);
    z       = (it != list.end()) ? *it++ : T(0);
}

template <typename T>
Quaternion<T>::Quaternion(const Vector3<T>& axis, T radians) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Axis–angle to quaternion
    //
    // Given a (typically unit) axis a and rotation angle θ:
    //   q = (cos(θ/2), a * sin(θ/2))
    //
    // Notes:
    //   - If axis is not unit-length, the quaternion's length will differ
    //     from 1; normalization may be needed.
    // ------------------------------------------------------------
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    w            = std::cos(half);
    x            = axis.x * s;
    y            = axis.y * s;
    z            = axis.z * s;
}

template <typename T>
Quaternion<T>::Quaternion(T rx, T ry, T rz) noexcept {
    // Build from Euler angles (XYZ order) by delegating to helper.
    // Convention used by from_euler_xyz(): apply X, then Y, then Z
    // in a composed quaternion product.
    *this = from_euler_xyz(rx, ry, rz);
}

template <typename T>
Quaternion<T>::Quaternion(const Matrix<T, 3, 3>& m) noexcept {
    // Build quaternion from a 3x3 rotation matrix via a numerically stable branch.
    *this = from_matrix3x3(m);
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_axis_angle(const Vector3<T>& axis, T radians) noexcept {
    // Same axis–angle conversion as the constructor, returned as a value.
    const T half = radians * T(0.5);
    const T s    = std::sin(half);
    return Quaternion(
        std::cos(half),
        axis.x * s,
        axis.y * s,
        axis.z * s);
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_euler_xyz(T rx, T ry, T rz) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Euler XYZ -> quaternion via half-angle sines/cosines
    //
    // For each axis rotation:
    //   qx = rotation about X by rx
    //   qy = rotation about Y by ry
    //   qz = rotation about Z by rz
    //
    // Composition:
    //   return qz * qy * qx
    //
    // This order means the X rotation is applied first, then Y, then Z
    // when using the convention v' = q * v * q^{-1}.
    // ------------------------------------------------------------
    const T hx = rx * T(0.5);
    const T hy = ry * T(0.5);
    const T hz = rz * T(0.5);

    const T cx = std::cos(hx), sx = std::sin(hx);
    const T cy = std::cos(hy), sy = std::sin(hy);
    const T cz = std::cos(hz), sz = std::sin(hz);

    Quaternion qx(cx, sx, 0, 0);
    Quaternion qy(cy, 0, sy, 0);
    Quaternion qz(cz, 0, 0, sz);

    return qz * qy * qx;
}

template <typename T>
Quaternion<T>
Quaternion<T>::from_matrix3x3(const Matrix<T, 3, 3>& m) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Rotation matrix -> quaternion (trace-based branching)
    //
    // Uses a common stable approach:
    //   - If trace is positive, compute w from trace.
    //   - Otherwise, pick the dominant diagonal element and compute
    //     the corresponding component first to reduce numerical error.
    //
    // Assumptions:
    //   - m is (approximately) a proper rotation matrix.
    // ------------------------------------------------------------
    const T tr = m.m00 + m.m11 + m.m22;

    if (tr > T(0)) {
        // Trace positive -> w has the largest magnitude (usually).
        const T s = std::sqrt(tr + T(1)) * T(2); // s = 4w
        return Quaternion(
            s * T(0.25),
            (m.m21 - m.m12) / s,
            (m.m02 - m.m20) / s,
            (m.m10 - m.m01) / s);
    } else if (m.m00 > m.m11 && m.m00 > m.m22) {
        // m00 largest -> x is largest component.
        const T s = std::sqrt(T(1) + m.m00 - m.m11 - m.m22) * T(2); // s = 4x
        return Quaternion(
            (m.m21 - m.m12) / s,
            s * T(0.25),
            (m.m01 + m.m10) / s,
            (m.m02 + m.m20) / s);
    } else if (m.m11 > m.m22) {
        // m11 largest -> y is largest component.
        const T s = std::sqrt(T(1) + m.m11 - m.m00 - m.m22) * T(2); // s = 4y
        return Quaternion(
            (m.m02 - m.m20) / s,
            (m.m01 + m.m10) / s,
            s * T(0.25),
            (m.m12 + m.m21) / s);
    } else {
        // m22 largest -> z is largest component.
        const T s = std::sqrt(T(1) + m.m22 - m.m00 - m.m11) * T(2); // s = 4z
        return Quaternion(
            (m.m10 - m.m01) / s,
            (m.m02 + m.m20) / s,
            (m.m12 + m.m21) / s,
            s * T(0.25));
    }
}

template <typename T>
const T*
Quaternion<T>::data() const noexcept {
    // Expose raw contiguous storage (w, x, y, z).
    return &w;
}

template <typename T>
T*
Quaternion<T>::data() noexcept {
    // Mutable raw storage access.
    return &w;
}

template <typename T>
T
Quaternion<T>::dot(const Quaternion& q) const noexcept {
    // Quaternion dot product:
    //   q · p = w*w' + x*x' + y*y' + z*z'
    // Useful for:
    //   - angle between orientations
    //   - slerp cosine term
    return w * q.w + x * q.x + y * q.y + z * q.z;
}

template <typename T>
T
Quaternion<T>::length_squared() const noexcept {
    // Squared norm of quaternion:
    //   |q|^2 = w^2 + x^2 + y^2 + z^2
    return w * w + x * x + y * y + z * z;
}

template <typename T>
T
Quaternion<T>::length() const noexcept {
    // Euclidean norm.
    return std::sqrt(length_squared());
}

template <typename T>
void
Quaternion<T>::normalize() noexcept {
    // Normalize to unit quaternion:
    //   q := q / |q|
    // Unit quaternions represent rotations; non-unit quaternions
    // will scale when used as rotations.
    const T len = length();
    if (len > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / len;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
    }
}

template <typename T>
Quaternion<T>
Quaternion<T>::normalized() const noexcept {
    // Return a normalized copy.
    Quaternion q = *this;
    q.normalize();
    return q;
}

template <typename T>
Quaternion<T>
Quaternion<T>::conjugate() const noexcept {
    // Conjugate:
    //   conj(w, x, y, z) = (w, -x, -y, -z)
    // For unit quaternions, conjugate == inverse.
    return Quaternion(w, -x, -y, -z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::inverse() const noexcept {
    // Inverse:
    //   q^{-1} = conj(q) / |q|^2
    // For unit quaternions, this simplifies to conjugate().
    const T ls = length_squared();
    if (ls > std::numeric_limits<T>::epsilon()) {
        const T inv = T(1) / ls;
        return Quaternion(w * inv, -x * inv, -y * inv, -z * inv);
    }
    // Degenerate case: return identity as a safe fallback.
    return Quaternion();
}

template <typename T>
bool
Quaternion<T>::is_identity(T eps) const noexcept {
    // Identity test with tolerance:
    //   q ≈ (1,0,0,0)
    return std::abs(w - T(1)) < eps && std::abs(x) < eps && std::abs(y) < eps && std::abs(z) < eps;
}

template <typename T>
Vector3<T>
Quaternion<T>::rotate(const Vector3<T>& v) const noexcept {
    // ------------------------------------------------------------
    // Rotate a 3D vector using the quaternion "sandwich product".
    //
    // We interpret this quaternion q = (w, x, y, z) as a rotation when it is
    // unit-length: ||q|| = 1.
    //
    // Core formula (for unit quaternions):
    //   v' = q * (0, v) * q^{-1}
    //
    // Where (0, v) is the "pure quaternion" formed from the vector.
    // The result is also a quaternion whose scalar part is ~0, and whose
    // vector part is the rotated vector.
    //
    // Important:
    // - If q is not normalized, using conjugate() instead of a true inverse
    //   introduces scaling by ||q||^2 (and can distort the rotation).
    // - For best results, ensure q is normalized before calling rotate().
    // ------------------------------------------------------------

    // Step 1) Embed the input vector into quaternion space.
    //
    // A 3D vector v = (vx, vy, vz) cannot be multiplied by a quaternion directly,
    // so we represent it as a "pure quaternion" (zero scalar part):
    //   qv = (0, vx, vy, vz)
    //
    // This lets us use quaternion multiplication to apply the rotation.
    Quaternion<T> qv(T(0), v.x, v.y, v.z);

    // Step 2) Compute the inverse rotation quaternion.
    //
    // For a general quaternion:
    //   q^{-1} = conjugate(q) / ||q||^2
    //
    // For a *unit* quaternion (||q|| = 1), this simplifies to:
    //   q^{-1} = conjugate(q)
    //
    // This implementation assumes q is unit-length and uses conjugate()
    // as a fast inverse.
    Quaternion<T> inv = conjugate();

    // Step 3) Apply the sandwich product: q * qv * q^{-1}.
    //
    // Intuition:
    // - Left-multiplying by q "pushes" qv into the rotated frame.
    // - Right-multiplying by q^{-1} "pulls back" while canceling the extra
    //   quaternion basis, leaving only the rotated vector part.
    //
    // Notes on order:
    // - Quaternion multiplication is NOT commutative.
    // - The order q * qv * inv implements the convention:
    //     v' = q v q^{-1}
    Quaternion<T> res = (*this) * qv * inv;

    // Step 4) Extract the rotated vector from the resulting quaternion.
    //
    // For ideal unit-rotation inputs:
    // - res.w should be approximately 0 (numerical error may leave a tiny value).
    // - (res.x, res.y, res.z) is the rotated vector.
    return Vector3<T>(res.x, res.y, res.z);
}

template <typename T>
Matrix<T, 3, 3>
Quaternion<T>::to_matrix3x3() const noexcept {
    // ------------------------------------------------------------
    // Algorithm: Quaternion -> 3x3 rotation matrix
    //
    // Uses the standard expanded form (assuming a unit quaternion):
    //   R = ...
    //
    // Precompute products to reduce multiplications.
    // ------------------------------------------------------------
    const T xx = x * x, yy = y * y, zz = z * z;
    const T xy = x * y, xz = x * z, yz = y * z;
    const T wx = w * x, wy = w * y, wz = w * z;

    Matrix<T, 3, 3> m;
    m.set(
        T(1) - T(2) * (yy + zz),
        T(2) * (xy - wz),
        T(2) * (xz + wy),

        T(2) * (xy + wz),
        T(1) - T(2) * (xx + zz),
        T(2) * (yz - wx),

        T(2) * (xz - wy),
        T(2) * (yz + wx),
        T(1) - T(2) * (xx + yy));
    return m;
}

template <typename T>
Matrix<T, 4, 4>
Quaternion<T>::to_matrix4x4() const noexcept {
    // Build a 4x4 homogeneous transform with rotation in the top-left.
    // Translation remains zero; bottom-right is 1.
    Matrix<T, 4, 4> m;
    m.set_identity();

    Matrix<T, 3, 3> r = to_matrix3x3();
    m.m00             = r.m00;
    m.m01             = r.m01;
    m.m02             = r.m02;
    m.m10             = r.m10;
    m.m11             = r.m11;
    m.m12             = r.m12;
    m.m20             = r.m20;
    m.m21             = r.m21;
    m.m22             = r.m22;
    return m;
}

template <typename T>
Quaternion<T>
Quaternion<T>::lerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    // Linear interpolation in 4D component space:
    //   (1-t)a + t b
    // Note:
    //   Not constant angular velocity and not necessarily unit-length.
    return a * (T(1) - t) + b * t;
}

template <typename T>
Quaternion<T>
Quaternion<T>::nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    // Normalized lerp:
    //   nlerp = normalize(lerp(a,b,t))
    // Faster than slerp and often good enough for animation.
    Quaternion r = lerp(a, b, t);
    r.normalize();
    return r;
}

template <typename T>
Quaternion<T>
Quaternion<T>::slerp(const Quaternion& a, const Quaternion& b, T t) noexcept {
    // ------------------------------------------------------------
    // Algorithm: Spherical linear interpolation (SLERP)
    //
    // Key idea:
    //   Interpolate along the great circle on the unit 4D sphere.
    //   Produces constant angular velocity and stays unit-length.
    //
    // Steps:
    //   1) cos_theta = dot(a, b)
    //   2) If cos_theta < 0, flip b to take the shorter arc
    //      (q and -q represent the same rotation).
    //   3) If cos_theta ~ 1, fall back to nlerp to avoid division by tiny sin(theta).
    //   4) Otherwise:
    //        theta = acos(cos_theta)
    //        w1 = sin((1-t)theta) / sin(theta)
    //        w2 = sin(t theta)     / sin(theta)
    //        return a*w1 + b*w2
    // ------------------------------------------------------------
    T cos_theta   = a.dot(b);
    Quaternion bb = b;

    // Flip to ensure shortest-path interpolation.
    if (cos_theta < T(0)) {
        bb        = Quaternion(-b.w, -b.x, -b.y, -b.z);
        cos_theta = -cos_theta;
    }

    // If angle is very small, use nlerp to avoid numerical issues.
    if (cos_theta > T(1) - T(1e-6)) {
        return nlerp(a, bb, t);
    }

    const T theta     = std::acos(cos_theta);
    const T sin_theta = std::sin(theta);

    const T w1 = std::sin((T(1) - t) * theta) / sin_theta;
    const T w2 = std::sin(t * theta) / sin_theta;

    return a * w1 + bb * w2;
}

template <typename T>
template <typename To>
Quaternion<To>
Quaternion<T>::cast_to() const noexcept {
    // Type conversion utility (e.g., float -> double).
    return Quaternion<To>(
        static_cast<To>(w),
        static_cast<To>(x),
        static_cast<To>(y),
        static_cast<To>(z));
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator+(const Quaternion& q) const noexcept {
    // Component-wise addition in 4D.
    return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator-(const Quaternion& q) const noexcept {
    // Component-wise subtraction in 4D.
    return Quaternion(w - q.w, x - q.x, y - q.y, z - q.z);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator*(const Quaternion& q) const noexcept {
    // ------------------------------------------------------------
    // Algorithm: Hamilton product (quaternion multiplication)
    //
    // This composes rotations when quaternions are unit-length.
    // Note the multiplication is not commutative (q1*q2 != q2*q1).
    // ------------------------------------------------------------
    return Quaternion(
        w * q.w - x * q.x - y * q.y - z * q.z,
        w * q.x + x * q.w + y * q.z - z * q.y,
        w * q.y - x * q.z + y * q.w + z * q.x,
        w * q.z + x * q.y - y * q.x + z * q.w);
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator*=(const Quaternion& q) noexcept {
    // In-place Hamilton product (rotation composition).
    *this = (*this) * q;
    return *this;
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator*(T s) const noexcept {
    // Scalar multiply in 4D.
    return Quaternion(w * s, x * s, y * s, z * s);
}

template <typename T>
Quaternion<T>
Quaternion<T>::operator/(T s) const noexcept {
    // Scalar divide in 4D.
    const T inv = T(1) / s;
    return Quaternion(w * inv, x * inv, y * inv, z * inv);
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator+=(const Quaternion& q) noexcept {
    // In-place component add.
    w += q.w;
    x += q.x;
    y += q.y;
    z += q.z;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator-=(const Quaternion& q) noexcept {
    // In-place component subtract.
    w -= q.w;
    x -= q.x;
    y -= q.y;
    z -= q.z;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator*=(T s) noexcept {
    // In-place scalar multiply.
    w *= s;
    x *= s;
    y *= s;
    z *= s;
    return *this;
}

template <typename T>
Quaternion<T>&
Quaternion<T>::operator/=(T s) noexcept {
    // In-place scalar divide.
    const T inv = T(1) / s;
    w *= inv;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
}

template <typename T>
bool
Quaternion<T>::operator==(const Quaternion& q) const noexcept {
    // Approximate equality with machine epsilon tolerance.
    // Note:
    //   For rotations, q and -q represent the same orientation,
    //   but this operator treats them as different values.
    return std::abs(w - q.w) < eps && std::abs(x - q.x) < eps && std::abs(y - q.y) < eps && std::abs(z - q.z) < eps;
}

template <typename T>
bool
Quaternion<T>::operator!=(const Quaternion& q) const noexcept {
    // Negation of approximate equality.
    return !(*this == q);
}

} // namespace atlas::math
