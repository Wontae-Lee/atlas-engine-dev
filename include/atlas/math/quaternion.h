#pragma once
#include <atlas/math/matrix/matrix.h>
#include <atlas/math/vector/vector3.h>
#include <initializer_list>
#include <type_traits>

namespace atlas ::math {

/**
 * @brief Quaternion rotation primitive for 3D orientation and interpolation.
 *
 * @details
 * `Quaternion<T>` is Atlas' compact, value-type representation of 3D rotations.
 * It stores four scalars `(w, x, y, z)` in contiguous memory and provides
 * construction, composition, conversion, and interpolation utilities commonly
 * used in simulation, geometry, and rendering pipelines.
 *
 * Mathematical model:
 * - Quaternion:
 *     q = (w, x, y, z) = w + x*i + y*j + z*k
 * - Norm:
 *     |q|^2 = w^2 + x^2 + y^2 + z^2
 * - Conjugate:
 *     conj(q) = (w, -x, -y, -z)
 * - Inverse:
 *     q^{-1} = conj(q) / |q|^2
 *   For unit quaternions (|q| = 1), q^{-1} = conj(q).
 *
 * What it does:
 * - Represents rotations as (typically) unit quaternions.
 * - Composes rotations via the Hamilton product (non-commutative).
 * - Rotates vectors with the sandwich product:
 *     v' = q * (0, v) * q^{-1}
 * - Converts between quaternion and rotation matrices (3x3 / 4x4).
 * - Interpolates orientations using LERP / NLERP / SLERP.
 *
 * Conventions used in this implementation:
 * - Rotation application convention:
 *     v' = q * v * q^{-1}
 *   where v is embedded as a pure quaternion (0, vx, vy, vz).
 * - Euler XYZ constructor uses:
 *     q = qz * qy * qx
 *   which corresponds to applying X, then Y, then Z (under the above convention).
 * - Double-cover property:
 *     q and -q represent the same physical rotation.
 *   Component equality does not account for this equivalence.
 *
 * Construction modes:
 * - Identity:
 *     (1, 0, 0, 0)
 * - Axis–angle:
 *     q = (cos(θ/2), a*sin(θ/2))
 *   where a is the rotation axis (typically unit length).
 * - Euler XYZ:
 *     built from half-angle sines/cosines and composed as qz*qy*qx.
 * - From 3x3 matrix:
 *     trace-based branching to improve numerical stability.
 *
 * Interpolation behavior:
 * - `lerp(a,b,t)`:
 *     linear interpolation in R^4; may drift from unit length.
 * - `nlerp(a,b,t)`:
 *     normalize(lerp(a,b,t)); fast, common default for many pipelines.
 * - `slerp(a,b,t)`:
 *     spherical interpolation on the unit 4D sphere; constant angular velocity.
 *   Uses shortest-arc handling by flipping b when dot(a,b) < 0, and falls back
 *   to NLERP when the angle is very small to avoid division by tiny sin(θ).
 *
 * Numerical notes:
 * - Unit length is not automatically enforced; normalize() after repeated
 *   composition or accumulation when using quaternions as rotations.
 * - `rotate()` uses conjugate() as the inverse; this is exact only for unit
 *   quaternions. Non-unit inputs may introduce scaling.
 * - `from_matrix3x3()` assumes the input is close to a proper rotation matrix.
 *
 * @tparam T Floating-point scalar type (e.g., `float`, `double`).
 *
 * @see Vector3
 * @see Matrix
 */
template <typename T>
class Quaternion {
    static_assert(std::is_floating_point_v<T>, "Quaternion requires a floating-point type");

public:
    /// @brief Scalar (real) part.
    T w;

    /// @brief X component of the vector (imaginary) part.
    T x;

    /// @brief Y component of the vector (imaginary) part.
    T y;

    /// @brief Z component of the vector (imaginary) part.
    T z;

    /**
     * @brief Default constructor (identity rotation).
     *
     * @details
     * Initializes to the identity quaternion:
     * \f[
     *   q = (1,0,0,0).
     * \f]
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion() noexcept;

    /**
     * @brief Constructs a quaternion from explicit components.
     *
     * @param w_ Scalar part.
     * @param x_ X component of vector part.
     * @param y_ Y component of vector part.
     * @param z_ Z component of vector part.
     *
     * @note
     * No normalization is performed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE constexpr Quaternion(T w_, T x_, T y_, T z_) noexcept;

    /**
     * @brief Constructs a quaternion from an initializer list.
     *
     * @details
     * Expected order is `{w, x, y, z}`. If fewer than 4 values are provided, missing
     * components are filled with identity defaults (implementation-defined; commonly
     * `{1,0,0,0}` with remaining zeros).
     *
     * @param list Initializer list of up to 4 scalars.
     *
     * @note
     * No normalization is performed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(std::initializer_list<T> list) noexcept;

    /**
     * @brief Constructs a rotation quaternion from axis-angle.
     *
     * @details
     * For a unit axis \f$\mathbf{a}\f$ and angle \f$\theta\f$ (radians),
     * a rotation quaternion is:
     * \f[
     *   q = \left(\cos\frac{\theta}{2},\; \mathbf{a}\sin\frac{\theta}{2}\right).
     * \f]
     *
     * @param axis Rotation axis (should be normalized for a pure rotation).
     * @param radians Rotation angle in radians.
     *
     * @note
     * If `axis` is not normalized, the resulting quaternion will generally not be unit length.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(const Vector3<T>& axis, T radians) noexcept;

    /**
     * @brief Constructs a quaternion from Euler angles (XYZ order).
     *
     * @details
     * Builds a quaternion representing a composition of rotations about X, then Y, then Z
     * (convention implied by the name). Angles are in radians.
     *
     * @param rx Rotation about X axis (radians).
     * @param ry Rotation about Y axis (radians).
     * @param rz Rotation about Z axis (radians).
     *
     * @note
     * Euler conventions are easy to misinterpret; confirm the exact multiplication order in
     * the implementation (`quaternion.hpp`) if strict compatibility is required.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Quaternion(T rx, T ry, T rz) noexcept;

    /**
     * @brief Constructs a quaternion from a 3×3 rotation matrix.
     *
     * @details
     * Converts a 3×3 matrix to a quaternion. The input matrix is expected to be a valid
     * rotation matrix (orthonormal with determinant +1) for best results.
     *
     * @param m 3×3 matrix.
     *
     * @note
     * Behavior is implementation-defined if `m` is not a rotation matrix.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Quaternion(const Matrix<T, 3, 3>& m) noexcept;

    Quaternion(const Quaternion&) noexcept = default;
    ~Quaternion() noexcept                 = default;
    Quaternion&
    operator=(const Quaternion&) noexcept = default;
    Quaternion&
    operator=(Quaternion&&) noexcept = default;

    /**
     * @brief Creates a quaternion from axis-angle (factory).
     *
     * @param axis Rotation axis.
     * @param radians Rotation angle (radians).
     * @return Quaternion representing the axis-angle rotation.
     *
     * @see Quaternion(const Vector3<T>&, T)
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_axis_angle(const Vector3<T>& axis, T radians) noexcept;

    /**
     * @brief Creates a quaternion from Euler XYZ angles (factory).
     *
     * @param rx Rotation about X axis (radians).
     * @param ry Rotation about Y axis (radians).
     * @param rz Rotation about Z axis (radians).
     * @return Quaternion representing the Euler rotation.
     *
     * @see Quaternion(T, T, T)
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_euler_xyz(T rx, T ry, T rz) noexcept;

    /**
     * @brief Creates a quaternion from a 3×3 matrix (factory).
     *
     * @param m 3×3 matrix (expected rotation matrix).
     * @return Quaternion converted from the matrix.
     *
     * @see Quaternion(const Matrix<T,3,3>&)
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    from_matrix3x3(const Matrix<T, 3, 3>& m) noexcept;

    /**
     * @brief Returns a pointer to the contiguous quaternion storage (const).
     *
     * @return Pointer to `w` (followed by `x`, `y`, `z`).
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const T*
    data() const noexcept;

    /**
     * @brief Returns a pointer to the contiguous quaternion storage (mutable).
     *
     * @return Pointer to `w` (followed by `x`, `y`, `z`).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T*
    data() noexcept;

    /**
     * @brief Dot product between two quaternions.
     *
     * @param q Other quaternion.
     * @return \f$w w' + x x' + y y' + z z'\f$.
     *
     * @note
     * For unit rotation quaternions, the dot product relates to the angular distance
     * between orientations.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    dot(const Quaternion& q) const noexcept;

    /**
     * @brief Squared Euclidean length of the quaternion.
     *
     * @return \f$\|q\|^2 = w^2 + x^2 + y^2 + z^2\f$.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length_squared() const noexcept;

    /**
     * @brief Euclidean length (magnitude) of the quaternion.
     *
     * @return \f$\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2}\f$.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    length() const noexcept;

    /**
     * @brief Normalizes the quaternion in-place.
     *
     * @details
     * Scales the quaternion so that \f$\|q\| = 1\f$ (if length is non-zero).
     *
     * @note
     * If the quaternion length is near zero, the behavior is implementation-defined
     * (commonly leaves it unchanged or resets to identity).
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    normalize() noexcept;

    /**
     * @brief Returns a normalized copy of this quaternion.
     *
     * @return Unit-length quaternion (if normalization succeeds).
     *
     * @see normalize()
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    normalized() const noexcept;

    /**
     * @brief Returns the conjugate quaternion.
     *
     * @details
     * Conjugation negates the vector part:
     * \f[
     *   \overline{q} = (w, -x, -y, -z).
     * \f]
     *
     * @return Conjugated quaternion.
     *
     * @note
     * For unit quaternions, conjugate equals inverse.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    conjugate() const noexcept;

    /**
     * @brief Returns the inverse quaternion.
     *
     * @details
     * The inverse is:
     * \f[
     *   q^{-1} = \frac{\overline{q}}{\|q\|^2}.
     * \f]
     *
     * @return Inverse quaternion.
     *
     * @note
     * For unit quaternions, this is the conjugate.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    inverse() const noexcept;

    /**
     * @brief Checks whether this quaternion is approximately the identity rotation.
     *
     * @details
     * A common identity test checks that the quaternion is close to `(1,0,0,0)`
     * within tolerance `eps`, typically component-wise or via angle.
     *
     * @param eps Tolerance (defaults to machine epsilon).
     * @return `true` if approximately identity; otherwise `false`.
     *
     * @note
     * The exact criterion is implementation-defined.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_identity(T eps = atlas::eps) const noexcept;

    /**
     * @brief Rotates a 3D vector by this quaternion.
     *
     * @details
     * This function applies a 3D rotation using the quaternion "sandwich product".
     * The input vector is embedded into quaternion space as a pure quaternion
     * (zero scalar part), then rotated by conjugation:
     *
     * \f[
     *   \mathbf{v}' = q \, (0,\mathbf{v}) \, q^{-1}.
     * \f]
     *
     * where \f$(0,\mathbf{v}) = (0, v_x, v_y, v_z)\f$.
     *
     * Geometric interpretation:
     * - Let \f$q = \left(\cos\frac{\theta}{2}, \mathbf{a}\sin\frac{\theta}{2}\right)\f$
     *   represent a rotation of angle \f$\theta\f$ around unit axis \f$\mathbf{a}\f$.
     * - The result \f$\mathbf{v}'\f$ is the vector \f$\mathbf{v}\f$ rotated by
     *   angle \f$\theta\f$ about axis \f$\mathbf{a}\f$ (right-hand rule).
     *
     * Equivalent closed-form (Rodrigues formula):
     * \f[
     *   \mathbf{v}' =
     *   \mathbf{v}\cos\theta
     *   + (\mathbf{a} \times \mathbf{v})\sin\theta
     *   + \mathbf{a}(\mathbf{a}\cdot\mathbf{v})(1-\cos\theta).
     * \f]
     *
     * Conceptual diagram:
     *
     * @verbatim
     *                a (rotation axis)
     *                ^
     *                |
     *                |
     *        v  ---->•------  v'
     *             \          /
     *              \        /
     *               \  θ   /
     *                \    /
     *                 \  /
     *                  \/
     *
     *   The vector v is rotated by angle θ about axis a,
     *   producing v'. The quaternion q encodes (a, θ/2).
     * @endverbatim
     *
     * Interpretation and conventions:
     * - Active rotation convention is used:
     *     v' = q v q^{-1}
     * - Quaternion multiplication is non-commutative.
     * - Changing multiplication order changes the meaning of the rotation.
     *
     * Requirements for a pure rotation:
     * - If \f$\|q\|=1\f$, then \f$q^{-1}=\overline{q}\f$ and vector length is preserved:
     *     \f$\|\mathbf{v}'\|=\|\mathbf{v}\|\f$.
     * - If q is not normalized, using only the conjugate instead of the full inverse
     *   may introduce a scaling factor (typically \f$\|q\|^2\f$).
     *
     * Numerical notes:
     * - Due to floating-point arithmetic, the intermediate scalar part may be
     *   slightly non-zero; only the vector part (x,y,z) is returned.
     * - Repeated composition of rotations may cause drift; periodic normalization
     *   is recommended.
     *
     * @param v Input vector.
     * @return Rotated vector \f$\mathbf{v}'\f$.
     *
     * @note
     * This quaternion should be normalized to represent a pure rotation.
     *
     * @see conjugate()
     * @see inverse()
     * @see normalize()
     * @see normalized()
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    rotate(const Vector3<T>& v) const noexcept;

    /**
     * @brief Converts this quaternion to a 3×3 rotation matrix.
     *
     * @return 3×3 matrix representing the same rotation (if quaternion is unit length).
     *
     * @note
     * If the quaternion is not normalized, the resulting matrix may include scaling/shear artifacts.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 3, 3>
    to_matrix3x3() const noexcept;

    /**
     * @brief Converts this quaternion to a 4×4 rotation matrix (homogeneous).
     *
     * @return 4×4 matrix whose upper-left 3×3 is the rotation.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Matrix<T, 4, 4>
    to_matrix4x4() const noexcept;

    /**
     * @brief Linear interpolation between two quaternions (no normalization).
     *
     * @details
     * Computes:
     * \f[
     *   (1-t)a + tb.
     * \f]
     *
     * @param a Start quaternion.
     * @param b End quaternion.
     * @param t Interpolation parameter in [0,1].
     * @return Interpolated quaternion (not normalized).
     *
     * @note
     * Not recommended for rotations unless followed by normalization.
     *
     * @see nlerp(), slerp()
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    lerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

    /**
     * @brief Normalized linear interpolation between two quaternions.
     *
     * @details
     * Equivalent to `normalize(lerp(a,b,t))`. Often used as a fast approximation to `slerp`.
     *
     * @param a Start quaternion.
     * @param b End quaternion.
     * @param t Interpolation parameter in [0,1].
     * @return Unit quaternion approximating spherical interpolation.
     *
     * @note
     * Many implementations flip `b` when `dot(a,b) < 0` to follow the shortest arc.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    nlerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

    /**
     * @brief Spherical linear interpolation between two quaternions.
     *
     * @details
     * Interpolates along the unit quaternion hypersphere, producing constant angular velocity
     * interpolation (for unit quaternions).
     *
     * @param a Start quaternion.
     * @param b End quaternion.
     * @param t Interpolation parameter in [0,1].
     * @return Interpolated quaternion (unit length for unit inputs).
     *
     * @note
     * Typical implementations handle the `dot(a,b) < 0` case by negating one input to ensure
     * shortest-path interpolation.
     */
    ATLAS_ALL_DEVICE static ATLAS_FORCE_INLINE Quaternion
    slerp(const Quaternion& a, const Quaternion& b, T t) noexcept;

    /**
     * @brief Casts quaternion components to a different floating-point type.
     *
     * @tparam To Destination scalar type.
     * @return Quaternion with components converted to `To`.
     */
    template <typename To>
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion<To>
    cast_to() const noexcept;

    /// @brief Component-wise addition.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator+(const Quaternion& q) const noexcept;

    /// @brief Component-wise subtraction.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator-(const Quaternion& q) const noexcept;

    /**
     * @brief Hamilton product of two quaternions.
     *
     * @details
     * Quaternion multiplication composes rotations (for unit quaternions). The exact composition
     * order depends on your convention (right-multiply vs left-multiply), but algebraically this
     * is the standard Hamilton product.
     *
     * @param q Right operand.
     * @return Product quaternion.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(const Quaternion& q) const noexcept;

    /// @brief Scalar multiplication.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator*(T s) const noexcept;

    /// @brief Scalar division.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion
    operator/(T s) const noexcept;

    /// @brief In-place addition.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator+=(const Quaternion& q) noexcept;

    /// @brief In-place subtraction.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator-=(const Quaternion& q) noexcept;

    /// @brief In-place Hamilton product.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(const Quaternion& q) noexcept;

    /// @brief In-place scalar multiplication.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator*=(T s) noexcept;

    /// @brief In-place scalar division.
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Quaternion&
    operator/=(T s) noexcept;

    /**
     * @brief Exact component-wise equality.
     *
     * @param q Other quaternion.
     * @return `true` if all components match exactly.
     *
     * @note
     * For floating-point types, prefer an epsilon-based comparison for robustness.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator==(const Quaternion& q) const noexcept;

    /**
     * @brief Exact component-wise inequality.
     *
     * @param q Other quaternion.
     * @return `true` if any component differs.
     */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    operator!=(const Quaternion& q) const noexcept;
};

} // namespace math

namespace atlas {

template <typename T>
using Quaternion  = math::Quaternion<T>;
using QuaternionF = math::Quaternion<float>;
using QuaternionD = math::Quaternion<double>;

} // namespace atlas

#include <atlas/math/quaternion.hpp>
