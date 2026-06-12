#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>

#include <optional>
#include <type_traits>

namespace atlas {

/**
 * @brief Represents a physical unit that couples geometry, synchronization state,
 *        and optional kinematic properties.
 *
 * This class stores a geometry operator, a synchronization operator, and optional
 * linear/angular kinematic quantities. Transform interpretation is delegated to
 * the synchronization operator.
 *
 * A unit can behave as:
 * - static, when no velocity information is present
 * - dynamic, when linear and/or angular velocity is present
 *
 * @tparam T Floating-point scalar type used for all numeric computations.
 */
template <typename T>
class Unit final {
    static_assert(std::is_floating_point_v<T>, "Unit requires a floating-point T");

public:
    /**
     * @brief Builder type used to construct a Unit instance safely on the host side.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * Constructs an empty unit with default-initialized operators and no kinematics.
     */
    Unit() = default;

    /**
     * @brief Default destructor.
     */
    ~Unit() = default;

    /**
     * @brief Constructs a unit with geometry and sync operators only.
     *
     * The created unit has no linear or angular kinematics.
     *
     * @param geometry_operator Geometry access/operator object.
     * @param sync_operator Synchronization/operator object containing transform state.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(atlas::GeometryOperator<T> geometry_operator,
         SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Constructs a unit with geometry, sync, and optional kinematic quantities.
     *
     * Missing velocity/acceleration pairs are canonicalized so that if one side of a
     * kinematic pair is present and the other is absent, the missing one is replaced
     * with a zero vector.
     *
     * @param geometry_operator Geometry access/operator object.
     * @param sync_operator Synchronization/operator object containing transform state.
     * @param velocity Optional linear velocity.
     * @param acceleration Optional linear acceleration.
     * @param angular_velocity Optional angular velocity vector.
     * @param angular_acceleration Optional angular acceleration vector.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(atlas::GeometryOperator<T> geometry_operator,
         SyncOperator<T> sync_operator,
         std::optional<Vector<T, 3>> velocity,
         std::optional<Vector<T, 3>> acceleration,
         std::optional<Vector<T, 3>> angular_velocity,
         std::optional<Vector<T, 3>> angular_acceleration) noexcept;

    /**
     * @brief Creates a builder instance for host-side construction.
     *
     * @return Builder object initialized for chained configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Replaces the geometry operator.
     *
     * @param geometry_operator New geometry operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept;

    /**
     * @brief Replaces the synchronization operator.
     *
     * @param sync_operator New sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_sync_operator(SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Advances the unit state by a time step.
     *
     * This function updates linear and angular velocities using the corresponding
     * accelerations, then applies translation and rotation to the sync state.
     *
     * If @p dt is not strictly positive, the function does nothing.
     *
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    /**
     * @brief Translates the unit by applying a displacement vector.
     *
     * This function updates the translation component stored in the synchronization
     * operator by adding the given displacement vector to the current translation.
     *
     * Mathematically, if the current translation is @f$\mathbf{x}@f$ and the input
     * displacement is @f$\Delta \mathbf{x}@f$, the updated translation is
     *
     * @f[
     *     \mathbf{x}_{\mathrm{new}}
     *     =
     *     \mathbf{x}_{\mathrm{old}}
     *     +
     *     \Delta \mathbf{x}.
     * @f]
     *
     * This operation only modifies the translational part of the synchronization
     * state. It does not directly modify orientation, velocity, acceleration, or
     * geometry data. Any coordinate-frame interpretation of the displacement is
     * determined by the synchronization operator and the surrounding simulation
     * convention.
     *
     * @param delta Displacement vector added to the current translation.
     *
     * @note This function performs a direct additive update and does not integrate
     *       velocity or acceleration by itself. Time integration is handled by
     *       update().
     * @note The geometry operator is not modified directly. The unit transform is
     *       changed through the synchronization operator.
     *
     * @see update()
     * @see SyncOperator<T>
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta) noexcept;

    /**
     * @brief Rotates the unit by applying an axis-angle rotation.
     *
     * This function updates the orientation component stored in the synchronization
     * operator by constructing an incremental quaternion from the given rotation
     * axis and angle, then composing it with the current orientation.
     *
     * Given an input axis @f$\mathbf{a}@f$ and rotation angle @f$\theta@f$, the axis
     * is first normalized:
     *
     * @f[
     *     \hat{\mathbf{a}}
     *     =
     *     \frac{\mathbf{a}}{\|\mathbf{a}\|}.
     * @f]
     *
     * The normalized axis and angle define an incremental unit quaternion:
     *
     * @f[
     *     q_{\Delta}
     *     =
     *     \left(
     *         \cos\frac{\theta}{2},
     *         \hat{a}_x \sin\frac{\theta}{2},
     *         \hat{a}_y \sin\frac{\theta}{2},
     *         \hat{a}_z \sin\frac{\theta}{2}
     *     \right).
     * @f]
     *
     * The current orientation quaternion @f$q_{\mathrm{old}}@f$ is then updated by
     * quaternion composition:
     *
     * @f[
     *     q_{\mathrm{new}}
     *     =
     *     \mathrm{normalize}
     *     \left(
     *         q_{\Delta} q_{\mathrm{old}}
     *     \right).
     * @f]
     *
     * The final normalization step reduces numerical drift caused by repeated
     * floating-point quaternion multiplications. After the orientation update, the
     * synchronization operator rebuilds its dependent transformation matrices so
     * that the matrix representation remains consistent with the quaternion state.
     *
     * If the input axis has zero length, no valid axis-angle rotation can be formed,
     * so the function returns without modifying the unit.
     *
     * This function does not assign world-space or local-space semantics to the
     * input axis. Coordinate-frame interpretation is delegated to the synchronization
     * operator and the simulation convention using it.
     *
     * @param axis Rotation axis. The vector does not need to be normalized by the
     *             caller.
     * @param angle_rad Rotation angle in radians.
     *
     * @note The angle is not clamped, wrapped, or converted. It is passed directly
     *       to Quaternion<T>::from_axis_angle().
     * @note This function modifies only the orientation stored in the synchronization
     *       operator and then rebuilds the associated transform matrices.
     * @note The geometry operator is not modified directly.
     *
     * @par Mathematical summary
     * @f[
     *     \mathbf{a} \neq \mathbf{0},
     *     \qquad
     *     \hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|},
     *     \qquad
     *     q_{\Delta}
     *     =
     *     \left(
     *         \cos\frac{\theta}{2},
     *         \hat{\mathbf{a}}\sin\frac{\theta}{2}
     *     \right),
     *     \qquad
     *     q_{\mathrm{new}}
     *     =
     *     \mathrm{normalize}
     *     \left(
     *         q_{\Delta} q_{\mathrm{old}}
     *     \right).
     * @f]
     *
     * @see Quaternion<T>::from_axis_angle()
     * @see SyncOperator<T>::rebuild_matrices()
     * @see update()
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis, T angle_rad) noexcept;

    /**
     * @brief Returns the geometry operator.
     *
     * @return Constant reference to the geometry operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::GeometryOperator<T>&
    geometry_operator() const noexcept;

    /**
     * @brief Returns the synchronization operator.
     *
     * @return Constant reference to the sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync_operator() const noexcept;

    /**
     * @brief Returns the optional linear velocity.
     *
     * @return Constant reference to the optional linear velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    velocity() const noexcept;

    /**
     * @brief Returns the optional linear acceleration.
     *
     * @return Constant reference to the optional linear acceleration.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    acceleration() const noexcept;

    /**
     * @brief Returns the optional angular velocity.
     *
     * @return Constant reference to the optional angular velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_velocity() const noexcept;

    /**
     * @brief Returns the optional angular acceleration.
     *
     * @return Constant reference to the optional angular acceleration.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_acceleration() const noexcept;

    /**
     * @brief Checks whether the unit has dynamic motion state.
     *
     * A unit is considered dynamic if it has either linear velocity or angular velocity.
     *
     * @return True if the unit is dynamic, false otherwise.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    dynamic() const noexcept;

private:
    /**
     * @brief Canonicalizes the kinematic option sets.
     *
     * This function ensures that velocity/acceleration and angular_velocity/
     * angular_acceleration are always paired consistently.
     *
     * Rules:
     * - If acceleration exists and velocity is missing, velocity becomes zero.
     * - If velocity exists and acceleration is missing, acceleration becomes zero.
     * - If angular acceleration exists and angular velocity is missing, angular velocity becomes zero.
     * - If angular velocity exists and angular acceleration is missing, angular acceleration becomes zero.
     *
     * @param velocity Optional linear velocity to normalize.
     * @param acceleration Optional linear acceleration to normalize.
     * @param angular_velocity Optional angular velocity to normalize.
     * @param angular_acceleration Optional angular acceleration to normalize.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                            std::optional<Vector<T, 3>>& acceleration,
                            std::optional<Vector<T, 3>>& angular_velocity,
                            std::optional<Vector<T, 3>>& angular_acceleration) noexcept;

    friend class Builder;

private:
    /**
     * @brief Geometry access/operator state.
     */
    atlas::GeometryOperator<T> _geometry_operator;

    /**
     * @brief Synchronization/operator state containing transform data.
     */
    SyncOperator<T> _sync_operator;

    /**
     * @brief Optional linear velocity.
     */
    std::optional<Vector<T, 3>> _velocity;

    /**
     * @brief Optional linear acceleration.
     */
    std::optional<Vector<T, 3>> _acceleration;

    /**
     * @brief Optional angular velocity.
     */
    std::optional<Vector<T, 3>> _angular_velocity;

    /**
     * @brief Optional angular acceleration.
     */
    std::optional<Vector<T, 3>> _angular_acceleration;
};

/**
 * @brief Host-side builder for Unit construction.
 *
 * This builder collects required dependencies and optional kinematic parameters,
 * validates them, and then constructs a finalized Unit object.
 *
 * @tparam T Floating-point scalar type used by the target Unit.
 */
template <typename T>
class Unit<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the geometry owner and derives its geometry operator.
     *
     * @param geometry Shared host pointer to geometry.
     * @return Reference to this builder.
     *
     * @throws std::runtime_error Thrown if @p geometry is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const atlas::GeometryHostPtr<T>& geometry);

    /**
     * @brief Sets the synchronization owner and derives its sync operator.
     *
     * @param sync Shared host pointer to sync state.
     * @return Reference to this builder.
     *
     * @throws std::runtime_error Thrown if @p sync is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync(const SyncHostPtr<T>& sync);

    /**
     * @brief Sets the linear velocity.
     *
     * @param v Linear velocity.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_velocity(const Vector<T, 3>& v) noexcept;

    /**
     * @brief Sets the linear acceleration.
     *
     * @param a Linear acceleration.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_acceleration(const Vector<T, 3>& a) noexcept;

    /**
     * @brief Sets the angular velocity.
     *
     * @param w Angular velocity.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_velocity(const Vector<T, 3>& w) noexcept;

    /**
     * @brief Sets the angular acceleration.
     *
     * @param alpha Angular acceleration.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_acceleration(const Vector<T, 3>& alpha) noexcept;

    /**
     * @brief Validates the collected state and builds a Unit instance.
     *
     * The builder state is reset after successful construction.
     *
     * @return Fully constructed Unit object.
     *
     * @throws std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Unit<T>
    build();

    /**
     * @brief Builds a Unit instance and wraps it in a host shared pointer.
     *
     * @return Shared host pointer owning the constructed Unit.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Unit<T>>
    make_host_shared();

private:
    /**
     * @brief Validates whether the builder contains enough data to build a valid unit.
     *
     * @throws std::runtime_error Thrown if any required dependency is missing
     *                            or if kinematic constraints are violated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Optional owning geometry pointer retained during build preparation.
     */
    std::optional<GeometryHostPtr<T>> _geometry;

    /**
     * @brief Optional geometry operator derived from the geometry owner.
     */
    std::optional<atlas::GeometryOperator<T>> _geometry_operator;

    /**
     * @brief Optional sync operator derived from the sync owner.
     */
    std::optional<SyncOperator<T>> _sync_operator;

    /**
     * @brief Optional linear velocity.
     */
    std::optional<Vector<T, 3>> _velocity;

    /**
     * @brief Optional linear acceleration.
     */
    std::optional<Vector<T, 3>> _acceleration;

    /**
     * @brief Optional angular velocity.
     */
    std::optional<Vector<T, 3>> _angular_velocity;

    /**
     * @brief Optional angular acceleration.
     */
    std::optional<Vector<T, 3>> _angular_acceleration;
};

} // namespace atlas

namespace atlas {


/**
 * @brief Host shared pointer alias for Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using UnitHostPtr = atlas::host_shared_ptr<Unit<T>>;

/**
 * @brief Device shared pointer alias for Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using UnitDevicePtr = atlas::device_shared_ptr<Unit<T>>;

} // namespace atlas

#include <atlas/unit/unit.hpp>