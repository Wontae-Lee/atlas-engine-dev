#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>
#include <atlas/sync/sync.h>

#include <optional>
#include <type_traits>

namespace atlas::physics {

/**
 * @brief Represents a physical unit that couples geometry, synchronization state,
 *        and optional kinematic properties.
 *
 * This class stores the geometry access operator, the synchronization/operator state
 * used for spatial transforms, and optional linear/angular kinematics.
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
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
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
     * @param velocity Optional linear velocity in world space.
     * @param acceleration Optional linear acceleration in world space.
     * @param angular_velocity Optional angular velocity vector in world space.
     * @param angular_acceleration Optional angular acceleration vector in world space.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
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
     * If the unit is not dynamic, the function does nothing.
     *
     * @param dt Time step in seconds.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    /**
     * @brief Applies a world-space translation to the unit.
     *
     * @param delta_world Translation vector in world coordinates.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta_world) noexcept;

    /**
     * @brief Applies a world-space rotation to the unit.
     *
     * The axis vector is normalized internally. If the axis length is zero,
     * the function does nothing.
     *
     * @param axis_world Rotation axis in world coordinates.
     * @param angle_rad Rotation angle in radians.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis_world, T angle_rad) noexcept;

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
     * @brief Optional linear velocity in world coordinates.
     */
    std::optional<Vector<T, 3>> _velocity;

    /**
     * @brief Optional linear acceleration in world coordinates.
     */
    std::optional<Vector<T, 3>> _acceleration;

    /**
     * @brief Optional angular velocity in world coordinates.
     */
    std::optional<Vector<T, 3>> _angular_velocity;

    /**
     * @brief Optional angular acceleration in world coordinates.
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
     * @param v Linear velocity in world coordinates.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_velocity(const Vector<T, 3>& v) noexcept;

    /**
     * @brief Sets the linear acceleration.
     *
     * @param a Linear acceleration in world coordinates.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_acceleration(const Vector<T, 3>& a) noexcept;

    /**
     * @brief Sets the angular velocity.
     *
     * @param w Angular velocity in world coordinates.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_velocity(const Vector<T, 3>& w) noexcept;

    /**
     * @brief Sets the angular acceleration.
     *
     * @param alpha Angular acceleration in world coordinates.
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

}

namespace atlas {

/**
 * @brief Alias for atlas::physics::Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Unit = atlas::physics::Unit<T>;

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

}

#include <atlas/unit/unit.hpp>