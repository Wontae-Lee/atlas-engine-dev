#pragma once

/**
 * @file unit.h
 * @brief Declares a movable geometry instance together with its pose and optional rigid-body kinematics.
 *
 * @details
 * This header defines @ref atlas::system::Unit, the basic scene/runtime object
 * that combines:
 * - a geometry operator describing the local-space shape,
 * - a sync operator describing the local-to-world rigid transform,
 * - optional linear and angular motion state.
 *
 * A unit is the canonical way to represent a placed geometry instance that can
 * be:
 * - translated,
 * - rotated,
 * - advanced in time,
 * - queried by higher-level systems such as sources, sinks, colliders, and visualization.
 *
 * ## Conceptual model
 * A unit couples **what the object is** with **where it is** and **how it moves**:
 * - geometry is represented by @ref atlas::GeometryOperator,
 * - pose is represented by @ref SyncOperator,
 * - motion is represented by optional velocity and acceleration terms.
 *
 * ## Local and world space
 * The stored geometry operator is interpreted in the unit's local coordinate
 * frame. The sync operator maps that local frame into world space.
 *
 * This separation allows:
 * - efficient reuse of analytic or mesh geometry,
 * - independent movement of multiple instances of the same underlying shape,
 * - runtime transformation without modifying the original geometry definition.
 *
 * ## Kinematic state
 * A unit may optionally carry:
 * - linear velocity,
 * - linear acceleration,
 * - angular velocity,
 * - angular acceleration.
 *
 * These are optional because some units are static while others are dynamic.
 *
 * ## Construction
 * A unit may be:
 * - default-constructed,
 * - directly constructed from geometry and sync operators,
 * - directly constructed from geometry, sync, and optional kinematics,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Couples geometry, pose, and optional rigid-body kinematics.
 *
 * @details
 * @ref Unit is the primary movable scene object in Atlas. It stores:
 * - a local-space geometry operator,
 * - a rigid transform operator,
 * - optional linear and angular motion state.
 *
 * ## Responsibilities
 * A unit can:
 * - expose its local geometry,
 * - expose its transform,
 * - advance its kinematic state over a timestep,
 * - be translated in world space,
 * - be rotated in world space,
 * - report whether it should be treated as dynamic.
 *
 * ## Dynamic vs static units
 * A unit is typically considered dynamic if it has any active linear or angular
 * motion term. Static units can still carry geometry and pose, but do not
 * evolve under @ref update unless explicitly transformed.
 *
 * ## Kinematic normalization
 * Internally, the implementation may normalize optional kinematic state through
 * @ref canonicalize_kinematics so that equivalent motion configurations are
 * stored consistently.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Unit final {
    static_assert(std::is_floating_point_v<T>, "Unit requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Unit.
     *
     * @details
     * The builder stages geometry, transform, and optional kinematic state,
     * validates them, and then constructs either:
     * - a unit by value, or
     * - a host-owned shared pointer to a unit.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a default-initialized unit with default geometry/transform
     * state and no active kinematic terms.
     */
    Unit() = default;

    /**
     * @brief Destructor.
     */
    ~Unit() = default;

    /**
     * @brief Construct a unit from geometry and transform operators.
     *
     * @details
     * Initializes a unit with:
     * - a local-space geometry operator,
     * - a local-to-world sync operator,
     * - no explicit kinematic state.
     *
     * @param geometry_operator Local-space geometry operator.
     * @param sync_operator Local-to-world rigid transform operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
        SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Construct a unit from geometry, transform, and optional kinematic state.
     *
     * @param geometry_operator Local-space geometry operator.
     * @param sync_operator Local-to-world rigid transform operator.
     * @param velocity Optional linear velocity.
     * @param acceleration Optional linear acceleration.
     * @param angular_velocity Optional angular velocity vector.
     * @param angular_acceleration Optional angular acceleration vector.
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
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Replace the stored local-space geometry operator.
     *
     * @param geometry_operator New local-space geometry operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept;

    /**
     * @brief Replace the stored sync operator.
     *
     * @param sync_operator New local-to-world rigid transform operator.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    set_sync_operator(SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Advance the unit's kinematic state by one timestep.
     *
     * @details
     * This updates the unit pose using the currently stored linear/angular
     * velocity and acceleration state according to the implementation in
     * `unit.hpp`.
     *
     * @param dt Timestep duration.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    /**
     * @brief Translate the unit in world space.
     *
     * @param delta_world World-space translation delta.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta_world) noexcept;

    /**
     * @brief Rotate the unit in world space.
     *
     * @param axis_world World-space rotation axis.
     * @param angle_rad Rotation angle in radians.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis_world, T angle_rad) noexcept;

    /**
     * @brief Return const access to the stored local-space geometry operator.
     *
     * @return Const reference to the geometry operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::GeometryOperator<T>&
    geometry_operator() const noexcept;

    /**
     * @brief Return const access to the stored sync operator.
     *
     * @return Const reference to the sync operator.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync_operator() const noexcept;

    /**
     * @brief Return const access to the optional linear velocity.
     *
     * @return Const reference to the optional velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    velocity() const noexcept;

    /**
     * @brief Return const access to the optional linear acceleration.
     *
     * @return Const reference to the optional acceleration.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    acceleration() const noexcept;

    /**
     * @brief Return const access to the optional angular velocity.
     *
     * @return Const reference to the optional angular velocity.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_velocity() const noexcept;

    /**
     * @brief Return const access to the optional angular acceleration.
     *
     * @return Const reference to the optional angular acceleration.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_acceleration() const noexcept;

    /**
     * @brief Return whether the unit carries active dynamic state.
     *
     * @details
     * A unit is typically considered dynamic if any linear or angular kinematic
     * term is present and contributes to time evolution.
     *
     * @return `true` if the unit is dynamic; otherwise `false`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    dynamic() const noexcept;

private:
    /**
     * @brief Normalize optional kinematic state into a canonical internal representation.
     *
     * @details
     * The implementation may use this helper to:
     * - eliminate redundant zero-valued optionals,
     * - ensure motion-state consistency,
     * - simplify downstream dynamic checks.
     *
     * @param velocity Optional linear velocity.
     * @param acceleration Optional linear acceleration.
     * @param angular_velocity Optional angular velocity.
     * @param angular_acceleration Optional angular acceleration.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                            std::optional<Vector<T, 3>>& acceleration,
                            std::optional<Vector<T, 3>>& angular_velocity,
                            std::optional<Vector<T, 3>>& angular_acceleration) noexcept;

    /// @brief Allow the builder to configure internals directly.
    friend class Builder;

private:
    /**
     * @brief Geometry queried in local space.
     *
     * @details
     * This geometry operator describes the unit shape before the sync transform
     * is applied.
     */
    atlas::GeometryOperator<T> _geometry_operator;

    /**
     * @brief Local-to-world rigid transform.
     *
     * @details
     * Maps the unit's local frame into world space.
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
     * @brief Optional angular velocity axis vector.
     *
     * @details
     * The exact magnitude/orientation convention is implementation-defined.
     */
    std::optional<Vector<T, 3>> _angular_velocity;

    /**
     * @brief Optional angular acceleration.
     */
    std::optional<Vector<T, 3>> _angular_acceleration;
};

/**
 * @brief Fluent builder for @ref Unit.
 *
 * @details
 * The builder provides a controlled construction path for a unit by staging:
 * - geometry,
 * - transform,
 * - optional linear motion state,
 * - optional angular motion state.
 *
 * ## Geometry input
 * Geometry may be supplied indirectly through a host geometry pointer via
 * @ref with_geometry, in which case the builder typically extracts or stores the
 * corresponding @ref atlas::GeometryOperator.
 *
 * ## Transform input
 * The builder stores a staged @ref SyncOperator describing the unit pose.
 *
 * ## Typical usage
 * @code
 * auto unit = atlas::Unit<float>::builder()
 *     .with_geometry(geometry)
 *     .with_sync(sync)
 *     .with_velocity({1.0f, 0.0f, 0.0f})
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - a valid geometry operator is available,
 * - a valid sync operator is available,
 * - staged kinematic state is internally consistent.
 *
 * The exact validation rules are implementation-defined in `unit.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Unit<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with no staged geometry, no staged transform, and no
     * optional kinematic state.
     */
    Builder() = default;

    /**
     * @brief Set the geometry from a host-side geometry object.
     *
     * @details
     * The builder may derive and cache the corresponding geometry operator from
     * the supplied geometry object.
     *
     * @param geometry Host-owned geometry pointer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const atlas::GeometryHostPtr<T>& geometry);

    /**
     * @brief Set the sync operator from a host-side sync object.
     *
     * @details
     * The builder may derive and cache the corresponding sync operator from the
     * supplied sync object.
     *
     * @param sync Host-owned sync pointer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync(const SyncHostPtr<T>& sync);

    /**
     * @brief Set the linear velocity.
     *
     * @param v Staged linear velocity.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_velocity(const Vector<T, 3>& v) noexcept;

    /**
     * @brief Set the linear acceleration.
     *
     * @param a Staged linear acceleration.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_acceleration(const Vector<T, 3>& a) noexcept;

    /**
     * @brief Set the angular velocity.
     *
     * @param w Staged angular velocity vector.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_velocity(const Vector<T, 3>& w) noexcept;

    /**
     * @brief Set the angular acceleration.
     *
     * @param alpha Staged angular acceleration vector.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_acceleration(const Vector<T, 3>& alpha) noexcept;

    /**
     * @brief Build a configured @ref Unit by value after validation.
     *
     * @return Constructed unit value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Unit<T>
    build();

    /**
     * @brief Build a configured @ref Unit in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Unit<T>>` owning the constructed unit.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Unit<T>>
    make_host_shared();

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on geometry, transform, and optional
     * kinematic state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending host-side geometry source.
     *
     * @details
     * May be used to derive a staged geometry operator.
     */
    std::optional<GeometryHostPtr<T>> _geometry;

    /**
     * @brief Pending geometry operator.
     *
     * @details
     * Stores the unit's local-space geometry once available.
     */
    std::optional<atlas::GeometryOperator<T>> _geometry_operator;

    /**
     * @brief Pending sync operator.
     *
     * @details
     * Stores the local-to-world rigid transform once available.
     */
    std::optional<SyncOperator<T>> _sync_operator;

    /**
     * @brief Pending linear velocity.
     */
    std::optional<Vector<T, 3>> _velocity;

    /**
     * @brief Pending linear acceleration.
     */
    std::optional<Vector<T, 3>> _acceleration;

    /**
     * @brief Pending angular velocity.
     */
    std::optional<Vector<T, 3>> _angular_velocity;

    /**
     * @brief Pending angular acceleration.
     */
    std::optional<Vector<T, 3>> _angular_acceleration;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Unit = atlas::system::Unit<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using UnitHostPtr = atlas::host_shared_ptr<Unit<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using UnitDevicePtr = atlas::device_shared_ptr<Unit<T>>;

} // namespace atlas

#include <atlas/unit/unit.hpp>