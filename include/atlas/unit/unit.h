#pragma once

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/geometry_operator.h>
#include <atlas/math/math.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Composite runtime object that binds geometry, spatial transform, and optional kinematics.
 *
 * @details
 * `Unit<T>` represents a single simulation/renderable object in world space.
 * It combines three orthogonal concerns:
 *
 * - a non-owning @ref GeometryOperator for distance and ray-style queries,
 * - an owning/value @ref SyncOperator describing translation + orientation.
 *
 * In addition, the type may carry simple first- and second-order linear/angular
 * kinematic state used by @ref update(T):
 *
 * - linear velocity / acceleration
 * - angular velocity / angular acceleration
 *
 * ## Lifetime model
 * The geometry operators used by Atlas geometry types are typically **non-owning**
 * wrappers around raw pointers into geometry storage. For builder-created units,
 * this class therefore retains the original @ref GeometryHostPtr internally so that
 * those operators remain valid for the lifetime of the `Unit`.
 *
 * Direct constructor/setter APIs that accept already-built operators intentionally
 * preserve the caller's ownership model. In that case, the caller is responsible for
 * ensuring the referenced geometry storage outlives the unit's use of those operators.
 *
 * ## Kinematic normalization
 * The type normalizes partially specified state into a solver-friendly form:
 *
 * - if acceleration exists without velocity, velocity is initialized to zero
 * - if velocity exists without acceleration, acceleration is initialized to zero
 * - the same rule applies to angular velocity / angular acceleration
 *
 * This avoids storing a public state that `update()` would otherwise silently ignore.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Unit final {
    static_assert(std::is_floating_point_v<T>, "Unit requires a floating-point T");

public:
    /**
     * @brief Fluent builder for constructing a validated @ref Unit.
     *
     * @details
     * The builder is the preferred API when the unit should own the geometry lifetime
     * transitively through stored non-owning operators.
     */
    class Builder;

public:
    /// @brief Default-constructed unit contains default-initialized operators and no kinematics.
    Unit()  = default;
    /// @brief Default destructor.
    ~Unit() = default;

    /**
     * @brief Construct a unit from a prebuilt geometry operator and a spatial transform.
     *
     * @details
     * This constructor stores the supplied operators as-is. No geometry owner is retained.
     * Use this only when the referenced geometry storage is guaranteed to outlive the unit.
     *
     * @param geometry_operator Geometry operator bound to geometry data.
     * @param sync_operator Initial rigid transform in world space.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Unit(
        atlas::GeometryOperator<T> geometry_operator,
        SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Construct a unit with explicit kinematic state.
     *
     * @details
     * Kinematic inputs are normalized into a consistent representation before storage.
     * For example, providing acceleration without velocity creates a zero velocity state
     * so that @ref update(T) can integrate the object forward consistently.
     *
     * @param geometry_operator Geometry operator bound to geometry data.
     * @param sync_operator Initial rigid transform in world space.
     * @param velocity Optional linear velocity.
     * @param acceleration Optional linear acceleration.
     * @param angular_velocity Optional angular velocity vector.
     * @param angular_acceleration Optional angular acceleration vector.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
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
     * @return A default-initialized builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Replace the stored geometry operator.
     *
     * @details
     * This does not update the internally retained geometry owner. If the new operator
     * references different geometry storage, the caller must manage that lifetime.
     *
     * @param geometry_operator New geometry operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept;

    /**
     * @brief Replace the stored rigid transform.
     *
     * @param sync_operator New rigid transform state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_sync_operator(SyncOperator<T> sync_operator) noexcept;

    /**
     * @brief Advance the unit's stored kinematics by a time step.
     *
     * @details
     * The integration policy is intentionally simple:
     *
     * - non-positive `dt` is ignored
     * - if linear velocity exists, velocity is first updated by acceleration, then position is advanced
     * - if angular velocity exists, angular velocity is first updated by angular acceleration,
     *   then orientation is advanced using the velocity vector's direction as the rotation axis
     *
     * This is a semi-implicit / symplectic-style update for the stored first-order state.
     *
     * @param dt Time step in simulation units.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt) noexcept;

    /**
     * @brief Translate the unit in world space.
     *
     * @param delta_world Translation delta expressed in world coordinates.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    move(const Vector<T, 3>& delta_world) noexcept;

    /**
     * @brief Rotate the unit around a world-space axis.
     *
     * @details
     * The provided axis is normalized internally. If the axis length is zero,
     * the function becomes a no-op.
     *
     * @param axis_world Rotation axis expressed in world coordinates.
     * @param angle_rad Rotation angle in radians.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rotate(const Vector<T, 3>& axis_world, T angle_rad) noexcept;

    /// @brief Read-only access to the stored geometry operator.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::GeometryOperator<T>&
    geometry_operator() const noexcept;

    /// @brief Read-only access to the stored rigid transform.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const atlas::SyncOperator<T>&
    sync_operator() const noexcept;

    /// @brief Read-only access to the optional linear velocity.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    velocity() const noexcept;

    /// @brief Read-only access to the optional linear acceleration.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    acceleration() const noexcept;

    /// @brief Read-only access to the optional angular velocity.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_velocity() const noexcept;

    /// @brief Read-only access to the optional angular acceleration.
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const std::optional<Vector<T, 3>>&
    angular_acceleration() const noexcept;

    /**
     * @brief Returns whether the unit currently has any first-order motion state.
     *
     * @details
     * A unit is considered dynamic when it has either:
     * - linear velocity, or
     * - angular velocity
     *
     * Because construction normalizes second-order-only inputs into zero-valued first-order
     * state, this predicate remains consistent with what @ref update(T) can integrate.
     *
     * @return `true` if the unit can move/rotate under @ref update(T), otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    dynamic() const noexcept;

private:
    /**
     * @brief Normalize partially specified kinematic state into a consistent representation.
     *
     * @details
     * The helper applies the following rules in-place:
     *
     * - acceleration without velocity creates zero velocity
     * - velocity without acceleration creates zero acceleration
     * - angular acceleration without angular velocity creates zero angular velocity
     * - angular velocity without angular acceleration creates zero angular acceleration
     *
     * The function is `private` because the normalization policy is an implementation detail
     * shared by constructors and the builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static void
    canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                            std::optional<Vector<T, 3>>& acceleration,
                            std::optional<Vector<T, 3>>& angular_velocity,
                            std::optional<Vector<T, 3>>& angular_acceleration) noexcept;

    friend class Builder;

private:
    /// @brief Non-owning geometry operator.
    atlas::GeometryOperator<T> _geometry_operator;
    /// @brief Owning value-type rigid transform state.
    SyncOperator<T> _sync_operator;
    /// @brief Geometry owner retained only for builder-created units to keep operators alive.
    GeometryHostPtr<T> _geometry_owner;

    /// @brief Optional linear velocity used by @ref update(T).
    std::optional<Vector<T, 3>> _velocity;
    /// @brief Optional linear acceleration used by @ref update(T).
    std::optional<Vector<T, 3>> _acceleration;
    /// @brief Optional angular velocity used by @ref update(T).
    std::optional<Vector<T, 3>> _angular_velocity;
    /// @brief Optional angular acceleration used by @ref update(T).
    std::optional<Vector<T, 3>> _angular_acceleration;
};

/**
 * @brief Builder for @ref Unit.
 *
 * @details
 * The builder collects the geometry owner, derived operators, sync state, and optional
 * kinematics, then validates and normalizes them before producing a `Unit`.
 *
 * Builder-created units retain the geometry owner internally, which is the safest way to
 * create long-lived units from Atlas geometry objects whose operators are non-owning views.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Unit<T>::Builder final {
public:
    /// @brief Default constructor.
    Builder() = default;

    /**
     * @brief Bind geometry and cache the corresponding geometry operator.
     *
     * @param geometry Host-owned geometry object.
     * @return `*this` for fluent chaining.
     *
     * @throws std::runtime_error if `geometry` is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const atlas::GeometryHostPtr<T>& geometry);

    /**
     * @brief Bind a sync object and copy out its sync operator.
     *
     * @param sync Host-owned sync object.
     * @return `*this` for fluent chaining.
     *
     * @throws std::runtime_error if `sync` is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_sync(const SyncHostPtr<T>& sync);

    /// @brief Set linear velocity.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_velocity(const Vector<T, 3>& v) noexcept;

    /// @brief Set linear acceleration.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_acceleration(const Vector<T, 3>& a) noexcept;

    /// @brief Set angular velocity.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_velocity(const Vector<T, 3>& w) noexcept;

    /// @brief Set angular acceleration.
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_angular_acceleration(const Vector<T, 3>& alpha) noexcept;

    /**
     * @brief Build a validated @ref Unit by value.
     *
     * @return Fully constructed unit.
     *
     * @throws std::runtime_error if required inputs are missing or inconsistent.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Unit<T>
    build();

    /**
     * @brief Build a validated @ref Unit on the host heap.
     *
     * @return Shared pointer owning the built unit.
     *
     * @throws std::runtime_error if required inputs are missing or inconsistent.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Unit<T>>
    make_host_shared();

private:
    /**
     * @brief Validate builder state before construction.
     *
     * @throws std::runtime_error if required components are missing or if
     *         second-order kinematics are provided without their first-order counterparts.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /// @brief Geometry owner to be retained by the resulting unit.
    std::optional<GeometryHostPtr<T>> _geometry;
    /// @brief Cached geometry operator derived from @ref _geometry.
    std::optional<atlas::GeometryOperator<T>> _geometry_operator;
    /// @brief Sync operator copied from the provided sync object.
    std::optional<SyncOperator<T>> _sync_operator;

    /// @brief Pending linear velocity.
    std::optional<Vector<T, 3>> _velocity;
    /// @brief Pending linear acceleration.
    std::optional<Vector<T, 3>> _acceleration;
    /// @brief Pending angular velocity.
    std::optional<Vector<T, 3>> _angular_velocity;
    /// @brief Pending angular acceleration.
    std::optional<Vector<T, 3>> _angular_acceleration;
};

} // namespace atlas::system

namespace atlas {

/// @brief Public alias for @ref atlas::system::Unit.
template <typename T>
using Unit = atlas::system::Unit<T>;

/// @brief Host shared pointer alias for @ref Unit.
template <typename T>
using UnitHostPtr = atlas::host_shared_ptr<Unit<T>>;

/// @brief Device shared pointer alias for @ref Unit.
template <typename T>
using UnitDevicePtr = atlas::device_shared_ptr<Unit<T>>;

} // namespace atlas

#include <atlas/unit/unit.hpp>
