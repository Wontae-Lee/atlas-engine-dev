#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Unit<T>::Unit(
    atlas::GeometryOperator<T> geometry_operator,
    SyncOperator<T> sync_operator) noexcept

    : _geometry_operator(std::move(geometry_operator))
    , _sync_operator(std::move(sync_operator)) {
    // Construct a unit from already-materialized value-type operators.
    //
    // This overload is intended for callers that already have:
    // - a geometry query operator describing the unit's shape, and
    // - a sync operator describing the unit's pose / coordinate transform.
    //
    // No dynamic kinematic state is configured here:
    // - linear velocity remains unset
    // - linear acceleration remains unset
    // - angular velocity remains unset
    // - angular acceleration remains unset
    //
    // As a consequence, the unit starts as static unless those optional
    // kinematic members are assigned later.
}

template <typename T>
Unit<T>::Unit(
    atlas::GeometryOperator<T> geometry_operator,
    SyncOperator<T> sync_operator,
    std::optional<Vector<T, 3>> velocity,
    std::optional<Vector<T, 3>> acceleration,
    std::optional<Vector<T, 3>> angular_velocity,
    std::optional<Vector<T, 3>> angular_acceleration) noexcept

    : _geometry_operator(std::move(geometry_operator))
    , _sync_operator(std::move(sync_operator)) {
    // Construct a unit from value-type geometry/sync operators plus optional
    // linear and angular kinematic state.
    //
    // The optional terms are normalized into a canonical paired form before
    // being stored:
    // - if acceleration exists but velocity does not, velocity becomes zero
    // - if velocity exists but acceleration does not, acceleration becomes zero
    // - the same rule is applied to angular velocity / angular acceleration
    //
    // This ensures update() can treat motion state as paired quantities.
    Unit<T>::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    // Store the canonicalized linear and angular motion state.
    _velocity             = std::move(velocity);
    _acceleration         = std::move(acceleration);
    _angular_velocity     = std::move(angular_velocity);
    _angular_acceleration = std::move(angular_acceleration);
}

template <typename T>
void
Unit<T>::set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept {
    // Replace the current geometry query operator.
    //
    // This changes how the unit is spatially queried without affecting:
    // - its transform
    // - its kinematic state
    _geometry_operator = std::move(geometry_operator);
}

template <typename T>
void
Unit<T>::set_sync_operator(SyncOperator<T> sync_operator) noexcept {
    // Replace the current sync operator.
    //
    // This updates the unit's transform state without affecting:
    // - its geometry definition
    // - its linear/angular kinematics
    _sync_operator = std::move(sync_operator);
}

template <typename T>
typename Unit<T>::Builder
Unit<T>::builder() noexcept {
    // Return a fresh builder object for staged Unit<T> construction.
    return Builder {};
}

template <typename T>
const atlas::GeometryOperator<T>&
Unit<T>::geometry_operator() const noexcept {
    // Return the stored geometry query operator by const reference.
    return _geometry_operator;
}

template <typename T>
const atlas::SyncOperator<T>&
Unit<T>::sync_operator() const noexcept {
    // Return the stored sync operator by const reference.
    return _sync_operator;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::velocity() const noexcept {
    // Return the optional linear velocity.
    //
    // When present, it is interpreted in world space.
    return _velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::acceleration() const noexcept {
    // Return the optional linear acceleration.
    //
    // When present, it is interpreted in world space.
    return _acceleration;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_velocity() const noexcept {
    // Return the optional angular velocity vector.
    //
    // Direction:
    // - rotation axis in world space
    //
    // Magnitude:
    // - angular speed in radians per unit time
    return _angular_velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_acceleration() const noexcept {
    // Return the optional angular acceleration vector.
    //
    // Direction:
    // - angular acceleration axis in world space
    //
    // Magnitude:
    // - angular acceleration rate in radians per unit time squared
    return _angular_acceleration;
}

template <typename T>
bool
Unit<T>::dynamic() const noexcept {
    // A unit is considered dynamic if it has either:
    // - linear velocity, or
    // - angular velocity
    //
    // Pure acceleration without its paired velocity should not exist once
    // canonicalize_kinematics() has been applied.
    return _velocity.has_value() || _angular_velocity.has_value();
}

template <typename T>
void
Unit<T>::update(T dt) noexcept {
    // Advance the unit's transform using its stored kinematic state over
    // one time step of duration dt.
    //
    // Integration policy used here:
    // - linear velocity is updated by linear acceleration, then translated
    // - angular velocity is updated by angular acceleration, then rotated
    //
    // This is a simple explicit step intended for transform evolution.

    // Reject non-positive time steps.
    if (!(dt > T(0))) return;

    // Static units do not move or rotate.
    if (!dynamic()) return;

    if (_velocity.has_value()) {
        // Integrate linear motion first.
        //
        // If linear acceleration exists, advance velocity assuming constant
        // acceleration over this step.
        if (_acceleration.has_value()) {
            *_velocity += (*_acceleration) * dt;
        }

        // Advance translation using the updated linear velocity.
        move((*_velocity) * dt);
    }

    if (_angular_velocity.has_value()) {
        // Integrate angular motion first.
        //
        // If angular acceleration exists, advance angular velocity assuming
        // constant angular acceleration over this step.
        if (_angular_acceleration.has_value()) {
            *_angular_velocity += (*_angular_acceleration) * dt;
        }

        // Convert angular velocity vector into:
        // - axis   : normalized direction
        // - angle  : |omega| * dt
        const T omega = _angular_velocity->length();
        if (omega > T(0)) {
            rotate(*_angular_velocity, omega * dt);
        }
    }
}

template <typename T>
void
Unit<T>::move(const atlas::math::Vector<T, 3>& delta_world) noexcept {
    // Translate the unit in world space by adding a displacement to the
    // sync operator's translation component.
    _sync_operator.translation += delta_world;
}

template <typename T>
void
Unit<T>::rotate(const atlas::math::Vector<T, 3>& axis_world, T angle_rad) noexcept {
    // Rotate the unit in world space by an axis-angle increment.
    //
    // Inputs:
    // - axis_world : rotation axis in world coordinates
    // - angle_rad  : rotation angle in radians
    //
    // The rotation is applied by constructing an incremental quaternion dq
    // and left-multiplying the current orientation:
    //   new_orientation = normalize(dq * old_orientation)

    atlas::math::Vector<T, 3> axis = axis_world;
    const T axis_len2              = axis.length_squared();

    // Degenerate axis -> no rotation.
    if (axis_len2 <= T(0)) return;

    // Normalize the rotation axis.
    axis *= (T(1) / static_cast<T>(std::sqrt(axis_len2)));

    // Standard axis-angle to quaternion conversion using half-angle terms.
    const T half = angle_rad * T(0.5);
    const T s    = static_cast<T>(std::sin(static_cast<double>(half)));
    const T c    = static_cast<T>(std::cos(static_cast<double>(half)));

    const atlas::math::Quaternion<T> dq(
        c,
        axis.x * s,
        axis.y * s,
        axis.z * s);
    // dq = [ cos(theta/2), axis * sin(theta/2) ]

    // Apply the incremental rotation and renormalize to control drift.
    _sync_operator.orientation = (dq * _sync_operator.orientation).normalized();

    // Rebuild any cached transform matrices derived from orientation/translation.
    _sync_operator.rebuild_matrices();
}

template <typename T>
void
Unit<T>::canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                                 std::optional<Vector<T, 3>>& acceleration,
                                 std::optional<Vector<T, 3>>& angular_velocity,
                                 std::optional<Vector<T, 3>>& angular_acceleration) noexcept {
    // Normalize optional linear/angular kinematic state into paired form.
    //
    // Goal:
    // - if one term of a velocity/acceleration pair exists, ensure the other
    //   also exists by promoting it to a zero vector.
    //
    // This keeps later runtime integration logic simpler and more uniform.

    if (acceleration.has_value() && !velocity.has_value()) {
        // Acceleration implies linear motion state should also have a velocity term.
        velocity = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (velocity.has_value() && !acceleration.has_value()) {
        // Velocity implies linear motion state should also have an acceleration term.
        acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (angular_acceleration.has_value() && !angular_velocity.has_value()) {
        // Angular acceleration implies angular velocity should exist.
        angular_velocity = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (angular_velocity.has_value() && !angular_acceleration.has_value()) {
        // Angular velocity implies angular acceleration should exist.
        angular_acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_geometry(const atlas::GeometryHostPtr<T>& geometry) {
    // Provide the geometry owner used to build the unit's value-type geometry operator.
    //
    // The host pointer is kept only during the builder phase so validity can
    // be checked before final materialization.

    if (!geometry) {
        atlas::logger::error()
            << "Unit::Builder: geometry must not be null.";
        throw std::runtime_error("Unit::Builder: geometry must not be null.");
    }

    _geometry = geometry;
    // Retain the source geometry owner for validation / staged construction.

    // Snapshot a value-type geometry operator immediately so the final Unit<T>
    // does not depend on dereferencing a host object at runtime.
    _geometry_operator = geometry->make_geometry_operator();
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_sync(const SyncHostPtr<T>& sync) {
    // Provide the sync owner used to build the unit's value-type transform operator.

    if (!sync) {
        atlas::logger::error()
            << "Unit::Builder: sync must not be null.";
        throw std::runtime_error("Unit::Builder: sync must not be null.");
    }

    // Snapshot the current sync state into a value-type operator so the final
    // Unit<T> stores only self-contained transform data.
    _sync_operator = sync->make_sync_operator();
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_velocity(const atlas::math::Vector<T, 3>& v) noexcept {
    // Stage linear velocity in world space.
    _velocity = v;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_acceleration(const atlas::math::Vector<T, 3>& a) noexcept {
    // Stage linear acceleration in world space.
    _acceleration = a;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_velocity(const atlas::math::Vector<T, 3>& w) noexcept {
    // Stage angular velocity vector in world space.
    _angular_velocity = w;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_acceleration(const atlas::math::Vector<T, 3>& alpha) noexcept {
    // Stage angular acceleration vector in world space.
    _angular_acceleration = alpha;
    return *this;
}

template <typename T>
Unit<T>
Unit<T>::Builder::build() {
    // Validate the staged builder state before constructing the final unit.
    validate();

    // Create local working copies of the staged optional motion state.
    auto geometry             = _geometry;
    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    // Normalize kinematics so paired velocity/acceleration terms are present.
    Unit<T>::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    // Materialize a value-type unit that is self-contained and independent
    // of the builder's temporary host-owned staging objects.
    Unit<T> u {};

    // Transfer the cached value-type operators into the final unit.
    u._geometry_operator = std::move(*_geometry_operator);
    u._sync_operator     = std::move(*_sync_operator);

    // Transfer the canonicalized kinematic state.
    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);

    // Clear builder-owned temporary state after successful materialization.
    _geometry.reset();
    _geometry_operator.reset();
    _sync_operator.reset();

    _velocity.reset();
    _acceleration.reset();
    _angular_velocity.reset();
    _angular_acceleration.reset();

    return u;
}

template <typename T>
atlas::host_shared_ptr<Unit<T>>
Unit<T>::Builder::make_host_shared() {
    // Build the unit by value, then move it into host-shared storage.
    auto u = build();
    return atlas::make_host_shared<Unit<T>>(std::move(u));
}

template <typename T>
void
Unit<T>::Builder::validate() const {
    // Validate that all required staged inputs exist and that kinematic
    // dependencies are consistent before build() proceeds.

    if (!_geometry.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: geometry owner is not initialized.";
        throw std::runtime_error("Unit::Builder: geometry owner is not initialized.");
    }

    if (!_geometry_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: geometry operator is not initialized.";
        throw std::runtime_error("Unit::Builder: geometry operator is not initialized.");
    }

    if (!_sync_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: sync operator is not initialized.";
        throw std::runtime_error("Unit::Builder: sync operator is not initialized.");
    }

    if (_acceleration.has_value() && !_velocity.has_value()) {
        // Builder policy requires explicit linear velocity whenever linear
        // acceleration is explicitly supplied.
        atlas::logger::error()
            << "Unit::Builder validation failed: acceleration is set but velocity is missing.";
        throw std::runtime_error("Unit::Builder: acceleration is set but velocity is missing.");
    }

    if (_angular_acceleration.has_value() && !_angular_velocity.has_value()) {
        // Builder policy requires explicit angular velocity whenever angular
        // acceleration is explicitly supplied.
        atlas::logger::error()
            << "Unit::Builder validation failed: angular_acceleration is set but angular_velocity is missing.";
        throw std::runtime_error("Unit::Builder: angular_acceleration is set but angular_velocity is missing.");
    }
}

} // namespace atlas::system