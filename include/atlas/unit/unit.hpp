#pragma once

#include <stdexcept>
#include <utility>

namespace atlas::physics {

template <typename T>
Unit<T>::Unit(atlas::GeometryOperator<T> geometry_operator,
              SyncOperator<T> sync_operator) noexcept
    : _geometry_operator(std::move(geometry_operator))
    , _sync_operator(std::move(sync_operator)) {
}

template <typename T>
Unit<T>::Unit(atlas::GeometryOperator<T> geometry_operator,
              SyncOperator<T> sync_operator,
              std::optional<Vector<T, 3>> velocity,
              std::optional<Vector<T, 3>> acceleration,
              std::optional<Vector<T, 3>> angular_velocity,
              std::optional<Vector<T, 3>> angular_acceleration) noexcept
    : _geometry_operator(std::move(geometry_operator))
    , _sync_operator(std::move(sync_operator)) {
    Unit<T>::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    _velocity             = std::move(velocity);
    _acceleration         = std::move(acceleration);
    _angular_velocity     = std::move(angular_velocity);
    _angular_acceleration = std::move(angular_acceleration);
}

template <typename T>
void
Unit<T>::set_geometry_operator(atlas::GeometryOperator<T> geometry_operator) noexcept {
    _geometry_operator = std::move(geometry_operator);
}

template <typename T>
void
Unit<T>::set_sync_operator(SyncOperator<T> sync_operator) noexcept {
    _sync_operator = std::move(sync_operator);
}

template <typename T>
typename Unit<T>::Builder
Unit<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
const atlas::GeometryOperator<T>&
Unit<T>::geometry_operator() const noexcept {
    return _geometry_operator;
}

template <typename T>
const atlas::SyncOperator<T>&
Unit<T>::sync_operator() const noexcept {
    return _sync_operator;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::velocity() const noexcept {
    return _velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::acceleration() const noexcept {
    return _acceleration;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_velocity() const noexcept {
    return _angular_velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_acceleration() const noexcept {
    return _angular_acceleration;
}

template <typename T>
bool
Unit<T>::dynamic() const noexcept {
    return _velocity.has_value() || _angular_velocity.has_value();
}

template <typename T>
void
Unit<T>::update(T dt) noexcept {
    // Ignore invalid or zero time steps.
    if (!(dt > T(0))) return;

    if (_velocity.has_value()) {
        // Integrate linear velocity from acceleration when acceleration is available.
        if (_acceleration.has_value()) {
            *_velocity += (*_acceleration) * dt;
        }

        // Apply linear displacement over the current time step.
        move((*_velocity) * dt);
    }

    if (_angular_velocity.has_value()) {
        // Integrate angular velocity from angular acceleration when available.
        if (_angular_acceleration.has_value()) {
            *_angular_velocity += (*_angular_acceleration) * dt;
        }

        // The angular velocity magnitude gives the angular speed.
        const T omega = _angular_velocity->length();

        // Apply rotation only when the angular speed is non-zero.
        if (omega > T(0)) {
            rotate(*_angular_velocity, omega * dt);
        }
    }
}

template <typename T>
void
Unit<T>::move(const atlas::math::Vector<T, 3>& delta) noexcept {
    // Accumulate the displacement into the synchronization transform.
    _sync_operator.translation += delta;
}

template <typename T>
void
Unit<T>::rotate(const atlas::math::Vector<T, 3>& axis, T angle_rad) noexcept {
    // Use the squared length to avoid an unnecessary square root for the zero-axis test.
    const T axis_len2 = axis.length_squared();

    // A zero-length axis cannot define a valid axis-angle rotation.
    if (axis_len2 <= T(0)) return;

    // Normalize the axis before constructing the incremental rotation quaternion.
    const atlas::math::Vector<T, 3> normalized_axis = atlas::math::normalized_or(
        axis,
        atlas::math::Vector<T, 3>(T(0), T(0), T(0)));

    // Build the incremental rotation represented by the normalized axis and angle.
    const atlas::math::Quaternion<T> rotation = atlas::math::Quaternion<T>::from_axis_angle(normalized_axis, angle_rad);

    // Apply the incremental rotation and renormalize to reduce numerical drift.
    _sync_operator.orientation = (rotation * _sync_operator.orientation).normalized();

    // Rebuild dependent transform matrices after the orientation update.
    _sync_operator.rebuild_matrices();
}
template <typename T>
void
Unit<T>::canonicalize_kinematics(
    std::optional<Vector<T, 3>>& velocity,
    std::optional<Vector<T, 3>>& acceleration,
    std::optional<Vector<T, 3>>& angular_velocity,
    std::optional<Vector<T, 3>>& angular_acceleration) noexcept {
    if (acceleration.has_value() && !velocity.has_value()) {
        velocity = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (velocity.has_value() && !acceleration.has_value()) {
        acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (angular_acceleration.has_value() && !angular_velocity.has_value()) {
        angular_velocity = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (angular_velocity.has_value() && !angular_acceleration.has_value()) {
        angular_acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_geometry(const atlas::GeometryHostPtr<T>& geometry) {
    if (!geometry) {
        throw std::runtime_error("Unit::Builder: geometry must not be null.");
    }

    _geometry          = geometry;
    _geometry_operator = geometry->make_geometry_operator();

    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_sync(const SyncHostPtr<T>& sync) {
    if (!sync) {
        throw std::runtime_error("Unit::Builder: sync must not be null.");
    }

    _sync_operator = sync->make_sync_operator();

    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_velocity(const atlas::math::Vector<T, 3>& v) noexcept {
    _velocity = v;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_acceleration(const atlas::math::Vector<T, 3>& a) noexcept {
    _acceleration = a;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_velocity(const atlas::math::Vector<T, 3>& w) noexcept {
    _angular_velocity = w;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_acceleration(
    const atlas::math::Vector<T, 3>& alpha) noexcept {
    _angular_acceleration = alpha;
    return *this;
}

template <typename T>
Unit<T>
Unit<T>::Builder::build() {
    validate();

    auto geometry             = _geometry;
    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    Unit<T>::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    Unit<T> u {};

    u._geometry_operator    = std::move(*_geometry_operator);
    u._sync_operator        = std::move(*_sync_operator);
    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);

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
    auto u = build();
    return atlas::make_host_shared<Unit<T>>(std::move(u));
}

template <typename T>
void
Unit<T>::Builder::validate() const {
    if (!_geometry.has_value()) {
        throw std::runtime_error("Unit::Builder: geometry owner is not initialized.");
    }

    if (!_geometry_operator.has_value()) {
        throw std::runtime_error("Unit::Builder: geometry operator is not initialized.");
    }

    if (!_sync_operator.has_value()) {
        throw std::runtime_error("Unit::Builder: sync operator is not initialized.");
    }

    if (_acceleration.has_value() && !_velocity.has_value()) {
        throw std::runtime_error("Unit::Builder: acceleration is set but velocity is missing.");
    }

    if (_angular_acceleration.has_value() && !_angular_velocity.has_value()) {
        throw std::runtime_error(
            "Unit::Builder: angular_acceleration is set but angular_velocity is missing.");
    }
}

} // namespace atlas::physics
