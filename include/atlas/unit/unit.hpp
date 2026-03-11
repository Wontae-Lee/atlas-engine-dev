#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
Unit<T>::Unit(
    QueryOperator<T> query_operator,
    TraceOperator<T> trace_operator,
    SyncOperator<T> sync_operator) noexcept
    : _query_operator(std::move(query_operator))
    , _trace_operator(std::move(trace_operator))
    , _sync_operator(std::move(sync_operator)) { }

template <typename T>
Unit<T>::Unit(
    QueryOperator<T> query_operator,
    TraceOperator<T> trace_operator,
    SyncOperator<T> sync_operator,
    std::optional<Vector<T, 3>> velocity,
    std::optional<Vector<T, 3>> acceleration,
    std::optional<Vector<T, 3>> angular_velocity,
    std::optional<Vector<T, 3>> angular_acceleration) noexcept
    : _query_operator(std::move(query_operator))
    , _trace_operator(std::move(trace_operator))
    , _sync_operator(std::move(sync_operator))
    , _velocity(std::move(velocity))
    , _acceleration(std::move(acceleration))
    , _angular_velocity(std::move(angular_velocity))
    , _angular_acceleration(std::move(angular_acceleration)) { }

template <typename T>
void
Unit<T>::set_query_operator(QueryOperator<T> query_operator) noexcept {
    _query_operator = std::move(query_operator);
}

template <typename T>
void
Unit<T>::set_trace_operator(TraceOperator<T> trace_operator) noexcept {
    _trace_operator = std::move(trace_operator);
}

template <typename T>
void
Unit<T>::set_sync_operator(SyncOperator<T> sync_operator) noexcept {
    _sync_operator = std::move(sync_operator);
}

template <typename T>
void
Unit<T>::set_operators(QueryOperator<T> query_operator,
                       TraceOperator<T> trace_operator,
                       SyncOperator<T> sync_operator) noexcept {
    _query_operator = std::move(query_operator);
    _trace_operator = std::move(trace_operator);
    _sync_operator  = std::move(sync_operator);
}

template <typename T>
typename Unit<T>::Builder
Unit<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
const atlas::QueryOperator<T>&
Unit<T>::query_operator() const noexcept {
    return _query_operator;
}

template <typename T>
const atlas::TraceOperator<T>&
Unit<T>::trace_operator() const noexcept {
    return _trace_operator;
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
    if (!(dt > T(0))) return;
    if (!dynamic()) return;

    if (_velocity.has_value()) {
        if (_acceleration.has_value()) {
            *_velocity += (*_acceleration) * dt;
        }
        move((*_velocity) * dt);
    }

    if (_angular_velocity.has_value()) {
        if (_angular_acceleration.has_value()) {
            *_angular_velocity += (*_angular_acceleration) * dt;
        }

        const T omega = _angular_velocity->length();
        if (omega > T(0)) {
            rotate(*_angular_velocity, omega * dt);
        }
    }
}

template <typename T>
void
Unit<T>::move(const atlas::math::Vector<T, 3>& delta_world) noexcept {
    _sync_operator.translation += delta_world;
}

template <typename T>
void
Unit<T>::rotate(const atlas::math::Vector<T, 3>& axis_world, T angle_rad) noexcept {
    atlas::math::Vector<T, 3> axis = axis_world;
    const T axis_len2              = axis.length_squared();
    if (axis_len2 <= T(0)) return;

    axis *= (T(1) / static_cast<T>(std::sqrt(axis_len2)));

    const T half = angle_rad * T(0.5);
    const T s    = static_cast<T>(std::sin(static_cast<double>(half)));
    const T c    = static_cast<T>(std::cos(static_cast<double>(half)));

    const atlas::math::Quaternion<T> dq(
        c,
        axis.x * s,
        axis.y * s,
        axis.z * s);

    _sync_operator.orientation = (dq * _sync_operator.orientation).normalized();
    _sync_operator.rebuild_matrices();
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_geometry(const atlas::GeometryHostPtr<T>& geometry) {
    if (!geometry) {
        atlas::logger::error()
            << "Unit::Builder: geometry must not be null.";
        throw std::runtime_error("Unit::Builder: geometry must not be null.");
    }

    _query_operator = geometry->make_query_operator();
    _trace_operator = geometry->make_trace_operator();
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_sync(const SyncHostPtr<T>& sync) {
    if (!sync) {
        atlas::logger::error()
            << "Unit::Builder: sync must not be null.";
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
Unit<T>::Builder::with_angular_acceleration(const atlas::math::Vector<T, 3>& alpha) noexcept {
    _angular_acceleration = alpha;
    return *this;
}

template <typename T>
Unit<T>
Unit<T>::Builder::build() {
    validate();

    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    if (velocity.has_value() && !acceleration.has_value()) {
        acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }

    if (angular_velocity.has_value() && !angular_acceleration.has_value()) {
        angular_acceleration = Vector<T, 3>(T(0), T(0), T(0));
    }

    Unit<T> u {};

    u._query_operator = std::move(*_query_operator);
    u._trace_operator = std::move(*_trace_operator);
    u._sync_operator  = std::move(*_sync_operator);

    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);

    _query_operator.reset();
    _trace_operator.reset();
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
    if (!_query_operator.has_value() || !_trace_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: geometry operators are not initialized.";
        throw std::runtime_error("Unit::Builder: geometry operators are not initialized.");
    }

    if (!_sync_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: sync operator is not initialized.";
        throw std::runtime_error("Unit::Builder: sync operator is not initialized.");
    }

    if (_acceleration.has_value() && !_velocity.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: acceleration is set but velocity is missing.";
        throw std::runtime_error("Unit::Builder: acceleration is set but velocity is missing.");
    }

    if (_angular_acceleration.has_value() && !_angular_velocity.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: angular_acceleration is set but angular_velocity is missing.";
        throw std::runtime_error("Unit::Builder: angular_acceleration is set but angular_velocity is missing.");
    }
}

} // namespace atlas::system