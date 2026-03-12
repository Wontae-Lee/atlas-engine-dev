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
    // This constructor takes already-materialized operators.
    // Because those operators are typically non-owning views, we deliberately do not
    // invent or infer a geometry owner here. The caller remains responsible for the
    // lifetime behind the supplied operator storage.
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
    // Store the operator/pose state first, then normalize the optional motion terms
    // into a representation that update() can interpret consistently.
    : _query_operator(std::move(query_operator))
    , _trace_operator(std::move(trace_operator))
    , _sync_operator(std::move(sync_operator)) {
    // Constructors are intentionally permissive about partially specified kinematics.
    // Rather than leaving the object in an inert "acceleration only" state, normalize
    // the values so first-order terms always exist when second-order terms do.
    canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    // Persist the normalized kinematic values.
    _velocity             = std::move(velocity);
    _acceleration         = std::move(acceleration);
    _angular_velocity     = std::move(angular_velocity);
    _angular_acceleration = std::move(angular_acceleration);
}

template <typename T>
void
Unit<T>::set_query_operator(QueryOperator<T> query_operator) noexcept {
    // Only the operator view is replaced here.
    // _geometry_owner is left untouched because the new operator may refer to
    // unrelated storage managed externally by the caller.
    _query_operator = std::move(query_operator);
}

template <typename T>
void
Unit<T>::set_trace_operator(TraceOperator<T> trace_operator) noexcept {
    // Same lifetime contract as set_query_operator().
    _trace_operator = std::move(trace_operator);
}

template <typename T>
void
Unit<T>::set_sync_operator(SyncOperator<T> sync_operator) noexcept {
    // SyncOperator is a value type, so assignment fully replaces the stored pose.
    _sync_operator = std::move(sync_operator);
}

template <typename T>
void
Unit<T>::set_operators(QueryOperator<T> query_operator,
                       TraceOperator<T> trace_operator,
                       SyncOperator<T> sync_operator) noexcept {
    // Bulk replacement is equivalent to calling the three individual setters.
    _query_operator = std::move(query_operator);
    _trace_operator = std::move(trace_operator);
    _sync_operator  = std::move(sync_operator);
}

template <typename T>
typename Unit<T>::Builder
Unit<T>::builder() noexcept {
    // Centralized entry point for validated fluent construction.
    return Builder {};
}

template <typename T>
const atlas::QueryOperator<T>&
Unit<T>::query_operator() const noexcept {
    // Return the exact stored query view.
    return _query_operator;
}

template <typename T>
const atlas::TraceOperator<T>&
Unit<T>::trace_operator() const noexcept {
    // Return the exact stored trace view.
    return _trace_operator;
}

template <typename T>
const atlas::SyncOperator<T>&
Unit<T>::sync_operator() const noexcept {
    // The sync operator is owned by value inside the unit.
    return _sync_operator;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::velocity() const noexcept {
    // May be empty for static units.
    return _velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::acceleration() const noexcept {
    // When velocity exists but acceleration was omitted, this may contain a normalized zero vector.
    return _acceleration;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_velocity() const noexcept {
    // May be empty for non-rotating units.
    return _angular_velocity;
}

template <typename T>
const std::optional<Vector<T, 3>>&
Unit<T>::angular_acceleration() const noexcept {
    // When angular velocity exists but angular acceleration was omitted, this may contain a normalized zero vector.
    return _angular_acceleration;
}

template <typename T>
bool
Unit<T>::dynamic() const noexcept {
    // update() can advance the unit only when first-order linear or angular state exists.
    // canonicalize_kinematics() guarantees that second-order-only input is converted
    // into this form, keeping the predicate aligned with runtime behavior.
    return _velocity.has_value() || _angular_velocity.has_value();
}

template <typename T>
void
Unit<T>::update(T dt) noexcept {
    // Ignore zero/negative timesteps to avoid undefined or accidental backwards integration.
    if (!(dt > T(0))) return;

    // Static units have nothing to integrate.
    if (!dynamic()) return;

    if (_velocity.has_value()) {
        // Semi-implicit linear step:
        // 1) velocity <- velocity + acceleration * dt
        // 2) position <- position + velocity * dt
        if (_acceleration.has_value()) {
            *_velocity += (*_acceleration) * dt;
        }
        move((*_velocity) * dt);
    }

    if (_angular_velocity.has_value()) {
        // Angular integration mirrors the linear path above.
        if (_angular_acceleration.has_value()) {
            *_angular_velocity += (*_angular_acceleration) * dt;
        }

        // The angular velocity vector encodes both axis direction and scalar speed.
        const T omega = _angular_velocity->length();
        if (omega > T(0)) {
            // rotate() normalizes the axis internally, so passing the full angular
            // velocity vector here preserves both axis and magnitude semantics.
            rotate(*_angular_velocity, omega * dt);
        }
    }
}

template <typename T>
void
Unit<T>::move(const atlas::math::Vector<T, 3>& delta_world) noexcept {
    // Translation is stored directly inside the sync operator.
    // No matrix rebuild is required because orientation stays unchanged.
    _sync_operator.translation += delta_world;
}

template <typename T>
void
Unit<T>::rotate(const atlas::math::Vector<T, 3>& axis_world, T angle_rad) noexcept {
    // Copy the input so normalization does not mutate the caller's vector.
    atlas::math::Vector<T, 3> axis = axis_world;
    const T axis_len2              = axis.length_squared();

    // A zero-length axis cannot define a valid rotation.
    if (axis_len2 <= T(0)) return;

    // Convert axis to unit length before building the delta quaternion.
    axis *= (T(1) / static_cast<T>(std::sqrt(axis_len2)));

    // Standard axis-angle to quaternion conversion.
    const T half = angle_rad * T(0.5);
    const T s    = static_cast<T>(std::sin(static_cast<double>(half)));
    const T c    = static_cast<T>(std::cos(static_cast<double>(half)));

    const atlas::math::Quaternion<T> dq(
        c,
        axis.x * s,
        axis.y * s,
        axis.z * s);

    // Left-multiplication applies a world-space incremental rotation.
    // Normalize the result to limit numerical drift after repeated updates.
    _sync_operator.orientation = (dq * _sync_operator.orientation).normalized();

    // SyncOperator caches orientation matrices, so refresh them after changing pose.
    _sync_operator.rebuild_matrices();
}

template <typename T>
void
Unit<T>::canonicalize_kinematics(std::optional<Vector<T, 3>>& velocity,
                                 std::optional<Vector<T, 3>>& acceleration,
                                 std::optional<Vector<T, 3>>& angular_velocity,
                                 std::optional<Vector<T, 3>>& angular_acceleration) noexcept {
    // Normalize partially specified state into an integrable representation.
    //
    // Motivation:
    // - update() depends on first-order state (velocity / angular_velocity).
    // - Public construction APIs accept second-order terms independently.
    // - Filling the missing paired terms avoids creating a public state that exists
    //   syntactically but has no effect at runtime.
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
    // Builder construction is the safe path for Atlas geometry because it lets the
    // final Unit retain the geometry owner while still storing lightweight operators.
    if (!geometry) {
        atlas::logger::error()
            << "Unit::Builder: geometry must not be null.";
        throw std::runtime_error("Unit::Builder: geometry must not be null.");
    }

    // Retain the owner so the built Unit can keep non-owning operators valid.
    _geometry       = geometry;

    // Cache the operator views immediately. build() then becomes simple assembly.
    _query_operator = geometry->make_query_operator();
    _trace_operator = geometry->make_trace_operator();
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_sync(const SyncHostPtr<T>& sync) {
    // Sync differs from geometry: make_sync_operator() returns a self-contained value,
    // so we only need to keep the copied operator and not the source Sync object itself.
    if (!sync) {
        atlas::logger::error()
            << "Unit::Builder: sync must not be null.";
        throw std::runtime_error("Unit::Builder: sync must not be null.");
    }

    // Snapshot the current pose into a plain value object.
    _sync_operator = sync->make_sync_operator();
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_velocity(const atlas::math::Vector<T, 3>& v) noexcept {
    // Validation is intentionally deferred to build() so setter order stays flexible.
    _velocity = v;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_acceleration(const atlas::math::Vector<T, 3>& a) noexcept {
    // Stored verbatim until validate()/canonicalize_kinematics() runs.
    _acceleration = a;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_velocity(const atlas::math::Vector<T, 3>& w) noexcept {
    // Stored verbatim until validate()/canonicalize_kinematics() runs.
    _angular_velocity = w;
    return *this;
}

template <typename T>
typename Unit<T>::Builder&
Unit<T>::Builder::with_angular_acceleration(const atlas::math::Vector<T, 3>& alpha) noexcept {
    // Stored verbatim until validate()/canonicalize_kinematics() runs.
    _angular_acceleration = alpha;
    return *this;
}

template <typename T>
Unit<T>
Unit<T>::Builder::build() {
    // Fail before any state is moved out of the builder.
    validate();

    // Work on local copies so normalization does not mutate builder state until
    // construction is known to succeed.
    auto geometry             = _geometry;
    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    // Reuse the same normalization policy as the direct constructor.
    Unit<T>::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    // Builder is a friend specifically so it can assemble a fully consistent unit
    // without exposing broad public mutators.
    Unit<T> u {};

    // Store the geometry owner first so the operator views assigned below will continue
    // to reference live geometry storage after the builder goes away.
    u._geometry_owner = std::move(*geometry);
    u._query_operator = std::move(*_query_operator);
    u._trace_operator = std::move(*_trace_operator);
    u._sync_operator  = std::move(*_sync_operator);

    // Persist the normalized motion state.
    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);

    // Reset the builder so reuse starts from a clean slate.
    _geometry.reset();
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
    // Keep build logic centralized in build(), then move the result into shared ownership.
    auto u = build();
    return atlas::make_host_shared<Unit<T>>(std::move(u));
}

template <typename T>
void
Unit<T>::Builder::validate() const {
    // Geometry ownership is mandatory because Unit retains it to keep the derived
    // non-owning operators valid for the unit's lifetime.
    if (!_geometry.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: geometry owner is not initialized.";
        throw std::runtime_error("Unit::Builder: geometry owner is not initialized.");
    }

    // with_geometry() is expected to populate both operator views together.
    if (!_query_operator.has_value() || !_trace_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: geometry operators are not initialized.";
        throw std::runtime_error("Unit::Builder: geometry operators are not initialized.");
    }

    // A complete unit always needs an initial pose.
    if (!_sync_operator.has_value()) {
        atlas::logger::error()
            << "Unit::Builder validation failed: sync operator is not initialized.";
        throw std::runtime_error("Unit::Builder: sync operator is not initialized.");
    }

    // Builder is intentionally stricter than the direct constructor:
    // second-order terms require the corresponding first-order term to be set explicitly.
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
