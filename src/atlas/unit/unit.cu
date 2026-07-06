#include <atlas/unit/unit.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Unit::Builder
Unit::builder() noexcept {
    return Builder {};
}

Unit::Builder&
Unit::Builder::with_geometry(const Geometry& geometry) {
    _geometry = geometry;

    return *this;
}

Unit::Builder&
Unit::Builder::with_sync(const SyncHostPtr& sync) {
    if (!sync) {
        throw std::runtime_error("Unit::Builder: sync must not be null.");
    }

    _sync = *sync;

    return *this;
}

Unit::Builder&
Unit::Builder::with_velocity(const Float3& v) noexcept {
    _velocity = v;
    return *this;
}

Unit::Builder&
Unit::Builder::with_acceleration(const Float3& a) noexcept {
    _acceleration = a;
    return *this;
}

Unit::Builder&
Unit::Builder::with_angular_velocity(const Float3& w) noexcept {
    _angular_velocity = w;
    return *this;
}

Unit::Builder&
Unit::Builder::with_angular_acceleration(const Float3& alpha) noexcept {
    _angular_acceleration = alpha;
    return *this;
}

Unit
Unit::Builder::build() {
    validate();

    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    Unit::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    Unit u {};

    u._geometry             = std::move(*_geometry);
    u._sync                 = std::move(*_sync);
    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);

    _geometry.reset();
    _sync.reset();
    _velocity.reset();
    _acceleration.reset();
    _angular_velocity.reset();
    _angular_acceleration.reset();

    return u;
}

atlas::host_shared_ptr<Unit>
Unit::Builder::make_host_shared() {
    auto u = build();
    return atlas::make_host_shared<Unit>(std::move(u));
}

void
Unit::Builder::validate() const {
    if (!_geometry.has_value()) {
        throw std::runtime_error("Unit::Builder: geometry is not initialized.");
    }

    if (!_sync.has_value()) {
        throw std::runtime_error("Unit::Builder: sync is not initialized.");
    }

    if (_acceleration.has_value() && !_velocity.has_value()) {
        throw std::runtime_error("Unit::Builder: acceleration is set but velocity is missing.");
    }

    if (_angular_acceleration.has_value() && !_angular_velocity.has_value()) {
        throw std::runtime_error(
            "Unit::Builder: angular_acceleration is set but angular_velocity is missing.");
    }
}

}
