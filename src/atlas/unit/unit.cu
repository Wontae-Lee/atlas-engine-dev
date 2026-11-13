#include <atlas/unit/unit.h>

#include <cmath>
#include <limits>
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

    // Store a value copy so the built Unit owns its pose independently of the
    // shared handle's later mutations or lifetime.
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

/** @brief Stages a bounded periodic angular-motion policy for builder validation. */
Unit::Builder&
Unit::Builder::with_angular_oscillation(const Float3& axis, const float amplitude, const float period) noexcept {
    _angular_oscillation = Unit::AngularOscillation { axis, amplitude, period };
    return *this;
}

Unit
Unit::Builder::build() {
    validate();

    // Canonicalize on local copies so the staged inputs are not mutated before
    // validation-through-build is fully committed.
    auto velocity             = _velocity;
    auto acceleration         = _acceleration;
    auto angular_velocity     = _angular_velocity;
    auto angular_acceleration = _angular_acceleration;

    Unit::canonicalize_kinematics(
        velocity,
        acceleration,
        angular_velocity,
        angular_acceleration);

    // Populate a bare Unit member-by-member (via friendship) rather than the
    // value constructor, because the kinematics are already canonicalized here.
    Unit u {};

    u._geometry             = std::move(*_geometry);
    u._sync                 = std::move(*_sync);
    u._velocity             = std::move(velocity);
    u._acceleration         = std::move(acceleration);
    u._angular_velocity     = std::move(angular_velocity);
    u._angular_acceleration = std::move(angular_acceleration);
    if (_angular_oscillation.has_value()) {
        u._angular_oscillation = *_angular_oscillation;
        auto& motion = *u._angular_oscillation;
        const float scale = atlas::abs(motion.axis).max();
        motion.axis = atlas::normalized_or(motion.axis / Float3(scale), Float3(0.0f));
        u.advance_angular_oscillation(0.0f);
    }

    // Reset staged state: the geometry/sync were moved-from, so the builder must
    // not be reused without reconfiguring it.
    _geometry.reset();
    _sync.reset();
    _velocity.reset();
    _acceleration.reset();
    _angular_velocity.reset();
    _angular_acceleration.reset();
    _angular_oscillation.reset();

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

    if (_angular_oscillation.has_value()) {
        const auto& motion = *_angular_oscillation;
        if (!atlas::isfinite(motion.axis)
            || (motion.axis.x == 0.0f && motion.axis.y == 0.0f && motion.axis.z == 0.0f)
            || !atlas::isfinite(motion.amplitude)
            || !atlas::isfinite(motion.period) || !(motion.period > 0.0f)) {
            throw std::runtime_error("Unit::Builder: angular oscillation requires a finite nonzero axis, finite amplitude, and finite positive period.");
        }
        if (_angular_velocity.has_value() || _angular_acceleration.has_value()) {
            throw std::runtime_error("Unit::Builder: angular oscillation cannot be combined with angular velocity or acceleration.");
        }
        const double frequency = 2.0 * atlas::pi_double / static_cast<double>(motion.period);
        const double maximum_speed = 0.5 * std::abs(static_cast<double>(motion.amplitude)) * frequency;
        const double maximum_acceleration = maximum_speed * frequency;
        if (maximum_speed > std::numeric_limits<float>::max()
            || maximum_acceleration > std::numeric_limits<float>::max()) {
            throw std::runtime_error("Unit::Builder: angular oscillation derivatives must fit the finite float range.");
        }
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
