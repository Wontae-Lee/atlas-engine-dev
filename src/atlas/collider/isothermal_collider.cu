#include <atlas/collider/isothermal_collider.h>

#include <stdexcept>
#include <utility>

namespace atlas {

IsothermalCollider::Builder
IsothermalCollider::builder() noexcept {
    return Builder {};
}

IsothermalCollider::Builder&
IsothermalCollider::Builder::with_unit(Unit unit) {
    _unit = std::move(unit);
    return *this;
}

IsothermalCollider::Builder&
IsothermalCollider::Builder::with_momentum_accommodation_coefficient(
    const float momentum_accommodation_coefficient) noexcept {
    _momentum_accommodation_coefficient = momentum_accommodation_coefficient;
    return *this;
}

IsothermalCollider::Builder&
IsothermalCollider::Builder::with_restitution(const float restitution) noexcept {
    _restitution = restitution;
    return *this;
}

IsothermalCollider::Builder&
IsothermalCollider::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {
    _diffuse_sampling = mode;
    return *this;
}

IsothermalCollider
IsothermalCollider::Builder::build() {
    validate();

    IsothermalCollider collider(
        std::move(*_unit),
        _momentum_accommodation_coefficient,
        _restitution,
        _diffuse_sampling);

    // Builder is single-use: reset so a stale copy of this state can't leak
    // into a second build() call from the same Builder instance.
    _unit.reset();
    _momentum_accommodation_coefficient = 1.0f;
    _restitution                        = 1.0f;
    _diffuse_sampling                   = DiffuseSampling::uniform;

    return collider;
}

atlas::host_shared_ptr<IsothermalCollider>
IsothermalCollider::Builder::make_host_shared() {
    return atlas::make_host_shared<IsothermalCollider>(build());
}

void
IsothermalCollider::Builder::validate() const {
    if (!_unit) {
        throw std::runtime_error("IsothermalCollider::Builder: unit must not be null.");
    }

    if (!atlas::isfinite(_momentum_accommodation_coefficient)
        || _momentum_accommodation_coefficient < 0.0f
        || _momentum_accommodation_coefficient > 1.0f) {
        throw std::runtime_error(
            "IsothermalCollider::Builder: momentum accommodation coefficient must be finite and within [0, 1].");
    }

    if (!atlas::isfinite(_restitution) || _restitution < 0.0f) {
        throw std::runtime_error(
            "IsothermalCollider::Builder: restitution must be finite and non-negative.");
    }
}

}
