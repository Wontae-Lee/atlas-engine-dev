#include <atlas/collider/interaction/isothermal_surface_kernel.h>

#include <stdexcept>

namespace atlas {

IsothermalSurfaceInteraction::Builder
IsothermalSurfaceInteraction::builder() noexcept {

    return Builder {};
}

void
IsothermalSurfaceInteraction::set_diffuse_sampling(const DiffuseSampling mode) noexcept {

    _diffuse_sampling = mode;
}

void
IsothermalSurfaceInteraction::set_restitution(const float restitution_coeff) noexcept {

    _restitution_coeff = restitution_coeff;
}

void
IsothermalSurfaceInteraction::set_momentum_acc(const float momentum_acc) noexcept {

    _momentum_acc = momentum_acc;
}

void
IsothermalSurfaceInteraction::set_temperature(const float temperature) noexcept {

    _temperature = temperature;
}

DiffuseSampling
IsothermalSurfaceInteraction::diffuse_sampling() const noexcept {

    return _diffuse_sampling;
}

float
IsothermalSurfaceInteraction::restitution() const noexcept {

    return _restitution_coeff;
}

float
IsothermalSurfaceInteraction::momentum_acc() const noexcept {

    return _momentum_acc;
}

float
IsothermalSurfaceInteraction::temperature() const noexcept {

    return _temperature;
}

IsothermalSurfaceInteraction::Builder&
IsothermalSurfaceInteraction::Builder::with_diffuse_sampling(const DiffuseSampling mode) noexcept {

    _diffuse_sampling = mode;
    return *this;
}

IsothermalSurfaceInteraction::Builder&
IsothermalSurfaceInteraction::Builder::with_restitution(const float restitution) noexcept {

    _restitution = restitution;
    return *this;
}

IsothermalSurfaceInteraction::Builder&
IsothermalSurfaceInteraction::Builder::with_momentum_acc(const float momentum_acc) noexcept {

    _momentum_acc = momentum_acc;
    return *this;
}

IsothermalSurfaceInteraction::Builder&
IsothermalSurfaceInteraction::Builder::with_temperature(const float temperature) noexcept {

    _temperature = temperature;
    return *this;
}

IsothermalSurfaceInteraction
IsothermalSurfaceInteraction::Builder::build() const {

    validate();

    IsothermalSurfaceInteraction interaction {};

    interaction.set_diffuse_sampling(_diffuse_sampling);
    interaction.set_restitution(_restitution);
    interaction.set_momentum_acc(_momentum_acc);
    interaction.set_temperature(_temperature);

    return interaction;
}

atlas::host_shared_ptr<IsothermalSurfaceInteraction>
IsothermalSurfaceInteraction::Builder::make_host_shared() const {

    return atlas::make_host_shared<IsothermalSurfaceInteraction>(build());
}

void
IsothermalSurfaceInteraction::Builder::validate() const {

    if (!atlas::isfinite(_restitution) || _restitution < 0.0f) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: restitution must be finite and non-negative.");
    }

    if (!atlas::isfinite(_momentum_acc) || _momentum_acc < 0.0f || _momentum_acc > 1.0f) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: momentum_acc must be finite and within [0, 1].");
    }

    if (!atlas::isfinite(_temperature) || _temperature < 0.0f) {
        throw std::runtime_error(
            "IsothermalSurfaceInteraction::Builder: temperature must be finite and non-negative.");
    }
}

}
