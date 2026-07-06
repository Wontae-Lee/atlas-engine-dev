#include <atlas/collider/interaction/maxwellian_surface_interaction.h>

#include <stdexcept>

namespace atlas {

MaxwellianSurfaceInteraction::Builder
MaxwellianSurfaceInteraction::builder() noexcept {
    return Builder {};
}

void
MaxwellianSurfaceInteraction::set_temperature(const float temperature) noexcept {
    _temperature = temperature;
}

void
MaxwellianSurfaceInteraction::set_molecular_mass(const float molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
}

void
MaxwellianSurfaceInteraction::set_momentum_acc(const float momentum_acc) noexcept {
    _momentum_acc = momentum_acc;
}

void
MaxwellianSurfaceInteraction::set_trans_acc(const float trans_acc) noexcept {
    _trans_acc = trans_acc;
}

void
MaxwellianSurfaceInteraction::set_rot_acc(const float rot_acc) noexcept {
    _rot_acc = rot_acc;
}

void
MaxwellianSurfaceInteraction::set_vib_acc(const float vib_acc) noexcept {
    _vib_acc = vib_acc;
}

void
MaxwellianSurfaceInteraction::set_rot_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _rot_style = style;
}

void
MaxwellianSurfaceInteraction::set_vib_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _vib_style = style;
}

void
MaxwellianSurfaceInteraction::set_accommodation(const float momentum_acc,
                                                const float trans_acc,
                                                const float rot_acc,
                                                const float vib_acc) noexcept {
    _momentum_acc = momentum_acc;
    _trans_acc    = trans_acc;
    _rot_acc      = rot_acc;
    _vib_acc      = vib_acc;
}

float
MaxwellianSurfaceInteraction::temperature() const noexcept {
    return _temperature;
}

float
MaxwellianSurfaceInteraction::molecular_mass() const noexcept {
    return _molecular_mass;
}

float
MaxwellianSurfaceInteraction::momentum_acc() const noexcept {
    return _momentum_acc;
}

float
MaxwellianSurfaceInteraction::trans_acc() const noexcept {
    return _trans_acc;
}

float
MaxwellianSurfaceInteraction::rot_acc() const noexcept {
    return _rot_acc;
}

float
MaxwellianSurfaceInteraction::vib_acc() const noexcept {
    return _vib_acc;
}

MaxwellianInternalEnergyStyle
MaxwellianSurfaceInteraction::rot_style() const noexcept {
    return _rot_style;
}

MaxwellianInternalEnergyStyle
MaxwellianSurfaceInteraction::vib_style() const noexcept {
    return _vib_style;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_temperature(const float temperature) noexcept {
    _temperature = temperature;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_molecular_mass(const float molecular_mass) noexcept {
    _molecular_mass = molecular_mass;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_momentum_acc(const float momentum_acc) noexcept {
    _momentum_acc = momentum_acc;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_trans_acc(const float trans_acc) noexcept {
    _trans_acc = trans_acc;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_rot_acc(const float rot_acc) noexcept {
    _rot_acc = rot_acc;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_vib_acc(const float vib_acc) noexcept {
    _vib_acc = vib_acc;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_rot_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _rot_style = style;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_vib_style(const MaxwellianInternalEnergyStyle style) noexcept {
    _vib_style = style;
    return *this;
}

MaxwellianSurfaceInteraction::Builder&
MaxwellianSurfaceInteraction::Builder::with_accommodation(const float momentum_acc,
                                                          const float trans_acc,
                                                          const float rot_acc,
                                                          const float vib_acc) noexcept {
    _momentum_acc = momentum_acc;
    _trans_acc    = trans_acc;
    _rot_acc      = rot_acc;
    _vib_acc      = vib_acc;
    return *this;
}

MaxwellianSurfaceInteraction
MaxwellianSurfaceInteraction::Builder::build() const {
    validate();

    MaxwellianSurfaceInteraction interaction {};
    interaction.set_temperature(_temperature);
    interaction.set_molecular_mass(_molecular_mass);
    interaction.set_momentum_acc(_momentum_acc);
    interaction.set_trans_acc(_trans_acc);
    interaction.set_rot_acc(_rot_acc);
    interaction.set_vib_acc(_vib_acc);
    interaction.set_rot_style(_rot_style);
    interaction.set_vib_style(_vib_style);
    return interaction;
}

atlas::host_shared_ptr<MaxwellianSurfaceInteraction>
MaxwellianSurfaceInteraction::Builder::make_host_shared() const {
    return atlas::make_host_shared<MaxwellianSurfaceInteraction>(build());
}

void
MaxwellianSurfaceInteraction::Builder::validate() const {
    if (!atlas::isfinite(_temperature) || _temperature <= 0.0f) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: temperature must be finite and positive.");
    }

    if (!atlas::isfinite(_molecular_mass) || _molecular_mass <= 0.0f) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: molecular_mass must be finite and positive.");
    }

    const bool invalid_accommodation = !atlas::isfinite(_momentum_acc) || _momentum_acc < 0.0f || _momentum_acc > 1.0f
        || !atlas::isfinite(_trans_acc) || _trans_acc < 0.0f || _trans_acc > 1.0f
        || !atlas::isfinite(_rot_acc) || _rot_acc < 0.0f || _rot_acc > 1.0f
        || !atlas::isfinite(_vib_acc) || _vib_acc < 0.0f || _vib_acc > 1.0f;

    if (invalid_accommodation) {
        throw std::runtime_error(
            "MaxwellianSurfaceInteraction::Builder: accommodation coefficients must be finite and within [0, 1].");
    }
}

}
