#pragma once

#include <atlas/core/macros.h>

namespace atlas {

class Molecule final {
public:
    Molecule() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Molecule(const float mass,
             const float translational_energy,
             const float rotational_energy,
             const float vibrational_energy,
             const float reference_diameter,
             const float reference_temperature,
             const float viscosity_index,
             const float scattering_parameter) noexcept
        : _mass(mass)
        , _translational_energy(translational_energy)
        , _rotational_energy(rotational_energy)
        , _vibrational_energy(vibrational_energy)
        , _reference_diameter(reference_diameter)
        , _reference_temperature(reference_temperature)
        , _viscosity_index(viscosity_index)
        , _scattering_parameter(scattering_parameter) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept {
        return _mass;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    translational_energy() const noexcept {
        return _translational_energy;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    rotational_energy() const noexcept {
        return _rotational_energy;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    vibrational_energy() const noexcept {
        return _vibrational_energy;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_diameter() const noexcept {
        return _reference_diameter;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_temperature() const noexcept {
        return _reference_temperature;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_index() const noexcept {
        return _viscosity_index;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    scattering_parameter() const noexcept {
        return _scattering_parameter;
    }

private:
    float _mass {};

    float _translational_energy {};

    float _rotational_energy {};

    float _vibrational_energy {};

    float _reference_diameter {};

    float _reference_temperature {};

    float _viscosity_index { 0.5f };

    float _scattering_parameter { 1.0f };
};

}