#pragma once

#include <atlas/core/macros.h>

namespace atlas {

class Solid final {
public:
    Solid() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit Solid(const float mass) noexcept
        : _mass(mass) {
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    mass() const noexcept {
        return _mass;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    translational_energy() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    rotational_energy() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    vibrational_energy() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_diameter() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    reference_temperature() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_index() const noexcept {
        return 1.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    scattering_parameter() const noexcept {
        return 1.0f;
    }

private:
    float _mass {};
};

}