#pragma once

#include <atlas/core/macros.h>

namespace atlas {

// A boundary, not a colliding species: it has mass and nothing else. The
// properties it does not carry read as one rather than being absent, which
// keeps every material accessor device-callable and keeps a solid out of the
// denominators a collision model divides by.
class Solid final {
public:
    Solid() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE explicit
    Solid(const float mass) noexcept
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
