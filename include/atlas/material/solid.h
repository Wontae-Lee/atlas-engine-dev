#pragma once

#include <atlas/core/macros.h>

#include <stdexcept>

namespace atlas {

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

    ATLAS_NODISCARD ATLAS_HOST static ATLAS_FORCE_INLINE float
    translational_energy() {
        throw std::runtime_error("Solid carries no translational energy.");
    }

    ATLAS_NODISCARD ATLAS_HOST static ATLAS_FORCE_INLINE float
    rotational_energy() {
        throw std::runtime_error("Solid carries no rotational energy.");
    }

    ATLAS_NODISCARD ATLAS_HOST static ATLAS_FORCE_INLINE float
    vibrational_energy() {
        throw std::runtime_error("Solid carries no vibrational energy.");
    }

private:
    float _mass {};
};

}
