#pragma once

#include <atlas/core/macros.h>

namespace atlas {

class Neutron final {
public:
    Neutron() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Neutron(const float mass,
            const float translational_energy,
            const float rotational_energy,
            const float vibrational_energy) noexcept
        : _mass(mass)
        , _translational_energy(translational_energy)
        , _rotational_energy(rotational_energy)
        , _vibrational_energy(vibrational_energy) {
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

private:
    float _mass {};

    float _translational_energy {};

    float _rotational_energy {};

    float _vibrational_energy {};
};

}
