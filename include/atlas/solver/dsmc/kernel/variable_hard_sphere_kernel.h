#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_scatter.h>
#include <atlas/material/material.h>

#include <cmath>

namespace atlas {

// The cross-section falls off with relative speed as (2kT_ref / m_r g^2)^(w-1/2),
// normalised by Gamma(5/2 - w), so the model reproduces a gas whose viscosity
// scales as T^w. Scattering stays isotropic.
class VariableHardSphereKernel final {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(const Material& lhs,
                  const Material& rhs,
                  const float relative_speed) noexcept {
        const float lhs_mass = lhs.mass();
        const float rhs_mass = rhs.mass();
        const float mass_sum = lhs_mass + rhs_mass;

        if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)
            || !(relative_speed > 0.0f)) {
            return 0.0f;
        }

        const float reference_diameter    = (lhs.reference_diameter() + rhs.reference_diameter()) * 0.5f;
        const float reference_temperature = (lhs.reference_temperature() + rhs.reference_temperature()) * 0.5f;
        const float viscosity_index       = (lhs.viscosity_index() + rhs.viscosity_index()) * 0.5f;

        if (!(reference_diameter > 0.0f) || !(reference_temperature > 0.0f)) {
            return 0.0f;
        }

        const float gamma_argument = 2.5f - viscosity_index;

        if (!(gamma_argument > 0.0f)) {
            return 0.0f;
        }

        const float reduced_mass = lhs_mass * rhs_mass / mass_sum;

        const float thermal_ratio = (2.0f * atlas::boltzmann_constant * reference_temperature)
            / (reduced_mass * relative_speed * relative_speed);

        if (!(thermal_ratio > 0.0f)) {
            return 0.0f;
        }

        const float gamma_value = static_cast<float>(std::tgamma(static_cast<double>(gamma_argument)));

        if (!(gamma_value > 0.0f)) {
            return 0.0f;
        }

        const float reference_area = atlas::pi * reference_diameter * reference_diameter;

        return reference_area * std::pow(thermal_ratio, viscosity_index - 0.5f) / gamma_value;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs) const noexcept {
        atlas::dsmc_scatter(lhs_velocity, rhs_velocity, lhs, rhs, 1.0f);
    }
};

}
