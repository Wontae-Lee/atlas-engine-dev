#pragma once

#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas::system {

template <typename T>
T
VariableSoftSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                           const MaterialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // Start from the hard-sphere cross section based on collision diameters.
    const T base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);

    // Missing or invalid diameter data disables the collision cross section.
    if (!(base_cross_section > T(0))) {
        return T(0);
    }

    // Use the arithmetic mean scattering parameter for the species pair.
    // Missing scattering-parameter data falls back to 1.
    const T lhs_scattering = lhs.scattering_parameter.value_or(T(1));
    const T rhs_scattering = rhs.scattering_parameter.value_or(T(1));

    // Apply the current VSS-style speed scaling:
    //
    //   sigma_vss = sigma_hs * g^(alpha_ij - 1)
    //
    // where alpha_ij = 0.5 * (alpha_lhs + alpha_rhs).
    const T speed_scale = relative_speed > T(0)
        ? std::pow(relative_speed, (lhs_scattering + rhs_scattering) * T(0.5) - T(1))
        : T(1);

    return base_cross_section * speed_scale;
}

template <typename T>
void
VariableSoftSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs) const noexcept {
    // Reuse the simple elastic hard-sphere velocity update.
    // VSS behavior is represented only through the speed-dependent cross section.
    HardSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system