#pragma once

#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas::system {

template <typename T>
T
VariableHardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                           const MaterialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // Start from the hard-sphere cross section based on collision diameters.
    const T base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);

    // Missing or invalid diameter data disables the collision cross section.
    if (!(base_cross_section > T(0))) {
        return T(0);
    }

    // Use the arithmetic mean viscosity index for the species pair.
    // Missing viscosity-index data falls back to 1.
    const T lhs_index = lhs.viscosity_index.value_or(T(1));
    const T rhs_index = rhs.viscosity_index.value_or(T(1));

    // Apply the VHS-style speed scaling:
    //
    //   sigma_vhs = sigma_hs * g^(omega_ij - 1)
    //
    // where omega_ij = 0.5 * (omega_lhs + omega_rhs).
    const T speed_scale = relative_speed > T(0)
        ? std::pow(relative_speed, (lhs_index + rhs_index) * T(0.5) - T(1))
        : T(1);

    return base_cross_section * speed_scale;
}

template <typename T>
void
VariableHardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MaterialProperties<T>& lhs,
                                        const MaterialProperties<T>& rhs) const noexcept {
    // Reuse the simple elastic hard-sphere velocity update.
    // VHS behavior is represented only through the velocity-dependent cross section.
    HardSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system