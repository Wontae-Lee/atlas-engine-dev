#pragma once

#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas::system {

template <typename T>
T
VariableSoftSphereKernel<T>::cross_section(const MatrialProperties<T>& lhs,
                                           const MatrialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    const T base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);
    if (!(base_cross_section > T(0))) {
        return T(0);
    }

    const T lhs_scattering = lhs.scattering_parameter.value_or(T(1));
    const T rhs_scattering = rhs.scattering_parameter.value_or(T(1));
    const T speed_scale = relative_speed > T(0)
        ? std::pow(relative_speed, (lhs_scattering + rhs_scattering) * T(0.5) - T(1))
        : T(1);

    return base_cross_section * speed_scale;
}

template <typename T>
void
VariableSoftSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MatrialProperties<T>& lhs,
                                        const MatrialProperties<T>& rhs) const noexcept {
    HardSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system
