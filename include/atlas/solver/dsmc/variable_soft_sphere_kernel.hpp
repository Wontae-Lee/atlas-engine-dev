#pragma once

#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas::system {

template <typename T>
T
VariableSoftSphereKernel<T>::cross_section(const MatrialProperties<T>& lhs,
                                           const MatrialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // Start from the hard-sphere baseline cross section.
    //
    // This implementation treats the variable-soft-sphere model as an extension
    // of the hard-sphere geometric collision area. The hard-sphere kernel
    // therefore provides the base cross section before any VSS-specific
    // speed-dependent scaling is applied.
    const T base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);

    // If no valid hard-sphere baseline exists, no valid VSS cross section can
    // be constructed either.
    //
    // This covers cases such as:
    //   - missing collision diameters,
    //   - non-positive effective collision diameter,
    //   - any upstream hard-sphere failure that produced a zero cross section.
    if (!(base_cross_section > T(0))) {
        return T(0);
    }

    // Read the scattering parameter of the left-hand species.
    //
    // The scattering parameter is optional in the material properties, so this
    // implementation falls back to 1 when the value is not explicitly provided.
    // This default keeps the model numerically well-defined even when the full
    // VSS parameter set is not available.
    const T lhs_scattering = lhs.scattering_parameter.value_or(T(1));

    // Read the scattering parameter of the right-hand species using the same
    // fallback rule.
    const T rhs_scattering = rhs.scattering_parameter.value_or(T(1));

    // Compute the relative-speed-dependent scaling factor used by this VSS
    // cross-section model.
    //
    // The exponent is formed from the average scattering parameter of the pair:
    //
    //     ((lhs_scattering + rhs_scattering) / 2) - 1
    //
    // The relative speed is then raised to this exponent so that the collision
    // cross section varies with the approach speed of the particle pair.
    //
    // When the relative speed is not strictly positive, the implementation
    // falls back to a neutral scaling factor of 1. This avoids invalid or
    // unstable power evaluations while preserving the hard-sphere baseline.
    const T speed_scale = relative_speed > T(0)
        ? std::pow(relative_speed, (lhs_scattering + rhs_scattering) * T(0.5) - T(1))
        : T(1);

    // Return the final VSS cross section.
    //
    // In this simplified formulation:
    //   - the hard-sphere kernel supplies the geometric reference area,
    //   - the speed_scale term injects the variable-soft-sphere dependence on
    //     relative velocity through the scattering parameters.
    return base_cross_section * speed_scale;
}

template <typename T>
void
VariableSoftSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MatrialProperties<T>& lhs,
                                        const MatrialProperties<T>& rhs) const noexcept {
    // Delegate the post-collision velocity update to the hard-sphere kernel.
    //
    // In this implementation, the VSS model modifies the collision rate through
    // its speed-dependent cross section, but it does not currently introduce a
    // distinct scattering-angle or post-collision velocity law. As a result,
    // the actual momentum exchange is handled using the same simplified
    // hard-sphere response operator.
    HardSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system