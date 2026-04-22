#pragma once

#include <atlas/solver/dsmc/hard_sphere_kernel.h>

#include <cmath>

namespace atlas::system {

template <typename T>
T
VariableHardSphereKernel<T>::cross_section(const MatrialProperties<T>& lhs,
                                           const MatrialProperties<T>& rhs,
                                           const T relative_speed) noexcept {
    // Start from the hard-sphere geometric cross section.
    //
    // The VHS model used here is built as a speed-dependent extension of the
    // hard-sphere model. Therefore, the hard-sphere cross section provides the
    // baseline interaction area before any viscosity-index-based scaling is
    // applied.
    const T base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);

    // If the hard-sphere baseline is unavailable or non-physical, no valid VHS
    // cross section can be constructed.
    //
    // This covers cases such as:
    //   - missing collision diameters,
    //   - non-positive effective diameter,
    //   - any other upstream condition that caused the hard-sphere kernel to
    //     report a zero cross section.
    if (!(base_cross_section > T(0))) {
        return T(0);
    }

    // Read the viscosity index of the left-hand species.
    //
    // When the property is not explicitly provided, default to 1. This choice
    // makes the speed exponent reduce to a neutral factor in many simple cases
    // and avoids failing the computation when optional viscosity metadata is
    // absent.
    const T lhs_index = lhs.viscosity_index.value_or(T(1));

    // Read the viscosity index of the right-hand species using the same fallback.
    const T rhs_index = rhs.viscosity_index.value_or(T(1));

    // Compute the multiplicative speed-scaling factor of the VHS model.
    //
    // This implementation uses the average of the two viscosity indices and
    // forms the exponent:
    //
    //     ((lhs_index + rhs_index) / 2) - 1
    //
    // The relative speed is then raised to that exponent, making the collision
    // cross section depend on the pair's approach speed.
    //
    // When the relative speed is not strictly positive, fall back to 1 so the
    // function remains numerically safe and preserves the hard-sphere baseline
    // instead of attempting an invalid power evaluation.
    const T speed_scale = relative_speed > T(0)
        ? std::pow(relative_speed, (lhs_index + rhs_index) * T(0.5) - T(1))
        : T(1);

    // Return the speed-adjusted VHS collision cross section.
    //
    // In this formulation:
    //   - the hard-sphere kernel provides the base geometric area,
    //   - the speed_scale term introduces the variable-hard-sphere dependence
    //     on relative velocity.
    return base_cross_section * speed_scale;
}

template <typename T>
void
VariableHardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                        Vector3<T>& rhs_velocity,
                                        const MatrialProperties<T>& lhs,
                                        const MatrialProperties<T>& rhs) const noexcept {
    // Reuse the hard-sphere collision response for the post-collision velocity update.
    //
    // In this implementation, the VHS model affects the collision probability
    // through its speed-dependent cross section, but it does not introduce a
    // distinct velocity-scattering rule at this stage. The actual velocity
    // update is therefore delegated directly to the hard-sphere kernel.
    HardSphereKernel<T> {}(lhs_velocity, rhs_velocity, lhs, rhs);
}

} // namespace atlas::system