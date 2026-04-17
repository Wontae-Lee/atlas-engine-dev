#pragma once

#include <atlas/math/vector/vector.h>

#include <numbers>

namespace atlas::system {

template <typename T>
T
HardSphereKernel<T>::cross_section(const MatrialProperties<T>& lhs,
                                   const MatrialProperties<T>& rhs) noexcept {
    if (!lhs.collision_diameter.has_value() || !rhs.collision_diameter.has_value()) {
        return T(0);
    }

    const T diameter = (lhs.collision_diameter.value() + rhs.collision_diameter.value()) * T(0.5);
    if (!(diameter > T(0))) {
        return T(0);
    }

    return static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
}

template <typename T>
void
HardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                Vector3<T>& rhs_velocity,
                                const MatrialProperties<T>& lhs,
                                const MatrialProperties<T>& rhs) const noexcept {
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return;
    }

    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();
    if (!(speed > T(0))) {
        return;
    }

    // Use the relative-velocity direction as a simple hard-sphere collision normal.
    const Vector3<T> normal          = relative / speed;
    const T normal_relative_velocity = atlas::math::dot(relative, normal);
    if (!(normal_relative_velocity > T(0))) {
        return;
    }

    lhs_velocity -= normal
        * (static_cast<T>(2) * rhs_mass / mass_sum * normal_relative_velocity);
    rhs_velocity += normal
        * (static_cast<T>(2) * lhs_mass / mass_sum * normal_relative_velocity);
}

} // namespace atlas::system
