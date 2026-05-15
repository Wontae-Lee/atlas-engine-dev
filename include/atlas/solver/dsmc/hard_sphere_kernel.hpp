#pragma once

#include <atlas/math/vector/vector.h>

#include <numbers>

namespace atlas::system {

template <typename T>
T
HardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                   const MaterialProperties<T>& rhs) noexcept {
    // Collision diameter is required for both species.
    if (!lhs.collision_diameter.has_value() || !rhs.collision_diameter.has_value()) {
        return T(0);
    }

    // Use the arithmetic mean pair diameter.
    const T diameter = (lhs.collision_diameter.value() + rhs.collision_diameter.value()) * T(0.5);

    // Non-positive diameter disables hard-sphere collision probability.
    if (!(diameter > T(0))) {
        return T(0);
    }

    // Hard-sphere cross section: sigma = pi * d^2.
    return static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
}

template <typename T>
void
HardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                Vector3<T>& rhs_velocity,
                                const MaterialProperties<T>& lhs,
                                const MaterialProperties<T>& rhs) const noexcept {
    // Both molecular masses must be positive for a valid elastic collision.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;
    const T mass_sum = lhs_mass + rhs_mass;

    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return;
    }

    // Compute relative velocity and its magnitude.
    const Vector3<T> relative = lhs_velocity - rhs_velocity;
    const T speed             = relative.length();

    if (!(speed > T(0))) {
        return;
    }

    // Use the current relative-velocity direction as the collision normal.
    const Vector3<T> normal = relative / speed;

    // Project relative velocity onto the collision normal.
    const T normal_relative_velocity = atlas::math::dot(relative, normal);

    if (!(normal_relative_velocity > T(0))) {
        return;
    }

    // Apply an elastic two-body update along the collision normal.
    lhs_velocity -= normal
        * (static_cast<T>(2) * rhs_mass / mass_sum * normal_relative_velocity);

    rhs_velocity += normal
        * (static_cast<T>(2) * lhs_mass / mass_sum * normal_relative_velocity);
}

} // namespace atlas::system