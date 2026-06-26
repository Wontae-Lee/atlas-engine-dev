#pragma once

#include <atlas/math/math.h>
#include <atlas/sampling/sampling.h>

namespace atlas {

template <typename T>
T
HardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                   const MaterialProperties<T>& rhs) noexcept {

    if (!lhs.reference_diameter.has_value() || !rhs.reference_diameter.has_value()) {
        return T(0);
    }

    const T diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * T(0.5);

    if (!(diameter > T(0))) {
        return T(0);
    }

    return static_cast<T>(atlas::pi) * diameter * diameter;
}

template <typename T>
T
HardSphereKernel<T>::cross_section(const MaterialProperties<T>& lhs,
                                   const MaterialProperties<T>& rhs,
                                   const T relative_speed) noexcept {
    static_cast<void>(relative_speed);
    return cross_section(lhs, rhs);
}

template <typename T>
void
HardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                Vector3<T>& rhs_velocity,
                                const MaterialProperties<T>& lhs,
                                const MaterialProperties<T>& rhs) const noexcept {

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

    const Vector3<T> center = (lhs_velocity * lhs_mass + rhs_velocity * rhs_mass) / mass_sum;

    const Vector3<T> axis = relative / speed;

    const Vector3<T> sample_seed = relative + center * T(atlas::RANDOM_HASH_NORMAL_SCALE_FOR_MIX)
        + Vector3<T>(lhs_mass, rhs_mass, lhs_mass + rhs_mass);

    const T u1 = atlas::sample_hashed_unit_interval(
        sample_seed,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U1));

    const T u2 = atlas::sample_hashed_unit_interval(
        sample_seed + axis,
        T(atlas::RANDOM_HASH_SALT_DIFFUSE_U2));

    const T cos_chi = T(2) * u1 - T(1);
    const T phi     = T(2) * static_cast<T>(atlas::pi) * u2;

    const Vector3<T> scattered_axis     = atlas::spherical_direction(axis, cos_chi, phi);
    const Vector3<T> scattered_relative = scattered_axis * speed;

    lhs_velocity = center + scattered_relative * (rhs_mass / mass_sum);
    rhs_velocity = center - scattered_relative * (lhs_mass / mass_sum);
}

}