#pragma once

#include <atlas/math/vector/vector.h>

#include <numbers>

namespace atlas::system {

template <typename T>
T
HardSphereKernel<T>::cross_section(const MatrialProperties<T>& lhs,
                                   const MatrialProperties<T>& rhs) noexcept {
    // A hard-sphere collision cross section can only be evaluated when both
    // materials provide a valid collision diameter.
    //
    // If either side does not define this quantity, the solver cannot derive
    // an effective interaction size for the particle pair, so it returns zero
    // to indicate that no valid hard-sphere cross section is available.
    if (!lhs.collision_diameter.has_value() || !rhs.collision_diameter.has_value()) {
        return T(0);
    }

    // Compute the effective collision diameter of the pair.
    //
    // For a simple hard-sphere model, a common approximation is to use the
    // arithmetic mean of the two particle collision diameters. This produces
    // one effective diameter representing the contact scale of the pairwise
    // interaction.
    const T diameter = (lhs.collision_diameter.value() + rhs.collision_diameter.value()) * T(0.5);

    // Reject non-positive effective diameters.
    //
    // A physical collision diameter must be strictly positive. Zero or negative
    // values would make the geometric collision area invalid, so the function
    // returns zero as a defensive fallback.
    if (!(diameter > T(0))) {
        return T(0);
    }

    // Return the hard-sphere collision cross section.
    //
    // In the hard-sphere model, the geometric cross section is:
    //
    //     sigma = pi * d^2
    //
    // where d is the effective collision diameter of the interacting pair.
    return static_cast<T>(std::numbers::pi_v<double>) * diameter * diameter;
}

template <typename T>
void
HardSphereKernel<T>::operator()(Vector3<T>& lhs_velocity,
                                Vector3<T>& rhs_velocity,
                                const MatrialProperties<T>& lhs,
                                const MatrialProperties<T>& rhs) const noexcept {
    // Read the molecular masses of the two colliding particles.
    //
    // The post-collision velocity update depends on both masses because the
    // momentum exchange along the collision normal is mass-weighted.
    const T lhs_mass = lhs.molecular_mass;
    const T rhs_mass = rhs.molecular_mass;

    // Precompute the total mass of the pair for reuse in the response formula.
    const T mass_sum = lhs_mass + rhs_mass;

    // Abort when the masses are not physically valid.
    //
    // A valid two-body collision response requires both masses to be strictly
    // positive, and therefore their sum must also be strictly positive.
    // If this condition is not satisfied, the velocity update is skipped.
    if (!(lhs_mass > T(0)) || !(rhs_mass > T(0)) || !(mass_sum > T(0))) {
        return;
    }

    // Compute the relative velocity between the two particles.
    //
    // This vector describes the motion of the left-hand particle in the
    // reference frame of the right-hand particle and determines both:
    //   - the approach direction,
    //   - the relative collision speed.
    const Vector3<T> relative = lhs_velocity - rhs_velocity;

    // Compute the magnitude of the relative velocity.
    const T speed             = relative.length();

    // If the relative speed is zero or non-positive, there is no meaningful
    // collision direction and no relative approach to resolve.
    //
    // In that case, the current operator performs no update.
    if (!(speed > T(0))) {
        return;
    }

    // Use the normalized relative-velocity direction as the collision normal.
    //
    // This is a simplified hard-sphere treatment. Rather than sampling a
    // stochastic post-collision scattering direction, this implementation uses
    // the current line of relative motion as the normal along which momentum
    // exchange is applied.
    const Vector3<T> normal = relative / speed;

    // Project the relative velocity onto the collision normal.
    //
    // Because the chosen normal is aligned with the relative-velocity direction,
    // this value is effectively the scalar normal component of the approach
    // speed. It determines how strongly the particles interact along the
    // collision axis.
    const T normal_relative_velocity = atlas::math::dot(relative, normal);

    // Skip the response when the particles are not approaching along the chosen
    // collision normal.
    //
    // A non-positive normal relative velocity indicates that the particles are
    // not moving toward one another in the modeled collision direction, so no
    // collision impulse is applied.
    if (!(normal_relative_velocity > T(0))) {
        return;
    }

    // Update the left-hand particle velocity by applying the hard-sphere
    // response along the collision normal.
    //
    // This expression corresponds to a one-dimensional elastic collision update
    // embedded along the chosen normal direction, with the impulse magnitude
    // weighted by the opposite particle's mass.
    lhs_velocity -= normal
        * (static_cast<T>(2) * rhs_mass / mass_sum * normal_relative_velocity);

    // Update the right-hand particle velocity with the complementary response.
    //
    // The two updates are symmetric and conserve the pairwise momentum exchange
    // under the assumptions of this simplified normal-direction collision model.
    rhs_velocity += normal
        * (static_cast<T>(2) * lhs_mass / mass_sum * normal_relative_velocity);
}

} // namespace atlas::system