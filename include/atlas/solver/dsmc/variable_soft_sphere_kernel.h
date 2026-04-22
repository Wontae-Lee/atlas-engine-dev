#pragma once

/**
 * @file variable_soft_sphere_kernel.h
 * @brief Declares a Variable Soft Sphere (VSS) collision kernel for DSMC solvers.
 *
 * This header defines `VariableSoftSphereKernel<T>`, a DSMC collision model used
 * to simulate binary particle collisions with both:
 * - velocity-dependent collision cross section
 * - adjustable scattering-angle behavior
 *
 * The Variable Soft Sphere (VSS) model extends the Variable Hard Sphere (VHS)
 * model by introducing an additional scattering parameter that controls the
 * angular distribution of post-collision relative velocity.
 *
 * Compared to simpler DSMC collision models:
 * - Hard Sphere (HS):
 *   uses a constant effective collision diameter
 * - Variable Hard Sphere (VHS):
 *   makes the collision cross section depend on relative speed
 * - Variable Soft Sphere (VSS):
 *   further modifies the scattering law to better match experimental transport
 *   properties such as diffusion and viscosity
 *
 * This kernel is designed to be:
 * - lightweight
 * - stateless
 * - callable from both host and device code
 * - usable through the tagged runtime wrapper `DsmcKernel<T>`
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Variable Soft Sphere DSMC collision kernel.
 *
 * This kernel provides two main operations:
 * - evaluation of the effective collision cross section
 * - application of a binary collision to two particle velocities
 *
 * The VSS model is typically preferred when the simulation needs more accurate
 * control over transport coefficients than the VHS model can provide, especially
 * because the angular scattering law is not fixed to the hard-sphere form.
 *
 * The kernel itself stores no mutable state. All collision behavior is derived
 * from:
 * - the input particle velocities
 * - the material properties of the colliding species
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class VariableSoftSphereKernel final {
public:
    /**
     * @brief Compute the VSS collision cross section.
     *
     * This function returns the effective collision cross section associated with
     * the Variable Soft Sphere model for a given relative speed and species pair.
     *
     * Like the VHS model, the VSS model generally uses a relative-speed-dependent
     * cross section. However, the full VSS model also introduces a separate
     * scattering parameter that influences post-collision angular behavior.
     *
     * This function is intended for use in collision-frequency or collision-count
     * estimation stages, where only the scalar cross section is required.
     *
     * Properties:
     * - stateless
     * - no side effects
     * - host/device callable
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of the relative velocity between the particles.
     * @return Effective VSS collision cross section.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MatrialProperties<T>& lhs,
                  const MatrialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Apply a VSS collision to a pair of particle velocities.
     *
     * This function performs a single binary collision update according to the
     * Variable Soft Sphere model. The two input velocities are modified in place.
     *
     * Conceptually, the implementation typically performs the following steps:
     * 1. Compute the center-of-mass velocity
     * 2. Compute the relative velocity vector
     * 3. Determine the post-collision scattering direction
     * 4. Apply the VSS angular-scattering rule using the model's scattering parameter
     * 5. Reconstruct post-collision particle velocities in the laboratory frame
     *
     * Relative to VHS, the key distinction is that the scattering-angle sampling
     * is modified to better represent real-gas transport behavior.
     *
     * Important properties:
     * - operates in place on `lhs_velocity` and `rhs_velocity`
     * - performs no dynamic memory allocation
     * - uses only local temporaries and input references
     * - is safe for high-frequency invocation in device-side collision loops
     *
     * @param lhs_velocity Velocity of the left-hand particle. Modified in place.
     * @param rhs_velocity Velocity of the right-hand particle. Modified in place.
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MatrialProperties<T>& lhs,
               const MatrialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_soft_sphere_kernel.hpp>