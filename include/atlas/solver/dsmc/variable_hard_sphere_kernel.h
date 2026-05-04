#pragma once

/**
 * @file variable_hard_sphere_kernel.h
 * @brief Declares a Variable Hard Sphere (VHS) collision kernel for DSMC solvers.
 *
 * This header defines `VariableHardSphereKernel<T>`, a collision model used in
 * Direct Simulation Monte Carlo (DSMC) methods.
 *
 * The Variable Hard Sphere (VHS) model extends the constant-diameter hard-sphere
 * model by allowing the effective collision cross section to depend on the
 * relative speed between particles. This enables better modeling of real gas
 * behavior, particularly viscosity and transport properties.
 *
 * Compared to the simple hard-sphere model:
 * - the collision diameter is no longer constant
 * - the cross section depends on relative velocity
 * - material-specific parameters (e.g., viscosity index) influence behavior
 *
 * This kernel is designed to be:
 * - usable in both host and device code
 * - lightweight and stateless
 * - callable via a uniform interface from `DsmcKernel<T>`
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Variable Hard Sphere DSMC collision kernel.
 *
 * This kernel implements:
 * - velocity-dependent collision cross section evaluation
 * - post-collision velocity update for two interacting particles
 *
 * It is intended to be used through `DsmcKernel<T>` for runtime dispatch.
 *
 * The kernel itself is stateless and relies entirely on:
 * - input particle velocities
 * - material properties of the interacting species
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class VariableHardSphereKernel final {
public:
    /**
     * @brief Compute the VHS collision cross section.
     *
     * The cross section determines the probability of collision between two
     * particles as a function of their relative speed and material properties.
     *
     * In the VHS model, the cross section typically follows a power-law relation:
     *     σ ∝ (relative_speed)^(2(1 - ω))
     * where ω is the viscosity index of the gas.
     *
     * The exact formulation depends on the implementation details and the
     * parameters stored in `MaterialProperties<T>`.
     *
     * This function is:
     * - pure (no side effects)
     * - stateless
     * - usable on both host and device
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of relative velocity between the particles.
     * @return Effective collision cross section.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Apply a VHS collision to a pair of particle velocities.
     *
     * This function performs a single DSMC collision step between two particles.
     * It updates the input velocities in place according to:
     * - conservation of momentum
     * - conservation of kinetic energy (for elastic collisions)
     * - scattering rules defined by the VHS model
     *
     * Typical steps inside the implementation:
     * 1. Compute relative velocity vector
     * 2. Transform to center-of-mass frame
     * 3. Sample a scattering direction
     * 4. Rotate relative velocity accordingly
     * 5. Transform back to lab frame
     * 6. Write updated velocities back to inputs
     *
     * Important properties:
     * - No dynamic memory allocation
     * - Operates only on local temporaries and input references
     * - Modifies `lhs_velocity` and `rhs_velocity` directly
     *
     * @param lhs_velocity Velocity of the left-hand particle (modified in place).
     * @param rhs_velocity Velocity of the right-hand particle (modified in place).
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_hard_sphere_kernel.hpp>