#pragma once

/**
 * @file variable_hard_sphere_kernel.h
 * @brief Declares a Variable Hard Sphere style collision kernel for DSMC solvers.
 *
 * @details
 * This file defines @ref atlas::system::VariableHardSphereKernel, a lightweight
 * DSMC collision-kernel component that evaluates the Bird VHS total collision
 * cross section and applies an elastic post-collision scattering update.
 *
 * The Variable Hard Sphere (VHS) model is commonly used in DSMC to represent
 * transport-property effects by making the effective collision cross section
 * depend on reference diameter, reference temperature, molecular masses,
 * relative speed, and viscosity index.
 *
 * @note
 * The kernel is stateless. All behavior is determined by the input velocities,
 * reference diameters, reference temperatures, molecular masses, viscosity
 * indices, and relative speed.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Variable Hard Sphere style DSMC collision kernel.
 *
 * @details
 * `VariableHardSphereKernel<T>` provides two operations used by the DSMC kernel
 * dispatch layer:
 *
 * - VHS collision cross-section evaluation,
 * - hash-based random elastic binary scattering.
 *
 * The class itself stores no state and can be copied into host or device
 * execution contexts.
 *
 * @tparam T Floating-point scalar type used for velocities, masses, diameters,
 *         viscosity indices, and cross-section calculations.
 */
template <typename T>
class VariableHardSphereKernel final {
public:
    /**
     * @brief Computes the VHS-style velocity-dependent collision cross section.
     *
     * @details
     * This function computes an effective collision cross section for two
     * species/material records and a given relative speed.
     *
     * The implementation uses the DSMC VHS total cross-section law:
     *
     * @f[
     *     \sigma_{\mathrm{VHS}}
     *     =
     *     \frac{\pi d_{ref,ij}^{2}}{\Gamma(2.5-\omega_{ij})}
     *     \left(
     *       \frac{2 k_B T_{ref,ij}}{m_r g^2}
     *     \right)^{\omega_{ij}-0.5},
     * @f]
     *
     * where:
     *
     * - @f$\sigma_{\mathrm{VHS}}@f$ is the returned cross section,
     * - @f$d_{ref,ij}@f$ is the averaged reference diameter,
     * - @f$T_{ref,ij}@f$ is the averaged reference temperature,
     * - @f$m_r@f$ is the reduced molecular mass,
     * - @f$g@f$ is @p relative_speed,
     * - @f$\omega_{ij}@f$ is the averaged viscosity index.
     *
     * If required reference data, masses, relative speed, or gamma argument are
     * invalid, this function returns zero.
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of the relative velocity between the two
     *        particles.
     *
     * @return VHS-style effective collision cross section.
     * @return `T(0)` if required VHS inputs are unavailable or invalid.
     *
     * @pre For a physical result, both materials should provide positive
     *      `reference_diameter` and `reference_temperature` values.
     * @pre For a velocity-dependent result, @p relative_speed should be positive.
     *
     * @note
     * The fallback viscosity index is `T(0.5)`, the hard-sphere limit.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Applies the binary velocity update for a VHS-style collision.
     *
     * @details
     * Randomly rotates the relative velocity using stateless hash-based samples
     * and reconstructs post-collision velocities from the center of mass.
     * Momentum and relative kinetic energy are preserved.
     *
     * @param lhs_velocity Velocity of the left-hand particle. Updated in place.
     * @param rhs_velocity Velocity of the right-hand particle. Updated in place.
     * @param lhs Material properties of the left-hand particle/species.
     *        `lhs.molecular_mass` is used by the elastic scattering update.
     * @param rhs Material properties of the right-hand particle/species.
     *        `rhs.molecular_mass` is used by the elastic scattering update.
     *
     * @pre For a physical velocity update, both molecular masses should be positive.
     *
     * @post If the update accepts the pair, both velocities contain
     *       post-collision values.
     * @post If the update rejects the pair because of invalid masses or zero
     *       relative speed, both velocities are left unchanged.
     *
     * @note
     * The viscosity index affects collision scheduling through
     * `cross_section()`, not the post-collision scattering angle.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_hard_sphere_kernel.hpp>
