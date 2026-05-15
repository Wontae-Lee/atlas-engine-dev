#pragma once

/**
 * @file variable_soft_sphere_kernel.h
 * @brief Declares a Variable Soft Sphere DSMC collision kernel.
 *
 * @details
 * This file defines @ref atlas::system::VariableSoftSphereKernel, a stateless
 * DSMC collision-kernel component that combines:
 *
 * - VHS total collision cross-section evaluation,
 * - VSS scattering-parameter-controlled elastic angular scattering.
 *
 * In the Variable Soft Sphere (VSS) model, the total collision cross section is
 * commonly evaluated with the same reference-temperature and viscosity-index
 * law used by the Variable Hard Sphere (VHS) model. The main additional feature
 * of VSS is the scattering parameter, which modifies the post-collision angular
 * distribution.
 *
 * This implementation follows that structure:
 *
 * - `cross_section()` delegates to `VariableHardSphereKernel<T>::cross_section()`,
 * - `operator()` uses the averaged VSS scattering parameter to sample the
 *   post-collision relative-velocity direction.
 *
 * @section vss_cross_section_model Cross-section model
 *
 * The total collision cross section is evaluated by the VHS kernel:
 *
 * @f[
 *     \sigma_{\mathrm{VSS}}
 *     =
 *     \sigma_{\mathrm{VHS}}.
 * @f]
 *
 * The delegated VHS law is:
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
 * - @f$d_{ref,ij}@f$ is the pair reference diameter,
 * - @f$T_{ref,ij}@f$ is the pair reference temperature,
 * - @f$m_r@f$ is the reduced molecular mass,
 * - @f$g@f$ is the relative speed,
 * - @f$\omega_{ij}@f$ is the pair viscosity index.
 *
 * @section vss_scattering_model Scattering model
 *
 * The VSS scattering parameter is computed as the arithmetic pair average:
 *
 * @f[
 *     \alpha_{ij}
 *     =
 *     \frac{\alpha_i + \alpha_j}{2}.
 * @f]
 *
 * Missing material scattering parameters fall back to:
 *
 * @f[
 *     \alpha = 1.
 * @f]
 *
 * Two hash-based deterministic samples are generated:
 *
 * @f[
 *     u_1,\ u_2 \in [0,1].
 * @f]
 *
 * The polar scattering angle is sampled through:
 *
 * @f[
 *     \cos\chi
 *     =
 *     2u_1^{1/\alpha_{ij}} - 1,
 * @f]
 *
 * and the azimuthal angle is:
 *
 * @f[
 *     \phi = 2\pi u_2.
 * @f]
 *
 * When @f$\alpha_{ij}=1@f$, this reduces to the isotropic hard-sphere angular
 * distribution:
 *
 * @f[
 *     \cos\chi = 2u_1 - 1.
 * @f]
 *
 * The scattered relative-velocity direction is then used to reconstruct both
 * particle velocities from the center-of-mass frame. This preserves total
 * momentum and relative kinetic energy for valid positive masses.
 *
 * @note
 * The kernel is stateless. The pseudo-random samples are deterministic functions
 * of the collision input state, so no mutable random-number-generator state is
 * required in host or device code.
 *
 * @note
 * The scattering parameter affects the post-collision angular distribution, not
 * the total collision cross section.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Variable Soft Sphere DSMC collision kernel.
 *
 * @details
 * `VariableSoftSphereKernel<T>` provides two operations used by DSMC collision
 * solvers:
 *
 * - total collision cross-section evaluation through the VHS law,
 * - VSS angular scattering controlled by the material scattering parameter.
 *
 * The class stores no runtime state. Its behavior is determined by:
 *
 * - the input particle velocities,
 * - molecular masses,
 * - reference diameters,
 * - reference temperatures,
 * - viscosity indices,
 * - scattering parameters,
 * - hash-based pseudo-random samples derived from the collision input.
 *
 * @tparam T Floating-point scalar type used for velocities, masses, diameters,
 *         temperatures, scattering parameters, and cross-section calculations.
 */
template <typename T>
class VariableSoftSphereKernel final {
public:
    /**
     * @brief Computes the VSS total collision cross section.
     *
     * @details
     * This function delegates directly to:
     *
     * @code
     * VariableHardSphereKernel<T>::cross_section(lhs, rhs, relative_speed)
     * @endcode
     *
     * Therefore, the total cross section follows the same reference-temperature
     * and viscosity-index law as the VHS model:
     *
     * @f[
     *     \sigma_{\mathrm{VSS}}
     *     =
     *     \sigma_{\mathrm{VHS}}.
     * @f]
     *
     * The VHS law used by the delegated function is:
     *
     * @f[
     *     \sigma_{\mathrm{VHS}}
     *     =
     *     \frac{\pi d_{ref,ij}^{2}}{\Gamma(2.5-\omega_{ij})}
     *     \left(
     *       \frac{2 k_B T_{ref,ij}}{m_r g^2}
     *     \right)^{\omega_{ij}-0.5}.
     * @f]
     *
     * The VSS scattering parameter does not appear in this scalar cross-section
     * calculation. It is used only by `operator()` to control the angular
     * scattering distribution.
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of the relative velocity between the two
     *        particles.
     *
     * @return VSS/VHS total collision cross section.
     * @return `T(0)` if the delegated VHS cross-section evaluation rejects the
     *         inputs.
     *
     * @pre For a physical result, both materials should provide positive
     *      `reference_diameter` and `reference_temperature` values.
     * @pre Both molecular masses should be positive.
     * @pre @p relative_speed should be positive.
     *
     * @note
     * Missing viscosity indices are handled by the delegated VHS implementation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Applies a VSS elastic post-collision velocity update.
     *
     * @details
     * This function updates two particle velocities in place. It preserves the
     * center-of-mass velocity and the magnitude of the relative velocity, while
     * changing the direction of the relative velocity according to the averaged
     * VSS scattering parameter.
     *
     * The pair scattering parameter is:
     *
     * @f[
     *     \alpha_{ij}
     *     =
     *     \frac{\alpha_i + \alpha_j}{2}.
     * @f]
     *
     * If either material does not provide `scattering_parameter`, `T(1)` is used
     * for that material.
     *
     * Let the pre-collision velocities be:
     *
     * @f[
     *     \mathbf{v}_i,\quad \mathbf{v}_j,
     * @f]
     *
     * and the molecular masses be:
     *
     * @f[
     *     m_i,\quad m_j.
     * @f]
     *
     * The center-of-mass velocity is:
     *
     * @f[
     *     \mathbf{c}
     *     =
     *     \frac{
     *         m_i\mathbf{v}_i + m_j\mathbf{v}_j
     *     }{
     *         m_i + m_j
     *     }.
     * @f]
     *
     * The incoming relative velocity is:
     *
     * @f[
     *     \mathbf{g}
     *     =
     *     \mathbf{v}_i - \mathbf{v}_j,
     *     \qquad
     *     g = \|\mathbf{g}\|.
     * @f]
     *
     * The incoming relative-velocity direction is:
     *
     * @f[
     *     \mathbf{e}_g
     *     =
     *     \frac{\mathbf{g}}{g}.
     * @f]
     *
     * The implementation builds two tangential directions
     * @f$\mathbf{t}_1@f$ and @f$\mathbf{t}_2@f$ around @f$\mathbf{e}_g@f$.
     *
     * It then generates two hash-based samples:
     *
     * @f[
     *     u_1,\ u_2 \in [0,1].
     * @f]
     *
     * The polar and azimuthal scattering angles are:
     *
     * @f[
     *     \cos\chi
     *     =
     *     2u_1^{1/\alpha_{ij}} - 1,
     * @f]
     *
     * @f[
     *     \phi
     *     =
     *     2\pi u_2.
     * @f]
     *
     * The scattered relative-velocity direction is:
     *
     * @f[
     *     \mathbf{e}'_g
     *     =
     *     \mathbf{e}_g\cos\chi
     *     +
     *     \left(
     *         \mathbf{t}_1\cos\phi
     *         +
     *         \mathbf{t}_2\sin\phi
     *     \right)
     *     \sin\chi.
     * @f]
     *
     * The scattered relative velocity is:
     *
     * @f[
     *     \mathbf{g}'
     *     =
     *     g\mathbf{e}'_g.
     * @f]
     *
     * Finally, post-collision velocities are reconstructed as:
     *
     * @f[
     *     \mathbf{v}'_i
     *     =
     *     \mathbf{c}
     *     +
     *     \frac{m_j}{m_i + m_j}
     *     \mathbf{g}',
     * @f]
     *
     * @f[
     *     \mathbf{v}'_j
     *     =
     *     \mathbf{c}
     *     -
     *     \frac{m_i}{m_i + m_j}
     *     \mathbf{g}'.
     * @f]
     *
     * The function returns without modifying either velocity when:
     *
     * - either molecular mass is not positive,
     * - the total mass is not positive,
     * - the averaged scattering parameter is not positive,
     * - the relative speed is not positive.
     *
     * @param lhs_velocity Velocity of the left-hand particle. Updated in place.
     * @param rhs_velocity Velocity of the right-hand particle. Updated in place.
     * @param lhs Material properties of the left-hand particle/species.
     *        `lhs.molecular_mass` and `lhs.scattering_parameter` are used.
     * @param rhs Material properties of the right-hand particle/species.
     *        `rhs.molecular_mass` and `rhs.scattering_parameter` are used.
     *
     * @pre For a physical velocity update, both molecular masses should be positive.
     * @pre For a physical VSS angular distribution, both scattering parameters
     *      should be positive when provided.
     *
     * @post If all validity checks pass, both velocities contain post-collision
     *       values.
     * @post If any validity check fails, both velocities are left unchanged.
     *
     * @note
     * Missing scattering parameters fall back to `T(1)`, which gives isotropic
     * hard-sphere-style scattering.
     *
     * @note
     * The pseudo-random samples are deterministic functions of the collision
     * inputs. This keeps the operator stateless and safe for parallel host/device
     * execution.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_soft_sphere_kernel.hpp>