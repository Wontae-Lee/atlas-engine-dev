#pragma once

/**
 * @file variable_hard_sphere_kernel.h
 * @brief Declares a Variable Hard Sphere style collision kernel for DSMC solvers.
 *
 * @details
 * This file defines @ref atlas::system::VariableHardSphereKernel, a lightweight
 * DSMC collision-kernel component that evaluates a velocity-dependent collision
 * cross section and reuses the simple hard-sphere elastic velocity update.
 *
 * The Variable Hard Sphere (VHS) model is commonly used in DSMC to represent
 * transport-property effects by making the effective collision cross section
 * depend on relative speed. In this implementation, the cross-section evaluation
 * is modified from the hard-sphere value by a power-law factor based on the
 * species viscosity indices.
 *
 * @section vhs_cross_section_model Cross-section model
 *
 * The implementation first computes the base hard-sphere cross section:
 *
 * @f[
 *     \sigma_{\mathrm{HS}}
 *     =
 *     \pi d_{ij}^{2},
 * @f]
 *
 * where:
 *
 * @f[
 *     d_{ij}
 *     =
 *     \frac{d_i + d_j}{2}.
 * @f]
 *
 * Here @f$d_i@f$ and @f$d_j@f$ are the collision diameters stored in the two
 * material-property records.
 *
 * The implementation then computes an averaged viscosity index:
 *
 * @f[
 *     \omega_{ij}
 *     =
 *     \frac{\omega_i + \omega_j}{2},
 * @f]
 *
 * where @f$\omega_i@f$ and @f$\omega_j@f$ are read from
 * `MaterialProperties<T>::viscosity_index`. If a material does not provide a
 * viscosity index, `T(1)` is used as the fallback.
 *
 * The returned cross section is:
 *
 * @f[
 *     \sigma_{\mathrm{VHS}}
 *     =
 *     \sigma_{\mathrm{HS}}
 *     g^{\omega_{ij} - 1},
 * @f]
 *
 * where @f$g@f$ is the relative speed supplied to `cross_section()`.
 *
 * In code form:
 *
 * @code
 * base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);
 * omega_ij = 0.5 * (lhs.viscosity_index + rhs.viscosity_index);
 * speed_scale = pow(relative_speed, omega_ij - 1);
 * sigma_vhs = base_cross_section * speed_scale;
 * @endcode
 *
 * If @p relative_speed is not positive, the implementation uses a speed scale of
 * `T(1)` instead of evaluating the power law.
 *
 * @section vhs_velocity_update Velocity update model
 *
 * The velocity update is delegated to @ref atlas::system::HardSphereKernel.
 * Therefore, this implementation does not sample a random VHS post-collision
 * scattering direction. It applies the same simple deterministic elastic
 * two-body velocity update used by the hard-sphere kernel.
 *
 * @note
 * This class is VHS-style in its cross-section calculation, but its velocity
 * update currently reuses the hard-sphere kernel. It should not be documented as
 * a complete stochastic VHS scattering implementation unless the operator is
 * extended accordingly.
 *
 * @note
 * The kernel is stateless. All behavior is determined by the input velocities,
 * collision diameters, molecular masses, viscosity indices, and relative speed.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Variable Hard Sphere style DSMC collision kernel.
 *
 * @details
 * `VariableHardSphereKernel<T>` provides two operations used by the DSMC kernel
 * dispatch layer:
 *
 * - velocity-dependent VHS-style collision cross-section evaluation,
 * - simple elastic binary velocity update delegated to `HardSphereKernel<T>`.
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
     * The base cross section is obtained from the hard-sphere kernel:
     *
     * @f[
     *     \sigma_{\mathrm{HS}}
     *     =
     *     \pi
     *     \left(
     *         \frac{d_i + d_j}{2}
     *     \right)^2.
     * @f]
     *
     * The VHS speed-dependent scaling uses the averaged viscosity index:
     *
     * @f[
     *     \omega_{ij}
     *     =
     *     \frac{\omega_i + \omega_j}{2}.
     * @f]
     *
     * The final cross section is:
     *
     * @f[
     *     \sigma_{\mathrm{VHS}}
     *     =
     *     \sigma_{\mathrm{HS}}
     *     g^{\omega_{ij} - 1},
     * @f]
     *
     * where:
     *
     * - @f$\sigma_{\mathrm{VHS}}@f$ is the returned cross section,
     * - @f$\sigma_{\mathrm{HS}}@f$ is the base hard-sphere cross section,
     * - @f$g@f$ is @p relative_speed,
     * - @f$\omega_{ij}@f$ is the averaged viscosity index.
     *
     * If either collision diameter is missing or invalid, the hard-sphere base
     * cross section is zero, and this function returns zero.
     *
     * If @p relative_speed is not positive, the speed scale is set to `T(1)`.
     * This avoids evaluating `pow()` at zero or negative speed and makes the
     * function fall back to the base hard-sphere cross section.
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of the relative velocity between the two
     *        particles.
     *
     * @return VHS-style effective collision cross section.
     * @return `T(0)` if the base hard-sphere cross section is unavailable or
     *         invalid.
     *
     * @pre For a physical result, both materials should provide positive
     *      `collision_diameter` values.
     * @pre For a velocity-dependent result, @p relative_speed should be positive.
     *
     * @note
     * The fallback viscosity index is `T(1)`. With
     * @f$\omega_{ij}=1@f$, the exponent becomes zero and the velocity scaling is
     * one, so the result reduces to the hard-sphere cross section.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Applies the binary velocity update for a VHS-style collision.
     *
     * @details
     * The current implementation delegates the velocity update to
     * `HardSphereKernel<T>`:
     *
     * @code
     * HardSphereKernel<T>{}(lhs_velocity, rhs_velocity, lhs, rhs);
     * @endcode
     *
     * Therefore, the update uses the same simple elastic two-body formula as the
     * hard-sphere kernel.
     *
     * Let the two velocities be:
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
     * The hard-sphere update computes the relative velocity:
     *
     * @f[
     *     \mathbf{g}
     *     =
     *     \mathbf{v}_i - \mathbf{v}_j,
     * @f]
     *
     * then uses:
     *
     * @f[
     *     \mathbf{n}
     *     =
     *     \frac{\mathbf{g}}{\|\mathbf{g}\|}
     * @f]
     *
     * as the collision normal. The updated velocities are:
     *
     * @f[
     *     \mathbf{v}_i'
     *     =
     *     \mathbf{v}_i
     *     -
     *     \frac{2m_j}{m_i + m_j}
     *     (\mathbf{g}\cdot\mathbf{n})
     *     \mathbf{n},
     * @f]
     *
     * @f[
     *     \mathbf{v}_j'
     *     =
     *     \mathbf{v}_j
     *     +
     *     \frac{2m_i}{m_i + m_j}
     *     (\mathbf{g}\cdot\mathbf{n})
     *     \mathbf{n}.
     * @f]
     *
     * This operation updates the two velocity references in place.
     *
     * @param lhs_velocity Velocity of the left-hand particle. Updated in place.
     * @param rhs_velocity Velocity of the right-hand particle. Updated in place.
     * @param lhs Material properties of the left-hand particle/species.
     *        `lhs.molecular_mass` is used by the delegated hard-sphere update.
     * @param rhs Material properties of the right-hand particle/species.
     *        `rhs.molecular_mass` is used by the delegated hard-sphere update.
     *
     * @pre For a physical velocity update, both molecular masses should be positive.
     *
     * @post If the delegated hard-sphere update accepts the pair, both velocities
     *       contain post-collision values.
     * @post If the delegated hard-sphere update rejects the pair because of
     *       invalid masses or zero relative speed, both velocities are left
     *       unchanged.
     *
     * @note
     * This function currently does not use the viscosity index directly. The
     * viscosity index affects collision scheduling through `cross_section()`, not
     * the post-collision velocity update.
     *
     * @note
     * This is not a full stochastic VHS scattering implementation. It does not
     * randomly sample a post-collision relative-velocity direction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_hard_sphere_kernel.hpp>