#pragma once

/**
 * @file variable_soft_sphere_kernel.h
 * @brief Declares a Variable Soft Sphere style collision kernel for DSMC solvers.
 *
 * @details
 * This file defines @ref atlas::system::VariableSoftSphereKernel, a lightweight
 * DSMC collision-kernel component that evaluates a relative-speed-dependent
 * collision cross section and reuses the simple hard-sphere elastic velocity
 * update.
 *
 * In a complete Variable Soft Sphere (VSS) DSMC model, the collision cross
 * section depends on relative speed and the post-collision scattering angle is
 * controlled by an additional scattering parameter. The current implementation
 * uses the scattering parameter only in the scalar cross-section scaling. The
 * post-collision velocity update itself is delegated to
 * @ref atlas::system::HardSphereKernel.
 *
 * @section vss_cross_section_model Cross-section model
 *
 * The implementation first computes the base hard-sphere cross section:
 *
 * @f[
 *     \sigma_{\mathrm{HS}}
 *     =
 *     \pi d_{ij}^{2},
 * @f]
 *
 * where the effective pair diameter is:
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
 * The implementation then computes an averaged scattering parameter:
 *
 * @f[
 *     \alpha_{ij}
 *     =
 *     \frac{\alpha_i + \alpha_j}{2},
 * @f]
 *
 * where @f$\alpha_i@f$ and @f$\alpha_j@f$ are read from
 * `MaterialProperties<T>::scattering_parameter`. If a material does not provide
 * a scattering parameter, `T(1)` is used as the fallback.
 *
 * The returned cross section is:
 *
 * @f[
 *     \sigma_{\mathrm{VSS}}
 *     =
 *     \sigma_{\mathrm{HS}}
 *     g^{\alpha_{ij} - 1},
 * @f]
 *
 * where @f$g@f$ is the relative speed supplied to `cross_section()`.
 *
 * In code form:
 *
 * @code
 * base_cross_section = HardSphereKernel<T>::cross_section(lhs, rhs);
 * alpha_ij = 0.5 * (lhs.scattering_parameter + rhs.scattering_parameter);
 * speed_scale = pow(relative_speed, alpha_ij - 1);
 * sigma_vss = base_cross_section * speed_scale;
 * @endcode
 *
 * If the relative speed is not positive, the implementation uses a speed scale
 * of `T(1)` and therefore falls back to the base hard-sphere cross section.
 *
 * @section vss_velocity_update Velocity update model
 *
 * The velocity update is delegated to @ref atlas::system::HardSphereKernel:
 *
 * @code
 * HardSphereKernel<T>{}(lhs_velocity, rhs_velocity, lhs, rhs);
 * @endcode
 *
 * Therefore, this implementation does not currently sample a VSS post-collision
 * angular distribution. It applies the same simple deterministic elastic
 * two-body velocity update used by the hard-sphere kernel.
 *
 * @note
 * This class is VSS-style in its cross-section scaling, but it is not a complete
 * stochastic VSS scattering implementation.
 *
 * @note
 * The scattering parameter affects collision scheduling through
 * `cross_section()`. It does not directly affect the post-collision velocity
 * direction in the current implementation.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Variable Soft Sphere style DSMC collision kernel.
 *
 * @details
 * `VariableSoftSphereKernel<T>` provides two operations used by DSMC collision
 * solvers:
 *
 * - VSS-style relative-speed-dependent collision cross-section evaluation,
 * - simple elastic binary velocity update delegated to `HardSphereKernel<T>`.
 *
 * The class is stateless. All behavior is determined by the input velocities and
 * the material properties of the two colliding species.
 *
 * @tparam T Floating-point scalar type used for velocities, masses, diameters,
 *         scattering parameters, and cross-section calculations.
 */
template <typename T>
class VariableSoftSphereKernel final {
public:
    /**
     * @brief Computes the VSS-style relative-speed-dependent collision cross section.
     *
     * @details
     * This function computes an effective collision cross section for two
     * species/material records and a supplied relative speed.
     *
     * The base cross section is obtained from the hard-sphere model:
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
     * The averaged scattering parameter is:
     *
     * @f[
     *     \alpha_{ij}
     *     =
     *     \frac{\alpha_i + \alpha_j}{2}.
     * @f]
     *
     * The final cross section is:
     *
     * @f[
     *     \sigma_{\mathrm{VSS}}
     *     =
     *     \sigma_{\mathrm{HS}}
     *     g^{\alpha_{ij} - 1},
     * @f]
     *
     * where:
     *
     * - @f$\sigma_{\mathrm{VSS}}@f$ is the returned cross section,
     * - @f$\sigma_{\mathrm{HS}}@f$ is the base hard-sphere cross section,
     * - @f$g@f$ is @p relative_speed,
     * - @f$\alpha_{ij}@f$ is the averaged scattering parameter.
     *
     * If either material lacks a valid collision diameter, the base hard-sphere
     * cross section is zero and this function returns zero.
     *
     * If @p relative_speed is not positive, the speed scaling factor is set to
     * `T(1)`. This avoids evaluating `pow()` at zero or negative speed.
     *
     * @param lhs Material properties of the left-hand particle/species.
     * @param rhs Material properties of the right-hand particle/species.
     * @param relative_speed Magnitude of the relative velocity between the two
     *        particles.
     *
     * @return VSS-style effective collision cross section.
     * @return `T(0)` if the base hard-sphere cross section is unavailable or
     *         invalid.
     *
     * @pre For a physical result, both materials should provide positive
     *      `collision_diameter` values.
     * @pre For a relative-speed-dependent result, @p relative_speed should be
     *      positive.
     *
     * @note
     * The fallback scattering parameter is `T(1)`. With
     * @f$\alpha_{ij}=1@f$, the exponent becomes zero and the result reduces to
     * the hard-sphere cross section.
     *
     * @note
     * In a full VSS model, the scattering parameter also controls angular
     * scattering. In this implementation, it only affects the scalar
     * cross-section scaling.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs,
                  T relative_speed) noexcept;

    /**
     * @brief Applies the binary velocity update for a VSS-style collision.
     *
     * @details
     * The current implementation delegates the velocity update to
     * `HardSphereKernel<T>`:
     *
     * @code
     * HardSphereKernel<T>{}(lhs_velocity, rhs_velocity, lhs, rhs);
     * @endcode
     *
     * Therefore, the post-collision velocity update follows the same simple
     * deterministic elastic two-body formula as the hard-sphere kernel.
     *
     * Let the input velocities be:
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
     * The delegated hard-sphere update computes:
     *
     * @f[
     *     \mathbf{g}
     *     =
     *     \mathbf{v}_i - \mathbf{v}_j,
     * @f]
     *
     * and uses:
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
     * This operation updates both velocity references in place.
     *
     * @param lhs_velocity Velocity of the left-hand particle. Updated in place.
     * @param rhs_velocity Velocity of the right-hand particle. Updated in place.
     * @param lhs Material properties of the left-hand particle/species.
     *        `lhs.molecular_mass` is used by the delegated hard-sphere update.
     * @param rhs Material properties of the right-hand particle/species.
     *        `rhs.molecular_mass` is used by the delegated hard-sphere update.
     *
     * @pre For a physical velocity update, both molecular masses should be
     *      positive.
     *
     * @post If the delegated hard-sphere update accepts the pair, both velocities
     *       contain post-collision values.
     * @post If the delegated hard-sphere update rejects the pair because of
     *       invalid masses or zero relative speed, both velocities are left
     *       unchanged.
     *
     * @note
     * This function does not currently use `scattering_parameter` directly.
     * The scattering parameter affects collision scheduling through
     * `cross_section()`.
     *
     * @note
     * This is not a full stochastic VSS angular-scattering implementation.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/variable_soft_sphere_kernel.hpp>