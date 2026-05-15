#pragma once

/**
 * @file hard_sphere_kernel.h
 * @brief Declares a simple hard-sphere collision kernel for DSMC solvers.
 *
 * @details
 * This file defines @ref atlas::system::HardSphereKernel, a binary collision
 * kernel used by DSMC solvers to evaluate hard-sphere collision cross sections
 * and to update the velocities of two colliding particles.
 *
 * The kernel provides two operations:
 *
 * - `cross_section()`: computes an effective hard-sphere collision cross section
 *   from the collision diameters of two material/species records.
 * - `operator()`: applies a simple elastic binary velocity update to two
 *   particle velocities.
 *
 * @section hard_sphere_cross_section Cross-section model
 *
 * For two collision diameters @f$d_i@f$ and @f$d_j@f$, this implementation first
 * computes the arithmetic mean diameter:
 *
 * @f[
 *     d_{ij}
 *     =
 *     \frac{d_i + d_j}{2}.
 * @f]
 *
 * The effective hard-sphere collision cross section is then:
 *
 * @f[
 *     \sigma_{ij}
 *     =
 *     \pi d_{ij}^{2}.
 * @f]
 *
 * Expanding the expression gives:
 *
 * @f[
 *     \sigma_{ij}
 *     =
 *     \pi
 *     \left(
 *         \frac{d_i + d_j}{2}
 *     \right)^2.
 * @f]
 *
 * If either species does not provide a collision diameter, or if the resulting
 * effective diameter is not positive, the cross section is reported as zero.
 *
 * @section hard_sphere_velocity_update Velocity update model
 *
 * The velocity-update operator applies a simple elastic two-body update along the
 * current relative-velocity direction.
 *
 * Let:
 *
 * @f[
 *     \mathbf{v}_i
 * @f]
 *
 * and:
 *
 * @f[
 *     \mathbf{v}_j
 * @f]
 *
 * be the two particle velocities, and let:
 *
 * @f[
 *     m_i,\quad m_j
 * @f]
 *
 * be their molecular masses.
 *
 * The relative velocity is:
 *
 * @f[
 *     \mathbf{g}
 *     =
 *     \mathbf{v}_i - \mathbf{v}_j.
 * @f]
 *
 * Its magnitude is:
 *
 * @f[
 *     g
 *     =
 *     \|\mathbf{g}\|.
 * @f]
 *
 * The implementation uses the normalized relative-velocity direction as the
 * collision normal:
 *
 * @f[
 *     \mathbf{n}
 *     =
 *     \frac{\mathbf{g}}{\|\mathbf{g}\|}.
 * @f]
 *
 * The normal relative velocity is then:
 *
 * @f[
 *     g_n
 *     =
 *     \mathbf{g} \cdot \mathbf{n}.
 * @f]
 *
 * Since @f$\mathbf{n}@f$ is chosen as the normalized relative velocity, this is
 * equivalent to:
 *
 * @f[
 *     g_n = \|\mathbf{g}\|.
 * @f]
 *
 * The elastic velocity update is:
 *
 * @f[
 *     \mathbf{v}_i'
 *     =
 *     \mathbf{v}_i
 *     -
 *     \frac{2m_j}{m_i + m_j}
 *     g_n
 *     \mathbf{n},
 * @f]
 *
 * and:
 *
 * @f[
 *     \mathbf{v}_j'
 *     =
 *     \mathbf{v}_j
 *     +
 *     \frac{2m_i}{m_i + m_j}
 *     g_n
 *     \mathbf{n}.
 * @f]
 *
 * This update preserves the center-of-mass velocity for valid positive masses
 * and performs an elastic exchange of the normal relative component.
 *
 * @note
 * This implementation is intentionally simple. It does not randomly sample a
 * post-collision scattering direction. A full stochastic DSMC hard-sphere model
 * often samples a random post-collision relative-velocity direction while
 * preserving relative speed and momentum. This kernel instead uses the current
 * relative-velocity direction as the collision normal.
 *
 * @note
 * The kernel updates velocity only. It does not update particle position,
 * collision time, collision probability, or particle species.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>

namespace atlas::system {

/**
 * @brief Simple hard-sphere binary collision kernel.
 *
 * @details
 * `HardSphereKernel<T>` provides the hard-sphere collision operations used by
 * DSMC collision solvers.
 *
 * The class has no runtime state. Its behavior is fully determined by:
 *
 * - the two input particle velocities,
 * - the two material-property records,
 * - the collision diameters,
 * - the molecular masses.
 *
 * The cross-section calculation uses the collision diameters stored in
 * `MaterialProperties<T>::collision_diameter`. The velocity update uses the
 * molecular masses stored in `MaterialProperties<T>::molecular_mass`.
 *
 * @tparam T Floating-point scalar type used for velocity, mass, diameter, and
 *         cross-section calculations.
 */
template <typename T>
class HardSphereKernel final {
public:
    /**
     * @brief Computes the effective hard-sphere collision cross section.
     *
     * @details
     * This function computes the hard-sphere collision cross section for a pair
     * of species/material records.
     *
     * If the two species have collision diameters:
     *
     * @f[
     *     d_i
     * @f]
     *
     * and:
     *
     * @f[
     *     d_j,
     * @f]
     *
     * the effective pair diameter is:
     *
     * @f[
     *     d_{ij}
     *     =
     *     \frac{d_i + d_j}{2}.
     * @f]
     *
     * The returned cross section is:
     *
     * @f[
     *     \sigma_{ij}
     *     =
     *     \pi d_{ij}^{2}
     *     =
     *     \pi
     *     \left(
     *         \frac{d_i + d_j}{2}
     *     \right)^2.
     * @f]
     *
     * With SI-consistent inputs:
     *
     * - @f$d_i@f$ and @f$d_j@f$ should be given in meters,
     * - @f$\sigma_{ij}@f$ is returned in square meters.
     *
     * The function returns zero when:
     *
     * - either material does not provide `collision_diameter`,
     * - the averaged collision diameter is not positive.
     *
     * Returning zero is useful because DSMC collision-count estimation commonly
     * uses:
     *
     * @f[
     *     \sigma_{ij} g_{ij},
     * @f]
     *
     * where @f$g_{ij}@f$ is the relative speed. A zero cross section therefore
     * disables collisions for pairs whose diameter data is unavailable or invalid.
     *
     * @param lhs Left species material properties.
     * @param rhs Right species material properties.
     *
     * @return Effective hard-sphere collision cross section.
     * @return `T(0)` if required diameter data is missing or invalid.
     *
     * @pre `lhs.collision_diameter` should be positive for a physical result.
     * @pre `rhs.collision_diameter` should be positive for a physical result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs) noexcept;

    /**
     * @brief Applies a simple elastic hard-sphere collision to two velocities.
     *
     * @details
     * This function updates two particle velocities in place using a deterministic
     * elastic binary collision formula.
     *
     * Let the two input velocities be:
     *
     * @f[
     *     \mathbf{v}_i,\quad \mathbf{v}_j.
     * @f]
     *
     * Let the two molecular masses be:
     *
     * @f[
     *     m_i,\quad m_j.
     * @f]
     *
     * The function first computes the relative velocity:
     *
     * @f[
     *     \mathbf{g}
     *     =
     *     \mathbf{v}_i - \mathbf{v}_j.
     * @f]
     *
     * If:
     *
     * @f[
     *     \|\mathbf{g}\| \le 0,
     * @f]
     *
     * no update is applied.
     *
     * Otherwise, the collision normal is chosen as:
     *
     * @f[
     *     \mathbf{n}
     *     =
     *     \frac{\mathbf{g}}{\|\mathbf{g}\|}.
     * @f]
     *
     * The normal relative velocity is:
     *
     * @f[
     *     g_n
     *     =
     *     \mathbf{g} \cdot \mathbf{n}.
     * @f]
     *
     * The post-collision velocities are:
     *
     * @f[
     *     \mathbf{v}_i'
     *     =
     *     \mathbf{v}_i
     *     -
     *     \frac{2m_j}{m_i + m_j}
     *     g_n
     *     \mathbf{n},
     * @f]
     *
     * @f[
     *     \mathbf{v}_j'
     *     =
     *     \mathbf{v}_j
     *     +
     *     \frac{2m_i}{m_i + m_j}
     *     g_n
     *     \mathbf{n}.
     * @f]
     *
     * For equal masses, this reduces to an exchange of the velocity component
     * along @f$\mathbf{n}@f$.
     *
     * The function returns without modifying either velocity when:
     *
     * - either molecular mass is not positive,
     * - the total mass is not positive,
     * - the relative speed is not positive,
     * - the normal relative velocity is not positive.
     *
     * @param lhs_velocity Velocity of the first particle. Updated in place.
     * @param rhs_velocity Velocity of the second particle. Updated in place.
     * @param lhs Left species material properties. `lhs.molecular_mass` is used.
     * @param rhs Right species material properties. `rhs.molecular_mass` is used.
     *
     * @pre `lhs.molecular_mass` should be positive for a physical collision.
     * @pre `rhs.molecular_mass` should be positive for a physical collision.
     *
     * @post If all validity checks pass, @p lhs_velocity and @p rhs_velocity
     *       contain the post-collision velocities.
     * @post If any validity check fails, both velocities are left unchanged.
     *
     * @note
     * This deterministic update is simpler than a full stochastic DSMC
     * hard-sphere collision, where the post-collision relative-velocity direction
     * is usually sampled randomly.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/hard_sphere_kernel.hpp>