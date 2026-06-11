#pragma once

/**
 * @file hard_sphere_kernel.h
 * @brief Declares a hard-sphere DSMC collision kernel with hash-based scattering.
 *
 * @details
 * This file defines @ref atlas::system::HardSphereKernel, a binary collision
 * kernel used by DSMC solvers to evaluate hard-sphere collision cross sections
 * and update the velocities of two colliding particles.
 *
 * The kernel provides two operations:
 *
 * - `cross_section()`: computes an effective hard-sphere collision cross section
 *   from the reference diameters of two material/species records.
 * - `operator()`: applies an elastic binary velocity update using a hash-based
 *   sampled post-collision relative-velocity direction.
 *
 * @section hard_sphere_cross_section Cross-section model
 *
 * For two reference diameters @f$d_i@f$ and @f$d_j@f$, this implementation first
 * computes the arithmetic mean diameter:
 *
 * @f[
 *     d_{ij}
 *     =
 *     \frac{d_i + d_j}{2}.
 * @f]
 *
 * The effective hard-sphere collision cross section is:
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
 * With SI-consistent inputs, @f$d_i@f$ and @f$d_j@f$ should be given in meters,
 * and @f$\sigma_{ij}@f$ is returned in square meters.
 *
 * If either species does not provide a reference diameter, or if the averaged
 * diameter is not positive, the cross section is reported as zero.
 *
 * @section hard_sphere_velocity_update Velocity update model
 *
 * The velocity-update operator performs an elastic two-body collision in the
 * center-of-mass frame.
 *
 * Let the pre-collision velocities be:
 *
 * @f[
 *     \mathbf{v}_i,\quad \mathbf{v}_j,
 * @f]
 *
 * and let the molecular masses be:
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
 * The pre-collision relative velocity is:
 *
 * @f[
 *     \mathbf{g}
 *     =
 *     \mathbf{v}_i - \mathbf{v}_j,
 * @f]
 *
 * with magnitude:
 *
 * @f[
 *     g = \|\mathbf{g}\|.
 * @f]
 *
 * A hash-based deterministic sampling step produces two pseudo-random values:
 *
 * @f[
 *     u_1,\ u_2 \in [0,1].
 * @f]
 *
 * These values define the scattering angles:
 *
 * @f[
 *     \cos\chi = 2u_1 - 1,
 * @f]
 *
 * @f[
 *     \phi = 2\pi u_2.
 * @f]
 *
 * The incoming relative-velocity direction:
 *
 * @f[
 *     \mathbf{e}_g
 *     =
 *     \frac{\mathbf{g}}{g}
 * @f]
 *
 * is used to build an orthonormal basis:
 *
 * @f[
 *     \mathbf{e}_g,\quad \mathbf{t}_1,\quad \mathbf{t}_2.
 * @f]
 *
 * The sampled post-collision relative-velocity direction is:
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
 * The post-collision relative velocity is:
 *
 * @f[
 *     \mathbf{g}'
 *     =
 *     g \mathbf{e}'_g.
 * @f]
 *
 * Finally, the post-collision velocities are reconstructed as:
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
 * This update preserves total momentum and relative kinetic energy for valid
 * positive molecular masses.
 *
 * @note
 * The scattering samples are hash-based deterministic functions of the collision
 * inputs. This keeps the operator stateless, reproducible, and safe for
 * parallel host/device execution, but it is not backed by a mutable random-number
 * generator state.
 *
 * @note
 * The kernel updates velocity only. It does not update particle position,
 * collision time, collision probability, or particle species.
 */

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Hard-sphere binary collision kernel with hash-based scattering.
 *
 * @details
 * `HardSphereKernel<T>` provides the hard-sphere collision operations used by
 * DSMC collision solvers.
 *
 * The class has no runtime state. Its behavior is determined by:
 *
 * - the two input particle velocities,
 * - the two material-property records,
 * - the reference diameters,
 * - the molecular masses,
 * - hash-based pseudo-random samples derived from the collision inputs.
 *
 * The cross-section calculation uses the reference diameters stored in
 * `MaterialProperties<T>::reference_diameter`. The velocity update uses the
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
     * If the two species have reference diameters @f$d_i@f$ and @f$d_j@f$, the
     * effective pair diameter is:
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
     * - either material does not provide `reference_diameter`,
     * - the averaged reference diameter is not positive.
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
     * @pre `lhs.reference_diameter` should be positive for a physical result.
     * @pre `rhs.reference_diameter` should be positive for a physical result.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section(const MaterialProperties<T>& lhs,
                  const MaterialProperties<T>& rhs) noexcept;

    /**
     * @brief Applies an elastic hard-sphere collision to two velocities.
     *
     * @details
     * This function updates two particle velocities in place by preserving the
     * center-of-mass velocity and rotating the relative velocity to a hash-sampled
     * post-collision direction.
     *
     * Let the two input velocities be:
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
     * The relative velocity is:
     *
     * @f[
     *     \mathbf{g}
     *     =
     *     \mathbf{v}_i - \mathbf{v}_j.
     * @f]
     *
     * If @f$g=\|\mathbf{g}\|@f$ is not positive, no update is applied.
     *
     * The function samples two deterministic pseudo-random values @f$u_1@f$ and
     * @f$u_2@f$ from the collision inputs and uses:
     *
     * @f[
     *     \cos\chi = 2u_1 - 1,\qquad
     *     \phi = 2\pi u_2.
     * @f]
     *
     * These angles define a new relative-velocity direction @f$\mathbf{e}'_g@f$.
     * The relative speed magnitude is preserved:
     *
     * @f[
     *     \mathbf{g}' = g\mathbf{e}'_g.
     * @f]
     *
     * The post-collision velocities are reconstructed as:
     *
     * @f[
     *     \mathbf{v}_i'
     *     =
     *     \mathbf{c}
     *     +
     *     \frac{m_j}{m_i + m_j}
     *     \mathbf{g}',
     * @f]
     *
     * @f[
     *     \mathbf{v}_j'
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
     * - the relative speed is not positive.
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
     * The random samples are deterministic functions of the collision inputs.
     * This keeps the stateless operator reproducible and thread-safe.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

#include <atlas/solver/dsmc/hard_sphere_kernel.hpp>
