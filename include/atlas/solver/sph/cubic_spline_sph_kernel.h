#pragma once

/**
 * @file cubic_spline_sph_kernel.h
 * @brief Declares the cubic-spline smoothing kernel used by SPH solvers.
 *
 * This header defines `CubicSplineSphKernel<T>`, a stateless collection of
 * smoothing-kernel functions used in Smoothed Particle Hydrodynamics (SPH).
 *
 * The cubic spline kernel is one of the most common SPH kernel choices because it:
 * - has compact support
 * - is inexpensive to evaluate
 * - provides smooth interpolation and force estimation
 *
 * In this implementation, the kernel interface is split into three physically
 * meaningful operations:
 * - `density_weight(...)`
 *   Used when accumulating scalar density contributions
 * - `pressure_gradient(...)`
 *   Used when evaluating pressure-force terms
 * - `viscosity_laplacian(...)`
 *   Used when evaluating viscosity diffusion terms
 *
 * The class is stateless and fully static, making it suitable for:
 * - host-side use
 * - device-side use
 * - repeated invocation inside tight particle-neighborhood loops
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Cubic-spline SPH kernel utilities.
 *
 * This type provides the kernel functions typically needed by an SPH solver.
 * It does not store any runtime state; all results depend only on:
 * - particle separation information
 * - the smoothing length
 *
 * The kernel is assumed to use compact support, so contributions typically vanish
 * when the inter-particle distance exceeds the support radius implied by
 * `smoothing_length`.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct CubicSplineSphKernel final {
    /**
     * @brief Evaluate the scalar density contribution weight of the cubic spline kernel.
     *
     * This function returns the scalar kernel value used in density accumulation,
     * typically in expressions of the form:
     *
     * \f[
     * \rho_i = \sum_j m_j W(r_{ij}, h)
     * \f]
     *
     * where:
     * - \f$ r_{ij} \f$ is the distance between particles
     * - \f$ h \f$ is the smoothing length
     * - \f$ W \f$ is the cubic spline kernel value
     *
     * This function is:
     * - stateless
     * - side-effect free
     * - inexpensive enough to be called inside dense neighbor loops
     *
     * @param radius Distance between two particles.
     * @param smoothing_length SPH smoothing length.
     * @return Scalar density weight contributed by the kernel at the given radius.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(T radius, T smoothing_length) noexcept;

    /**
     * @brief Evaluate the pressure-force kernel gradient for the cubic spline kernel.
     *
     * This function returns the gradient of the smoothing kernel with respect to
     * the particle separation vector. It is typically used in pressure-force terms
     * such as:
     *
     * \f[
     * \mathbf{f}^{pressure}_i \propto -\sum_j m_j
     * \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}\right)
     * \nabla W(\mathbf{r}_{ij}, h)
     * \f]
     *
     * The input `delta` usually represents the relative displacement vector between
     * two particles, for example:
     *
     * \f[
     * \mathbf{r}_{ij} = \mathbf{x}_i - \mathbf{x}_j
     * \f]
     *
     * The scalar `radius` is typically the magnitude of `delta`.
     *
     * @param delta Relative displacement vector between two particles.
     * @param radius Magnitude of the relative displacement.
     * @param smoothing_length SPH smoothing length.
     * @return Pressure kernel gradient vector.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T smoothing_length) noexcept;

    /**
     * @brief Evaluate the viscosity Laplacian term of the cubic spline kernel.
     *
     * This function returns the scalar Laplacian of the smoothing kernel, which is
     * commonly used in viscosity-force formulations such as:
     *
     * \f[
     * \mathbf{f}^{viscosity}_i \propto
     * \sum_j m_j \frac{\mu_j}{\rho_j}
     * (\mathbf{v}_j - \mathbf{v}_i)\nabla^2 W(r_{ij}, h)
     * \f]
     *
     * The exact SPH viscosity formulation may differ across implementations, but
     * this function provides the kernel-side Laplacian quantity needed for such
     * diffusion-like terms.
     *
     * @param radius Distance between two particles.
     * @param smoothing_length SPH smoothing length.
     * @return Scalar viscosity Laplacian value of the kernel.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(T radius, T smoothing_length) noexcept;
};

} // namespace atlas::system

#include <atlas/solver/sph/cubic_spline_sph_kernel.hpp>