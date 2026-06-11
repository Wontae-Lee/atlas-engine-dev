#pragma once

/**
 * @file wendland_quintic_sph_kernel.h
 * @brief Declares the Wendland quintic smoothing kernel used by SPH solvers.
 *
 * This header defines `WendlandQuinticSphKernel<T>`, a smoothing-kernel
 * implementation commonly used in Smoothed Particle Hydrodynamics (SPH).
 *
 * The Wendland quintic kernel is widely used due to its favorable numerical
 * properties:
 * - compact support
 * - high smoothness (C2 continuous)
 * - improved particle stability
 * - reduced tensile instability compared to spline-based kernels
 *
 * Like other SPH kernels in this library, this kernel exposes:
 * - `density_weight(...)`        → scalar kernel value
 * - `pressure_gradient(...)`     → gradient of the kernel
 * - `viscosity_laplacian(...)`   → Laplacian of the kernel
 *
 * The implementation is stateless and suitable for:
 * - host-side execution
 * - device-side execution
 * - repeated use in dense particle-neighborhood loops
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Wendland quintic SPH smoothing-kernel utilities.
 *
 * This kernel provides the core building blocks required for SPH computations.
 * All functions are static and depend only on:
 * - inter-particle distance
 * - cell size
 *
 * The Wendland quintic kernel is often preferred over cubic spline kernels in
 * simulations requiring higher stability and smoother force behavior, especially
 * in incompressible or weakly compressible SPH formulations.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct WendlandQuinticSphKernel final {
    /**
     * @brief Evaluate the scalar density weight of the Wendland quintic kernel.
     *
     * This function returns the smoothing-kernel value used in density
     * accumulation:
     *
     * \f[
     * \rho_i = \sum_j m_j W(r_{ij}, h)
     * \f]
     *
     * Compared to other kernels, the Wendland quintic formulation:
     * - avoids negative lobes
     * - provides smoother density estimates
     * - improves particle distribution regularity
     *
     * @param radius Distance between two particles.
     * @param cell_size SPH cell size.
     * @return Scalar kernel value used for density accumulation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(T radius, T cell_size) noexcept;

    /**
     * @brief Evaluate the pressure-force kernel gradient.
     *
     * This function computes the gradient of the Wendland quintic kernel and is
     * typically used in pressure-force calculations:
     *
     * \f[
     * \mathbf{f}^{pressure}_i \propto -\sum_j m_j
     * \left(\frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2}\right)
     * \nabla W(\mathbf{r}_{ij}, h)
     * \f]
     *
     * The Wendland kernel's smooth gradient helps:
     * - reduce noise in pressure forces
     * - improve stability in highly dynamic flows
     *
     * @param delta Relative displacement vector between two particles.
     * @param radius Magnitude of the displacement vector.
     * @param cell_size SPH cell size.
     * @return Gradient of the kernel evaluated at the given position.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T cell_size) noexcept;

    /**
     * @brief Evaluate the viscosity Laplacian of the Wendland quintic kernel.
     *
     * This function returns the scalar Laplacian term used in viscosity-force
     * computations:
     *
     * \f[
     * \mathbf{f}^{viscosity}_i \propto
     * \sum_j m_j \frac{\mu_j}{\rho_j}
     * (\mathbf{v}_j - \mathbf{v}_i)\nabla^2 W(r_{ij}, h)
     * \f]
     *
     * The smoothness of the Wendland kernel improves numerical stability for
     * diffusion-like terms such as viscosity.
     *
     * @param radius Distance between two particles.
     * @param cell_size SPH cell size.
     * @return Scalar Laplacian value of the kernel.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(T radius, T cell_size) noexcept;
};

} // namespace atlas::system

#include <atlas/solver/sph/wendland_quintic_sph_kernel.hpp>