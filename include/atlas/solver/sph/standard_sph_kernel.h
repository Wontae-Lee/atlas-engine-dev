#pragma once

/**
 * @file standard_sph_kernel.h
 * @brief Declares the standard smoothing kernel used by SPH solvers.
 *
 * This header defines `StandardSphKernel<T>`, a stateless collection of kernel
 * functions used by Smoothed Particle Hydrodynamics (SPH) solvers.
 *
 * The standard SPH kernel interface is split into three physically meaningful
 * operations:
 * - `density_weight(...)`
 *   Returns the scalar kernel weight used for density accumulation
 * - `pressure_gradient(...)`
 *   Returns the gradient of the kernel used in pressure-force evaluation
 * - `viscosity_laplacian(...)`
 *   Returns the scalar Laplacian used in viscosity-force evaluation
 *
 * This kernel type is designed to be:
 * - lightweight
 * - stateless
 * - usable in both host and device code
 * - callable repeatedly inside dense particle-neighborhood loops
 *
 * The exact analytic form is defined in the accompanying implementation file.
 */

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas::system {

/**
 * @brief Standard SPH smoothing-kernel utilities.
 *
 * `StandardSphKernel<T>` provides the kernel-side building blocks typically
 * needed by an SPH solver. It does not store any runtime state; all outputs
 * depend only on:
 * - the inter-particle separation
 * - the cell size
 *
 * The kernel is typically assumed to have compact support, which means its
 * contribution becomes zero outside the support radius implied by the smoothing
 * length.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct StandardSphKernel final {
    /**
     * @brief Evaluate the scalar density weight of the standard SPH kernel.
     *
     * This function returns the scalar smoothing-kernel value used during density
     * accumulation, typically in expressions such as:
     *
     * \f[
     * \rho_i = \sum_j m_j W(r_{ij}, h)
     * \f]
     *
     * where:
     * - \f$ r_{ij} \f$ is the distance between particles
     * - \f$ h \f$ is the cell size
     * - \f$ W \f$ is the smoothing-kernel value
     *
     * The returned value is:
     * - stateless
     * - side-effect free
     * - suitable for repeated evaluation in neighbor loops
     *
     * @param radius Distance between two particles.
     * @param cell_size SPH cell size.
     * @return Scalar kernel weight used for density estimation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    density_weight(T radius, T cell_size) noexcept;

    /**
     * @brief Evaluate the pressure-force kernel gradient.
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
     * The input `delta` typically represents the relative displacement:
     *
     * \f[
     * \mathbf{r}_{ij} = \mathbf{x}_i - \mathbf{x}_j
     * \f]
     *
     * and `radius` is usually the magnitude of `delta`.
     *
     * @param delta Relative displacement vector between two particles.
     * @param radius Magnitude of the relative displacement.
     * @param cell_size SPH cell size.
     * @return Kernel gradient vector used in pressure-force accumulation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<T>
    pressure_gradient(const Vector3<T>& delta,
                      T radius,
                      T cell_size) noexcept;

    /**
     * @brief Evaluate the viscosity Laplacian of the standard SPH kernel.
     *
     * This function returns the scalar Laplacian of the kernel, typically used in
     * viscosity-force terms of the form:
     *
     * \f[
     * \mathbf{f}^{viscosity}_i \propto
     * \sum_j m_j \frac{\mu_j}{\rho_j}
     * (\mathbf{v}_j - \mathbf{v}_i)\nabla^2 W(r_{ij}, h)
     * \f]
     *
     * The exact SPH viscosity formulation may vary by solver, but this function
     * provides the kernel-side Laplacian quantity needed for diffusion-like
     * viscosity models.
     *
     * @param radius Distance between two particles.
     * @param cell_size SPH cell size.
     * @return Scalar kernel Laplacian used in viscosity-force evaluation.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    viscosity_laplacian(T radius, T cell_size) noexcept;
};

} // namespace atlas::system

#include <atlas/solver/sph/standard_sph_kernel.hpp>