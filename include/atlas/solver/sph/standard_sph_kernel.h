#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

/**
 * @file standard_sph_kernel.h
 * @brief The original Müller/Charypar/Gross (2003) real-time SPH kernel
 *        trio: a different, purpose-shaped kernel for each of density,
 *        pressure gradient, and viscosity Laplacian.
 *
 * @details
 * ### Background
 * A single smoothing kernel differentiated three ways (for `W`, `grad
 * W`, `laplacian W`) has known practical failure modes in SPH: the
 * common "Poly6" density kernel's gradient vanishes at `r = 0`, so a
 * pressure force built from its own gradient fails to repel particles
 * that have clumped together (attractive numerical clustering); a
 * naive gradient-based viscosity Laplacian can go negative and inject
 * energy instead of damping it. Müller, Charypar, and Gross's
 * influential "Particle-Based Fluid Simulation for Interactive
 * Applications" (2003) sidesteps both problems by using *three
 * different, independently designed* kernels — the actual smoothness
 * (`W`) needed only for density, a kernel with a well-behaved,
 * non-vanishing gradient for pressure ("Spiky"), and one with a
 * simple, always-non-negative Laplacian for viscosity — rather than
 * deriving all three from one kernel's calculus. See
 * `CubicSplineSphKernel`/`WendlandQuinticSphKernel` for the alternative
 * (single self-consistent kernel) approach.
 *
 * ### Derivation — the three kernels (`h` = `cell_size`, `q = r/h`)
 * - `density_weight` ("Poly6"): `W(r,h) = (315 / (64*pi*h^9)) *
 *   (h^2 - r^2)^3` for `0 <= r <= h`. Smooth (`C2`) and cheap (no
 *   square root needed if callers already have `r^2`, though this
 *   overload takes `r` directly), but its radial derivative
 *   `dW/dr = -(945/(32*pi*h^9)) * r * (h^2-r^2)^2` vanishes at `r=0` —
 *   exactly the clustering problem "Spiky" avoids.
 * - `pressure_gradient` ("Spiky" gradient): built from
 *   `W_spiky(r,h) = (15/(pi*h^6)) * (h-r)^3`, whose gradient
 *   `grad(W_spiky) = -(45/(pi*h^6)) * (h-r)^2 * (delta/r)` (`delta` the
 *   vector from neighbor to particle, `delta/r` the unit direction)
 *   stays strictly negative (repulsive) for all `0 < r < h`, including
 *   as `r -> 0` — the property that prevents particle clustering under
 *   pressure forces.
 * - `viscosity_laplacian`: built from a third kernel
 *   `W_visc(r,h) = (15/(2*pi*h^3)) * (-r^3/(2h^3) + r^2/h^2 + h/(2r) - 1)`,
 *   chosen specifically because its Laplacian
 *   `laplacian(W_visc) = (45/(pi*h^6)) * (h - r)` is *linear and always
 *   non-negative* on `[0,h]` — guaranteeing the viscosity force always
 *   damps (never amplifies) relative motion between neighbors,
 *   regardless of their exact separation.
 *
 * ### References
 * - M. Müller, D. Charypar, and M. Gross, "Particle-Based Fluid
 *   Simulation for Interactive Applications," ACM
 *   SIGGRAPH/Eurographics Symposium on Computer Animation, 2003.
 *   (defines exactly this Poly6/Spiky/viscosity kernel trio)
 */

namespace atlas {

/**
 * @brief Müller et al. (2003) Poly6 (density) / Spiky (pressure
 *        gradient) / linear-Laplacian (viscosity) kernel trio. See this
 *        file's top-of-file documentation for each formula's derivation
 *        and why three different kernels are used.
 */
struct StandardSphKernel final {

    /** @brief Poly6 density weight `(315/(64*pi*h^9)) * (h^2-r^2)^3`;
     *  `0` outside `[0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    density_weight(float radius, float cell_size) noexcept;

    /** @brief Spiky-kernel gradient, scaled along `delta`; chosen for
     *  its non-vanishing repulsive gradient as `radius -> 0` (see this
     *  file's top-of-file documentation). `0` outside `(0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    pressure_gradient(const Float3& delta,
                      float radius,
                      float cell_size) noexcept;

    /** @brief Linear, always-non-negative viscosity Laplacian
     *  `(45/(pi*h^6)) * (h-r)`; `0` outside `[0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    viscosity_laplacian(float radius, float cell_size) noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
StandardSphKernel::density_weight(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float cell_size_squared = cell_size * cell_size;

    const float support = cell_size_squared - radius * radius;

    const float coeff = static_cast<float>(315.0 / (64.0 * atlas::pi));

    return coeff * support * support * support
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
StandardSphKernel::pressure_gradient(const Float3& delta,
                                     const float radius,
                                     const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius > 0.0f) || radius > cell_size) {
        return Float3(0.0f, 0.0f, 0.0f);
    }

    const float coeff = static_cast<float>(-45.0 / atlas::pi);

    const float support = (cell_size - radius) * (cell_size - radius);

    return delta * (coeff * support / (cell_size * cell_size * cell_size * cell_size * cell_size * cell_size * radius));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
StandardSphKernel::viscosity_laplacian(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float coeff = static_cast<float>(45.0 / atlas::pi);

    return coeff * (cell_size - radius)
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

}
