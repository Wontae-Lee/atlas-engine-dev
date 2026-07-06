#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

/**
 * @file cubic_spline_sph_kernel.h
 * @brief Monaghan (1992) cubic B-spline ("M4") SPH kernel: a single,
 *        self-consistent kernel whose own analytic gradient and
 *        Laplacian are used directly (unlike `StandardSphKernel`'s
 *        three independently-designed kernels).
 *
 * @details
 * ### Background
 * The cubic B-spline is the most widely used SPH kernel in the
 * "traditional"/astrophysical SPH literature (Monaghan 1992):
 * `C2`-smooth, compactly supported, and — unlike `StandardSphKernel`'s
 * Poly6 — has a well-behaved (non-vanishing, sign-correct) gradient, so
 * the *same* kernel can consistently be differentiated for both the
 * pressure-gradient and viscosity-Laplacian operators without the
 * clustering/negative-viscosity problems that motivated Müller et al.'s
 * three-kernel split. This implementation reparametrizes Monaghan's
 * original piecewise definition (breakpoints at `q = r/h = 1` and `2`,
 * support `2h`) so the kernel's *entire* compact support is exactly
 * `[0, cell_size]` — i.e. `q = r / cell_size` here — with the shape
 * breakpoint at `q = 0.5` instead of `q = 1`; substituting
 * `q_here = q_Monaghan / 2` into Monaghan's piecewise cubic recovers
 * exactly the formulas below, so this is the same M4 spline, just with
 * `cell_size` playing the role of Monaghan's `2h`.
 *
 * ### Derivation (`h` = `cell_size`, `q = r/h`)
 * - `density_weight`: `alpha = 1/(pi*h^3)`, and
 *   `W(q) = alpha * (6q^3 - 6q^2 + 1)` for `q < 0.5`,
 *   `W(q) = alpha * 2*(1-q)^3` for `0.5 <= q <= 1`. Continuous and `C1`
 *   at `q = 0.5` (both branches and their first derivatives agree
 *   there, by construction of the cubic spline).
 * - `pressure_gradient`: the exact radial derivative `dW/dq * (1/h)` of
 *   the density kernel above (chain rule through `q = r/h`), scaled
 *   along `delta/radius` — not a separately designed kernel.
 * - `viscosity_laplacian`: the exact second radial derivative of the
 *   same kernel (in a radially-symmetric 3D Laplacian form), again
 *   derived analytically rather than substituted.
 *
 * ### References
 * - J. J. Monaghan, "Smoothed particle hydrodynamics," Annual Review of
 *   Astronomy and Astrophysics, 30, 1992, pp. 543-574. (the cubic
 *   B-spline / M4 kernel)
 */

namespace atlas {

/**
 * @brief Monaghan (1992) cubic B-spline (M4) kernel, single
 *        self-consistent kernel for density/pressure/viscosity. See
 *        this file's top-of-file documentation for the formula and its
 *        relation to Monaghan's original parametrization.
 */
struct CubicSplineSphKernel final {

    /** @brief Cubic B-spline density weight; see this file's
     *  top-of-file Derivation. `0` outside `[0, cell_size]`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    density_weight(float radius, float cell_size) noexcept;

    /** @brief Exact analytic gradient of `density_weight`, scaled along
     *  `delta`. `0` outside `(0, cell_size]`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3
    pressure_gradient(const Vector3& delta,
                      float radius,
                      float cell_size) noexcept;

    /** @brief Exact analytic (radially-symmetric) Laplacian of
     *  `density_weight`. `0` outside `[0, cell_size]`. */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static float
    viscosity_laplacian(float radius, float cell_size) noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
CubicSplineSphKernel::density_weight(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float q = radius / cell_size;

    const float alpha = static_cast<float>(1.0 / atlas::pi)
        / (cell_size * cell_size * cell_size);

    if (q < 0.5f) {

        return alpha * (6.0f * q * q * q - 6.0f * q * q + 1.0f);
    }

    const float one_minus_q = 1.0f - q;
    return alpha * (2.0f * one_minus_q * one_minus_q * one_minus_q);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
CubicSplineSphKernel::pressure_gradient(const Vector3& delta,
                                        const float radius,
                                        const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius > 0.0f) || radius > cell_size) {
        return Vector3(0.0f, 0.0f, 0.0f);
    }

    const float q = radius / cell_size;

    const float alpha = static_cast<float>(6.0 / atlas::pi)
        / (cell_size * cell_size * cell_size * cell_size);

    float radial_derivative = 0.0f;

    if (q < 0.5f) {

        radial_derivative = alpha * (3.0f * q * q - 2.0f * q);
    } else {

        const float one_minus_q = 1.0f - q;
        radial_derivative       = -alpha * one_minus_q * one_minus_q;
    }

    return delta * (radial_derivative / radius);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
CubicSplineSphKernel::viscosity_laplacian(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float q = radius / cell_size;

    const float alpha = static_cast<float>(6.0 / atlas::pi)
        / (cell_size * cell_size * cell_size * cell_size * cell_size);

    if (q < 0.5f) {

        return alpha * (6.0f * q - 2.0f);
    }

    return alpha * (2.0f - 2.0f * q);
}

}
