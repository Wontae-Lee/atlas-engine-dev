#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

/**
 * @file wendland_quintic_sph_kernel.h
 * @brief Wendland (1995) C2 quintic SPH kernel: a single self-consistent
 *        kernel (same design philosophy as `CubicSplineSphKernel`) with
 *        markedly better resistance to the SPH "pairing instability"
 *        than the cubic spline.
 *
 * @details
 * ### Background
 * The cubic B-spline's second derivative changes sign within its
 * support, which lets particle pairs settle into an unphysically close,
 * clumped equilibrium separation (the "pairing" or "tensile"
 * instability) under certain conditions. Wendland's compactly-supported
 * radial basis functions are constructed to be positive-definite with a
 * strictly monotonic derivative profile, which Dehnen and Aly (2012)
 * showed empirically suppresses pairing far more effectively than the
 * cubic spline, at the cost of needing more neighbors for the same
 * accuracy (a wider, smoother kernel needs a larger neighbor count to
 * resolve). It is now a common default in modern SPH codes for this
 * reason.
 *
 * ### Derivation (`h` = `cell_size`, `q = r/h`) — the 3D Wendland C2 kernel
 * `alpha = 21/(2*pi*h^3)`, `W(q) = alpha * (1-q)^4 * (1+4q)` for
 * `0 <= q <= 1` — this is exactly the standard 3D Wendland C2 radial
 * function (degree-4 polynomial in `(1-q)`, `C2` continuous, i.e. its
 * value, first, and second derivatives all vanish smoothly at the
 * support boundary `q=1`, which is what gives it its improved stability
 * properties over the cubic spline's only-`C1` boundary behavior).
 * `pressure_gradient`/`viscosity_laplacian` are, as in
 * `CubicSplineSphKernel`, the exact analytic first/second radial
 * derivatives of this same `W`, not separately substituted kernels.
 *
 * ### References
 * - H. Wendland, "Piecewise polynomial, positive definite and
 *   compactly supported radial functions of minimal degree," Advances
 *   in Computational Mathematics, 4(1), 1995, pp. 389-396. (the
 *   Wendland radial basis function family)
 * - W. Dehnen and H. Aly, "Improving convergence in smoothed particle
 *   hydrodynamics simulations without pairing instability," Monthly
 *   Notices of the Royal Astronomical Society, 425(2), 2012,
 *   pp. 1068-1082. (establishes Wendland kernels' pairing-instability
 *   resistance for SPH)
 */

namespace atlas {

/**
 * @brief Wendland (1995) C2 quintic kernel, single self-consistent
 *        kernel with strong pairing-instability resistance (Dehnen &
 *        Aly 2012). See this file's top-of-file documentation for the
 *        formula and why it improves on the cubic spline.
 */
struct WendlandQuinticSphKernel final {

    /** @brief Wendland C2 density weight `alpha*(1-q)^4*(1+4q)`,
     *  `alpha = 21/(2*pi*h^3)`. `0` outside `[0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    density_weight(float radius, float cell_size) noexcept;

    /** @brief Exact analytic gradient of `density_weight`, scaled along
     *  `delta`. `0` outside `(0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    pressure_gradient(const Float3& delta,
                      float radius,
                      float cell_size) noexcept;

    /** @brief Exact analytic (radially-symmetric) Laplacian of
     *  `density_weight`. `0` outside `[0, cell_size]`. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    viscosity_laplacian(float radius, float cell_size) noexcept;
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
WendlandQuinticSphKernel::density_weight(const float radius,
                                         const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float q           = radius / cell_size;
    const float one_minus_q = 1.0f - q;

    const float alpha = static_cast<float>(21.0 / (2.0 * atlas::pi))
        / (cell_size * cell_size * cell_size);

    return alpha
        * one_minus_q * one_minus_q * one_minus_q * one_minus_q
        * (1.0f + 4.0f * q);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
WendlandQuinticSphKernel::pressure_gradient(const Float3& delta,
                                            const float radius,
                                            const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius > 0.0f) || radius > cell_size) {
        return Float3(0.0f, 0.0f, 0.0f);
    }

    const float q           = radius / cell_size;
    const float one_minus_q = 1.0f - q;

    const float alpha = static_cast<float>(-210.0 / atlas::pi)
        / (cell_size * cell_size
           * cell_size * cell_size);

    const float radial_derivative = alpha * q * one_minus_q * one_minus_q * one_minus_q;

    return delta * (radial_derivative / radius);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
WendlandQuinticSphKernel::viscosity_laplacian(const float radius,
                                              const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float q           = radius / cell_size;
    const float one_minus_q = 1.0f - q;

    const float alpha = static_cast<float>(210.0 / atlas::pi)
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size);

    return alpha
        * one_minus_q * one_minus_q
        * (1.0f - 4.0f * q);
}

}
