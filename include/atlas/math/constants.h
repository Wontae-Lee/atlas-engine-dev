#pragma once
#include <atlas/core/macros.h>

#include <cmath>
#include <limits>

namespace atlas {

/**
 * @brief The mathematical constant pi in single precision.
 *
 * Truncated to float precision from the full double-precision literal; used
 * across the engine for angular conversions and solid-angle sampling. Declared
 * `inline constexpr` so it is a single compile-time value shared by every
 * translation unit that includes this header, usable on host and device.
 */
inline constexpr float pi = 3.14159265358979323846f;

/**
 * @brief The square root of two in single precision.
 *
 * Truncated to float precision. Provided as a compile-time constant so the most
 * thermal-velocity math (which scales by sqrt(2)) avoids a runtime sqrt.
 */
inline constexpr float SQRT_TWO = 1.41421356237309504880f;

/**
 * @brief Boltzmann constant in SI units (joules per kelvin).
 *
 * The exact 2019-redefinition value, narrowed to float. Relates gas temperature
 * to per-molecule kinetic energy in the DSMC collision and sampling kernels.
 */
inline constexpr float boltzmann_constant = 1.380649e-23f;

/**
 * @brief Standard gravitational acceleration at Earth's surface (m/s^2).
 *
 * The conventional standard value. Applied as a uniform body acceleration where
 * gravity is enabled.
 */
inline constexpr float gravity = 9.80665f;

/**
 * @brief Small positive epsilon used as an equality/near-zero threshold.
 *
 * Chosen for single-precision geometry at engine scale: two quantities within
 * this distance are treated as equal (see Quaternion::operator== and
 * is_identity). Not tied to the machine epsilon of float.
 */
inline constexpr float eps = 1e-6f;

/**
 * @brief General-purpose tolerance for iterative or geometric convergence.
 *
 * Numerically equal to @ref eps but named separately so call sites can express
 * intent (a convergence tolerance rather than an equality epsilon) without
 * coupling the two if one later needs tuning.
 */
inline constexpr float tol = 1e-6f;

/**
 * @brief A very large finite float used as a stand-in for "effectively infinite".
 *
 * Preferred over @ref inf where a finite sentinel is required so that
 * subsequent arithmetic (differences, comparisons) stays finite and does not
 * produce NaN. Used for far-plane / unbounded-ray distances.
 *
 * @warning The identifier `far` collides with a legacy keyword in some Windows
 * headers; this is the existing project name and is only noted here.
 */
inline constexpr float far = 1e30f;

/**
 * @brief Positive floating-point infinity.
 *
 * The IEEE-754 infinity for float. Returned by solve_quadratic() for roots that
 * lie at infinity (a degenerate leading coefficient), and used wherever a true
 * unbounded value is intended rather than the finite @ref far sentinel.
 */
inline constexpr float inf = std::numeric_limits<float>::infinity();

/**
 * @brief Host/device wrapper around std::isfinite for a single float.
 *
 * @param value The value to test.
 * @return True when @p value is neither infinite nor NaN.
 * @note Provided so device code has a name-resolvable finiteness check that
 * matches the host result exactly.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const float value) noexcept {
    return std::isfinite(value);
}

/**
 * @brief Square root that clamps non-positive inputs to zero instead of NaN.
 *
 * A numerical guard: callers pass quantities that are mathematically
 * non-negative but may dip slightly below zero from rounding (for example
 * 1 - cos^2). Feeding such a value to std::sqrt would yield NaN; this returns
 * 0 instead.
 *
 * @param value The radicand.
 * @return sqrt(value) when value > 0, otherwise 0.0f. Note that exactly-zero
 * input also returns 0.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sqrt_nonnegative(const float value) noexcept {
    return value > 0.0f ? std::sqrt(value) : 0.0f;
}

/**
 * @brief Solve the real quadratic a*t^2 + b*t + c = 0 with a stable formula.
 *
 * Uses the sign-aware form q = -0.5*(b + sign(b)*sqrt(disc)) to avoid the
 * catastrophic cancellation of the naive quadratic formula, then recovers the
 * two roots as c/q and q/a. The roots are returned sorted so @p t0 <= @p t1.
 *
 * Degenerate coefficients are handled explicitly: when q == 0 the root c/q is
 * reported as @ref inf, and when a == 0 (no quadratic term) the root q/a is
 * reported as @ref inf. This keeps the function total rather than dividing by
 * zero.
 *
 * @param a Coefficient of t^2.
 * @param b Coefficient of t.
 * @param c Constant term.
 * @param t0 Out: the smaller real root (or @ref inf for a degenerate branch).
 * @param t1 Out: the larger real root (or @ref inf for a degenerate branch).
 * @return True when the discriminant is non-negative and real roots exist;
 * false when the discriminant is negative, in which case @p t0 and @p t1 are
 * left unmodified.
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
solve_quadratic(const float a,
                const float b,
                const float c,
                float& t0,
                float& t1) noexcept {
    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = sqrt_nonnegative(discriminant);
    // Add sqrt with the sign of b so the two terms never subtract: this is the
    // numerically stable (Citardauq) form that dodges cancellation when b is
    // large relative to sqrt(discriminant).
    const float sign_b            = (b >= 0.0f) ? 1.0f : -1.0f;
    const float q                 = -0.5f * (b + sign_b * sqrt_discriminant);

    // Recover roots from the product/quotient identities; guard each divisor so
    // a degenerate q or leading coefficient yields a well-defined infinity.
    t0 = (q == 0.0f) ? inf : (c / q);
    t1 = (a == 0.0f) ? inf : (q / a);

    if (t0 > t1) {
        const float tmp = t0;
        t0              = t1;
        t1              = tmp;
    }

    return true;
}

}