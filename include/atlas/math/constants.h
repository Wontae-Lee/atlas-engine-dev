/**
 * @file   constants.h
 * @brief  Common numerical constants and small scalar math helpers shared across the math module.
 *
 * Free-standing values and functions used by Float3, Float3x3, Quaternion,
 * and other math/geometry code. All functions explicitly declare their
 * compilation target (host/device) via ATLAS_ALL_DEVICE, and force
 * inlining via ATLAS_FORCE_INLINE to eliminate call overhead.
 */

#pragma once
#include <atlas/core/macros.h>

#include <cmath>
#include <limits>

namespace atlas {

/** @brief The mathematical constant pi. */
inline constexpr float pi = 3.14159265358979323846f;

/** @brief The square root of 2. */
inline constexpr float SQRT_TWO = 1.41421356237309504880f;

/** @brief The Boltzmann constant, in J/K. */
inline constexpr float boltzmann_constant = 1.380649e-23f;

/** @brief Standard gravitational acceleration, in m/s^2. */
inline constexpr float gravity = 9.80665f;

/** @brief General-purpose small epsilon for approximate comparisons. */
inline constexpr float eps = 1e-6f;

/** @brief General-purpose small tolerance for convergence/degeneracy checks. */
inline constexpr float tol = 1e-6f;

/** @brief A large finite distance used as a practical "far" sentinel (e.g. ray tmax). */
inline constexpr float far = 1e30f;

/** @brief Positive floating-point infinity. */
inline constexpr float inf = std::numeric_limits<float>::infinity();

/**
 * @brief Checks whether a scalar value is finite (excludes inf, NaN).
 * @param value  Value to check
 * @return       true if value passes std::isfinite
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const float value) noexcept {
    return std::isfinite(value);
}

/**
 * @brief Square root that is safe against small negative inputs.
 *
 * Returns 0 instead of NaN when value <= 0, which is convenient after
 * subtractions that should be non-negative but may dip slightly below
 * zero due to floating-point error.
 *
 * @param value  Input value
 * @return       sqrt(value) if value > 0, otherwise 0
 */
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sqrt_nonnegative(const float value) noexcept {
    return value > 0.0f ? std::sqrt(value) : 0.0f;
}

/**
 * @brief Solves the quadratic equation a*t^2 + b*t + c = 0 for real roots.
 *
 * Uses the numerically stable form that avoids catastrophic cancellation
 * (computing q from the sign of b, then deriving both roots from q),
 * and always returns t0 <= t1.
 *
 * @param[in]  a   Quadratic coefficient
 * @param[in]  b   Linear coefficient
 * @param[in]  c   Constant coefficient
 * @param[out] t0  Smaller root (inf if q == 0)
 * @param[out] t1  Larger root (inf if a == 0)
 * @return     true if real roots exist (discriminant >= 0), false otherwise
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
    const float sign_b            = (b >= 0.0f) ? 1.0f : -1.0f;
    const float q                 = -0.5f * (b + sign_b * sqrt_discriminant);

    t0 = (q == 0.0f) ? inf : (c / q);
    t1 = (a == 0.0f) ? inf : (q / a);

    if (t0 > t1) {
        const float tmp = t0;
        t0              = t1;
        t1              = tmp;
    }

    return true;
}

} // namespace atlas
