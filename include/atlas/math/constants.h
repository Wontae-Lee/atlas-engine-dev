#pragma once
#include <atlas/core/macros.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace atlas {

// ------------------------------------------------------------
// Numerical constants (math)
// ------------------------------------------------------------
    // Double-precision epsilon used as a small tolerance in comparisons.
    // Typical use: avoid treating near-zero as non-zero in robust predicates.
    constexpr double k_epsilon_d = 1e-12;

    // Single-precision epsilon used as a small tolerance in comparisons.
    // Typical use: intersection tests, normalization guards, etc.
    constexpr float k_epsilon_f = 1e-6f;

    // "Very far" sentinel distance in double precision.
    // Used as a large finite placeholder instead of infinity when algorithms
    // prefer finite arithmetic (e.g., iterative minimization / BVH traversal).
    constexpr double k_farthest_d = 1e30;

    // "Very far" sentinel distance in single precision.
    constexpr float k_farthest_f = 1e30f;

// ------------------------------------------------------------
// Global convenience aliases
// ------------------------------------------------------------

// Mathematical constant π.
// Uses the C math macro M_PI (platform/defines dependent).
constexpr double pi = M_PI;

// Square root of two.
constexpr double SQRT_TWO = 1.4142135623730950488;

// Boltzmann constant in SI units (J/K = kg m^2 s^-2 K^-1).
// Useful for Maxwell-Boltzmann thermal velocity and kinetic theory formulas.
constexpr double boltzmann_constant = 1.380649e-23;

// Default floating epsilon used across the codebase (single-precision).
// Handy for "nearly zero" checks without spelling out the namespace.
constexpr double eps = k_epsilon_d;

// Default "far distance" sentinel (single-precision).
// Often used as an initial "best t" for ray hits, etc.
constexpr double far = k_farthest_d;

// IEEE +infinity for float (useful as an unbounded sentinel).
constexpr double inf = std::numeric_limits<double>::infinity();

// Generic double-precision tolerance used for tighter comparisons.
// Often used in geometry_operator predicates where float epsilon is too loose.
constexpr double tol = 1e-6;

constexpr double gravity = 9.80665;

// ------------------------------------------------------------
// Scalar helpers (math)
// ------------------------------------------------------------

/**
 * @brief Returns the absolute value of a scalar.
 */
template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<T>, T>
abs(const T value) noexcept {
    return value < T(0) ? -value : value;
}

/**
 * @brief Returns true when a scalar is finite.
 */
template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<T>, bool>
isfinite(const T value) noexcept {
    return std::isfinite(static_cast<double>(value));
}

/**
 * @brief Returns sqrt(value) for positive values and zero otherwise.
 *
 * Useful for numerically guarded formulas where small negative roundoff
 * should collapse to zero before the square root.
 */
template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sqrt_nonnegative(const T value) noexcept {
    using std::sqrt;
    return value > T(0) ? static_cast<T>(sqrt(value)) : T(0);
}
} // namespace atlas
