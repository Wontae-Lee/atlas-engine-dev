#pragma once
#include <cmath>
#include <limits>

namespace atlas {

// ------------------------------------------------------------
// Numerical constants (math)
// ------------------------------------------------------------
namespace math {
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
}

// ------------------------------------------------------------
// Global convenience aliases
// ------------------------------------------------------------

// Mathematical constant π.
// Uses the C math macro M_PI (platform/defines dependent).
constexpr double pi = M_PI;

// Boltzmann constant in SI units (J/K = kg m^2 s^-2 K^-1).
// Useful for Maxwell-Boltzmann thermal velocity and kinetic theory formulas.
constexpr double boltzmann_constant = 1.380649e-23;

// Default floating epsilon used across the codebase (single-precision).
// Handy for "nearly zero" checks without spelling out the namespace.
constexpr double eps = math::k_epsilon_d;

// Default "far distance" sentinel (single-precision).
// Often used as an initial "best t" for ray hits, etc.
constexpr double far = math::k_farthest_d;

// IEEE +infinity for float (useful as an unbounded sentinel).
constexpr double inf = std::numeric_limits<double>::infinity();

// Generic double-precision tolerance used for tighter comparisons.
// Often used in geometry_operator predicates where float epsilon is too loose.
constexpr double tol = eps;

} // namespace atlas
