#pragma once
#include <atlas/core/macros.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace atlas {

constexpr double k_epsilon_d = 1e-12;

constexpr float k_epsilon_f = 1e-6f;

constexpr double k_farthest_d = 1e30;

constexpr float k_farthest_f = 1e30f;

constexpr double pi = M_PI;

constexpr double SQRT_TWO = 1.4142135623730950488;

constexpr double boltzmann_constant = 1.380649e-23;

constexpr double eps = k_epsilon_d;

constexpr double far = k_farthest_d;

constexpr double inf = std::numeric_limits<double>::infinity();

constexpr double tol = 1e-6;

constexpr double gravity = 9.80665;

template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<T>, T>
abs(const T value) noexcept {
    return value < T(0) ? -value : value;
}

template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<T>, bool>
isfinite(const T value) noexcept {
    return std::isfinite(static_cast<double>(value));
}

template <typename T>
ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
sqrt_nonnegative(const T value) noexcept {
    using std::sqrt;
    return value > T(0) ? static_cast<T>(sqrt(value)) : T(0);
}

}