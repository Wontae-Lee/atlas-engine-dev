#pragma once
#include <atlas/core/macros.h>

#include <cmath>
#include <limits>

namespace atlas {

inline constexpr float pi = 3.14159265358979323846f;

inline constexpr float SQRT_TWO = 1.41421356237309504880f;

inline constexpr float boltzmann_constant = 1.380649e-23f;

inline constexpr float gravity = 9.80665f;

inline constexpr float eps = 1e-6f;

inline constexpr float tol = 1e-6f;

inline constexpr float far = 1e30f;

inline constexpr float inf = std::numeric_limits<float>::infinity();

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
isfinite(const float value) noexcept {
    return std::isfinite(value);
}

ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
sqrt_nonnegative(const float value) noexcept {
    return value > 0.0f ? std::sqrt(value) : 0.0f;
}

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

}
