#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas {

struct CubicSplineSphKernel final {

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    density_weight(float radius, float cell_size) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    pressure_gradient(const Float3& delta,
                      float radius,
                      float cell_size) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
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

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
CubicSplineSphKernel::pressure_gradient(const Float3& delta,
                                        const float radius,
                                        const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius > 0.0f) || radius > cell_size) {
        return Float3(0.0f, 0.0f, 0.0f);
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
