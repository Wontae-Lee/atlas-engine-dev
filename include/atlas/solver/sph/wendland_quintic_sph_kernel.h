#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas {

struct WendlandQuinticSphKernel final {

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
