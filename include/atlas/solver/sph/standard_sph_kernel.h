#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>

namespace atlas {

struct StandardSphKernel final {

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
StandardSphKernel::density_weight(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float cell_size_squared = cell_size * cell_size;

    const float support = cell_size_squared - radius * radius;

    const float coeff = static_cast<float>(315.0 / (64.0 * atlas::pi));

    return coeff * support * support * support
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
StandardSphKernel::pressure_gradient(const Float3& delta,
                                     const float radius,
                                     const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius > 0.0f) || radius > cell_size) {
        return Float3(0.0f, 0.0f, 0.0f);
    }

    const float coeff = static_cast<float>(-45.0 / atlas::pi);

    const float support = (cell_size - radius) * (cell_size - radius);

    return delta * (coeff * support / (cell_size * cell_size * cell_size * cell_size * cell_size * cell_size * radius));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
StandardSphKernel::viscosity_laplacian(const float radius, const float cell_size) noexcept {

    if (!(cell_size > 0.0f) || !(radius >= 0.0f) || radius > cell_size) {
        return 0.0f;
    }

    const float coeff = static_cast<float>(45.0 / atlas::pi);

    return coeff * (cell_size - radius)
        / (cell_size * cell_size * cell_size
           * cell_size * cell_size * cell_size);
}

}
