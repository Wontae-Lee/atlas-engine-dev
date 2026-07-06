#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/solver/sph/cubic_spline_sph_kernel.h>
#include <atlas/solver/sph/standard_sph_kernel.h>
#include <atlas/solver/sph/wendland_quintic_sph_kernel.h>

#include <type_traits>

namespace atlas {

enum struct SphKernelType : int {

    standard,

    cubic_spline,

    wendland_quintic
};

struct SphKernel final {

    SphKernelType type = SphKernelType::standard;

    union {

        StandardSphKernel standard;

        CubicSplineSphKernel cubic_spline;

        WendlandQuinticSphKernel wendland_quintic;
    };

    ATLAS_ALL_DEVICE
    SphKernel() noexcept;

    ATLAS_ALL_DEVICE explicit SphKernel(SphKernelType type) noexcept;

    ATLAS_ALL_DEVICE
    SphKernel(const SphKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE SphKernel&
    operator=(const SphKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE ~SphKernel() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit SphKernel(const Payload& op);

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    density_weight(SphKernelType type,
                   float radius,
                   float cell_size) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Float3
    pressure_gradient(SphKernelType type,
                      const Float3& delta,
                      float radius,
                      float cell_size) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    viscosity_laplacian(SphKernelType type,
                        float radius,
                        float cell_size) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    density_weight(float radius, float cell_size) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    pressure_gradient(const Float3& delta,
                      float radius,
                      float cell_size) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    viscosity_laplacian(float radius, float cell_size) const noexcept;
};

using SphKernelVariant = DeviceVariant<
    SphKernel,
    SphKernelType,
    SphKernelType::standard,
    DeviceVariantCase<SphKernelType::standard, &SphKernel::standard>,
    DeviceVariantCase<SphKernelType::cubic_spline, &SphKernel::cubic_spline>,
    DeviceVariantCase<SphKernelType::wendland_quintic, &SphKernel::wendland_quintic>>;

struct SphDensityWeight {
    float radius;
    float cell_size;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(Tag) const noexcept {
        using Kernel = typename Tag::type;
        return Kernel::density_weight(radius, cell_size);
    }
};
struct SphPressureGradient {
    const Float3& delta;
    float radius;
    float cell_size;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(Tag) const noexcept {
        using Kernel = typename Tag::type;
        return Kernel::pressure_gradient(delta, radius, cell_size);
    }
};
struct SphViscosityLaplacian {
    float radius;
    float cell_size;
    template <typename Tag>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(Tag) const noexcept {
        using Kernel = typename Tag::type;
        return Kernel::viscosity_laplacian(radius, cell_size);
    }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel() noexcept {
    SphKernelVariant::construct(*this, SphKernelType::standard);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel(const SphKernelType type) noexcept {
    SphKernelVariant::construct(*this, type);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SphKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SphKernel::SphKernel(const Payload& op) {
    SphKernelVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::density_weight(const SphKernelType type,
                          const float radius,
                          const float cell_size) noexcept {
    return SphKernelVariant::visit_type(
        type,
        SphDensityWeight { radius, cell_size },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
SphKernel::pressure_gradient(const SphKernelType type,
                             const Float3& delta,
                             const float radius,
                             const float cell_size) noexcept {
    return SphKernelVariant::visit_type(
        type,
        SphPressureGradient { delta, radius, cell_size },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::viscosity_laplacian(const SphKernelType type,
                               const float radius,
                               const float cell_size) noexcept {
    return SphKernelVariant::visit_type(
        type,
        SphViscosityLaplacian { radius, cell_size },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::density_weight(const float radius, const float cell_size) const noexcept {
    return density_weight(type, radius, cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
SphKernel::pressure_gradient(const Float3& delta,
                             const float radius,
                             const float cell_size) const noexcept {
    return pressure_gradient(type, delta, radius, cell_size);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
SphKernel::viscosity_laplacian(const float radius, const float cell_size) const noexcept {
    return viscosity_laplacian(type, radius, cell_size);
}

}
