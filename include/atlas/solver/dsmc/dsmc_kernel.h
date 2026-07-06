#pragma once

#include <atlas/core/detail/device_variant.h>
#include <atlas/solver/dsmc/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <cstddef>
#include <type_traits>

namespace atlas {

enum struct DsmcKernelType : int {

    hard_sphere,

    variable_hard_sphere,

    variable_soft_sphere
};

struct DsmcPairParameters final {
    float reference_diameter {};
    float reference_temperature {};
    float viscosity_index { 0.5f };
    float scattering_parameter { 1.0f };
    float reduced_mass {};
    bool valid {};
};

struct DsmcKernel final {

    DsmcKernelType type = DsmcKernelType::hard_sphere;

    union {

        HardSphereKernel hard_sphere;

        VariableHardSphereKernel variable_hard_sphere;

        VariableSoftSphereKernel variable_soft_sphere;
    };

    ATLAS_ALL_DEVICE
    DsmcKernel() noexcept;

    ATLAS_ALL_DEVICE explicit DsmcKernel(DsmcKernelType type) noexcept;

    ATLAS_ALL_DEVICE
    DsmcKernel(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE ~DsmcKernel() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, DsmcKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit DsmcKernel(const Payload& op);

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static float
    cross_section(DsmcKernelType type,
                  const MaterialProperties& lhs,
                  const MaterialProperties& rhs,
                  float relative_speed) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static DsmcPairParameters
    pair_parameters(const MaterialProperties& lhs,
                    const MaterialProperties& rhs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const MaterialProperties& lhs,
               const MaterialProperties& rhs) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sigma_g(const MaterialProperties* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            float relative_speed_squared) const noexcept;
};

namespace detail {

    using DsmcKernelVariant = DeviceVariant<
        DsmcKernel,
        DsmcKernelType,
        DsmcKernelType::hard_sphere,
        DeviceVariantCase<DsmcKernelType::hard_sphere, &DsmcKernel::hard_sphere>,
        DeviceVariantCase<DsmcKernelType::variable_hard_sphere, &DsmcKernel::variable_hard_sphere>,
        DeviceVariantCase<DsmcKernelType::variable_soft_sphere, &DsmcKernel::variable_soft_sphere>>;

    struct DsmcCrossSection {
        const MaterialProperties& lhs;
        const MaterialProperties& rhs;
        float relative_speed;
        template <typename Tag>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
        operator()(Tag) const noexcept {
            using Kernel = typename Tag::type;
            return Kernel::cross_section(lhs, rhs, relative_speed);
        }
    };
    struct DsmcCollide {
        Float3& lhs_velocity;
        Float3& rhs_velocity;
        const MaterialProperties& lhs;
        const MaterialProperties& rhs;
        template <typename K>
        ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
        operator()(const K& kernel) const noexcept { kernel(lhs_velocity, rhs_velocity, lhs, rhs); }
    };

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel() noexcept {
    detail::DsmcKernelVariant::construct(*this, DsmcKernelType::hard_sphere);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const DsmcKernelType type) noexcept {
    detail::DsmcKernelVariant::construct(*this, type);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, DsmcKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const Payload& op) {
    detail::DsmcKernelVariant::construct_payload(*this, op);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE DsmcPairParameters
DsmcKernel::pair_parameters(const MaterialProperties& lhs,
                            const MaterialProperties& rhs) noexcept {
    DsmcPairParameters pair {};
    const float lhs_mass = lhs.molecular_mass;
    const float rhs_mass = rhs.molecular_mass;
    const float mass_sum = lhs_mass + rhs_mass;
    if (!(lhs_mass > 0.0f) || !(rhs_mass > 0.0f) || !(mass_sum > 0.0f)) {
        return pair;
    }

    pair.reduced_mass    = lhs_mass * rhs_mass / mass_sum;
    pair.viscosity_index = (lhs.viscosity_index.value_or(0.5f)
                            + rhs.viscosity_index.value_or(0.5f))
        * 0.5f;
    pair.scattering_parameter = (lhs.scattering_parameter.value_or(1.0f)
                                 + rhs.scattering_parameter.value_or(1.0f))
        * 0.5f;

    if (lhs.reference_diameter.has_value() && rhs.reference_diameter.has_value()) {
        pair.reference_diameter = (lhs.reference_diameter.value() + rhs.reference_diameter.value()) * 0.5f;
    }
    if (lhs.reference_temperature.has_value() && rhs.reference_temperature.has_value()) {
        pair.reference_temperature = (lhs.reference_temperature.value() + rhs.reference_temperature.value()) * 0.5f;
    }

    pair.valid = pair.reduced_mass > 0.0f;
    return pair;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::cross_section(const DsmcKernelType type,
                          const MaterialProperties& lhs,
                          const MaterialProperties& rhs,
                          const float relative_speed) noexcept {
    return detail::DsmcKernelVariant::visit_type(
        type,
        detail::DsmcCrossSection { lhs, rhs, relative_speed },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcKernel::operator()(Float3& lhs_velocity,
                       Float3& rhs_velocity,
                       const MaterialProperties& lhs,
                       const MaterialProperties& rhs) const noexcept {
    detail::DsmcKernelVariant::apply(
        *this,
        detail::DsmcCollide { lhs_velocity, rhs_velocity, lhs, rhs });
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::sigma_g(const MaterialProperties* properties_ptr,
                    const std::size_t species_i,
                    const std::size_t species_j,
                    const float relative_speed_squared) const noexcept {
    if (!(relative_speed_squared > 0.0f)) {
        return 0.0f;
    }

    const float relative_speed = atlas::sqrt_nonnegative(relative_speed_squared);
    return DsmcKernel::cross_section(
               type,
               properties_ptr[species_i],
               properties_ptr[species_j],
               relative_speed)
        * relative_speed;
}

}
