#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/core/macros.h>
#include <atlas/material/material.h>
#include <atlas/math/math.h>
#include <atlas/solver/dsmc/kernel/dsmc_kernel_type.h>
#include <atlas/solver/dsmc/kernel/hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_hard_sphere_kernel.h>
#include <atlas/solver/dsmc/kernel/variable_soft_sphere_kernel.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace atlas {

template <typename K>
concept ConceptDsmcKernel = requires(const K kernel,
                                     Float3 velocity,
                                     const Material material,
                                     float relative_speed) {
    { K::cross_section(material, material, relative_speed) } -> std::same_as<float>;
    { kernel(velocity, velocity, material, material) } -> std::same_as<void>;
};

static_assert(ConceptDsmcKernel<HardSphereKernel>);
static_assert(ConceptDsmcKernel<VariableHardSphereKernel>);
static_assert(ConceptDsmcKernel<VariableSoftSphereKernel>);

class DsmcKernel final {
public:
    DsmcKernelType type = DsmcKernelType::hard_sphere;

    union {

        HardSphereKernel hard_sphere;

        VariableHardSphereKernel variable_hard_sphere;

        VariableSoftSphereKernel variable_soft_sphere;
    };

    ATLAS_ALL_DEVICE
    DsmcKernel() noexcept;

    ATLAS_ALL_DEVICE explicit DsmcKernel(DsmcKernelType kernel_type) noexcept;

    ATLAS_ALL_DEVICE
    DsmcKernel(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE DsmcKernel&
    operator=(const DsmcKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~DsmcKernel() noexcept = default;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    cross_section(const Material& lhs, const Material& rhs, float relative_speed) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    sigma_g(const Material* materials,
            std::size_t lhs_species,
            std::size_t rhs_species,
            float relative_speed_squared) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Float3& lhs_velocity,
               Float3& rhs_velocity,
               const Material& lhs,
               const Material& rhs) const noexcept;
};

using DsmcKernelVariant = DeviceVariant<
    DsmcKernel,
    DsmcKernelType,
    DsmcKernelType::hard_sphere,
    DeviceVariantCase<DsmcKernelType::hard_sphere, &DsmcKernel::hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_hard_sphere, &DsmcKernel::variable_hard_sphere>,
    DeviceVariantCase<DsmcKernelType::variable_soft_sphere, &DsmcKernel::variable_soft_sphere>>;

class DsmcCrossSection {
public:
    const Material& lhs;
    const Material& rhs;
    float relative_speed;
    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    operator()(const K&) const noexcept { return K::cross_section(lhs, rhs, relative_speed); }
};

class DsmcCollide {
public:
    Float3& lhs_velocity;
    Float3& rhs_velocity;
    const Material& lhs;
    const Material& rhs;
    template <typename K>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(const K& kernel) const noexcept { kernel(lhs_velocity, rhs_velocity, lhs, rhs); }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel() noexcept {
    DsmcKernelVariant::construct(*this, DsmcKernelType::hard_sphere);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
DsmcKernel::DsmcKernel(const DsmcKernelType kernel_type) noexcept {
    DsmcKernelVariant::construct(*this, kernel_type);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::cross_section(const Material& lhs,
                          const Material& rhs,
                          const float relative_speed) const noexcept {
    return DsmcKernelVariant::visit(
        *this,
        DsmcCrossSection { lhs, rhs, relative_speed },
        0.0f);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
DsmcKernel::sigma_g(const Material* materials,
                    const std::size_t lhs_species,
                    const std::size_t rhs_species,
                    const float relative_speed_squared) const noexcept {
    if (!(relative_speed_squared > 0.0f)) {
        return 0.0f;
    }

    const float relative_speed = atlas::sqrt_nonnegative(relative_speed_squared);

    return cross_section(materials[lhs_species], materials[rhs_species], relative_speed)
        * relative_speed;
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
DsmcKernel::operator()(Float3& lhs_velocity,
                       Float3& rhs_velocity,
                       const Material& lhs,
                       const Material& rhs) const noexcept {
    DsmcKernelVariant::apply(
        *this,
        DsmcCollide { lhs_velocity, rhs_velocity, lhs, rhs });
}

}