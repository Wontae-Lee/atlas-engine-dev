#pragma once

#include <atlas/core/device_variant.h>
#include <atlas/collider/interaction/isothermal_surface_kernel.h>
#include <atlas/collider/interaction/maxwellian_surface_interaction.h>

#include <type_traits>

namespace atlas {

enum struct SurfaceInteractionType : int {

    isothermal,

    maxwellian
};

struct SurfaceInteractionKernel final {

    SurfaceInteractionType type = SurfaceInteractionType::isothermal;

    union {

        IsothermalSurfaceInteraction isothermal;

        MaxwellianSurfaceInteraction maxwellian;
    };

    ATLAS_ALL_DEVICE
    SurfaceInteractionKernel() noexcept;

    ATLAS_ALL_DEVICE
    SurfaceInteractionKernel(const SurfaceInteractionKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE SurfaceInteractionKernel&
    operator=(const SurfaceInteractionKernel& other) noexcept = default;

    ATLAS_ALL_DEVICE
    ~SurfaceInteractionKernel() noexcept = default;

    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SurfaceInteractionKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit SurfaceInteractionKernel(const Payload& interaction) noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const Float3& incident, const Float3& normal) const noexcept;

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal,
                    const MaterialProperties& material) const noexcept;
};

using SurfaceInteractionVariant = DeviceVariant<
    SurfaceInteractionKernel,
    SurfaceInteractionType,
    SurfaceInteractionType::isothermal,
    DeviceVariantCase<SurfaceInteractionType::isothermal, &SurfaceInteractionKernel::isothermal>,
    DeviceVariantCase<SurfaceInteractionType::maxwellian, &SurfaceInteractionKernel::maxwellian>>;

struct SurfaceInteractionApply {
    Float3 incident;
    Float3 normal;
    template <typename I>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const I& interaction) const noexcept { return interaction(incident, normal); }
};

struct SurfaceInteractionInternalEnergy {
    FluidInternalEnergy incident_energy;
    Float3 incident_velocity;
    Float3 normal;
    MaterialProperties material;
    template <typename I>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    operator()(const I& interaction) const noexcept {
        return interaction.internal_energy(incident_energy, incident_velocity, normal, material);
    }
};

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SurfaceInteractionKernel::SurfaceInteractionKernel() noexcept {
    SurfaceInteractionVariant::construct(*this, SurfaceInteractionType::isothermal);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SurfaceInteractionKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SurfaceInteractionKernel::SurfaceInteractionKernel(const Payload& interaction) noexcept {
    SurfaceInteractionVariant::construct_payload(*this, interaction);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
SurfaceInteractionKernel::operator()(const Float3& incident,
                                     const Float3& normal) const noexcept {
    return SurfaceInteractionVariant::visit(
        *this,
        SurfaceInteractionApply { incident, normal },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
SurfaceInteractionKernel::internal_energy(
    const FluidInternalEnergy& incident_energy,
    const Float3& incident_velocity,
    const Float3& normal,
    const MaterialProperties& material) const noexcept {
    return SurfaceInteractionVariant::visit(
        *this,
        SurfaceInteractionInternalEnergy { incident_energy, incident_velocity, normal, material },
        incident_energy);
}

}
