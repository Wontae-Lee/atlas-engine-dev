#pragma once

#include <atlas/collider/interaction/isothermal_surface_kernel.h>
#include <atlas/collider/interaction/maxwellian_surface_interaction.h>
#include <atlas/core/detail/device_variant.h>

#include <type_traits>

/**
 * @file surface_interaction_kernel.h
 * @brief Runtime-selectable dispatcher over the two wall-interaction
 *        models (`IsothermalSurfaceInteraction`,
 *        `MaxwellianSurfaceInteraction`), so a `Collider`/`Unit` can be
 *        configured with either model at build time without templating
 *        every kernel that calls into it.
 *
 * @details
 * Follows the same tagged-union `DeviceVariant` pattern as
 * `PostColliderKernel` (see that file's top-of-file documentation for why
 * this pattern exists instead of virtual dispatch): a
 * `SurfaceInteractionType` tag selects which of the `union`'s two payload
 * models is currently constructed, and `detail::SurfaceInteractionVariant`
 * switches on it inside `operator()`/`internal_energy()`. The templated
 * converting constructor lets a `Collider::Builder` be handed a
 * `HostBuffer<IsothermalSurfaceInteraction>` or
 * `HostBuffer<MaxwellianSurfaceInteraction>` directly and have each
 * element wrapped into a `SurfaceInteractionKernel` automatically.
 */

namespace atlas {

/**
 * @brief Selects which wall-interaction model a `SurfaceInteractionKernel`
 *        holds; see `isothermal_surface_kernel.h` /
 *        `maxwellian_surface_interaction.h` for what each model does.
 */
enum struct SurfaceInteractionType : int {

    /** Restitution-scaled speed, specular/diffuse-direction blend
     *  (`IsothermalSurfaceInteraction`). */
    isothermal,

    /** Full Maxwell-model thermal reflection with Larsen-Borgnakke
     *  internal-energy exchange (`MaxwellianSurfaceInteraction`). */
    maxwellian
};

/**
 * @brief Tagged-union wrapper letting `ColliderCollisionKernel`/
 *        `FastColliderKernel`/... call `operator()`/`internal_energy()`
 *        on whichever surface-interaction model a `Unit` was configured
 *        with. See this file's top-of-file documentation for the
 *        `DeviceVariant` pattern.
 */
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

    /**
     * @brief Wraps a concrete `IsothermalSurfaceInteraction` or
     *        `MaxwellianSurfaceInteraction` value, setting `type` to
     *        match (SFINAE-disabled for `Payload = SurfaceInteractionKernel`
     *        itself so this does not shadow the copy constructor).
     */
    template <typename Payload,
              std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SurfaceInteractionKernel>, int> = 0>
    ATLAS_ALL_DEVICE explicit SurfaceInteractionKernel(const Payload& interaction) noexcept;

    /** @brief Dispatches to the active payload's reflection sampling. */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    operator()(const Float3& incident, const Float3& normal) const noexcept;

    /** @brief Dispatches to the active payload's internal-energy
     *  exchange (a no-op passthrough for `isothermal`). */
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
    internal_energy(const FluidInternalEnergy& incident_energy,
                    const Float3& incident_velocity,
                    const Float3& normal,
                    const MaterialProperties& material) const noexcept;
};

namespace detail {

    using SurfaceInteractionVariant = DeviceVariant<
        SurfaceInteractionKernel,
        SurfaceInteractionType,
        SurfaceInteractionType::isothermal,
        DeviceVariantCase<SurfaceInteractionType::isothermal, &SurfaceInteractionKernel::isothermal>,
        DeviceVariantCase<SurfaceInteractionType::maxwellian, &SurfaceInteractionKernel::maxwellian>>;

    // Functor visitors instead of generic device lambdas (nvcc forbids
    // generic / by-reference-capturing extended `__host__ __device__` lambdas).
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

}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SurfaceInteractionKernel::SurfaceInteractionKernel() noexcept {
    detail::SurfaceInteractionVariant::construct(*this, SurfaceInteractionType::isothermal);
}

template <typename Payload,
          std::enable_if_t<!std::is_same_v<std::decay_t<Payload>, SurfaceInteractionKernel>, int>>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
SurfaceInteractionKernel::SurfaceInteractionKernel(const Payload& interaction) noexcept {
    detail::SurfaceInteractionVariant::construct_payload(*this, interaction);
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
SurfaceInteractionKernel::operator()(const Float3& incident,
                                     const Float3& normal) const noexcept {
    return detail::SurfaceInteractionVariant::visit(
        *this,
        detail::SurfaceInteractionApply { incident, normal },
        Float3(0.0f, 0.0f, 0.0f));
}

ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE FluidInternalEnergy
SurfaceInteractionKernel::internal_energy(
    const FluidInternalEnergy& incident_energy,
    const Float3& incident_velocity,
    const Float3& normal,
    const MaterialProperties& material) const noexcept {
    return detail::SurfaceInteractionVariant::visit(
        *this,
        detail::SurfaceInteractionInternalEnergy { incident_energy, incident_velocity, normal, material },
        incident_energy);
}

}
