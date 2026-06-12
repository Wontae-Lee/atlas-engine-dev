#pragma once

/**
 * @file surface_interaction_kernel.h
 * @brief Declares tagged device-callable surface interaction kernels for Collider.
 */

#include <atlas/core/detail/device_variant.h>
#include <atlas/collider/interaction/isothermal_surface_kernel.h>
#include <atlas/collider/interaction/maxwellian_surface_interaction.h>

namespace atlas {

/**
 * @brief Selects the active surface interaction behavior.
 */
enum struct SurfaceInteractionType : int {
    /**
     * @brief Isothermal restitution/momentum-accommodation hemisphere interaction.
     */
    isothermal,

    /**
     * @brief SPARTA-compatible Maxwellian diffuse/specular interaction.
     */
    maxwellian
};

/**
 * @brief Tagged runtime wrapper for device-callable surface interactions.
 *
 * This wrapper gives Collider a value-type polymorphic interaction model that
 * can be copied into DeviceBuffer and dispatched inside device kernels without
 * virtual functions or host-only ownership.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
struct SurfaceInteractionKernel final {
    /**
     * @brief Type tag selecting the active interaction behavior.
     */
    SurfaceInteractionType type = SurfaceInteractionType::isothermal;

    union {
        IsothermalSurfaceInteraction<T> isothermal;
        MaxwellianSurfaceInteraction<T> maxwellian;
    };

    /**
     * @brief Construct a default isothermal surface interaction.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel() noexcept;

    /**
     * @brief Construct a kernel from an isothermal interaction.
     *
     * @param interaction Interaction payload to store.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const IsothermalSurfaceInteraction<T>& interaction) noexcept;

    /**
     * @brief Construct a kernel from a Maxwellian interaction.
     *
     * @param interaction Interaction payload to store.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const MaxwellianSurfaceInteraction<T>& interaction) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const SurfaceInteractionKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SurfaceInteractionKernel&
    operator=(const SurfaceInteractionKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    ~SurfaceInteractionKernel() noexcept;

    /**
     * @brief Computes the outgoing velocity for the active surface model.
     *
     * @param incident Incident velocity relative to the surface.
     * @param normal Surface normal defining the outgoing hemisphere.
     * @return Outgoing velocity relative to the surface.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    /**
     * @brief Computes outgoing internal energy for the active surface model.
     *
     * @param incident_energy Incident per-particle internal energy.
     * @param incident_velocity Incident velocity relative to the surface.
     * @param normal Surface normal defining the outgoing hemisphere.
     * @return Outgoing per-particle internal energy.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal,
                    const MaterialProperties<T>& material) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    destroy_active() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    copy_from(const SurfaceInteractionKernel& other) noexcept;
};

} // namespace atlas

#include <atlas/collider/interaction/surface_interaction_kernel.hpp>
