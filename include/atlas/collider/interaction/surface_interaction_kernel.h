#pragma once

#include <atlas/collider/interaction/isothermal_surface_kernel.h>
#include <atlas/collider/interaction/maxwellian_surface_interaction.h>
#include <atlas/core/detail/device_variant.h>

namespace atlas {

enum struct SurfaceInteractionType : int {

    isothermal,

    maxwellian
};

template <typename T>
struct SurfaceInteractionKernel final {

    SurfaceInteractionType type = SurfaceInteractionType::isothermal;

    union {
        IsothermalSurfaceInteraction<T> isothermal;
        MaxwellianSurfaceInteraction<T> maxwellian;
    };

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const IsothermalSurfaceInteraction<T>& interaction) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const MaxwellianSurfaceInteraction<T>& interaction) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    SurfaceInteractionKernel(const SurfaceInteractionKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE SurfaceInteractionKernel&
    operator=(const SurfaceInteractionKernel& other) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~SurfaceInteractionKernel() noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

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

}

#include <atlas/collider/interaction/surface_interaction_kernel.hpp>