#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas {

enum class DiffuseSampling {

    CosineWeighted,

    Uniform
};

template <typename T>
class IsothermalSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "IsothermalSurfaceInteraction requires a floating-point T");

public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~IsothermalSurfaceInteraction() noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_restitution(T restitution_coeff) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_momentum_acc(T momentum_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DiffuseSampling
    diffuse_sampling() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    restitution() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    momentum_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal,
                    const MaterialProperties<T>& material) const noexcept;

private:
    T _restitution_coeff { T(1) };

    T _momentum_acc { T(1) };

    T _temperature { T(273.15) };

    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };
};

template <typename T>
class IsothermalSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_diffuse_sampling(DiffuseSampling mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_restitution(T restitution) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_momentum_acc(T momentum_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE IsothermalSurfaceInteraction<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<IsothermalSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DiffuseSampling _diffuse_sampling { DiffuseSampling::Uniform };

    T _restitution { T(1) };

    T _momentum_acc { T(1) };

    T _temperature { T(273.15) };
};

}

namespace atlas {

template <typename T>
using IsothermalSurfaceInteractionHostPtr = atlas::host_shared_ptr<IsothermalSurfaceInteraction<T>>;

template <typename T>
using IsothermalSurfaceInteractionDevicePtr = atlas::device_shared_ptr<IsothermalSurfaceInteraction<T>>;

}

#include <atlas/collider/interaction/isothermal_surface_kernel.hpp>