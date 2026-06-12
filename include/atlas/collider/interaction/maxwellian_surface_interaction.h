#pragma once

#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas {

enum struct MaxwellianInternalEnergyStyle : int {
    none,
    smooth,
    discrete
};

template <typename T>
class MaxwellianSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "MaxwellianSurfaceInteraction requires a floating-point T");

public:
    class Builder;

public:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE ~MaxwellianSurfaceInteraction() noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_momentum_acc(T momentum_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_trans_acc(T trans_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_rot_acc(T rot_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vib_acc(T vib_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_accommodation(T momentum_acc, T trans_acc, T rot_acc, T vib_acc) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    molecular_mass() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    momentum_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    trans_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    rot_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    vib_acc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE MaxwellianInternalEnergyStyle
    rot_style() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE MaxwellianInternalEnergyStyle
    vib_style() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    most_probable_speed() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    sample(const Vector3<T>& incident,
           const Vector3<T>& normal,
           T branch_sample,
           T perpendicular_sample,
           T theta_sample,
           T tangent_sample,
           const Vector3<T>& tangent_seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    internal_energy(const FluidInternalEnergy<T>& incident_energy,
                    const Vector3<T>& incident_velocity,
                    const Vector3<T>& normal,
                    const MaterialProperties<T>& material) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE FluidInternalEnergy<T>
    sample_internal_energy(const FluidInternalEnergy<T>& incident,
                           T trans_sample,
                           T trans_theta_sample,
                           T rot_sample,
                           T rot_theta_sample,
                           T vib_sample,
                           T vib_theta_sample) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_internal_energy_mode(T incident, T acc, T sample, T theta_sample) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_rotational_energy(const MaterialProperties<T>& material,
                                     const Vector3<T>& seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_vibrational_energy(const MaterialProperties<T>& material,
                                      const Vector3<T>& seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sample_diffuse_smooth_energy(int dof, const Vector3<T>& seed, T salt) const noexcept;

private:
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    T _momentum_acc { T(1) };
    T _trans_acc { T(1) };
    T _rot_acc { T(1) };
    T _vib_acc { T(1) };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

template <typename T>
class MaxwellianSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_momentum_acc(T momentum_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_trans_acc(T trans_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rot_acc(T rot_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vib_acc(T vib_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rot_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vib_style(MaxwellianInternalEnergyStyle style) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_accommodation(T momentum_acc, T trans_acc, T rot_acc, T vib_acc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE MaxwellianSurfaceInteraction<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaxwellianSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    T _momentum_acc { T(1) };
    T _trans_acc { T(1) };
    T _rot_acc { T(1) };
    T _vib_acc { T(1) };
    MaxwellianInternalEnergyStyle _rot_style { MaxwellianInternalEnergyStyle::smooth };
    MaxwellianInternalEnergyStyle _vib_style { MaxwellianInternalEnergyStyle::smooth };
};

}

namespace atlas {
template <typename T>
using MaxwellianSurfaceInteractionHostPtr = atlas::host_shared_ptr<MaxwellianSurfaceInteraction<T>>;

template <typename T>
using MaxwellianSurfaceInteractionDevicePtr = atlas::device_shared_ptr<MaxwellianSurfaceInteraction<T>>;

}

#include <atlas/collider/interaction/maxwellian_surface_interaction.hpp>