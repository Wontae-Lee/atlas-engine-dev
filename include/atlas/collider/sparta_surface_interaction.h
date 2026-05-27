#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class SpartaSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "SpartaSurfaceInteraction requires a floating-point T");

public:
    class Builder;

    struct InternalEnergyState {
        T rotational {};
        T vibrational {};
    };

    struct ParticleState {
        Vector3<T> position {};
        Vector3<T> velocity {};
        InternalEnergyState internal_energy {};
    };

public:
    SpartaSurfaceInteraction() = default;

    ~SpartaSurfaceInteraction() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_particle_properties(const MaterialProperties<T>& properties) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_wall_properties(const MaterialProperties<T>& properties) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_wall_velocity(const Vector3<T>& velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_wall_rotation_origin(const Vector3<T>& origin) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_no_slip(bool no_slip) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    molecular_mass() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MaterialProperties<T>&
    particle_properties() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const MaterialProperties<T>&
    wall_properties() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<T>&
    wall_velocity() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<T>&
    wall_angular_velocity() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Vector3<T>&
    wall_rotation_origin() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    no_slip() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE ParticleState
    collide(const ParticleState& state,
            const Vector3<T>& point_of_intersection,
            const Vector3<T>& normal) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    effective_molecular_mass() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    wall_velocity_at(const Vector3<T>& point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    diffuse_reflection(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE InternalEnergyState
    diffuse_internal_energy(const InternalEnergyState& old_energy) const noexcept;

private:
    T _accommodation { T(1) };
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    MaterialProperties<T> _particle_properties {};
    MaterialProperties<T> _wall_properties {};
    Vector3<T> _wall_velocity {};
    Vector3<T> _wall_angular_velocity {};
    Vector3<T> _wall_rotation_origin {};
    bool _no_slip { false };
};

template <typename T>
class SpartaSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T molecular_mass) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_particle_properties(const MaterialProperties<T>& properties) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_wall_properties(const MaterialProperties<T>& properties) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_wall_velocity(const Vector3<T>& velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_wall_angular_velocity(const Vector3<T>& angular_velocity) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_wall_rotation_origin(const Vector3<T>& origin) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_no_slip(bool no_slip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE SpartaSurfaceInteraction<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<SpartaSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _accommodation { T(1) };
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    MaterialProperties<T> _particle_properties {};
    MaterialProperties<T> _wall_properties {};
    Vector3<T> _wall_velocity {};
    Vector3<T> _wall_angular_velocity {};
    Vector3<T> _wall_rotation_origin {};
    bool _no_slip { false };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using SpartaSurfaceInteraction = atlas::system::SpartaSurfaceInteraction<T>;

template <typename T>
using SpartaSurfaceInteractionHostPtr = atlas::host_shared_ptr<atlas::system::SpartaSurfaceInteraction<T>>;

template <typename T>
using SpartaSurfaceInteractionDevicePtr = atlas::device_shared_ptr<atlas::system::SpartaSurfaceInteraction<T>>;

} // namespace atlas

#include <atlas/collider/sparta_surface_interaction.hpp>
