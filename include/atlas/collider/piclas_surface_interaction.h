#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/math.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/memory/memory.h>

#include <type_traits>

namespace atlas::system {

template <typename T>
class PiclasSurfaceInteraction final {
    static_assert(std::is_floating_point_v<T>,
                  "PiclasSurfaceInteraction requires a floating-point T");

public:
    class Builder;

    struct InternalEnergyState {
        T vibrational {};
        T rotational {};
        T electronic {};
    };

    struct InternalEnergyParameters {
        T rotational_wall_energy {};
        T electronic_wall_energy {};
        T characteristic_vibrational_temperature {};
        T gamma_quant {};
        int max_vibrational_quantum {};
        bool enable_vibrational_relaxation {};
        bool enable_electronic_relaxation {};
        bool use_dsmc { true };
        int collision_mode { 2 };
        int interaction_id { 2 };
        bool fully_ionized {};
    };

    struct ParticleState {
        Vector3<T> position {};
        Vector3<T> last_position {};
        Vector3<T> velocity {};
        Vector3<T> trajectory {};
        T trajectory_length {};
        InternalEnergyState internal_energy {};
    };

public:
    PiclasSurfaceInteraction() = default;

    ~PiclasSurfaceInteraction() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_momentum_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_translational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vibrational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_rotational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_electronic_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_only_specular(bool only_specular) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_only_diffuse(bool only_diffuse) noexcept;

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
    set_normal_points_out_of_domain(bool points_out_of_domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_use_dsmc(bool use_dsmc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_collision_mode(int collision_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_vibrational_relaxation_enabled(bool enabled) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_electronic_relaxation_enabled(bool enabled) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    momentum_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    translational_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    vibrational_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    rotational_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    electronic_accommodation() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    only_specular() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    only_diffuse() const noexcept;

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
    normal_points_out_of_domain() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    use_dsmc() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    collision_mode() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    vibrational_relaxation_enabled() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    electronic_relaxation_enabled() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    operator()(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE ParticleState
    collide(const ParticleState& state,
            const Vector3<T>& point_of_intersection,
            const Vector3<T>& normal,
            T remaining_time) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE InternalEnergyState
    accommodate_internal_energy(const InternalEnergyState& old_energy,
                                const InternalEnergyParameters& parameters,
                                const Vector3<T>& seed) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE InternalEnergyState
    accommodate_internal_energy(const InternalEnergyState& old_energy,
                                const Vector3<T>& seed) const noexcept;

private:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    effective_molecular_mass() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    should_accommodate_rotational_vibrational(int interaction_id) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    should_accommodate_electronic(int interaction_id, bool fully_ionized) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    wall_velocity_at(const Vector3<T>& point) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    diffuse_normal(const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    diffuse_reflection(const Vector3<T>& incident, const Vector3<T>& normal) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    sample_vibrational_quantum(T sample,
                               T characteristic_vibrational_temperature,
                               int max_vibrational_quantum) const noexcept;

private:
    T _momentum_accommodation { T(1) };
    T _translational_accommodation { T(1) };
    T _vibrational_accommodation { T(1) };
    T _rotational_accommodation { T(1) };
    T _electronic_accommodation { T(1) };
    // TODO: Add PICLas-style adapted wall temperature support using side-local
    // surface heat-flux sampling and radiative-emissivity feedback.
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    MaterialProperties<T> _particle_properties {};
    MaterialProperties<T> _wall_properties {};
    Vector3<T> _wall_velocity {};
    Vector3<T> _wall_angular_velocity {};
    Vector3<T> _wall_rotation_origin {};
    bool _only_specular { false };
    bool _only_diffuse { false };
    bool _normal_points_out_of_domain { false };
    bool _use_dsmc { true };
    int _collision_mode { 2 };
    bool _vibrational_relaxation_enabled { true };
    bool _electronic_relaxation_enabled { true };
};

template <typename T>
class PiclasSurfaceInteraction<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_momentum_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_translational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_accommodation(T accommodation) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_only_specular(bool only_specular) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_only_diffuse(bool only_diffuse) noexcept;

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
    with_normal_points_out_of_domain(bool points_out_of_domain) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_use_dsmc(bool use_dsmc) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collision_mode(int collision_mode) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_relaxation_enabled(bool enabled) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_relaxation_enabled(bool enabled) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE PiclasSurfaceInteraction<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<PiclasSurfaceInteraction<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    T _momentum_accommodation { T(1) };
    T _translational_accommodation { T(1) };
    T _vibrational_accommodation { T(1) };
    T _rotational_accommodation { T(1) };
    T _electronic_accommodation { T(1) };
    // TODO: Add PICLas-style adapted wall temperature support using side-local
    // surface heat-flux sampling and radiative-emissivity feedback.
    T _temperature { T(273.15) };
    T _molecular_mass { T(1) };
    MaterialProperties<T> _particle_properties {};
    MaterialProperties<T> _wall_properties {};
    Vector3<T> _wall_velocity {};
    Vector3<T> _wall_angular_velocity {};
    Vector3<T> _wall_rotation_origin {};
    bool _only_specular { false };
    bool _only_diffuse { false };
    bool _normal_points_out_of_domain { false };
    bool _use_dsmc { true };
    int _collision_mode { 2 };
    bool _vibrational_relaxation_enabled { true };
    bool _electronic_relaxation_enabled { true };
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using PiclasSurfaceInteraction = atlas::system::PiclasSurfaceInteraction<T>;

template <typename T>
using PiclasSurfaceInteractionHostPtr = atlas::host_shared_ptr<atlas::system::PiclasSurfaceInteraction<T>>;

template <typename T>
using PiclasSurfaceInteractionDevicePtr = atlas::device_shared_ptr<atlas::system::PiclasSurfaceInteraction<T>>;

} // namespace atlas

#include <atlas/collider/piclas_surface_interaction.hpp>
