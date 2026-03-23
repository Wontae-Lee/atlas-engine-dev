#pragma once

#include <atlas/matter/matter.h>
#include <atlas/memory/memory.h>

#include <optional>

namespace atlas::system {

template <typename T>
class FluidicParticle final : public Matter<T> {
public:
    class Builder;

public:
    T molecular_mass {};

    std::optional<T> statistical_weight;

    std::optional<T> translational_energy;

    std::optional<T> rotational_energy;

    std::optional<T> vibrational_energy;

    std::optional<int> species_id;

    std::optional<T> collision_diameter;

    std::optional<T> viscosity_index;

    std::optional<T> scattering_parameter;

    std::optional<T> electronic_energy;

    std::optional<int> charge;

public:
    FluidicParticle() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

template <typename T>
class FluidicParticle<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE FluidicParticle<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<FluidicParticle<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T mass);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_statistical_weight(T w);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_translational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_species_id(int id);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_collision_diameter(T d_ref);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_viscosity_index(T omega);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_scattering_parameter(T alpha);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_charge(int q);

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<T> _molecular_mass;

    std::optional<T> _statistical_weight;

    std::optional<T> _translational_energy;

    std::optional<T> _rotational_energy;

    std::optional<T> _vibrational_energy;

    std::optional<int> _species_id;

    std::optional<T> _collision_diameter;

    std::optional<T> _viscosity_index;

    std::optional<T> _scattering_parameter;

    std::optional<T> _electronic_energy;

    std::optional<int> _charge;
};

}

namespace atlas {

template <typename T>
using FluidicParticle = system::FluidicParticle<T>;

template <typename T>
using FluidicParticleHostPtr = atlas::host_shared_ptr<system::FluidicParticle<T>>;

template <typename T>
using FluidicParticleDevicePtr = atlas::device_shared_ptr<system::FluidicParticle<T>>;

}

#include <atlas/matter/fluidic_particle.hpp>