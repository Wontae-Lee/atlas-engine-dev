#pragma once

#include <atlas/memory/memory.h>

#include <optional>

namespace atlas::system {

struct MaterialType final {

    enum Value : int {

        Molecule,

        Atom,

        Ion,

        Neutron,

        Solid
    };
};

template <typename T>
class MatrialProperties final {
public:
    class Builder;

public:
    MaterialType::Value type {};

    T mass {};

    std::optional<T> molecular_mass;

    std::optional<T> statistical_weight;

    std::optional<T> translational_energy;

    std::optional<T> rotational_energy;

    std::optional<T> vibrational_energy;

    std::optional<int> species_id;

    std::optional<T> collision_diameter;

    std::optional<T> viscosity_index;

    std::optional<T> scattering_parameter;

    std::optional<T> rest_density;

    std::optional<T> pressure_coefficient;

    std::optional<T> dynamic_viscosity;

    std::optional<T> smoothing_length;

    std::optional<T> electronic_energy;

    std::optional<int> charge;

public:
    MatrialProperties() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

template <typename T>
class MatrialProperties<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE MatrialProperties<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MatrialProperties<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(MaterialType::Value t);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_mass(T m);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T m);

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
    with_rest_density(T rho0);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_pressure_coefficient(T k);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dynamic_viscosity(T mu);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_smoothing_length(T h);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_electronic_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_charge(int q);

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<MaterialType::Value> _type;

    std::optional<T> _mass;

    std::optional<T> _molecular_mass;

    std::optional<T> _statistical_weight;

    std::optional<T> _translational_energy;

    std::optional<T> _rotational_energy;

    std::optional<T> _vibrational_energy;

    std::optional<int> _species_id;

    std::optional<T> _collision_diameter;

    std::optional<T> _viscosity_index;

    std::optional<T> _scattering_parameter;

    std::optional<T> _rest_density;

    std::optional<T> _pressure_coefficient;

    std::optional<T> _dynamic_viscosity;

    std::optional<T> _smoothing_length;

    std::optional<T> _electronic_energy;

    std::optional<int> _charge;
};

}

namespace atlas {

using MaterialType = system::MaterialType;

template <typename T>
using MatrialProperties = system::MatrialProperties<T>;

template <typename T>
using MatrialPropertiesHostPtr = atlas::host_shared_ptr<system::MatrialProperties<T>>;

template <typename T>
using MatrialPropertiesDevicePtr = atlas::device_shared_ptr<system::MatrialProperties<T>>;

}

#include <atlas/material/matrial_properties.hpp>