#pragma once

#include <atlas/memory/memory.h>

#include <optional>

namespace atlas {

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
class MaterialProperties final {
public:
    class Builder;

public:
    MaterialType::Value type {};

    T mass {};

    T molecular_mass {};

    std::optional<T> translational_energy;

    std::optional<T> rotational_energy;

    std::optional<T> vibrational_energy;

    std::optional<int> rotational_dof;

    std::optional<int> vibrational_dof;

    std::optional<T> rotational_temperature;

    std::optional<T> characteristic_vibrational_temperature;

    std::optional<int> max_vibrational_quantum;

    std::optional<T> gamma_quant;

    std::optional<int> interaction_id;

    std::optional<bool> fully_ionized;

    std::optional<bool> polyatomic_molecule;

    std::optional<int> species_id;

    std::optional<T> reference_diameter;

    std::optional<T> reference_temperature;

    std::optional<T> viscosity_index;

    std::optional<T> scattering_parameter;

    std::optional<T> rotational_relaxation_probability;

    std::optional<T> vibrational_relaxation_probability;

    std::optional<T> rotational_relaxation_c1;

    std::optional<T> rotational_relaxation_c2;

    std::optional<T> rotational_relaxation_c3;

    std::optional<T> vibrational_relaxation_c1;

    std::optional<T> vibrational_relaxation_c2;

    std::optional<T> rest_density;

    std::optional<T> pressure_coefficient;

    std::optional<T> dynamic_viscosity;

    std::optional<T> electronic_energy;

    std::optional<int> charge;

public:
    MaterialProperties() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;
};

template <typename T>
class MaterialProperties<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE MaterialProperties<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<MaterialProperties<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(MaterialType::Value t);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_mass(T m);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_molecular_mass(T m);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_translational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_energy(T e);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_dof(int dof);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_dof(int dof);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_temperature(T temperature);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_characteristic_vibrational_temperature(T temperature);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_max_vibrational_quantum(int quantum);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_gamma_quant(T gamma);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_interaction_id(int id);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fully_ionized(bool value);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_polyatomic_molecule(bool value);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_species_id(int id);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_reference_diameter(T d_ref);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_reference_temperature(T t_ref);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_viscosity_index(T omega);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_scattering_parameter(T alpha);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_relaxation_probability(T probability);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_relaxation_probability(T probability);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rotational_relaxation_coefficients(T c1, T c2, T c3);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_vibrational_relaxation_coefficients(T c1, T c2);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_rest_density(T rho0);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_pressure_coefficient(T k);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_dynamic_viscosity(T mu);

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

    T _molecular_mass {};

    std::optional<T> _translational_energy;

    std::optional<T> _rotational_energy;

    std::optional<T> _vibrational_energy;

    std::optional<int> _rotational_dof;

    std::optional<int> _vibrational_dof;

    std::optional<T> _rotational_temperature;

    std::optional<T> _characteristic_vibrational_temperature;

    std::optional<int> _max_vibrational_quantum;

    std::optional<T> _gamma_quant;

    std::optional<int> _interaction_id;

    std::optional<bool> _fully_ionized;

    std::optional<bool> _polyatomic_molecule;

    std::optional<int> _species_id;

    std::optional<T> _reference_diameter;

    std::optional<T> _reference_temperature;

    std::optional<T> _viscosity_index;

    std::optional<T> _scattering_parameter;

    std::optional<T> _rotational_relaxation_probability;

    std::optional<T> _vibrational_relaxation_probability;

    std::optional<T> _rotational_relaxation_c1;

    std::optional<T> _rotational_relaxation_c2;

    std::optional<T> _rotational_relaxation_c3;

    std::optional<T> _vibrational_relaxation_c1;

    std::optional<T> _vibrational_relaxation_c2;

    std::optional<T> _rest_density;

    std::optional<T> _pressure_coefficient;

    std::optional<T> _dynamic_viscosity;

    std::optional<T> _electronic_energy;

    std::optional<int> _charge;
};

}

namespace atlas {

template <typename T>
using MatrialPropertiesHostPtr = atlas::host_shared_ptr<MaterialProperties<T>>;

template <typename T>
using MatrialPropertiesDevicePtr = atlas::device_shared_ptr<MaterialProperties<T>>;

}

#include <atlas/material/material_properties.hpp>