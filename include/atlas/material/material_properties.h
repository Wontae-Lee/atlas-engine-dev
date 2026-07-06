#pragma once

#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>

#include <optional>

namespace atlas {

struct MaterialType final {

    enum Value : int {

        molecule,

        atom,

        ion,

        neutron,

        solid
    };
};

class MaterialProperties final {
public:
    class Builder;

public:
    MaterialType::Value type {};

    float mass {};

    float molecular_mass {};

    std::optional<float> translational_energy;

    std::optional<float> rotational_energy;

    std::optional<float> vibrational_energy;

    std::optional<int> rotational_dof;

    std::optional<int> vibrational_dof;

    std::optional<float> rotational_temperature;

    std::optional<float> characteristic_vibrational_temperature;

    std::optional<int> max_vibrational_quantum;

    std::optional<float> gamma_quant;

    std::optional<int> interaction_id;

    std::optional<bool> fully_ionized;

    std::optional<bool> polyatomic_molecule;

    std::optional<int> species_id;

    std::optional<float> reference_diameter;

    std::optional<float> reference_temperature;

    std::optional<float> viscosity_index;

    std::optional<float> scattering_parameter;

    std::optional<float> rotational_relaxation_probability;

    std::optional<float> vibrational_relaxation_probability;

    std::optional<float> rotational_relaxation_c1;

    std::optional<float> rotational_relaxation_c2;

    std::optional<float> rotational_relaxation_c3;

    std::optional<float> vibrational_relaxation_c1;

    std::optional<float> vibrational_relaxation_c2;

    std::optional<float> rest_density;

    std::optional<float> pressure_coefficient;

    std::optional<float> dynamic_viscosity;

    std::optional<float> electronic_energy;

    std::optional<int> charge;

public:
    MaterialProperties() = default;

    ATLAS_HOST static Builder
    builder() noexcept;
};

class MaterialProperties::Builder final {
public:
    Builder() = default;

    ATLAS_HOST MaterialProperties
    build() const;

    ATLAS_HOST atlas::host_shared_ptr<MaterialProperties>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_type(MaterialType::Value t);

    ATLAS_HOST Builder&
    with_mass(float m);

    ATLAS_HOST Builder&
    with_molecular_mass(float m);

    ATLAS_HOST Builder&
    with_translational_energy(float e);

    ATLAS_HOST Builder&
    with_rotational_energy(float e);

    ATLAS_HOST Builder&
    with_vibrational_energy(float e);

    ATLAS_HOST Builder&
    with_rotational_dof(int dof);

    ATLAS_HOST Builder&
    with_vibrational_dof(int dof);

    ATLAS_HOST Builder&
    with_rotational_temperature(float temperature);

    ATLAS_HOST Builder&
    with_characteristic_vibrational_temperature(float temperature);

    ATLAS_HOST Builder&
    with_max_vibrational_quantum(int quantum);

    ATLAS_HOST Builder&
    with_gamma_quant(float gamma);

    ATLAS_HOST Builder&
    with_interaction_id(int id);

    ATLAS_HOST Builder&
    with_fully_ionized(bool value);

    ATLAS_HOST Builder&
    with_polyatomic_molecule(bool value);

    ATLAS_HOST Builder&
    with_species_id(int id);

    ATLAS_HOST Builder&
    with_reference_diameter(float d_ref);

    ATLAS_HOST Builder&
    with_reference_temperature(float t_ref);

    ATLAS_HOST Builder&
    with_viscosity_index(float omega);

    ATLAS_HOST Builder&
    with_scattering_parameter(float alpha);

    ATLAS_HOST Builder&
    with_rotational_relaxation_probability(float probability);

    ATLAS_HOST Builder&
    with_vibrational_relaxation_probability(float probability);

    ATLAS_HOST Builder&
    with_rotational_relaxation_coefficients(float c1, float c2, float c3);

    ATLAS_HOST Builder&
    with_vibrational_relaxation_coefficients(float c1, float c2);

    ATLAS_HOST Builder&
    with_rest_density(float rho0);

    ATLAS_HOST Builder&
    with_pressure_coefficient(float k);

    ATLAS_HOST Builder&
    with_dynamic_viscosity(float mu);

    ATLAS_HOST Builder&
    with_electronic_energy(float e);

    ATLAS_HOST Builder&
    with_charge(int q);

private:
    ATLAS_HOST void
    validate() const;

private:
    std::optional<MaterialType::Value> _type;

    std::optional<float> _mass;

    float _molecular_mass {};

    std::optional<float> _translational_energy;

    std::optional<float> _rotational_energy;

    std::optional<float> _vibrational_energy;

    std::optional<int> _rotational_dof;

    std::optional<int> _vibrational_dof;

    std::optional<float> _rotational_temperature;

    std::optional<float> _characteristic_vibrational_temperature;

    std::optional<int> _max_vibrational_quantum;

    std::optional<float> _gamma_quant;

    std::optional<int> _interaction_id;

    std::optional<bool> _fully_ionized;

    std::optional<bool> _polyatomic_molecule;

    std::optional<int> _species_id;

    std::optional<float> _reference_diameter;

    std::optional<float> _reference_temperature;

    std::optional<float> _viscosity_index;

    std::optional<float> _scattering_parameter;

    std::optional<float> _rotational_relaxation_probability;

    std::optional<float> _vibrational_relaxation_probability;

    std::optional<float> _rotational_relaxation_c1;

    std::optional<float> _rotational_relaxation_c2;

    std::optional<float> _rotational_relaxation_c3;

    std::optional<float> _vibrational_relaxation_c1;

    std::optional<float> _vibrational_relaxation_c2;

    std::optional<float> _rest_density;

    std::optional<float> _pressure_coefficient;

    std::optional<float> _dynamic_viscosity;

    std::optional<float> _electronic_energy;

    std::optional<int> _charge;
};

using MaterialPropertiesHostPtr = atlas::host_shared_ptr<MaterialProperties>;

using MaterialPropertiesDevicePtr = atlas::device_shared_ptr<MaterialProperties>;

}
