#include <atlas/material/material_properties.h>

#include <stdexcept>
#include <utility>

namespace atlas {

MaterialProperties::Builder
MaterialProperties::builder() noexcept {
    return Builder {};
}

MaterialProperties
MaterialProperties::Builder::build() const {
    validate();
    MaterialProperties p {};
    p.type                                   = _type.value_or(MaterialType::molecule);
    p.mass                                   = *_mass;
    p.molecular_mass                         = _molecular_mass;
    p.translational_energy                   = _translational_energy;
    p.rotational_energy                      = _rotational_energy;
    p.vibrational_energy                     = _vibrational_energy;
    p.rotational_dof                         = _rotational_dof;
    p.vibrational_dof                        = _vibrational_dof;
    p.rotational_temperature                 = _rotational_temperature;
    p.characteristic_vibrational_temperature = _characteristic_vibrational_temperature;
    p.max_vibrational_quantum                = _max_vibrational_quantum;
    p.gamma_quant                            = _gamma_quant;
    p.interaction_id                         = _interaction_id;
    p.fully_ionized                          = _fully_ionized;
    p.polyatomic_molecule                    = _polyatomic_molecule;
    p.species_id                             = _species_id;
    p.reference_diameter                     = _reference_diameter;
    p.reference_temperature                  = _reference_temperature;
    p.viscosity_index                        = _viscosity_index;
    p.scattering_parameter                   = _scattering_parameter;
    p.rotational_relaxation_probability      = _rotational_relaxation_probability;
    p.vibrational_relaxation_probability     = _vibrational_relaxation_probability;
    p.rotational_relaxation_c1               = _rotational_relaxation_c1;
    p.rotational_relaxation_c2               = _rotational_relaxation_c2;
    p.rotational_relaxation_c3               = _rotational_relaxation_c3;
    p.vibrational_relaxation_c1              = _vibrational_relaxation_c1;
    p.vibrational_relaxation_c2              = _vibrational_relaxation_c2;
    p.rest_density                           = _rest_density;
    p.pressure_coefficient                   = _pressure_coefficient;
    p.dynamic_viscosity                      = _dynamic_viscosity;
    p.electronic_energy                      = _electronic_energy;
    p.charge                                 = _charge;
    return p;
}

atlas::host_shared_ptr<MaterialProperties>
MaterialProperties::Builder::make_host_shared() const {
    auto p = build();
    return atlas::make_host_shared<MaterialProperties>(std::move(p));
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_type(const MaterialType::Value t) {
    _type = t;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_mass(const float m) {
    if (!(m > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: mass must be > 0.");
    }
    _mass = m;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_molecular_mass(const float m) {
    if (!(m > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: molecular_mass must be > 0.");
    }
    _molecular_mass = m;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_translational_energy(const float e) {
    _translational_energy = e;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rotational_energy(const float e) {
    _rotational_energy = e;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_vibrational_energy(const float e) {
    _vibrational_energy = e;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rotational_dof(const int dof) {
    if (dof != 0 && dof != 2 && dof != 3) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_dof must be 0, 2, or 3.");
    }
    _rotational_dof = dof;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_vibrational_dof(const int dof) {
    if (dof < 0 || dof % 2 != 0) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_dof must be non-negative and even.");
    }
    _vibrational_dof = dof;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rotational_temperature(const float temperature) {
    if (!(temperature > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_temperature must be > 0.");
    }
    _rotational_temperature = temperature;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_characteristic_vibrational_temperature(const float temperature) {
    if (!(temperature > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: characteristic_vibrational_temperature must be > 0.");
    }
    _characteristic_vibrational_temperature = temperature;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_max_vibrational_quantum(const int quantum) {
    _max_vibrational_quantum = quantum;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_gamma_quant(const float gamma) {
    _gamma_quant = gamma;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_interaction_id(const int id) {
    _interaction_id = id;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_fully_ionized(const bool value) {
    _fully_ionized = value;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_polyatomic_molecule(const bool value) {
    _polyatomic_molecule = value;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_species_id(const int id) {
    _species_id = id;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_reference_diameter(const float d_ref) {
    _reference_diameter = d_ref;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_reference_temperature(const float t_ref) {
    _reference_temperature = t_ref;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_viscosity_index(const float omega) {
    _viscosity_index = omega;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_scattering_parameter(const float alpha) {
    _scattering_parameter = alpha;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rotational_relaxation_probability(const float probability) {
    if (probability < 0.0f || probability > 1.0f) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_relaxation_probability must be in [0, 1].");
    }
    _rotational_relaxation_probability = probability;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_vibrational_relaxation_probability(const float probability) {
    if (probability < 0.0f || probability > 1.0f) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_relaxation_probability must be in [0, 1].");
    }
    _vibrational_relaxation_probability = probability;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rotational_relaxation_coefficients(const float c1, const float c2, const float c3) {
    if (!(c1 > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_relaxation_c1 must be > 0.");
    }
    _rotational_relaxation_c1 = c1;
    _rotational_relaxation_c2 = c2;
    _rotational_relaxation_c3 = c3;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_vibrational_relaxation_coefficients(const float c1, const float c2) {
    if (!(c1 > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_relaxation_c1 must be > 0.");
    }
    _vibrational_relaxation_c1 = c1;
    _vibrational_relaxation_c2 = c2;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_rest_density(const float rho0) {
    _rest_density = rho0;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_pressure_coefficient(const float k) {
    _pressure_coefficient = k;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_dynamic_viscosity(const float mu) {
    _dynamic_viscosity = mu;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_electronic_energy(const float e) {
    _electronic_energy = e;
    return *this;
}

MaterialProperties::Builder&
MaterialProperties::Builder::with_charge(const int q) {
    _charge = q;
    return *this;
}

void
MaterialProperties::Builder::validate() const {
    if (!_mass.has_value()) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: mass must be provided.");
    }
    if (!(_molecular_mass > 0.0f)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: molecular_mass must be provided.");
    }
}

}
