#pragma once
#include <stdexcept>
#include <utility>
namespace atlas {
template <typename T>
typename MaterialProperties<T>::Builder
MaterialProperties<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
MaterialProperties<T>
MaterialProperties<T>::Builder::build() const {
    validate();
    MaterialProperties<T> p {};
    p.type                 = _type.value_or(MaterialType::Molecule);
    p.mass                 = *_mass;
    p.molecular_mass       = _molecular_mass;
    p.translational_energy = _translational_energy;
    p.rotational_energy    = _rotational_energy;
    p.vibrational_energy   = _vibrational_energy;
    p.rotational_dof       = _rotational_dof;
    p.vibrational_dof      = _vibrational_dof;
    p.rotational_temperature = _rotational_temperature;
    p.characteristic_vibrational_temperature = _characteristic_vibrational_temperature;
    p.max_vibrational_quantum = _max_vibrational_quantum;
    p.gamma_quant = _gamma_quant;
    p.interaction_id = _interaction_id;
    p.fully_ionized = _fully_ionized;
    p.polyatomic_molecule = _polyatomic_molecule;
    p.species_id           = _species_id;
    p.reference_diameter    = _reference_diameter;
    p.reference_temperature = _reference_temperature;
    p.viscosity_index       = _viscosity_index;
    p.scattering_parameter  = _scattering_parameter;
    p.rotational_relaxation_probability = _rotational_relaxation_probability;
    p.vibrational_relaxation_probability = _vibrational_relaxation_probability;
    p.rotational_relaxation_c1 = _rotational_relaxation_c1;
    p.rotational_relaxation_c2 = _rotational_relaxation_c2;
    p.rotational_relaxation_c3 = _rotational_relaxation_c3;
    p.vibrational_relaxation_c1 = _vibrational_relaxation_c1;
    p.vibrational_relaxation_c2 = _vibrational_relaxation_c2;
    p.rest_density          = _rest_density;
    p.pressure_coefficient  = _pressure_coefficient;
    p.dynamic_viscosity     = _dynamic_viscosity;
    p.electronic_energy     = _electronic_energy;
    p.charge                = _charge;
    return p;
}

template <typename T>
atlas::host_shared_ptr<MaterialProperties<T>>
MaterialProperties<T>::Builder::make_host_shared() const {
    auto p = build();
    return atlas::make_host_shared<MaterialProperties<T>>(std::move(p));
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_type(const MaterialType::Value t) {
    _type = t;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_mass(T m) {
    if (!(m > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: mass must be > 0.");
    }
    _mass = m;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_molecular_mass(T m) {
    if (!(m > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: molecular_mass must be > 0.");
    }
    _molecular_mass = m;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_translational_energy(T e) {
    _translational_energy = e;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rotational_energy(T e) {
    _rotational_energy = e;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_vibrational_energy(T e) {
    _vibrational_energy = e;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rotational_dof(int dof) {
    if (dof != 0 && dof != 2 && dof != 3) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_dof must be 0, 2, or 3.");
    }
    _rotational_dof = dof;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_vibrational_dof(int dof) {
    if (dof < 0 || dof % 2 != 0) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_dof must be non-negative and even.");
    }
    _vibrational_dof = dof;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rotational_temperature(T temperature) {
    if (!(temperature > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_temperature must be > 0.");
    }
    _rotational_temperature = temperature;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_characteristic_vibrational_temperature(T temperature) {
    if (!(temperature > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: characteristic_vibrational_temperature must be > 0.");
    }
    _characteristic_vibrational_temperature = temperature;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_max_vibrational_quantum(int quantum) {
    _max_vibrational_quantum = quantum;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_gamma_quant(T gamma) {
    _gamma_quant = gamma;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_interaction_id(int id) {
    _interaction_id = id;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_fully_ionized(bool value) {
    _fully_ionized = value;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_polyatomic_molecule(bool value) {
    _polyatomic_molecule = value;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_species_id(int id) {
    _species_id = id;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_reference_diameter(T d_ref) {
    _reference_diameter = d_ref;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_reference_temperature(T t_ref) {
    _reference_temperature = t_ref;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_viscosity_index(T omega) {
    _viscosity_index = omega;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_scattering_parameter(T alpha) {
    _scattering_parameter = alpha;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rotational_relaxation_probability(T probability) {
    if (probability < T(0) || probability > T(1)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_relaxation_probability must be in [0, 1].");
    }
    _rotational_relaxation_probability = probability;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_vibrational_relaxation_probability(T probability) {
    if (probability < T(0) || probability > T(1)) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_relaxation_probability must be in [0, 1].");
    }
    _vibrational_relaxation_probability = probability;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rotational_relaxation_coefficients(T c1, T c2, T c3) {
    if (!(c1 > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: rotational_relaxation_c1 must be > 0.");
    }
    _rotational_relaxation_c1 = c1;
    _rotational_relaxation_c2 = c2;
    _rotational_relaxation_c3 = c3;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_vibrational_relaxation_coefficients(T c1, T c2) {
    if (!(c1 > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: vibrational_relaxation_c1 must be > 0.");
    }
    _vibrational_relaxation_c1 = c1;
    _vibrational_relaxation_c2 = c2;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_rest_density(T rho0) {
    _rest_density = rho0;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_pressure_coefficient(T k) {
    _pressure_coefficient = k;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_dynamic_viscosity(T mu) {
    _dynamic_viscosity = mu;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_electronic_energy(T e) {
    _electronic_energy = e;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_charge(int q) {
    _charge = q;
    return *this;
}

template <typename T>
void
MaterialProperties<T>::Builder::validate() const {
    if (!_mass.has_value()) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: mass must be provided.");
    }
    if (!(_molecular_mass > T(0))) {
        throw std::invalid_argument(
            "MaterialProperties::Builder: molecular_mass must be provided.");
    }
}

}
