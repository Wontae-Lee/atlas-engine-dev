#pragma once
#include <stdexcept>
#include <utility>
namespace atlas::system {
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
    p.species_id           = _species_id;
    p.collision_diameter   = _collision_diameter;
    p.viscosity_index      = _viscosity_index;
    p.scattering_parameter = _scattering_parameter;
    p.rest_density         = _rest_density;
    p.pressure_coefficient = _pressure_coefficient;
    p.dynamic_viscosity    = _dynamic_viscosity;
    p.smoothing_length     = _smoothing_length;
    p.electronic_energy    = _electronic_energy;
    p.charge               = _charge;
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
MaterialProperties<T>::Builder::with_species_id(int id) {
    _species_id = id;
    return *this;
}

template <typename T>
typename MaterialProperties<T>::Builder&
MaterialProperties<T>::Builder::with_collision_diameter(T d_ref) {
    _collision_diameter = d_ref;
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
MaterialProperties<T>::Builder::with_smoothing_length(T h) {
    _smoothing_length = h;
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