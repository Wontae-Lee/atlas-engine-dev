#pragma once

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename FluidicParticle<T>::Builder
FluidicParticle<T>::builder() noexcept {

    return Builder {};
}

template <typename T>
FluidicParticle<T>
FluidicParticle<T>::Builder::build() const {

    validate();

    FluidicParticle<T> p {};

    p.molecular_mass = *_molecular_mass;

    p.statistical_weight = _statistical_weight;

    p.translational_energy = _translational_energy;
    p.rotational_energy    = _rotational_energy;
    p.vibrational_energy   = _vibrational_energy;

    p.species_id = _species_id;

    p.collision_diameter   = _collision_diameter;
    p.viscosity_index      = _viscosity_index;
    p.scattering_parameter = _scattering_parameter;

    p.electronic_energy = _electronic_energy;
    p.charge            = _charge;

    return p;
}

template <typename T>
atlas::host_shared_ptr<FluidicParticle<T>>
FluidicParticle<T>::Builder::make_host_shared() const {

    auto p = build();
    return atlas::make_host_shared<FluidicParticle<T>>(std::move(p));
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_molecular_mass(T mass) {
    if (!(mass > T(0))) {
        throw std::invalid_argument(
            "FluidicParticle::Builder: molecular_mass must be > 0.");
    }
    _molecular_mass = mass;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_statistical_weight(T w) {
    _statistical_weight = w;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_translational_energy(T e) {
    _translational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_rotational_energy(T e) {
    _rotational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_vibrational_energy(T e) {
    _vibrational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_species_id(int id) {
    _species_id = id;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_collision_diameter(T d_ref) {
    _collision_diameter = d_ref;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_viscosity_index(T omega) {
    _viscosity_index = omega;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_scattering_parameter(T alpha) {
    _scattering_parameter = alpha;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_electronic_energy(T e) {
    _electronic_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_charge(int q) {
    _charge = q;
    return *this;
}

template <typename T>
void
FluidicParticle<T>::Builder::validate() const {
    if (!_molecular_mass.has_value()) {
        throw std::invalid_argument(
            "FluidicParticle::Builder: molecular_mass is required.");
    }
}

}
