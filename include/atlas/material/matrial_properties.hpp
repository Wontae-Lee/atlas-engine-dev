#pragma once

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
typename MatrialProperties<T>::Builder
MatrialProperties<T>::builder() noexcept {
    // Return a fresh builder object for staged construction of
    // `MatrialProperties<T>`.
    //
    // The builder path is useful when material parameters are supplied
    // progressively and validated before the final object is created.
    return Builder {};
}

template <typename T>
MatrialProperties<T>
MatrialProperties<T>::Builder::build() const {
    // Validate the staged builder state before constructing
    // the final material-properties object.
    validate();

    // Start from a default-constructed material-properties instance.
    MatrialProperties<T> p {};

    // Use the explicitly configured material type when present.
    // Otherwise fall back to `MaterialType::Molecule` as the default.
    p.type = _type.value_or(MaterialType::Molecule);

    // Mass or molecular mass and statistical weight must be provided
    if (!_mass.has_value() && (_molecular_mass.has_value() && _statistical_weight.has_value())) {
        p.mass = *_molecular_mass * *_statistical_weight;
    } else {
        p.mass = *_mass;
    }

    // Copy the molecular mass when provided.
    p.molecular_mass = _molecular_mass;

    // Copy statistical or weighting metadata.
    p.statistical_weight = _statistical_weight;

    // Copy energy-related material properties.
    p.translational_energy = _translational_energy;
    p.rotational_energy    = _rotational_energy;
    p.vibrational_energy   = _vibrational_energy;

    // Copy species classification identifier.
    p.species_id = _species_id;

    // Copy transport / collision / continuum-style parameters.
    p.collision_diameter   = _collision_diameter;
    p.viscosity_index      = _viscosity_index;
    p.scattering_parameter = _scattering_parameter;
    p.rest_density         = _rest_density;
    p.pressure_coefficient = _pressure_coefficient;
    p.dynamic_viscosity    = _dynamic_viscosity;
    p.smoothing_length     = _smoothing_length;

    // Copy electro-physical properties.
    p.electronic_energy = _electronic_energy;
    p.charge            = _charge;

    // Return the fully materialized properties object.
    return p;
}

template <typename T>
atlas::host_shared_ptr<MatrialProperties<T>>
MatrialProperties<T>::Builder::make_host_shared() const {
    // Build the validated material-properties object by value first.
    auto p = build();

    // Move the built object into host-shared managed storage.
    return atlas::make_host_shared<MatrialProperties<T>>(std::move(p));
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_type(const MaterialType::Value t) {
    // Store the material type in the builder's staged state.
    _type = t;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_mass(T m) {
    // Reject non-positive mass immediately because mass is a required
    // physical parameter and must be strictly greater than zero.
    if (!(m > T(0))) {
        throw std::invalid_argument(
            "MatrialProperties::Builder: mass must be > 0.");
    }

    // Store the validated mass in the builder.
    _mass = m;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_molecular_mass(T m) {
    // Reject non-positive molecular mass immediately because it must be
    // physically meaningful when provided.
    if (!(m > T(0))) {
        throw std::invalid_argument(
            "MatrialProperties::Builder: molecular_mass must be > 0.");
    }

    // Store the validated molecular mass in the builder.
    _molecular_mass = m;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_statistical_weight(T w) {
    // Store the statistical weight used by the material model.
    _statistical_weight = w;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_translational_energy(T e) {
    // Store translational energy metadata.
    _translational_energy = e;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_rotational_energy(T e) {
    // Store rotational energy metadata.
    _rotational_energy = e;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_vibrational_energy(T e) {
    // Store vibrational energy metadata.
    _vibrational_energy = e;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_species_id(int id) {
    // Store the species identifier used to classify this material.
    _species_id = id;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_collision_diameter(T d_ref) {
    // Store the reference collision diameter.
    _collision_diameter = d_ref;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_viscosity_index(T omega) {
    // Store the viscosity index or related transport exponent.
    _viscosity_index = omega;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_scattering_parameter(T alpha) {
    // Store the scattering parameter used by the interaction model.
    _scattering_parameter = alpha;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_rest_density(T rho0) {
    // Store the reference or rest density.
    _rest_density = rho0;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_pressure_coefficient(T k) {
    // Store the pressure-law coefficient.
    _pressure_coefficient = k;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_dynamic_viscosity(T mu) {
    // Store the dynamic viscosity.
    _dynamic_viscosity = mu;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_smoothing_length(T h) {
    // Store the smoothing length used by kernel- or particle-based models.
    _smoothing_length = h;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_electronic_energy(T e) {
    // Store electronic energy metadata.
    _electronic_energy = e;
    return *this;
}

template <typename T>
typename MatrialProperties<T>::Builder&
MatrialProperties<T>::Builder::with_charge(int q) {
    // Store the material or particle charge state.
    _charge = q;
    return *this;
}

template <typename T>
void
MatrialProperties<T>::Builder::validate() const {
    // Mass is the only mandatory parameter in the current builder contract.
    //
    // Without it, the material-properties object is considered incomplete.
    if ((!_mass.has_value()) || (!_molecular_mass.has_value() && !_statistical_weight.has_value())) {
        throw std::invalid_argument(
            "MatrialProperties::Builder: either mass or molecular_mass and statistical_weight must be provided.");
    }

    if (_molecular_mass.has_value() && _statistical_weight.has_value()
        && (*_molecular_mass * *_statistical_weight != *_mass)) {
        throw std::invalid_argument(
            "MatrialProperties::Builder: mass must equal molecular_mass * statistical_weight.");
    }
}

} // namespace atlas::system
