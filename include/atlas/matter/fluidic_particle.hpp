#pragma once

#include <stdexcept> // std::invalid_argument
#include <utility>   // std::move

namespace atlas::system {

/* =========================
 * FluidicParticle<T>
 * ========================= */

template <typename T>
typename FluidicParticle<T>::Builder
FluidicParticle<T>::builder() noexcept {
    // Factory for a fluent Builder.
    //
    // Why a static-ish builder()?
    // - Keeps construction readable when there are many optional fields:
    //     auto p = FluidicParticle<float>::builder()
    //                 .with_molecular_mass(...)
    //                 .with_species_id(...)
    //                 .build();
    //
    // noexcept rationale:
    // - Returning a default-constructed Builder is guaranteed not to throw.
    // - Any validation/throwing happens later in build()/validate().
    return Builder {};
}

/* =========================
 * FluidicParticle<T>::Builder
 * ========================= */

template <typename T>
FluidicParticle<T>
FluidicParticle<T>::Builder::build() const {
    // Construct a fully-formed FluidicParticle<T> from the builder state.
    //
    // Strong guarantee:
    // - If validate() throws, no object is returned.
    // - After validation passes, we only do plain assignments.

    // Ensure required fields exist and builder state is coherent.
    validate();

    // Start from value-initialized particle.
    //
    // Why value-init?
    // - Ensures fundamental fields get zero-init.
    // - Ensures std::optional fields start disengaged unless explicitly assigned.
    FluidicParticle<T> p {};

    // Required field(s):
    // - molecular_mass is mandatory; validated above so dereference is safe.
    p.molecular_mass = *_molecular_mass;

    // Optional fields:
    //
    // Assigning optionals:
    // - Copying std::optional copies engagement state and value if present.
    // - This preserves "unset" vs "set" semantics in the final object.
    p.statistical_weight = _statistical_weight;

    // Internal energy modes.
    p.translational_energy = _translational_energy;
    p.rotational_energy    = _rotational_energy;
    p.vibrational_energy   = _vibrational_energy;

    // Species / state.
    p.species_id = _species_id;

    // Collision model parameters.
    p.collision_diameter   = _collision_diameter;
    p.viscosity_index      = _viscosity_index;
    p.scattering_parameter = _scattering_parameter;

    // Advanced / optional models.
    p.electronic_energy = _electronic_energy;
    p.charge            = _charge;

    // Return by value:
    // - NRVO / move elision typically makes this cheap.
    // - Keeps ownership clear and avoids lifetime issues.
    return p;
}

template <typename T>
atlas::host_shared_ptr<FluidicParticle<T>>
FluidicParticle<T>::Builder::make_host_shared() const {
    // Convenience: produce a shared, owning host pointer.
    //
    // Pattern:
    // - Build by value (reuses validation and ensures correctness).
    // - Move into heap-allocated object managed by atlas::host_shared_ptr.
    //
    // Why std::move?
    // - Avoids copying potentially large particle payloads (if expanded in the future).
    // - Leaves the temporary 'p' in a valid but unspecified state, which is fine since we
    //   don't use it after moving.
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

} // namespace atlas::system
