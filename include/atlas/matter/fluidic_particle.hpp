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
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_molecular_mass(T mass) {
    // Mandatory property setter: molecular mass.
    //
    // Physical meaning:
    // - Molecular mass is the particle's per-molecule mass (or per "simulated molecule"
    //   depending on your DSMC interpretation).
    //
    // Why validate here (early) instead of only in validate()?
    // - Fails fast at the call site, making bugs easier to locate.
    //
    // Policy choice:
    // - We require mass > 0.
    // - If your code uses sentinel values or allows 0 for "unset", change this rule.
    if (!(mass > T(0))) {
        throw std::invalid_argument(
            "FluidicParticle::Builder: molecular_mass must be > 0.");
    }

    // Store into optional backing field.
    // - Using std::optional allows us to distinguish "unset" vs "set".
    _molecular_mass = mass;

    // Fluent API: return self so calls can be chained.
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_statistical_weight(T w) {
    // Optional DSMC parameter: statistical weight.
    //
    // Interpretation:
    // - One simulated particle may represent N real molecules.
    // - This can be used to control resolution / variance.
    //
    // Note:
    // - This setter does not enforce constraints (e.g., w > 0) in this minimal version.
    // - If you want stricter safety, add a check similar to molecular_mass:
    //     if (!(w > 0)) throw ...
    _statistical_weight = w;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_translational_energy(T e) {
    // Optional energy mode: translational energy.
    //
    // Typical meaning:
    // - Kinetic energy associated with center-of-mass motion.
    //
    // Constraints:
    // - Often expected to be >= 0.
    // - Not enforced here to keep the builder lightweight; enforce in validate()
    //   if you want consistent invariants.
    _translational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_rotational_energy(T e) {
    // Optional energy mode: rotational energy.
    //
    // Typical meaning:
    // - Energy stored in rotational degrees of freedom (diatomic/polyatomic gases).
    _rotational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_vibrational_energy(T e) {
    // Optional energy mode: vibrational energy.
    //
    // Typical meaning:
    // - Vibrational excitation energy (relevant at higher temperatures).
    _vibrational_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_species_id(int id) {
    // Optional identifier: species id.
    //
    // Typical usage:
    // - Index into a species table (mass, collision params, internal modes, etc.).
    //
    // Policy:
    // - Not validated here (e.g., id >= 0) because valid ranges are
    //   often application-defined.
    _species_id = id;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_collision_diameter(T d_ref) {
    // Optional collision model parameter: reference collision diameter.
    //
    // Used by:
    // - VHS/VSS hard-sphere style collision models.
    //
    // Expected constraints:
    // - Typically d_ref > 0.
    // - Not enforced here (keep minimal); enforce in validate() if desired.
    _collision_diameter = d_ref;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_viscosity_index(T omega) {
    // Optional collision model parameter: viscosity index (omega).
    //
    // Used by:
    // - VHS/VSS to define temperature dependence of cross section / viscosity.
    //
    // Expected constraints:
    // - Often omega >= 0 (model-dependent).
    _viscosity_index = omega;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_scattering_parameter(T alpha) {
    // Optional collision model parameter: scattering parameter (alpha).
    //
    // Used by:
    // - VSS to shape the angular scattering distribution.
    //
    // Model consistency note:
    // - Some codes require alpha only when VSS is enabled.
    // - You can enforce cross-field constraints in validate().
    _scattering_parameter = alpha;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_electronic_energy(T e) {
    // Optional energy mode: electronic energy.
    //
    // Typical meaning:
    // - Electronic excitation energy.
    // - Often relevant for plasma / high-temperature flows.
    _electronic_energy = e;
    return *this;
}

template <typename T>
typename FluidicParticle<T>::Builder&
FluidicParticle<T>::Builder::with_charge(int q) {
    // Optional property: charge state.
    //
    // Interpretation:
    // - Could represent elementary charges (e.g., -1, 0, +1),
    //   or an application-specific charge unit.
    //
    // Not validated:
    // - Some applications allow wide ranges; validation is domain-specific.
    _charge = q;
    return *this;
}

template <typename T>
void
FluidicParticle<T>::Builder::validate_or_throw() const {
    // Centralized validation hook.
    //
    // Why validate in one place?
    // - Ensures build(), make_host_shared(), make_device_*() etc. all share the same rules.
    // - Lets setters remain cheap and minimal while still guaranteeing invariants at build time.

    // 1) Required field check: molecular_mass must be provided.
    //    - Using optional lets us distinguish "never set" from "set to some value".
    if (!_molecular_mass.has_value()) {
        throw std::invalid_argument(
            "FluidicParticle::Builder: molecular_mass is required.");
    }

    // 2) Optional consistency checks (examples).
    //
    // Keep them commented unless you want strict invariants:
    //
    //   if (_statistical_weight && !(*_statistical_weight > T(0))) {
    //       throw std::invalid_argument("... statistical_weight must be > 0");
    //   }
    //
    //   if (_collision_diameter && !(*_collision_diameter > T(0))) { ... }
    //
    // Cross-field consistency example (VSS):
    //   if (_scattering_parameter && !_viscosity_index) { ... }
    //
    // In this minimal version, we only enforce the required presence of molecular_mass.
}

template <typename T>
FluidicParticle<T>
FluidicParticle<T>::Builder::build() const {
    // Construct a fully-formed FluidicParticle<T> from the builder state.
    //
    // Strong guarantee:
    // - If validate() throws, no object is returned.
    // - After validation passes, we only do plain assignments.

    // Ensure required fields exist and builder state is coherent.
    validate_or_throw();

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

} // namespace atlas::system
