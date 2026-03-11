#pragma once

#include <atlas/logging/logging.h>

namespace atlas::system {

template <typename T>
Matter<T>::Matter(T mass_)
    : mass(mass_) {
    // Construct a "Matter" base object with an explicit mass.
    //
    // This class appears to act as a lightweight physical-property base for
    // derived types (e.g., Element<T>, Compound<T>, Mixture<T>) by providing a
    // common `mass` field and a consistent validation rule.
    //
    // Why validate here?
    // - Many downstream computations (momentum, kinetic energy, weighting, etc.)
    //   implicitly assume mass > 0.
    // - Enforcing this invariant at construction time prevents subtle bugs
    //   from propagating into solver kernels or thermodynamic routines.
    atlas::check<std::invalid_argument>(mass > T(0))
        << "Matter: mass must be positive.";
}

template <typename T>
typename Matter<T>::Builder
Matter<T>::builder() noexcept {
    // Builder entry point.
    //
    // Returns a default-initialized Builder that can be configured fluently:
    //   auto m = Matter<float>::builder().with_mass(1.0f).build();
    //
    // This pattern keeps the main type minimal while supporting readable,
    // validated configuration.
    return Builder {};
}

// ============================================================
// Builder
// ============================================================

template <typename T>
typename Matter<T>::Builder&
Matter<T>::Builder::with_mass(T mass) noexcept {
    // Set the mass to be used when constructing Matter<T>.
    //
    // Notes:
    // - This function does not validate immediately to keep the fluent
    //   interface lightweight and to allow the user to set multiple fields
    //   before a single validation point.
    // - Validation happens in validate() and is called by build()
    //   and make_host_shared().
    _mass = mass;
    return *this;
}

template <typename T>
void
Matter<T>::Builder::validate_or_throw() const {
    // Validate builder state before constructing the object.
    //
    // Ensures the same invariant enforced by the Matter constructor:
    // - mass must be strictly positive.
    //
    // Centralizing validation here gives:
    // - consistent error messages for all build paths,
    // - the ability to extend validation later without touching call sites.
    atlas::check<std::invalid_argument>(_mass > T(0))
        << "Matter::Builder: mass must be positive.";
}

template <typename T>
Matter<T>
Matter<T>::Builder::build() const {
    // Construct a Matter<T> instance by value.
    //
    // Steps:
    //  1) Validate parameters (throws if invalid).
    //  2) Forward the validated mass into the Matter constructor.
    //
    // Returning by value enables NRVO/move elision and keeps usage simple.
    validate_or_throw();
    return Matter<T>(_mass);
}

template <typename T>
atlas::host_shared_ptr<Matter<T>>
Matter<T>::Builder::make_host_shared() const {
    // Construct a heap-allocated Matter<T> wrapped in a host_shared_ptr.
    //
    // This is useful when:
    // - the object is shared across multiple owners (systems/components),
    // - polymorphic usage requires dynamic lifetime management,
    // - you want to pass around an owning handle rather than a value type.
    //
    // Validation is performed before allocation to fail fast and avoid
    // allocating objects that would immediately be invalid.
    validate_or_throw();

    // Note:
    // This uses atlas::make_host_shared to construct the object efficiently
    // (typically a single allocation for control block + object, depending on
    // implementation), and ensures consistent ownership semantics on the host.
    return atlas::make_host_shared<Matter<T>>(_mass);
}

} // namespace atlas::system
