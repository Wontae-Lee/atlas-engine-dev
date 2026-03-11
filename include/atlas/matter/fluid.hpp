// <atlas/matter/fluid.hpp>
#pragma once

#include <utility>      // std::move
#include <stdexcept>    // std::runtime_error

namespace atlas::system {

/* =========================
 * Fluid<T>
 * ========================= */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder
Fluid<T>::builder() noexcept {
    // Entry point for constructing a Fluid<T> via the Builder pattern.
    //
    // Design rationale:
    // - Fluids often have non-trivial construction rules
    //   (e.g. non-empty, matching particle/amount buffers).
    // - The builder centralizes these rules and keeps Fluid<T> itself simple.
    //
    // noexcept:
    // - Returning a default-constructed Builder is guaranteed not to throw.
    return Builder{};
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
int Fluid<T>::size() const noexcept {
    // Number of particle entries stored in this fluid.
    //
    // Important:
    // - Each entry corresponds to one particle *type* or instance,
    //   not necessarily one physical molecule.
    // - The actual quantity is tracked separately in `_amounts`.
    //
    // Note on return type:
    // - Internally size() is size_t; we expose int for convenience
    //   in tight simulation loops.
    return static_cast<int>(_particles.size());
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
bool Fluid<T>::empty() const noexcept {
    // Convenience wrapper.
    //
    // Semantically equivalent to:
    //   return _particles.empty();
    //
    // Implemented via size() to keep logic centralized.
    return size() == 0;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
const HostBuffer<FluidicParticleHostPtr<T>>&
Fluid<T>::particles() const noexcept {
    // Read-only access to particle definitions.
    //
    // Ownership model:
    // - Each entry is a shared pointer to a FluidicParticle<T>.
    // - Copying this buffer or its elements is cheap (shared ownership).
    //
    // Intended usage:
    // - Inspection, iteration, read-only physics kernels.
    return _particles;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
HostBuffer<FluidicParticleHostPtr<T>>&
Fluid<T>::particles() noexcept {
    // Mutable access to particle definitions.
    //
    // Caution:
    // - Direct mutation can break invariants assumed by solvers
    //   (e.g. particle/amount alignment).
    // - Prefer using the Builder for construction-time modifications.
    return _particles;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
const HostBuffer<T>&
Fluid<T>::amounts() const noexcept {
    // Read-only access to per-particle amounts.
    //
    // Interpretation:
    // - `_amounts[i]` corresponds to `_particles[i]`.
    // - Represents quantity, weight, or number density multiplier,
    //   depending on the physical model.
    return _amounts;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
HostBuffer<T>&
Fluid<T>::amounts() noexcept {
    // Mutable access to per-particle amounts.
    //
    // Same caution as particles():
    // - Mutating directly may break consistency.
    return _amounts;
}

/* =========================
 * Fluid<T>::Builder
 * ========================= */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particle(const FluidicParticle<T>& p) {
    // Add a particle definition with a default amount of 1.
    //
    // Delegates to the (particle, amount) overload to keep logic unified.
    return add_particle(p, T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particle(const FluidicParticle<T>& p, T amount) {
    // Add a particle by value, with an explicit amount.
    //
    // Semantics:
    // - The particle is copied into a heap-allocated FluidicParticle<T>
    //   managed by a host_shared_ptr.
    // - The amount is stored in a parallel buffer.
    //
    // Design invariant:
    // - `_particles.size()` must always match `_amounts.size()`.
    _particles.push_back(
        atlas::make_host_shared<FluidicParticle<T>>(p));
    _amounts.push_back(amount);
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particle(FluidicParticleHostPtr<T> p) {
    // Add a particle via shared pointer with default amount = 1.
    //
    // Delegates to pointer+amount overload.
    return add_particle(std::move(p), T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particle(FluidicParticleHostPtr<T> p, T amount) {
    // Add a particle via shared pointer with an explicit amount.
    //
    // Semantics:
    // - Shared ownership of the particle is transferred/copied into the fluid.
    // - No deep copy of the particle occurs.
    //
    // Note:
    // - Null pointer handling is deferred to validate(),
    //   depending on builder policy flags.
    _particles.push_back(std::move(p));
    _amounts.push_back(amount);
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particles(const HostBuffer<FluidicParticle<T>>& ps) {
    // Bulk-add particles by value with default amount = 1.
    //
    // Complexity:
    // - O(n) allocations, where n = ps.size().
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], T(1));
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particles(
    const HostBuffer<FluidicParticle<T>>& ps,
    const HostBuffer<T>& amounts) {
    // Bulk-add particles by value with explicit amounts.
    //
    // Strong requirement:
    // - Particle buffer and amount buffer must have identical sizes.
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(amounts.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_particles(values, amounts): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], amounts[i]);
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particles(
    const HostBuffer<FluidicParticleHostPtr<T>>& ps) {
    // Bulk-add particles via shared pointers with default amount = 1.
    //
    // Complexity:
    // - O(n) pointer copies, no allocations.
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], T(1));
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::add_particles(
    const HostBuffer<FluidicParticleHostPtr<T>>& ps,
    const HostBuffer<T>& amounts) {
    // Bulk-add shared particles with explicit amounts.
    //
    // Size consistency is mandatory.
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(amounts.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_particles(ptrs, amounts): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], amounts[i]);
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::require_non_empty(bool on) noexcept {
    // Configure whether the built Fluid<T> must be non-empty.
    //
    // Typical use:
    // - Solvers that assume at least one species/particle.
    _require_non_empty = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::reject_null_particles(bool on) noexcept {
    // Configure whether null particle pointers are rejected at build time.
    //
    // When enabled:
    // - validate() will scan the particle buffer
    //   and reject any null shared_ptr.
    _reject_null_particles = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
typename Fluid<T>::Builder&
Fluid<T>::Builder::reject_negative_amounts(bool on) noexcept {
    // Configure whether negative amounts are rejected at build time.
    //
    // Physical rationale:
    // - Negative particle amounts are almost always nonsensical.
    _reject_negative_amounts = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
void Fluid<T>::Builder::validate_or_throw() const {
    // Centralized validation of builder invariants.
    //
    // This function enforces:
    // - Structural consistency (buffer sizes)
    // - Optional semantic constraints controlled by flags.

    // 1) Structural invariant:
    //    - Particles and amounts must be 1:1 aligned.
    if (_particles.size() != _amounts.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/amounts size mismatch.");
    }

    // 2) Non-empty requirement.
    if (_require_non_empty && _particles.size() == 0) {
        throw std::runtime_error(
            "Fluid::Builder: fluid must contain at least one particle.");
    }

    // 3) Null particle rejection (optional).
    if (_reject_null_particles) {
        const int n = static_cast<int>(_particles.size());
        for (int i = 0; i < n; ++i) {
            if (!_particles[i]) {
                throw std::runtime_error(
                    "Fluid::Builder: null particle pointer encountered.");
            }
        }
    }

    // 4) Negative amount rejection (optional).
    if (_reject_negative_amounts) {
        const int n = static_cast<int>(_amounts.size());
        for (int i = 0; i < n; ++i) {
            if (_amounts[i] < T(0)) {
                throw std::runtime_error(
                    "Fluid::Builder: negative amount encountered.");
            }
        }
    }
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
Fluid<T> Fluid<T>::Builder::build() const {
    // Construct a Fluid<T> after validation.
    //
    // Strong exception guarantee:
    // - If validation fails, no Fluid is produced.
    validate_or_throw();

    Fluid<T> f{};

    // Shallow copies:
    // - Shared pointers are copied (cheap).
    // - Amount buffer is copied by value.
    f._particles = _particles;
    f._amounts   = _amounts;

    return f;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
atlas::host_shared_ptr<Fluid<T>>
Fluid<T>::Builder::make_host_shared() const {
    // Convenience helper:
    // - Build a Fluid<T>
    // - Move it into a shared, heap-allocated object.
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

} // namespace atlas::system
