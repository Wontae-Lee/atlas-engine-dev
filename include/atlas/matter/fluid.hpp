// <atlas/matter/fluid.hpp>
#pragma once

#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

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
    return Builder {};
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
Fluid<T>::size() const noexcept {
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
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
Fluid<T>::empty() const noexcept {
    // Convenience wrapper.
    //
    // Semantically equivalent to:
    //   return _particles.empty();
    //
    // Implemented via size() to keep logic centralized.
    return size() == 0;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<FluidicParticleHostPtr<T>>&
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
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<T>&
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

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<GeneratorHostPtr<T>>&
Fluid<T>::generators() const noexcept {
    return _generators;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<GeneratorHostPtr<T>>&
Fluid<T>::generators() noexcept {
    return _generators;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
Fluid<T>::normalized(const T eps) const noexcept {
    if (_amounts.empty()) return false;

    T sum = T(0);
    for (const T amount : _amounts) {
        if (!std::isfinite(amount) || amount < T(0)) return false;
        sum += amount;
    }

    return std::abs(sum - T(1)) <= eps;
}

/* =========================
 * Fluid<T>::Builder
 * ========================= */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    Fluid<T>
    Fluid<T>::Builder::build() const {
    // Construct a Fluid<T> after validation.
    //
    // Strong exception guarantee:
    // - If validation fails, no Fluid is produced.
    validate();

    Fluid<T> f {};

    // Shallow copies:
    // - Shared pointers are copied (cheap).
    // - Amount buffer is copied by value.
    f._particles = _particles;
    f._amounts   = _amounts;
    f._generators = _generators;

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

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(const FluidicParticle<T>& p) {
    return add_particle(p, T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(const FluidicParticle<T>& p, T amount) {
    return add_particle(p, amount, nullptr);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(const FluidicParticle<T>& p,
                                    T amount,
                                    GeneratorHostPtr<T> generator) {
    _particles.push_back(atlas::make_host_shared<FluidicParticle<T>>(p));
    _amounts.push_back(amount);
    _generators.push_back(std::move(generator));
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(FluidicParticleHostPtr<T> p) {
    return add_particle(std::move(p), T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(FluidicParticleHostPtr<T> p, T amount) {
    return add_particle(std::move(p), amount, nullptr);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particle(FluidicParticleHostPtr<T> p,
                                    T amount,
                                    GeneratorHostPtr<T> generator) {
    _particles.push_back(std::move(p));
    _amounts.push_back(amount);
    _generators.push_back(std::move(generator));
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particles(const HostBuffer<FluidicParticle<T>>& ps) {
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
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_particles(ps, amounts, generators);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particles(
        const HostBuffer<FluidicParticle<T>>& ps,
        const HostBuffer<T>& amounts,
        const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(amounts.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_particles(values, amounts, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], amounts[i], generators[i]);
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particles(
        const HostBuffer<FluidicParticleHostPtr<T>>& ps) {
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
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_particles(ps, amounts, generators);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_particles(
        const HostBuffer<FluidicParticleHostPtr<T>>& ps,
        const HostBuffer<T>& amounts,
        const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(amounts.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_particles(ptrs, amounts, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_particle(ps[i], amounts[i], generators[i]);
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::require_non_empty(bool on) noexcept {
    _require_non_empty = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::reject_null_particles(bool on) noexcept {
    _reject_null_particles = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::reject_negative_amounts(bool on) noexcept {
    _reject_negative_amounts = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::require_normalized_amounts(bool on) noexcept {
    _require_normalized_amounts = on;
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE void
Fluid<T>::Builder::validate() const {
    if (_particles.size() != _amounts.size() || _particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/amounts/generators size mismatch.");
    }

    if (_require_non_empty && _particles.size() == 0) {
        throw std::runtime_error(
            "Fluid::Builder: fluid must contain at least one particle.");
    }

    if (_reject_null_particles) {
        const int n = static_cast<int>(_particles.size());
        for (int i = 0; i < n; ++i) {
            if (!_particles[i]) {
                throw std::runtime_error(
                    "Fluid::Builder: null particle pointer encountered.");
            }
        }
    }

    if (_reject_negative_amounts) {
        const int n = static_cast<int>(_amounts.size());
        for (int i = 0; i < n; ++i) {
            if (_amounts[i] < T(0)) {
                throw std::runtime_error(
                    "Fluid::Builder: negative amount encountered.");
            }
        }
    }

    if (_require_normalized_amounts) {
        if (_amounts.empty()) {
            throw std::runtime_error(
                "Fluid::Builder: normalized amounts require a non-empty fluid.");
        }

        T sum = T(0);
        const int n = static_cast<int>(_amounts.size());
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(_amounts[i]) || _amounts[i] < T(0)) {
                throw std::runtime_error(
                    "Fluid::Builder: normalized amounts must be finite and non-negative.");
            }
            sum += _amounts[i];
        }

        if (std::abs(sum - T(1)) > static_cast<T>(1e-4)) {
            throw std::runtime_error(
                "Fluid::Builder: normalized amounts must sum to 1.");
        }
    }
}

} // namespace atlas::system
