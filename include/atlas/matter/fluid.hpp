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
    return Builder {};
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
Fluid<T>::size() const noexcept {
    return static_cast<int>(_particles.size());
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
Fluid<T>::empty() const noexcept {
    return size() == 0;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<FluidicParticleHostPtr<T>>&
Fluid<T>::particles() const noexcept {
    return _particles;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
    HostBuffer<FluidicParticleHostPtr<T>>&
    Fluid<T>::particles() noexcept {
    return _particles;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<T>&
Fluid<T>::mole_fractions() const noexcept {
    return _mole_fractions;
}

template <typename T>
ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE
    HostBuffer<T>&
    Fluid<T>::mole_fractions() noexcept {
    return _mole_fractions;
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

/* =========================
 * Fluid<T>::Builder
 * ========================= */

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    Fluid<T>
    Fluid<T>::Builder::build() const {
    validate();

    Fluid<T> f {};
    f._particles      = _particles;
    f._mole_fractions = _mole_fractions;
    f._generators     = _generators;

    return f;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    atlas::host_shared_ptr<Fluid<T>>
    Fluid<T>::Builder::make_host_shared() const {
    auto f = build();
    return atlas::make_host_shared<Fluid<T>>(std::move(f));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(const FluidicParticle<T>& p) {
    return add_species(p, T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(const FluidicParticle<T>& p, T mole_fraction) {
    return add_species(p, mole_fraction, nullptr);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(const FluidicParticle<T>& p,
                                   T mole_fraction,
                                   GeneratorHostPtr<T> generator) {
    _particles.push_back(atlas::make_host_shared<FluidicParticle<T>>(p));
    _mole_fractions.push_back(mole_fraction);
    _generators.push_back(std::move(generator));
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(FluidicParticleHostPtr<T> p) {
    return add_species(std::move(p), T(1));
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(FluidicParticleHostPtr<T> p, T mole_fraction) {
    return add_species(std::move(p), mole_fraction, nullptr);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species(FluidicParticleHostPtr<T> p,
                                   T mole_fraction,
                                   GeneratorHostPtr<T> generator) {
    _particles.push_back(std::move(p));
    _mole_fractions.push_back(mole_fraction);
    _generators.push_back(std::move(generator));
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps) {
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(
        const HostBuffer<FluidicParticle<T>>& ps,
        const HostBuffer<T>& mole_fractions) {
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(
        const HostBuffer<FluidicParticle<T>>& ps,
        const HostBuffer<T>& mole_fractions,
        const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(values, mole_fractions, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(
        const HostBuffer<FluidicParticleHostPtr<T>>& ps) {
    const int n = static_cast<int>(ps.size());
    for (int i = 0; i < n; ++i) {
        add_species(ps[i], T(1));
    }
    return *this;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(
        const HostBuffer<FluidicParticleHostPtr<T>>& ps,
        const HostBuffer<T>& mole_fractions) {
    const HostBuffer<GeneratorHostPtr<T>> generators(ps.size(), nullptr);
    return add_species_bulk(ps, mole_fractions, generators);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE
    typename Fluid<T>::Builder&
    Fluid<T>::Builder::add_species_bulk(
        const HostBuffer<FluidicParticleHostPtr<T>>& ps,
        const HostBuffer<T>& mole_fractions,
        const HostBuffer<GeneratorHostPtr<T>>& generators) {
    const int n = static_cast<int>(ps.size());
    if (static_cast<int>(mole_fractions.size()) != n || static_cast<int>(generators.size()) != n) {
        throw std::runtime_error(
            "Fluid::Builder::add_species_bulk(ptrs, mole_fractions, generators): size mismatch.");
    }

    for (int i = 0; i < n; ++i) {
        add_species(ps[i], mole_fractions[i], generators[i]);
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
ATLAS_HOST ATLAS_FORCE_INLINE void
Fluid<T>::Builder::validate() const {
    if (_particles.size() != _mole_fractions.size() || _particles.size() != _generators.size()) {
        throw std::runtime_error(
            "Fluid::Builder: particles/mole_fractions/generators size mismatch.");
    }

    if (_require_non_empty && _particles.size() == 0) {
        throw std::runtime_error(
            "Fluid::Builder: fluid must contain at least one species.");
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

    // Normalize mole fractions
    T sum       = T(0);
    const int n = static_cast<int>(_mole_fractions.size());
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(_mole_fractions[i]) || _mole_fractions[i] < T(0)) {
            throw std::runtime_error(
                "Fluid::Builder: mole fractions must be finite and non-negative.");
        }
        sum += _mole_fractions[i];
    }

    if (n > 0 && sum > T(0)) {
        for (int i = 0; i < n; ++i) {
            const_cast<HostBuffer<T>&>(_mole_fractions)[i] /= sum;
        }
    } else if (n > 0) {
        throw std::runtime_error(
            "Fluid::Builder: mole fractions sum to zero.");
    }
}

} // namespace atlas::system
