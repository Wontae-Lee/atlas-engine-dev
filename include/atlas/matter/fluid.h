#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/generator/generator.h>
#include <atlas/matter/fluidic_particle.h>
#include <atlas/memory/memory.h>

namespace atlas::system {

template <typename T>
class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Fluid() = default;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<FluidicParticleHostPtr<T>>&
    particles() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<FluidicParticleHostPtr<T>>&
    particles() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<T>&
    mole_fractions() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<T>&
    mole_fractions() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<GeneratorHostPtr<T>>&
    generators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE HostBuffer<GeneratorHostPtr<T>>&
    generators() noexcept;

private:
    friend class Builder;

private:
    HostBuffer<FluidicParticleHostPtr<T>> _particles;

    HostBuffer<T> _mole_fractions;

    HostBuffer<GeneratorHostPtr<T>> _generators;
};

template <typename T>
class Fluid<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const FluidicParticle<T>& p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const FluidicParticle<T>& p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const FluidicParticle<T>& p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(FluidicParticleHostPtr<T> p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticle<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<FluidicParticleHostPtr<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<FluidicParticleHostPtr<T>> _particles;

    HostBuffer<T> _mole_fractions;

    HostBuffer<GeneratorHostPtr<T>> _generators;

    bool _require_non_empty = false;

    bool _reject_null_particles = true;
};

}

namespace atlas {

template <typename T>
using Fluid = system::Fluid<T>;

template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

}

#include <atlas/matter/fluid.hpp>