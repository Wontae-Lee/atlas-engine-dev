#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/generator/generator.h>
#include <atlas/logging/logging.h>
#include <atlas/material/matrial_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <cstdint>

namespace atlas::system {

template <typename T>
struct FluidDeviceProbe {
    MatrialProperties<T>* particle_property { nullptr };

    Vector3<T>* pos { nullptr };

    Vector3<T>* vel { nullptr };

    T* temperature { nullptr };

    size_t* species { nullptr };

    int* active { nullptr };

    int particle_count { 0 };

    size_t buffer_size { 0 };
};

template <typename T>
class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ~Fluid() = default;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MatrialProperties<T>>&
    particles() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MatrialProperties<T>>&
    particles() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<T>&
    mole_fractions() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<T>&
    mole_fractions() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>
    make_device_probe() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    positions() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    velocities() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<T>&
    temperatures() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<size_t>&
    species_ids() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    active() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

private:
    friend class Builder;

private:
    DeviceBuffer<MatrialProperties<T>> _particle_properties;

    DeviceBuffer<T> _mole_fractions;

    DeviceBuffer<GenerateOperator<T>> _generators;

    DeviceBuffer<Vector3<T>> d_pos;

    DeviceBuffer<Vector3<T>> d_vel;

    DeviceBuffer<T> d_temperature;

    DeviceBuffer<size_t> d_species;

    DeviceBuffer<int> d_active;

    size_t _buffer_size = 0;

    std::uint64_t _probe_count = 0;
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
    add_species(const MatrialProperties<T>& p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction, GeneratorHostPtr<T> generator);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps, const HostBuffer<T>& mole_fractions);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DeviceBuffer<MatrialProperties<T>> _particles;

    DeviceBuffer<T> _mole_fractions;

    DeviceBuffer<GenerateOperator<T>> _generators;

    size_t _buffer_size = 0;

    bool _require_non_empty = false;

    bool _reject_null_particles = true;
};

}

namespace atlas {

template <typename T>
using Fluid = system::Fluid<T>;

template <typename T>
using FluidDeviceProbe = system::FluidDeviceProbe<T>;

template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>
