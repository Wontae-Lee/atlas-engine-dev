#pragma once

/**
 * @file fluid.h
 * @brief Declares particle storage, fluid probes, and fluid builder utilities.
 *
 * Fluid owns the SoA-style particle buffers used by the simulation runtime
 * together with species properties, per-species mole fractions, and generation
 * operators. It also provides a compact device probe used by system stages and
 * backend kernels.
 */

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

/**
 * @brief Device-facing view of particle storage.
 *
 * FluidDeviceProbe carries raw pointers to the particle arrays and metadata
 * needed by kernels. The probe represents the allocated capacity together with
 * the currently active particle prefix.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct FluidDeviceProbe {
    MatrialProperties<T>* particle_property { nullptr }; ///< Per-species material properties.

    Vector3<T>* pos { nullptr }; ///< Particle positions.

    Vector3<T>* vel { nullptr }; ///< Particle velocities.

    T* temperature { nullptr }; ///< Particle temperatures.

    size_t* species { nullptr }; ///< Species index for each particle.

    int* active { nullptr }; ///< Optional active flags per particle slot.

    int particle_count { 0 }; ///< Number of active particles stored in the prefix.

    size_t buffer_size { 0 }; ///< Total allocated particle capacity.
};

/**
 * @brief Owns the particle buffers and per-species metadata for a fluid.
 *
 * Fluid stores species definitions and the particle arrays that simulation
 * stages operate on. Particle data is held in backend-native device buffers so
 * runtime systems can access it without additional marshaling.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Fluid final {
public:
    /**
     * @brief Fluent builder for Fluid.
     */
    class Builder;

    /**
     * @brief Default constructor.
     */
    Fluid() = default;

    /**
     * @brief Allocates particle buffers for a fixed capacity.
     *
     * @param buffer_size Total particle capacity to allocate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    /**
     * @brief Returns a builder initialized with default values.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Destructor.
     */
    ~Fluid() = default;

    /**
     * @brief Returns the number of configured species.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    size() const noexcept;

    /**
     * @brief Returns whether any species have been configured.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    /**
     * @brief Returns const access to per-species material properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MatrialProperties<T>>&
    particles() const noexcept;

    /**
     * @brief Returns mutable access to per-species material properties.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MatrialProperties<T>>&
    particles() noexcept;

    /**
     * @brief Returns const access to normalized mole fractions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<T>&
    mole_fractions() const noexcept;

    /**
     * @brief Returns mutable access to normalized mole fractions.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<T>&
    mole_fractions() noexcept;

    /**
     * @brief Returns const access to per-species generation operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    /**
     * @brief Returns mutable access to per-species generation operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    /**
     * @brief Creates the device probe consumed by runtime kernels.
     *
     * The system owns the single authoritative probe for a live simulation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Returns mutable access to particle positions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    positions() noexcept;

    /**
     * @brief Returns mutable access to particle velocities.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    velocities() noexcept;

    /**
     * @brief Returns mutable access to particle temperatures.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<T>&
    temperatures() noexcept;

    /**
     * @brief Returns mutable access to particle species identifiers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<size_t>&
    species_ids() noexcept;

    /**
     * @brief Returns mutable access to particle activity flags.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    active() noexcept;

    /**
     * @brief Returns the total allocated particle capacity.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

private:
    friend class Builder;

private:
    DeviceBuffer<MatrialProperties<T>> _particle_properties; ///< Per-species material definitions.

    DeviceBuffer<T> _mole_fractions; ///< Normalized per-species mole fractions.

    DeviceBuffer<GenerateOperator<T>> _generators; ///< Per-species spawning operators.

    DeviceBuffer<Vector3<T>> d_pos; ///< Particle positions.

    DeviceBuffer<Vector3<T>> d_vel; ///< Particle velocities.

    DeviceBuffer<T> d_temperature; ///< Particle temperatures.

    DeviceBuffer<size_t> d_species; ///< Particle species ids.

    DeviceBuffer<int> d_active; ///< Particle activity flags.

    size_t _buffer_size = 0; ///< Allocated particle capacity.

    std::uint64_t _probe_count = 0; ///< Tracks runtime probe creation.
};

template <typename T>
class Fluid<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a value instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    /**
     * @brief Builds a host-shared fluid instance after validation.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    /**
     * @brief Adds a species with a default mole fraction of 1.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p);

    /**
     * @brief Adds a species with an explicit mole fraction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction);

    /**
     * @brief Adds a species with an explicit mole fraction and generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction, GeneratorHostPtr<T> generator);

    /**
     * @brief Adds a species from a host-shared material pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p);

    /**
     * @brief Adds a species pointer with an explicit mole fraction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction);

    /**
     * @brief Adds a species pointer with an explicit mole fraction and generator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction, GeneratorHostPtr<T> generator);

    /**
     * @brief Adds multiple value-based species with default mole fractions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps);

    /**
     * @brief Adds multiple value-based species with explicit mole fractions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps, const HostBuffer<T>& mole_fractions);

    /**
     * @brief Adds multiple value-based species with explicit mole fractions and generators.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Adds multiple pointer-based species with default mole fractions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps);

    /**
     * @brief Adds multiple pointer-based species with explicit mole fractions.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps, const HostBuffer<T>& mole_fractions);

    /**
     * @brief Adds multiple pointer-based species with explicit mole fractions and generators.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Sets the particle capacity to allocate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    /**
     * @brief Requires at least one species before build().
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    /**
     * @brief Controls whether null material pointers are rejected.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

private:
    /**
     * @brief Validates builder state before construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    DeviceBuffer<MatrialProperties<T>> _particles; ///< Pending species definitions.

    DeviceBuffer<T> _mole_fractions; ///< Pending mole fractions.

    DeviceBuffer<GenerateOperator<T>> _generators; ///< Pending generation operators.

    size_t _buffer_size = 0; ///< Pending particle capacity.

    bool _require_non_empty = false; ///< Whether empty species lists are forbidden.

    bool _reject_null_particles = true; ///< Whether null material pointers are rejected.
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::Fluid.
 */
template <typename T>
using Fluid = system::Fluid<T>;

/**
 * @brief Convenience alias for atlas::system::FluidDeviceProbe.
 */
template <typename T>
using FluidDeviceProbe = system::FluidDeviceProbe<T>;

/**
 * @brief Host shared pointer alias for Fluid.
 */
template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

}

#include <atlas/fluid/fluid.hpp>
