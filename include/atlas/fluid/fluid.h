#pragma once

/**
 * @file fluid.h
 * @brief Declares particle storage, fluid device probes, and builder utilities for particle-based simulation state.
 *
 * @details
 * This header defines @ref atlas::system::Fluid, the runtime object responsible
 * for owning particle-resolved storage and species-level metadata used by Atlas
 * simulation stages.
 *
 * A fluid combines two related layers of state:
 * - **species metadata**, such as material properties, mole fractions, and
 *   generation operators,
 * - **particle storage**, such as position, velocity, temperature, species
 *   identity, and activity flags.
 *
 * ## Storage model
 * Particle data is stored in a structure-of-arrays style using backend-native
 * device buffers so that simulation kernels can access contiguous field arrays
 * efficiently. The allocated buffers represent a fixed particle capacity, while
 * the active particles are typically interpreted as a prefix of that capacity.
 *
 * ## Species model
 * The fluid also stores per-species metadata required to create, classify, or
 * evolve particles:
 * - @ref MatrialProperties for each species,
 * - normalized per-species mole fractions,
 * - per-species generation operators used by emission logic.
 *
 * ## Probe model
 * Since backend kernels should not carry host-side ownership structures, the
 * fluid can create a compact @ref FluidDeviceProbe containing:
 * - raw pointers to particle arrays,
 * - per-species material property pointers,
 * - the currently active particle count,
 * - the total allocated capacity.
 *
 * The system runtime typically owns the single authoritative fluid probe for a
 * live simulation and reuses it across update stages.
 *
 * ## Capacity semantics
 * The fluid allocates a fixed number of particle slots through its buffer size.
 * The probe distinguishes between:
 * - the total allocated capacity, and
 * - the currently active particle count.
 *
 * This makes it possible for runtime stages to:
 * - append newly emitted particles,
 * - deactivate particles without immediate compaction,
 * - operate on only the active prefix while preserving spare capacity.
 *
 * ## Construction
 * A fluid may be:
 * - default-constructed,
 * - directly constructed with an explicit particle capacity, or
 * - configured through the nested fluent @ref Builder, which stages species,
 *   mole fractions, generators, and buffer capacity before validation.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used for particle state and species data.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/generator/generator.h>
#include <atlas/material/matrial_properties.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>

#include <cstdint>

namespace atlas::system {

/**
 * @brief Lightweight device-facing view of particle storage.
 *
 * @details
 * @ref FluidDeviceProbe is the compact runtime structure passed to device kernels
 * and backend-parallel execution code.
 *
 * It exposes the raw particle-array pointers and minimal metadata needed by
 * low-level simulation stages without carrying host-side ownership or other
 * heavyweight runtime structures.
 *
 * The probe distinguishes between:
 * - the total allocated particle capacity, and
 * - the number of currently active particles stored in the active prefix.
 *
 * ## Typical usage
 * This probe is commonly consumed by stages that need to:
 * - iterate over active particles,
 * - read or update particle positions and velocities,
 * - access particle temperatures and species ids,
 * - test particle activity,
 * - look up per-species material properties.
 *
 * ## Ownership
 * All pointers stored in the probe are non-owning. The corresponding fluid-owned
 * buffers must remain valid for the lifetime of any backend execution using the probe.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
struct FluidDeviceProbe {
    /**
     * @brief Pointer to the per-species material property array.
     *
     * @details
     * Points to the species-level material definitions used to interpret particle
     * species ids and associated physical parameters.
     */
    MatrialProperties<T>* particle_property { nullptr };

    /**
     * @brief Pointer to the particle position array.
     *
     * @details
     * Stores one world-space position per particle slot.
     */
    Vector3<T>* pos { nullptr };

    /**
     * @brief Pointer to the particle velocity array.
     *
     * @details
     * Stores one velocity vector per particle slot.
     */
    Vector3<T>* vel { nullptr };

    /**
     * @brief Pointer to the particle temperature array.
     *
     * @details
     * Stores one temperature value per particle slot.
     */
    T* temperature { nullptr };

    /**
     * @brief Pointer to the particle species-id array.
     *
     * @details
     * Each entry identifies which configured species the corresponding particle belongs to.
     */
    size_t* species { nullptr };

    /**
     * @brief Pointer to the optional particle activity-flag array.
     *
     * @details
     * Stores one activity flag per particle slot. Depending on implementation
     * policy, this may be used to distinguish active, inactive, dead, or reusable
     * particle entries.
     */
    int* active { nullptr };

    /**
     * @brief Number of active particles stored in the active prefix.
     *
     * @details
     * Runtime stages generally iterate over particle indices in the range
     * `[0, particle_count)`.
     */
    int particle_count { 0 };

    /**
     * @brief Total allocated particle capacity.
     *
     * @details
     * Indicates the number of slots available in the particle arrays, whether
     * currently active or not.
     */
    size_t buffer_size { 0 };
};

/**
 * @brief Owns particle buffers and per-species metadata for a fluid.
 *
 * @details
 * @ref Fluid is the primary runtime container for particle-based simulation state.
 *
 * It owns:
 * - per-species material definitions,
 * - per-species mole fractions,
 * - per-species generation operators,
 * - device-resident particle positions,
 * - device-resident particle velocities,
 * - device-resident particle temperatures,
 * - device-resident particle species ids,
 * - device-resident particle activity flags.
 *
 * ## Structure of arrays
 * Particle state is stored in separate device buffers per attribute rather than
 * as an array of particle structs. This layout is typically advantageous for:
 * - memory coalescing in backend kernels,
 * - attribute-specific update stages,
 * - efficient bulk operations over positions, velocities, or temperatures.
 *
 * ## Species and composition
 * The fluid tracks the set of configured species together with their normalized
 * mole fractions and optional generators. This metadata allows runtime stages to:
 * - sample species during emission,
 * - interpret species-dependent material behavior,
 * - couple generators to species-specific spawning logic.
 *
 * ## Runtime probes
 * The fluid can create a @ref FluidDeviceProbe through @ref make_device_probe.
 * The system runtime commonly caches and reuses that probe as the canonical
 * device-facing view of the live fluid state.
 *
 * ## Capacity and activity
 * The fluid owns a fixed-capacity particle buffer set. Activity flags and the
 * active prefix allow stages to manage live particle subsets without necessarily
 * reallocating or compacting storage every time particle membership changes.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Fluid final {
public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Fluid.
     *
     * @details
     * The builder stages species definitions, mole fractions, generation operators,
     * and particle capacity, validates them, and then materializes either:
     * - a value instance of @ref Fluid, or
     * - a host-owned shared pointer to such an instance.
     */
    class Builder;

    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs a fluid with default-initialized empty storage and zero particle
     * capacity.
     */
    Fluid() = default;

    /**
     * @brief Allocate particle buffers for a fixed capacity.
     *
     * @details
     * Constructs a fluid with particle arrays sized to the specified capacity.
     * Species metadata may still be configured separately depending on the usage path.
     *
     * @param buffer_size Total particle capacity to allocate.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE explicit Fluid(size_t buffer_size);

    /**
     * @brief Builder entry point.
     *
     * @details
     * Returns a default-initialized @ref Builder for fluent fluid construction.
     *
     * @return Fresh builder instance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Default destructor.
     *
     * @details
     * Since resources are owned through RAII-managed member objects, the destructor
     * is defaulted.
     */
    ~Fluid() = default;

    /**
     * @brief Return the number of configured species.
     *
     * @details
     * This refers to the number of material/species definitions stored in the fluid,
     * not the number of active particles.
     *
     * @return Number of configured species.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    size() const noexcept;

    /**
     * @brief Return whether any species have been configured.
     *
     * @return `true` if no species definitions are stored; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    /**
     * @brief Return const access to per-species material properties.
     *
     * @return Const reference to the species material-property buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<MatrialProperties<T>>&
    particles() const noexcept;

    /**
     * @brief Return mutable access to per-species material properties.
     *
     * @return Mutable reference to the species material-property buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<MatrialProperties<T>>&
    particles() noexcept;

    /**
     * @brief Return const access to normalized mole fractions.
     *
     * @details
     * Stores the per-species composition weights associated with the configured
     * material-property list.
     *
     * @return Const reference to the mole-fraction buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<T>&
    mole_fractions() const noexcept;

    /**
     * @brief Return mutable access to normalized mole fractions.
     *
     * @return Mutable reference to the mole-fraction buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<T>&
    mole_fractions() noexcept;

    /**
     * @brief Return const access to per-species generation operators.
     *
     * @return Const reference to the generation-operator buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<GenerateOperator<T>>&
    generators() const noexcept;

    /**
     * @brief Return mutable access to per-species generation operators.
     *
     * @return Mutable reference to the generation-operator buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<GenerateOperator<T>>&
    generators() noexcept;

    /**
     * @brief Create the device probe consumed by runtime kernels.
     *
     * @details
     * Builds and returns a compact @ref FluidDeviceProbe containing raw pointers
     * to particle arrays and metadata describing the active and allocated particle range.
     *
     * The system runtime typically owns the single authoritative probe for a live
     * simulation and reuses it across multiple stages.
     *
     * @return Device-facing probe referencing this fluid's runtime state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE FluidDeviceProbe<T>
    make_device_probe() noexcept;

    /**
     * @brief Return mutable access to particle positions.
     *
     * @return Mutable reference to the position buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    positions() noexcept;

    /**
     * @brief Return mutable access to particle velocities.
     *
     * @return Mutable reference to the velocity buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<Vector3<T>>&
    velocities() noexcept;

    /**
     * @brief Return mutable access to particle temperatures.
     *
     * @return Mutable reference to the temperature buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<T>&
    temperatures() noexcept;

    /**
     * @brief Return mutable access to particle species identifiers.
     *
     * @return Mutable reference to the species-id buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE DeviceBuffer<size_t>&
    species_ids() noexcept;

    /**
     * @brief Return mutable access to particle activity flags.
     *
     * @return Mutable reference to the activity-flag buffer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<int>&
    active() noexcept;

    /**
     * @brief Return the total allocated particle capacity.
     *
     * @return Number of allocated particle slots.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE size_t
    buffer_size() const noexcept;

private:
    /// @brief Allow the builder to populate internal storage directly.
    friend class Builder;

private:
    /**
     * @brief Per-species material definitions.
     *
     * @details
     * Stores one material-property record for each configured species.
     */
    DeviceBuffer<MatrialProperties<T>> _particle_properties;

    /**
     * @brief Normalized per-species mole fractions.
     *
     * @details
     * Stores one composition weight per configured species.
     */
    DeviceBuffer<T> _mole_fractions;

    /**
     * @brief Per-species spawning operators.
     *
     * @details
     * Stores one generation operator per configured species.
     */
    DeviceBuffer<GenerateOperator<T>> _generators;

    /**
     * @brief Particle position buffer.
     *
     * @details
     * Stores one world-space position per particle slot.
     */
    DeviceBuffer<Vector3<T>> d_pos;

    /**
     * @brief Particle velocity buffer.
     *
     * @details
     * Stores one velocity vector per particle slot.
     */
    DeviceBuffer<Vector3<T>> d_vel;

    /**
     * @brief Particle temperature buffer.
     *
     * @details
     * Stores one temperature value per particle slot.
     */
    DeviceBuffer<T> d_temperature;

    /**
     * @brief Particle species-id buffer.
     *
     * @details
     * Stores one species index per particle slot.
     */
    DeviceBuffer<size_t> d_species;

    /**
     * @brief Particle activity-flag buffer.
     *
     * @details
     * Stores one activity flag per particle slot.
     */
    DeviceBuffer<int> d_active;

    /**
     * @brief Allocated particle capacity.
     *
     * @details
     * Stores the number of particle slots allocated across all particle arrays.
     */
    size_t _buffer_size = 0;

    /**
     * @brief Internal counter tracking probe creation.
     *
     * @details
     * May be used by the implementation to monitor probe refresh activity.
     */
    std::uint64_t _probe_count = 0;
};

/**
 * @brief Fluent builder for @ref Fluid.
 *
 * @details
 * The builder provides a controlled construction path for @ref Fluid while
 * staging species metadata and particle-storage configuration.
 *
 * It supports:
 * - adding species by value or by shared pointer,
 * - specifying explicit mole fractions,
 * - attaching optional generators,
 * - bulk insertion of multiple species,
 * - configuring particle capacity,
 * - enabling validation policies such as non-empty requirements and null checks.
 *
 * ## Typical usage
 * @code
 * auto fluid = atlas::Fluid<float>::builder()
 *     .add_species(species_a, 0.7f, generator_a)
 *     .add_species(species_b, 0.3f, generator_b)
 *     .with_buffer_size(100000)
 *     .require_non_empty(true)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - species list requirements are satisfied,
 * - null material pointers are rejected when configured,
 * - mole-fraction buffers are consistent with species counts,
 * - generator lists are compatible with species counts,
 * - particle capacity is acceptable for the intended runtime.
 *
 * The exact validation and normalization rules are implementation-defined in
 * `fluid.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Fluid<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Creates a builder with empty staged species metadata, zero particle
     * capacity, optional empty-species allowance, and null-material rejection enabled.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Fluid by value after validation.
     *
     * @details
     * Validates the staged builder state, allocates particle storage, transfers
     * species metadata, and constructs the final fluid object.
     *
     * @return Fully constructed fluid value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Fluid<T>
    build() const;

    /**
     * @brief Build a configured @ref Fluid in a host_shared_ptr after validation.
     *
     * @details
     * Validates the staged builder state, constructs the fluid, and returns it
     * in a host-owned shared pointer.
     *
     * @return `atlas::host_shared_ptr<Fluid<T>>` owning the constructed fluid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Fluid<T>>
    make_host_shared() const;

    /**
     * @brief Add a species with a default mole fraction of `1`.
     *
     * @details
     * Stages a species definition by value and assigns it a default composition weight.
     *
     * @param p Material-property record to add.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p);

    /**
     * @brief Add a species with an explicit mole fraction.
     *
     * @param p Material-property record to add.
     * @param mole_fraction Composition weight associated with the species.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction);

    /**
     * @brief Add a species with an explicit mole fraction and generator.
     *
     * @param p Material-property record to add.
     * @param mole_fraction Composition weight associated with the species.
     * @param generator Host-side generator associated with the species.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(const MatrialProperties<T>& p, T mole_fraction, GeneratorHostPtr<T> generator);

    /**
     * @brief Add a species from a host-shared material pointer.
     *
     * @param p Host-side shared pointer to the material-property record.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p);

    /**
     * @brief Add a species pointer with an explicit mole fraction.
     *
     * @param p Host-side shared pointer to the material-property record.
     * @param mole_fraction Composition weight associated with the species.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction);

    /**
     * @brief Add a species pointer with an explicit mole fraction and generator.
     *
     * @param p Host-side shared pointer to the material-property record.
     * @param mole_fraction Composition weight associated with the species.
     * @param generator Host-side generator associated with the species.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species(MatrialPropertiesHostPtr<T> p, T mole_fraction, GeneratorHostPtr<T> generator);

    /**
     * @brief Add multiple value-based species with default mole fractions.
     *
     * @param ps Host buffer containing species definitions.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps);

    /**
     * @brief Add multiple value-based species with explicit mole fractions.
     *
     * @param ps Host buffer containing species definitions.
     * @param mole_fractions Host buffer containing per-species mole fractions.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps, const HostBuffer<T>& mole_fractions);

    /**
     * @brief Add multiple value-based species with explicit mole fractions and generators.
     *
     * @param ps Host buffer containing species definitions.
     * @param mole_fractions Host buffer containing per-species mole fractions.
     * @param generators Host buffer containing per-species generators.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialProperties<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Add multiple pointer-based species with default mole fractions.
     *
     * @param ps Host buffer containing species material-property pointers.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps);

    /**
     * @brief Add multiple pointer-based species with explicit mole fractions.
     *
     * @param ps Host buffer containing species material-property pointers.
     * @param mole_fractions Host buffer containing per-species mole fractions.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps, const HostBuffer<T>& mole_fractions);

    /**
     * @brief Add multiple pointer-based species with explicit mole fractions and generators.
     *
     * @param ps Host buffer containing species material-property pointers.
     * @param mole_fractions Host buffer containing per-species mole fractions.
     * @param generators Host buffer containing per-species generators.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    add_species_bulk(const HostBuffer<MatrialPropertiesHostPtr<T>>& ps,
                     const HostBuffer<T>& mole_fractions,
                     const HostBuffer<GeneratorHostPtr<T>>& generators);

    /**
     * @brief Set the particle capacity to allocate.
     *
     * @details
     * Stages the number of particle slots that should be allocated in the final fluid.
     *
     * @param buffer_size Particle capacity to allocate.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_buffer_size(size_t buffer_size) noexcept;

    /**
     * @brief Require at least one species before building.
     *
     * @details
     * Enables or disables the validation rule that forbids empty species lists.
     *
     * @param on Whether the non-empty requirement should be enabled.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    require_non_empty(bool on = true) noexcept;

    /**
     * @brief Control whether null material pointers are rejected.
     *
     * @details
     * Enables or disables validation of null material pointers in pointer-based
     * species insertion paths.
     *
     * @param on Whether null material pointers should be rejected.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    reject_null_particles(bool on = true) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on the staged species metadata, mole
     * fractions, generator lists, and particle capacity.
     *
     * @note
     * The exact validation policy is implementation-defined in `fluid.hpp`.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending species definitions.
     *
     * @details
     * Staged material-property records to be stored in the final fluid.
     */
    DeviceBuffer<MatrialProperties<T>> _particles;

    /**
     * @brief Pending mole fractions.
     *
     * @details
     * Staged per-species composition weights aligned with @ref _particles.
     */
    DeviceBuffer<T> _mole_fractions;

    /**
     * @brief Pending generation operators.
     *
     * @details
     * Staged per-species emission/generation operators aligned with @ref _particles.
     */
    DeviceBuffer<GenerateOperator<T>> _generators;

    /**
     * @brief Pending particle capacity.
     *
     * @details
     * Number of particle slots to allocate in the final fluid.
     */
    size_t _buffer_size = 0;

    /**
     * @brief Whether empty species lists are forbidden.
     */
    bool _require_non_empty = false;

    /**
     * @brief Whether null material pointers are rejected.
     */
    bool _reject_null_particles = true;
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Fluid.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Fluid = system::Fluid<T>;

/**
 * @brief Convenience alias for @ref atlas::system::FluidDeviceProbe.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using FluidDeviceProbe = system::FluidDeviceProbe<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Fluid.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using FluidHostPtr = atlas::host_shared_ptr<system::Fluid<T>>;

} // namespace atlas

#include <atlas/fluid/fluid.hpp>