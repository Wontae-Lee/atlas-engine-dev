#pragma once

/**
 * @file source.h
 * @brief Declares particle emission sources and builder utilities for spawning particles into a fluid buffer.
 *
 * @details
 * This header defines @ref atlas::system::Source, a runtime object responsible
 * for emitting new particles from one or more configured units into an existing
 * fluid particle buffer.
 *
 * A source combines:
 * - spawn geometry through @ref Unit,
 * - per-unit spawn classification through @ref SpawnType,
 * - per-unit spawn behavior through @ref SpawnOperator,
 * - species composition and velocity generation from an associated @ref Fluid.
 *
 * ## High-level responsibilities
 * A source can:
 * - sample candidate local positions on configured units,
 * - cache those local positions for reuse across repeated emissions,
 * - determine which candidates should emit particles,
 * - assign emitted particles to species according to the fluid composition,
 * - generate emitted velocities using the fluid's configured generators,
 * - write the new particles into inactive capacity of a fluid buffer.
 *
 * ## Caching model
 * Geometry-dependent local sample positions are cached per unit in
 * @ref _local_positions so that repeated calls to @ref emit do not need to
 * recompute the geometric sampling pattern unless relevant source configuration
 * has changed.
 *
 * The cache is invalidated when properties affecting geometric sampling or unit
 * configuration are modified.
 *
 * ## Species assignment
 * Species are staged through a deterministic cache and then decorrelated per
 * emission using shuffle keys generated from an evolving internal seed. This
 * allows repeated source emissions to preserve composition while avoiding a
 * trivially repeating species order.
 *
 * ## Coordinate convention
 * Candidate positions are initially sampled in unit-local space and are then
 * interpreted through the unit's sync operator when particles are emitted into
 * world space.
 *
 * ## Construction
 * A source may be:
 * - default-constructed,
 * - constructed directly from device-resident units/spawn data and a fluid,
 * - configured through the nested fluent @ref Builder.
 *
 * ---
 *
 * @tparam T Floating-point scalar type used by the simulation.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Emits new particles from configured units into a fluid buffer.
 *
 * @details
 * @ref Source is the particle-emission stage used to populate inactive particle
 * capacity from a set of spawn units and associated spawn rules.
 *
 * The source stores:
 * - the units from which particles may be emitted,
 * - the spawn type for each unit,
 * - the spawn operator for each unit,
 * - the fluid that defines species composition and velocity generation,
 * - source-wide emission parameters such as spacing, tolerance, flip mode, and temperature.
 *
 * ## Typical emission workflow
 * A call to @ref emit typically performs the following conceptual steps:
 * 1. ensure that cached local sample positions exist,
 * 2. determine which local sample positions produce emitted particles,
 * 3. transform local positions into world-space particle states,
 * 4. assign species according to the fluid mole-fraction configuration,
 * 5. generate outgoing velocities using the fluid generators,
 * 6. write the resulting particles into available inactive slots.
 *
 * ## Update semantics
 * The @ref update function may be used to advance the source or its units over
 * time before emission occurs. Exact update semantics are implementation-defined
 * in `source.hpp`.
 *
 * ## Flip mode
 * The @ref flip flag allows spawn decisions to be inverted according to the
 * source's geometric/spawn rule logic. This is useful for complement-style
 * spawn regions or negative-space emission rules.
 *
 * ## Temperature
 * The configured source temperature is assigned to newly emitted particles unless
 * the implementation chooses to derive or override it through another active rule.
 *
 * ---
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    /**
     * @brief Fluent builder for configuring and constructing @ref Source.
     *
     * @details
     * The builder stages units, spawn metadata, fluid configuration, and source
     * parameters, validates them, and constructs either:
     * - a source by value, or
     * - a host-owned shared pointer to a source.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty source with default emission parameters and no staged
     * unit/spawn/fluid dependencies.
     */
    Source() = default;

    /**
     * @brief Destructor.
     */
    ~Source() = default;

    /**
     * @brief Construct a source from explicit unit, spawn, and fluid dependencies.
     *
     * @param units Device-resident spawn units.
     * @param spawn_types Spawn type per unit.
     * @param spawn_operators Spawn operator per unit.
     * @param fluid Fluid providing species composition and generators.
     * @param flip Whether spawn classification should be inverted.
     * @param spacing Sampling spacing used to generate local candidate points.
     * @param tolerance Geometric tolerance used in spawn tests.
     * @param temperature Temperature assigned to emitted particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(DeviceBuffer<Unit<T>> units,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<SpawnOperator<T>> spawn_operators,
           FluidHostPtr<T> fluid,
           bool flip     = false,
           T spacing     = T(0.1),
           T tolerance   = T(0),
           T temperature = T(273.15)) noexcept;

    /**
     * @brief Builder entry point.
     *
     * @return A default-initialized @ref Builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Update the source state by one timestep.
     *
     * @details
     * This may update time-dependent source behavior and/or unit motion according
     * to the implementation in `source.hpp`.
     *
     * @param dt Simulation timestep.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Emit new particles into a fluid probe.
     *
     * @details
     * Writes newly emitted particles into the supplied fluid device probe using
     * the source's configured units, spawn rules, cached sample positions,
     * species layout, and velocity generators.
     *
     * @param particle_probe Mutable fluid device probe to update in-place.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(FluidDeviceProbe<T>& particle_probe);

    /**
     * @brief Replace the source units with a device buffer.
     *
     * @details
     * Typically invalidates geometry-derived caches.
     *
     * @param units New device-resident unit buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    /**
     * @brief Replace the source units with host-provided values.
     *
     * @details
     * Typically uploads the supplied units and invalidates geometry-derived caches.
     *
     * @param units New host-side unit buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Replace the associated fluid dependency.
     *
     * @param fluid New fluid dependency.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Replace the spawn-type buffer with a device buffer.
     *
     * @param spawn_types New device-resident spawn-type buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_types(DeviceBuffer<SpawnType> spawn_types) noexcept;

    /**
     * @brief Replace the spawn types with host-provided values.
     *
     * @param spawn_types New host-side spawn-type buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    /**
     * @brief Replace the spawn operators with a device buffer.
     *
     * @param spawn_operators New device-resident spawn-operator buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operators(DeviceBuffer<SpawnOperator<T>> spawn_operators) noexcept;

    /**
     * @brief Replace the spawn operators with host-provided values.
     *
     * @param spawn_operators New host-side spawn-operator buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    /**
     * @brief Set the geometric spawn tolerance.
     *
     * @param tolerance New source tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    /**
     * @brief Set whether spawn decisions should be inverted.
     *
     * @param flip New flip mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    /**
     * @brief Set the local sampling spacing.
     *
     * @details
     * Typically invalidates cached local sample positions.
     *
     * @param spacing New sampling spacing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spacing(T spacing) noexcept;

    /**
     * @brief Set the emission temperature assigned to new particles.
     *
     * @param temperature New emission temperature.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    /**
     * @brief Return mutable access to the spawn units.
     *
     * @return Mutable device buffer of units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    /**
     * @brief Return const access to the spawn units.
     *
     * @return Const device buffer of units.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    /**
     * @brief Return const access to the associated fluid dependency.
     *
     * @return Const host shared pointer to the fluid.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    /**
     * @brief Return mutable access to the spawn types.
     *
     * @return Mutable device buffer of spawn types.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<SpawnType>&
    spawn_types() noexcept;

    /**
     * @brief Return const access to the spawn types.
     *
     * @return Const device buffer of spawn types.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<SpawnType>&
    spawn_types() const noexcept;

    /**
     * @brief Return mutable access to the spawn operators.
     *
     * @return Mutable device buffer of spawn operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<SpawnOperator<T>>&
    spawn_operators() noexcept;

    /**
     * @brief Return const access to the spawn operators.
     *
     * @return Const device buffer of spawn operators.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<SpawnOperator<T>>&
    spawn_operators() const noexcept;

    /**
     * @brief Return the configured geometric spawn tolerance.
     *
     * @return Source tolerance.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    /**
     * @brief Return whether spawn decisions are inverted.
     *
     * @return `true` if flip mode is enabled; otherwise `false`.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

    /**
     * @brief Return the configured local sampling spacing.
     *
     * @return Sampling spacing.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    spacing() const noexcept;

    /**
     * @brief Return the configured emission temperature.
     *
     * @return Emission temperature.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    /**
     * @brief Return const access to the cached local sample positions.
     *
     * @details
     * The outer buffer is indexed per unit, while each stored device buffer
     * contains local candidate positions associated with that unit.
     *
     * @return Const host buffer of per-unit device buffers.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<DeviceBuffer<Vector3<T>>>&
    local_positions() const noexcept;

    /**
     * @brief Force a rebuild of cached geometry-derived sampling data.
     *
     * @details
     * Recomputes local sample positions and refreshes any other geometry-dependent
     * cached source data required by emission.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

private:
    /**
     * @brief Shuffle the staged species layout for an emission batch.
     *
     * @details
     * Uses internally generated shuffle keys and the evolving shuffle seed to
     * decorrelate the species order while preserving the intended composition.
     *
     * @param count Number of species entries to shuffle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shuffle_species(std::size_t count);

private:
    /**
     * @brief Spawn units.
     *
     * @details
     * Each unit contributes geometry and pose for candidate emission positions.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Spawn type per unit.
     *
     * @details
     * Selects how each unit is interpreted for emission classification.
     */
    DeviceBuffer<SpawnType> _spawn_types;

    /**
     * @brief Spawn rule per unit.
     *
     * @details
     * Encodes the geometric or logical emission rule used for each unit.
     */
    DeviceBuffer<SpawnOperator<T>> _spawn_operators;

    /**
     * @brief Fluid dependency for species composition and velocity generation.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Whether spawn decisions are inverted.
     */
    bool _flip = false;

    /**
     * @brief Grid spacing used to sample candidate local positions.
     */
    T _spacing = T(0.1);

    /**
     * @brief Geometric tolerance used during spawn tests.
     */
    T _tolerance = T(0);

    /**
     * @brief Emission temperature assigned to new particles.
     */
    T _temperature { T(273.15) };

    /**
     * @brief Cached local spawn positions per unit.
     *
     * @details
     * Each entry stores a device buffer of local-space candidate positions for
     * the corresponding source unit.
     */
    HostBuffer<DeviceBuffer<Vector3<T>>> _local_positions;

    /**
     * @brief Cached deterministic species layout.
     *
     * @details
     * Represents the base composition pattern derived from the associated fluid.
     */
    DeviceBuffer<size_t> _species_cache;

    /**
     * @brief Per-emission shuffled species assignment.
     *
     * @details
     * Derived from @ref _species_cache using shuffle keys and the current seed.
     */
    DeviceBuffer<size_t> _shuffled_species;

    /**
     * @brief Shuffle keys used to randomize emitted species order.
     */
    DeviceBuffer<std::uint64_t> _shuffle_keys;

    /**
     * @brief Monotonic seed used to decorrelate repeated emissions.
     */
    std::uint64_t _shuffle_seed = 0;

    /**
     * @brief Whether cached geometry-derived data must be rebuilt.
     */
    bool _is_invalidated_cache = true;
};

/**
 * @brief Fluent builder for @ref Source.
 *
 * @details
 * The builder provides a controlled construction path for particle sources by
 * staging:
 * - spawn units,
 * - fluid dependency,
 * - per-unit spawn types,
 * - per-unit spawn operators,
 * - source-wide emission parameters.
 *
 * ## Typical usage
 * @code
 * auto source = atlas::Source<float>::builder()
 *     .with_units(units)
 *     .with_fluid(fluid)
 *     .with_spawn_types(spawn_types)
 *     .with_spawn_operators(spawn_ops)
 *     .with_spacing(0.1f)
 *     .build();
 * @endcode
 *
 * ## Validation policy
 * `validate()` is invoked by @ref build and @ref make_host_shared. Typical checks
 * may include:
 * - a valid fluid is present,
 * - the unit list is not empty,
 * - spawn-type and spawn-operator counts are compatible with the unit count,
 * - spacing and tolerance values are admissible.
 *
 * The exact validation rules are implementation-defined in `source.hpp`.
 *
 * ---
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Source<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the builder with empty unit/spawn buffers and default source
     * parameters.
     */
    Builder() = default;

    /**
     * @brief Build a configured @ref Source by value after validation.
     *
     * @return Constructed source value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Source<T>
    build();

    /**
     * @brief Build a configured @ref Source in a host_shared_ptr after validation.
     *
     * @return `atlas::host_shared_ptr<Source<T>>` owning the constructed source.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Source<T>>
    make_host_shared();

    /**
     * @brief Set the source units from host-side values.
     *
     * @param units Host-side unit buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Set the associated fluid dependency.
     *
     * @param fluid Host-owned fluid pointer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the per-unit spawn types.
     *
     * @param spawn_types Host-side spawn-type buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    /**
     * @brief Set one spawn operator to be applied uniformly.
     *
     * @details
     * The builder may replicate this operator as needed for all configured units.
     *
     * @param spawn_operator Spawn operator to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept;

    /**
     * @brief Set the per-unit spawn operators.
     *
     * @param spawn_operators Host-side spawn-operator buffer.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    /**
     * @brief Set the geometric spawn tolerance.
     *
     * @param tolerance Tolerance to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Set whether spawn decisions should be inverted.
     *
     * @param flip Flip mode to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    /**
     * @brief Set the local sampling spacing.
     *
     * @param spacing Sampling spacing to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    /**
     * @brief Set the emission temperature.
     *
     * @param temperature Temperature to stage.
     * @return `*this` for fluent chaining.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    /**
     * @brief Validate staged builder state before construction.
     *
     * @details
     * Performs pre-construction checks on units, fluid dependency, spawn metadata,
     * and source-wide parameters.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Pending host-side unit list.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Pending fluid dependency.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Pending per-unit spawn types.
     */
    HostBuffer<SpawnType> _spawn_types;

    /**
     * @brief Pending per-unit spawn operators.
     */
    HostBuffer<SpawnOperator<T>> _spawn_operators;

    /**
     * @brief Pending flip mode.
     */
    bool _flip = false;

    /**
     * @brief Pending local sampling spacing.
     */
    T _spacing = T(0.1);

    /**
     * @brief Pending geometric tolerance.
     */
    T _tolerance = T(0);

    /**
     * @brief Pending emission temperature.
     */
    T _temperature { T(273.15) };
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Convenience alias for @ref atlas::system::Source.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using Source = atlas::system::Source<T>;

/**
 * @brief Convenience alias for a host-owned shared pointer to @ref atlas::system::Source.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

/**
 * @brief Convenience alias for a device-owned shared pointer to @ref atlas::system::Source.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

} // namespace atlas

#include <atlas/source/source.hpp>