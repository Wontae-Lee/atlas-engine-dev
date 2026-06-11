#pragma once

/**
 * @file source.h
 * @brief Declares the Source class used to emit particles into a fluid from configured source units.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/source/source_probe.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace atlas::fluid {

/**
 * @brief Particle emission source for a fluid.
 *
 * A Source manages one or more source units and emits particles into an
 * associated Fluid instance.
 *
 * Its responsibilities include:
 * - storing source units and spawn policies,
 * - generating and caching local candidate emission positions,
 * - assigning and shuffling species indices,
 * - writing emitted particle data into fluid state buffers.
 *
 * Emission is typically performed through @ref update, which advances
 * time-dependent unit state and then emits particles for the same step.
 *
 * The lower-level @ref emit function remains available when a caller wants to
 * trigger emission directly after managing unit state separately.
 *
 * Cached local emission positions are rebuilt lazily through rebuild_cache()
 * when the source configuration becomes invalidated.
 *
 * @tparam T Floating-point scalar type used by the source and fluid.
 */
template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    using SourceProbe = atlas::fluid::SourceProbe<T>;

    /**
     * @brief Builder for configuring and constructing Source objects.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Source() = default;

    /**
     * @brief Destructor.
     */
    ~Source() = default;

    /**
     * @brief Constructs a source from fully prepared buffers and configuration.
     *
     * This constructor directly installs all source data, including units,
     * spawn configuration, target fluid, and emission parameters.
     *
     * @param units Device buffer containing source units.
     * @param spawn_types Device buffer containing spawn type configuration.
     * @param spawn_operators Device buffer containing spawn operators.
     * @param fluid Host shared pointer to the target fluid.
     * @param flip Whether spawn acceptance should be inverted.
     * @param spacing Grid spacing used to generate candidate local spawn positions.
     * @param tolerance Geometric tolerance used during spawn acceptance tests.
     * @param temperature Temperature parameter forwarded to particle generators.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(DeviceBuffer<Unit<T>> units,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<SpawnOperator<T>> spawn_operators,
           atlas::host_shared_ptr<atlas::Fluid<T>> fluid,
           bool flip                = false,
           T spacing                = T(0.1),
           T tolerance              = T(0),
           T temperature            = T(273.15),
           ObserverHostPtr observer = nullptr) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent Source construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Updates source units and emits particles for the given time step.
     *
     * This function advances time-dependent state within each source unit and
     * then emits particles into the target fluid for the same simulation step.
     *
     * @param dt Time step.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    /**
     * @brief Emits particles into the associated fluid.
     *
     * This function rebuilds caches when necessary, shuffles species assignment,
     * transforms local emission positions into world space, generates particle
     * velocities, and writes the resulting attributes into the fluid state buffers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit();

    /**
     * @brief Rebuilds cached local emission data when invalidated.
     *
     * This function regenerates local candidate positions for each source unit,
     * rebuilds species caches, and prepares internal buffers needed by emit().
     *
     * If the cache is already valid, the function performs no work.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

    /**
     * @brief Shuffles cached species assignments for an emission pass.
     *
     * This function permutes the cached species sequence using shuffle keys so
     * emitted particles do not always follow the same deterministic species order.
     *
     * @param count Number of species entries to shuffle.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    shuffle_species(std::size_t count);

    /**
     * @brief Refreshes the cached probe for source emission data.
     *
     * @return True when all required source and fluid state exists.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    make_probe() noexcept;

private:
    /**
     * @brief Source units used to define local emission regions and transforms.
     */
    DeviceBuffer<Unit<T>> _units;

    /**
     * @brief Spawn type configuration associated with the source units.
     */
    DeviceBuffer<SpawnType> _spawn_types;

    /**
     * @brief Spawn operators used to accept or reject candidate spawn positions.
     */
    DeviceBuffer<SpawnOperator<T>> _spawn_operators;

    /**
     * @brief Target fluid receiving emitted particles.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Optional observer used to record source metrics.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Whether spawn acceptance should be inverted.
     */
    bool _flip = false;

    /**
     * @brief Grid spacing used when sampling candidate local emission positions.
     */
    T _spacing = T(0.1);

    /**
     * @brief Tolerance used for geometric spawn acceptance tests.
     */
    T _tolerance = T(0);

    /**
     * @brief Temperature forwarded to particle generators during emission.
     */
    T _temperature { T(273.15) };

    /**
     * @brief Cached accepted particle count for each source unit.
     */
    HostBuffer<int> _local_unit_counts;

    /**
     * @brief Flat contiguous device buffer of all local emission positions.
     *
     * Built directly by rebuild_cache() so emit() can launch a single GPU
     * kernel without per-unit staging buffers.
     */
    DeviceBuffer<Vector3<T>> _flat_local_positions;

    /**
     * @brief Unit index for each flat local emission position.
     *
     * Entry i stores the source unit that owns _flat_local_positions[i].
     */
    DeviceBuffer<int> _flat_unit_indices;

    /**
     * @brief Total number of cached local particle positions across all units.
     */
    std::size_t _local_particle_count = 0;

    /**
     * @brief Base cached species sequence before shuffling.
     */
    DeviceBuffer<std::size_t> _species_cache;

    /**
     * @brief Species sequence used for the current emission pass after shuffling.
     */
    DeviceBuffer<std::size_t> _shuffled_species;

    /**
     * @brief Temporary shuffle keys used to permute species assignments.
     */
    DeviceBuffer<std::uint64_t> _shuffle_keys;

    /**
     * @brief Cached probe populated by @ref make_probe.
     */
    SourceProbe _probe {};

    /**
     * @brief Monotonically increasing seed used across shuffle passes.
     */
    std::uint64_t _shuffle_seed = 0;

    /**
     * @brief Indicates whether cached local emission data must be rebuilt.
     */
    bool _is_invalidated_cache = true;

    /**
     * @brief Monotonic source-step index used by source metric recording.
     */
    std::size_t _step_index = 0;
};

/**
 * @brief Builder for Source.
 *
 * This builder collects source units, spawn configuration, target fluid, and
 * emission parameters before constructing a validated Source object.
 *
 * Structural validation ensures that:
 * - source units are provided,
 * - a target fluid exists,
 * - spawn configuration is non-empty,
 * - spawn types and spawn operators either have size 1 or match unit count,
 * - numeric parameters are finite and valid.
 *
 * @tparam T Floating-point scalar type used by the source and fluid.
 */
template <typename T>
class Source<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Builds a validated Source object.
     *
     * @return Constructed Source object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Source<T>
    build();

    /**
     * @brief Builds a host-side shared Source object.
     *
     * @return Host shared pointer to a constructed Source object.
     *
     * @throw std::runtime_error Thrown if the builder configuration is invalid.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Source<T>>
    make_host_shared();

    /**
     * @brief Appends source units to the builder.
     *
     * @param units Host buffer containing source units.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p units is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_units(const HostBuffer<Unit<T>>& units);

    /**
     * @brief Sets the target fluid.
     *
     * @param fluid Host shared pointer to the target fluid.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(atlas::host_shared_ptr<atlas::Fluid<T>> fluid) noexcept;

    /**
     * @brief Sets the optional observer used to record source metrics.
     *
     * @param observer Host shared pointer to the observer.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /**
     * @brief Appends spawn type configuration entries.
     *
     * Spawn types must either contain exactly one entry shared by all units or
     * one entry per unit.
     *
     * @param spawn_types Host buffer containing spawn type entries.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p spawn_types is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    /**
     * @brief Appends a single spawn operator.
     *
     * @param spawn_operator Spawn operator to append.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept;

    /**
     * @brief Appends multiple spawn operators.
     *
     * Spawn operators must either contain exactly one entry shared by all units
     * or one entry per unit.
     *
     * @param spawn_operators Host buffer containing spawn operators.
     * @return Reference to this builder.
     *
     * @throw std::runtime_error Thrown if @p spawn_operators is empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    /**
     * @brief Sets the geometric spawn tolerance.
     *
     * @param tolerance Tolerance used during spawn acceptance tests.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Sets whether spawn acceptance should be inverted.
     *
     * @param flip Whether spawn acceptance is flipped.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    /**
     * @brief Sets the candidate sampling spacing.
     *
     * @param spacing Grid spacing used to sample local candidate positions.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    /**
     * @brief Sets the emission temperature.
     *
     * @param temperature Temperature forwarded to particle generators.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if validation fails.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Host-side source units collected by the builder.
     */
    HostBuffer<Unit<T>> _units;

    /**
     * @brief Target fluid collected by the builder.
     */
    FluidHostPtr<T> _fluid;

    /**
     * @brief Optional observer collected by the builder.
     */
    ObserverHostPtr _observer {};

    /**
     * @brief Host-side spawn type configuration.
     */
    HostBuffer<SpawnType> _spawn_types;

    /**
     * @brief Host-side spawn operators.
     */
    HostBuffer<SpawnOperator<T>> _spawn_operators;

    /**
     * @brief Whether spawn acceptance should be inverted.
     */
    bool _flip = false;

    /**
     * @brief Grid spacing used to sample candidate local positions.
     */
    T _spacing = T(0.1);

    /**
     * @brief Tolerance used during geometric spawn acceptance tests.
     */
    T _tolerance = T(0);

    /**
     * @brief Temperature forwarded to particle generators during emission.
     */
    T _temperature { T(273.15) };
};

} // namespace atlas::fluid

namespace atlas {

/**
 * @brief Alias for atlas::fluid::Source.
 *
 * @tparam T Floating-point scalar type used by the source.
 */
template <typename T>
using Source = atlas::fluid::Source<T>;

/**
 * @brief Host-side shared pointer alias for Source.
 *
 * @tparam T Floating-point scalar type used by the source.
 */
template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

/**
 * @brief Device-side shared pointer alias for Source.
 *
 * @tparam T Floating-point scalar type used by the source.
 */
template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

} // namespace atlas

#include <atlas/source/source.hpp>
