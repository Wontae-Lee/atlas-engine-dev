#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/data/particle_data.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/matter/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/sampling/sampling.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/sync/sync_operator.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <optional>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Runtime particle source that spawns particles based on geometric spawn conditions.
 *
 * @details
 * `Source<T>` mirrors @ref Sink structure:
 * - Bound @ref Unit supplies geometry and transform
 * - Local positions sampled as grid using @ref sampling::sample_spawn_grid
 * - @ref SpawnOperator classifies particles (Surface/Volume)
 * - @ref GenerateOperator produces velocity distributions
 * - @ref SyncOperator transforms local to world coordinates
 * - @ref Fluid determines species distribution via amounts
 *
 * Iterates through local_positions and evaluates spawn criteria. For accepted positions:
 * - Transforms to world space via `SyncOperator::sync_to_world`
 * - Generates velocity via `GenerateOperator::generate`
 * - Assigns species proportionally based on `Fluid::amounts()`
 * - Adds particles to @ref ParticleDeviceProbe
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    class Builder;

public:
    Source()  = default;
    ~Source() = default;

    /**
     * @brief Constructs a source from a unit, fluid, spawn type, and tolerance.
     *
     * @param unit Unit defining geometry and transform.
     * @param fluid Fluid defining species and amounts.
     * @param spawn_type Surface or Volume spawn mode.
     * @param tolerance Classification tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(Unit<T> unit,
           FluidHostPtr<T> fluid,
           SpawnType spawn_type = SpawnType::Surface,
           T tolerance          = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Spawns particles matching spawn criteria and adds them to the probe.
     *
     * @details
     * Iterates through local_positions, tests against geometry using the spawn
     * operator (Surface/Volume). For accepted positions:
     * - Transforms to world space
     * - Generates velocity
     * - Assigns species based on fluid amounts
     * - Adds to particle_probe
     *
     * @param particle_probe Target particle storage probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(ParticleDeviceProbe<T>& particle_probe);

    /**
     * @brief Replace the bound unit and invalidate cached spawn data.
     *
     * @param unit New unit supplying geometry and transform state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(Unit<T> unit) noexcept;

    /**
     * @brief Replace the bound fluid and invalidate cached species/layout data.
     *
     * @param fluid New fluid mixture used for species assignment.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Switch between surface and volume spawn classification.
     *
     * @param spawn_type Spawn classification mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_type(SpawnType spawn_type) noexcept;

    /**
     * @brief Replace the full runtime spawn operator.
     *
     * @param spawn_operator Explicit operator overriding the current spawn mode object.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operator(SpawnOperator<T> spawn_operator) noexcept;

    /**
     * @brief Set the spawn classification tolerance and invalidate caches.
     *
     * @param tolerance Classification tolerance forwarded to the spawn operator.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    /**
     * @brief Set grid spacing for cached spawn samples and invalidate caches.
     *
     * @param spacing Uniform sample spacing in local coordinates.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spacing(T spacing) noexcept;

    /**
     * @brief Replace the runtime velocity generator used during emission.
     *
     * @param generate_operator Generator applied to emitted particles.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_generate_operator(GenerateOperator<T> generate_operator) noexcept;

    /**
     * @brief Return the currently bound unit.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Unit<T>&
    unit() const noexcept;

    /**
     * @brief Return the currently bound fluid.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    /**
     * @brief Return the active spawn mode tag.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SpawnType
    spawn_type() const noexcept;

    /**
     * @brief Return the full runtime spawn operator.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SpawnOperator<T>&
    spawn_operator() const noexcept;

    /**
     * @brief Return the current spawn tolerance.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    /**
     * @brief Return the current sample spacing used for cache construction.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    spacing() const noexcept;

    /**
     * @brief Return the velocity generator applied to emitted particles.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const GenerateOperator<T>&
    generate_operator() const noexcept;

    /**
     * @brief Return the cached local-space spawnable positions.
     *
     * @details
     * The returned buffer contains only positions that already satisfy the
     * current spawn predicate for the bound unit/spacing/tolerance.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Vector3<T>>&
    local_positions() const noexcept;

private:
    /**
     * @brief Rebuild cached spawn positions and species layout when invalidated.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

private:
    Unit<T> _unit;
    FluidHostPtr<T> _fluid;
    SpawnOperator<T> _spawn_operator { SpawnType::Surface };
    GenerateOperator<T> _generate_operator { GenerateType::uniform };
    T _tolerance = T(0);
    T _spacing   = T(0.1);
    DeviceBuffer<Vector3<T>> _local_positions;
    DeviceBuffer<size_t> _species_cache;
    DeviceBuffer<size_t> _shuffled_species;
    DeviceBuffer<std::uint64_t> _shuffle_keys;
    std::uint64_t _shuffle_seed = 0;
    bool _is_invalidated_cache = true;
};

template <typename T>
class Source<T>::Builder final {
public:
    /// @brief Default-construct an empty builder.
    Builder() = default;

    /**
     * @brief Validate staged inputs and build a @ref Source by value.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Source<T>
    build();

    /**
     * @brief Build a @ref Source wrapped in a host shared pointer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Source<T>>
    make_host_shared();

    /**
     * @brief Copy a unit into the builder state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(const Unit<T>& unit) noexcept;

    /**
     * @brief Move a unit into the builder state.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(Unit<T>&& unit) noexcept;

    /**
     * @brief Set the fluid mixture used by the source.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    /**
     * @brief Set the spawn classification mode.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_type(SpawnType spawn_type) noexcept;

    /**
     * @brief Set spawn tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    /**
     * @brief Set local-space sample spacing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    /**
     * @brief Set the generator used for emitted particle velocities.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_generate_operator(GenerateOperator<T> generate_operator) noexcept;

private:
    /**
     * @brief Validate staged builder inputs or throw on invalid configuration.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<Unit<T>> _unit;
    FluidHostPtr<T> _fluid;
    SpawnType _spawn_type = SpawnType::Surface;
    GenerateOperator<T> _generate_operator { GenerateType::uniform };
    T _tolerance          = T(0);
    T _spacing            = T(0.1);
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using Source = atlas::system::Source<T>;

template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

} // namespace atlas

#include <atlas/source/source.hpp>
