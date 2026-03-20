#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/data/particle_data.h>
#include <atlas/matter/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <optional>
#include <type_traits>

namespace atlas::system {

/**
 * @brief Runtime particle emitter that spawns particles based on geometric spawn conditions.
 *
 * @details
 * `Source<T>` mirrors @ref Sink structure:
 * - Bound @ref Unit supplies geometry and transform
 * - @ref SpawnOperator classifies spawn regions (Surface/Volume)
 * - Emits particles into available slots via @ref ParticleDeviceProbe
 *
 * Generates local-space positions from unit geometry, transforms them to world space
 * via `SyncOperator::sync_to_world`, and fills inactive particle slots.
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
     * @brief Constructs a source from a unit, spawn type, spacing, and tolerance.
     *
     * @param unit Unit defining geometry and transform.
     * @param spawn_type Surface or Volume spawn mode.
     * @param spacing Particle spacing.
     * @param tolerance Classification tolerance.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(Unit<T> unit,
           SpawnType spawn_type = SpawnType::Surface,
           T spacing            = T(0),
           T tolerance          = T(0)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Emits particles into available probe slots based on spawn criteria.
     *
     * @details
     * Generates local positions from unit geometry using the spawn operator
     * (Surface/Volume), transforms them to world space, and fills inactive slots
     * (`active == 0`) in the target probe. Initializes velocities/species from fluid.
     *
     * @param particle_probe Target particle storage probe.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(ParticleDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_unit(Unit<T> unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_type(SpawnType spawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Unit<T>&
    unit() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE SpawnType
    spawn_type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    spacing() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE int
    build_available_slot_indices(const ParticleDeviceProbe<T>& particle_probe,
                                 DeviceBuffer<int>& slot_indices,
                                 std::size_t requested_count) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update_species_distribution();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    initialize_emitted_particles(const ParticleDeviceProbe<T>& particle_probe,
                                 const DeviceBuffer<int>& slot_indices,
                                 int emit_count,
                                 DeviceBuffer<std::size_t>& emitted_species) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    assign_generated_velocities(const ParticleDeviceProbe<T>& particle_probe,
                                const DeviceBuffer<int>& slot_indices,
                                int emit_count,
                                const DeviceBuffer<std::size_t>& emitted_species) const;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    assign_species_velocities(const ParticleDeviceProbe<T>& particle_probe,
                              const DeviceBuffer<int>& slot_indices,
                              int emit_count,
                              const DeviceBuffer<std::size_t>& emitted_species,
                              std::size_t species,
                              const GeneratorHostPtr<T>& generator) const;

private:
    Unit<T> _unit;
    SpawnOperator<T> _spawn_operator { SpawnType::Surface };
    T _spacing   = T(0);
    T _tolerance = T(0);
    FluidHostPtr<T> _fluid;

    // Caching for performance
    DeviceBuffer<Vector3<T>> _local_positions_cache;
    DeviceBuffer<T> _species_cdf_device;
    int _species_count { 0 };
};

template <typename T>
class Source<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Source<T>
    build();

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Source<T>>
    make_host_shared();

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(const Unit<T>& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_unit(Unit<T>&& unit) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_type(SpawnType spawn_type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(const FluidHostPtr<T>& fluid) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    std::optional<Unit<T>> _unit;
    SpawnType _spawn_type = SpawnType::Surface;
    T _spacing            = T(0);
    T _tolerance          = T(0);
    FluidHostPtr<T> _fluid;
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
