#pragma once

/**
 * @file source.h
 * @brief Declares particle source logic and source builder utilities.
 *
 * A Source samples positions on one or more units, chooses species according to
 * fluid mole fractions, generates outgoing velocities, and writes new particles
 * into inactive fluid capacity.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/fluid/fluid.h>
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
 * @brief Emits new particles from configured units into a fluid buffer.
 *
 * Source combines spawn geometry, spawn operators, species composition, and
 * velocity generators. It also caches sampled local positions so repeated emit
 * calls can reuse geometry-dependent work until relevant configuration changes.
 *
 * @tparam T Floating-point scalar used by the simulation.
 */
template <typename T>
class Source final {
    static_assert(std::is_floating_point_v<T>, "Source requires a floating-point T");

public:
    class Builder;

public:
    Source()  = default;
    ~Source() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Source(DeviceBuffer<Unit<T>> units,
           DeviceBuffer<SpawnType> spawn_types,
           DeviceBuffer<SpawnOperator<T>> spawn_operators,
           FluidHostPtr<T> fluid,
           bool flip     = false,
           T tolerance   = T(0),
           T temperature = T(273.15)) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    update(T dt);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    emit(FluidDeviceProbe<T>& particle_probe);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(DeviceBuffer<Unit<T>> units) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_types(DeviceBuffer<SpawnType> spawn_types) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operators(DeviceBuffer<SpawnOperator<T>> spawn_operators) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_temperature(T temperature) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<Unit<T>>&
    units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Unit<T>>&
    units() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const FluidHostPtr<T>&
    fluid() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<SpawnType>&
    spawn_types() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<SpawnType>&
    spawn_types() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DeviceBuffer<SpawnOperator<T>>&
    spawn_operators() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<SpawnOperator<T>>&
    spawn_operators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    tolerance() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    flip() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    spacing() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    temperature() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Vector3<T>>&
    local_positions() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild_cache() noexcept;

private:
    DeviceBuffer<Unit<T>> _units;                    ///< Spawn units.
    DeviceBuffer<SpawnType> _spawn_types;            ///< Spawn types per unit.
    DeviceBuffer<SpawnOperator<T>> _spawn_operators; ///< Spawn rules per unit.
    FluidHostPtr<T> _fluid;                          ///< Fluid dependency for species and generators.
    bool _flip   = false;                            ///< Whether spawn decisions are inverted.
    T _tolerance = T(0);                             ///< Geometric tolerance used during spawn tests.
    T _spacing   = T(0.1);                           ///< Grid spacing used to sample candidate local positions.
    T _temperature { T(273.15) };                    ///< Emission temperature assigned to new particles.
    DeviceBuffer<Vector3<T>> _local_positions;       ///< Cached local spawn positions.
    DeviceBuffer<int> _local_position_unit_indices;  ///< Owning unit index per cached position.
    DeviceBuffer<size_t> _species_cache;             ///< Cached deterministic species layout.
    DeviceBuffer<size_t> _shuffled_species;          ///< Per-emission shuffled species assignment.
    DeviceBuffer<std::uint64_t> _shuffle_keys;       ///< Keys used to shuffle emitted species.
    std::uint64_t _shuffle_seed = 0;                 ///< Monotonic seed used to decorrelate emissions.
    bool _is_invalidated_cache  = true;              ///< Whether cached geometry-derived data must be rebuilt.
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
    with_units(const HostBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_fluid(FluidHostPtr<T> fluid) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_types(const HostBuffer<SpawnType>& spawn_types);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operator(const SpawnOperator<T>& spawn_operator) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spawn_operators(const HostBuffer<SpawnOperator<T>>& spawn_operators);

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_tolerance(T tolerance) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_flip(bool flip) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_spacing(T spacing) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    HostBuffer<Unit<T>> _units;
    FluidHostPtr<T> _fluid;
    HostBuffer<SpawnType> _spawn_types;
    HostBuffer<SpawnOperator<T>> _spawn_operators;
    bool _flip   = false;
    T _tolerance = T(0);
    T _spacing   = T(0.1);
    T _temperature { T(273.15) };
};

}

namespace atlas {

/**
 * @brief Convenience alias for atlas::system::Source.
 */
template <typename T>
using Source = atlas::system::Source<T>;

template <typename T>
using SourceHostPtr = atlas::host_shared_ptr<Source<T>>;

template <typename T>
using SourceDevicePtr = atlas::device_shared_ptr<Source<T>>;

}

#include <atlas/source/source.hpp>
