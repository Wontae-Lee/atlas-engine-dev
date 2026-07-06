#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/material_properties.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/**
 * @file fluid.h
 * @brief The particle population every solver/collider/sink/source/
 *        measurer in Atlas ultimately operates on: a fixed-capacity
 *        structure-of-arrays store of per-particle attribute buffers
 *        (`FluidState`s) plus per-species material/generator tables.
 *
 * @details
 * ### Operating principle
 * `Fluid` separates *capacity* from *live count*: `buffer_size()` is
 * the fixed allocation size every registered `FluidState` is sized to,
 * while `particle_count()` is how many of those slots currently hold a
 * live particle (`<= buffer_size()`). This split is what lets
 * `Source::emit()` add particles cheaply (write into
 * `[particle_count, particle_count + emit_count)`, then bump the count —
 * no reallocation as long as slots remain) and `Sink::compact_fluid_particles()`
 * shrink the live count without touching the underlying allocation.
 *
 * `_states` (`FluidStateStore`, a `TypeStore<FluidState>` — a
 * type-indexed heterogeneous container, see `container/type_store.h`)
 * holds whichever `FluidState`s this fluid actually needs
 * (`FluidPositionState`, `FluidVelocityState`, ..., see
 * `fluid_state.h`); the templated `emplace_state`/`state`/`has_state`/
 * `remove_state` accessors are inline in the class body (the
 * `TypeStore`-backed template-member convention shared with `Universe`
 * — see `docs/updates/updates.md` §2.12), so a caller can add exactly
 * the states its simulation needs without every `Fluid` paying for
 * every possible state.
 *
 * `_particle_properties`/`_generators` are parallel per-species arrays
 * (indexed by `FluidSpeciesState`): `particle_properties()[species]` is
 * that species' `MaterialProperties` (mass, cross-section, ...) and
 * `generators()[species]` is the `Generate` `Source` draws new
 * particles' velocities from for that species.
 * `statistical_weight()` (`F_N`) is the real-molecules-per-simulated-
 * particle scaling used throughout the DSMC number-density/collision-rate
 * formulas (see `dsmc_solver.h`, `KnudsenCodec::knudsen_number`).
 */

namespace atlas {

using FluidStateStore = TypeStore<FluidState>;

/**
 * @brief Fixed-capacity particle population: per-particle state buffers
 *        plus per-species material/generator tables. See this file's
 *        top-of-file documentation for the buffer-size/particle-count
 *        split and the state-store pattern.
 */
class Fluid final {
public:
    class Builder;

    Fluid() = default;

    ATLAS_HOST explicit Fluid(std::size_t buffer_size);

    Fluid(const Fluid&) = delete;

    Fluid(Fluid&&) noexcept = default;

    ~Fluid() = default;

    Fluid&
    operator=(const Fluid&)
        = delete;

    Fluid&
    operator=(Fluid&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    /** Per-species velocity generator table, indexed by
     *  `FluidSpeciesState`; see `generate.h`. */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<Generate>&
    generators() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<Generate>&
    generators() noexcept;

    /** Per-species material properties table, indexed by
     *  `FluidSpeciesState`; see `material_properties.h`. */
    ATLAS_HOST ATLAS_NODISCARD const DeviceBuffer<MaterialProperties>&
    particle_properties() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD DeviceBuffer<MaterialProperties>&
    particle_properties() noexcept;

    /** @brief Sets the live particle count (must not exceed
     *  `buffer_size()`); used by `Source`/`Sink` after emitting/removing
     *  particles. */
    ATLAS_HOST void
    set_particle_count(std::size_t particle_count);

    /** @brief Constructs and registers a `StateT` (e.g.
     *  `FluidVelocityState`) in this fluid's state store, replacing any
     *  existing one of the same type. */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    /** @brief Installs an already-constructed `StateT`, replacing any
     *  existing one of the same type. */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

    /** @brief The registered `StateT`, or `nullptr` if this fluid does
     *  not have one. */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE StateT*
    state() noexcept {
        return _states.template get<StateT>();
    }

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const StateT*
    state() const noexcept {
        return _states.template get<StateT>();
    }

    /** @brief Whether a `StateT` is currently registered. */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    /** @brief Unregisters and returns a `StateT`, or `nullptr` if none
     *  was registered. */
    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    /** @brief The underlying heterogeneous state store; for callers
     *  that need to iterate every registered state generically (e.g.
     *  `Sink::compact_fluid_particles`). */
    ATLAS_HOST ATLAS_NODISCARD FluidStateStore&
    states() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const FluidStateStore&
    states() const noexcept;

    /** @brief Fixed allocation size every registered `FluidState` is
     *  sized to; see this file's top-of-file documentation for the
     *  buffer-size/particle-count split. */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    buffer_size() const noexcept;

    /** @brief Current live particle count (`<= buffer_size()`). */
    ATLAS_HOST ATLAS_NODISCARD std::size_t
    particle_count() const noexcept;

    /** @brief Real molecules per simulated particle (`F_N`); see this
     *  file's top-of-file documentation. */
    ATLAS_HOST ATLAS_NODISCARD float
    statistical_weight() const noexcept;

    /** @brief Optional attached observer for sensor-metrics recording
     *  (e.g. `SinkSensorMetrics`/`SourceSensorMetrics`); may be null. */
    ATLAS_HOST ATLAS_NODISCARD const ObserverHostPtr&
    observer() const noexcept;

    /** @brief Serializes this fluid's particle state to a binary
     *  snapshot at `path` (see `serialization/protobuf_snapshot.h`). */
    ATLAS_HOST void
    save(std::string_view path) const;

private:
    friend class Builder;

    DeviceBuffer<MaterialProperties> _particle_properties;

    DeviceBuffer<Generate> _generators;

    std::size_t _particle_count = 0;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    ObserverHostPtr _observer {};

    FluidStateStore _states;
};

/**
 * @brief Fluent builder for `Fluid`. `with_properties`/`with_generators`
 *        set the parallel per-species tables (must end up the same
 *        length); `with_binary` restores particle state from a saved
 *        snapshot instead of starting empty.
 */
class Fluid::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_NODISCARD Fluid
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<Fluid>
    make_host_shared() const;

    /** @brief Per-species material properties table. */
    ATLAS_HOST Builder&
    with_properties(const HostBuffer<MaterialProperties>& properties);

    /** @brief Per-species velocity generators, converted to their
     *  device-callable `Generate` values. */
    ATLAS_HOST Builder&
    with_generators(const HostBuffer<GeneratorHostPtr>& generators);

    /** @brief Fixed particle-buffer capacity; required. */
    ATLAS_HOST Builder&
    with_buffer_size(std::size_t buffer_size) noexcept;

    /** @brief Real molecules per simulated particle (`F_N`); defaults
     *  to `1.0`. */
    ATLAS_HOST Builder&
    with_statistical_weight(float statistical_weight) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /** @brief Restores particle state (position/velocity/species/
     *  active/temperature, particle count) from a saved binary
     *  snapshot at `path`, instead of an empty fluid. */
    ATLAS_HOST Builder&
    with_binary(const std::string& path);

private:
    ATLAS_HOST void
    validate() const;

private:
    HostBuffer<MaterialProperties> _particles;

    HostBuffer<Generate> _generators;

    std::size_t _buffer_size = 0;

    float _statistical_weight = 1.0f;

    ObserverHostPtr _observer {};

    std::optional<HostBuffer<Vector3>> _position_state;

    std::optional<HostBuffer<Vector3>> _velocity_state;

    std::optional<HostBuffer<std::size_t>> _species_state;

    std::optional<HostBuffer<int>> _active_state;

    std::optional<HostBuffer<float>> _temperature_state;

    std::optional<std::size_t> _particle_count;
};

using FluidHostPtr = atlas::host_shared_ptr<Fluid>;

using FluidDevicePtr = atlas::device_shared_ptr<Fluid>;

}
