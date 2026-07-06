#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/observer/observer.h>
#include <atlas/unit/unit.h>
#include <atlas/unit/unit_field.h>
#include <atlas/universe/universe_state.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

/**
 * @file universe.h
 * @brief The fixed, axis-aligned uniform grid every spatially-indexed
 *        system in Atlas partitions particles/cells over (`Searcher`,
 *        `DsmcSolver`, `SphSolver`, `KnudsenCodec`, the measurers, ...),
 *        plus its heterogeneous per-cell state store.
 *
 * @details
 * ### Operating principle
 * A `Universe` is defined by `lower_corner`/`upper_corner` (the
 * simulated domain's world-space bounds) and `cell_size` (a single
 * scalar — cells are uniform cubes, not an adaptive/non-uniform grid).
 * `compute_grid_size()` derives the integer cell counts per axis:
 * `floor((upper - lower) * inverse_cell_size) + 1` — the `+1` ensures
 * the grid always covers at least one cell along each axis and rounds
 * up any partial trailing cell rather than truncating it away (the
 * domain's actual extent may not be an exact multiple of `cell_size`).
 * `cell_count()` is the product of the three per-axis grid sizes, and
 * `cell_volume()` is `cell_size^3`.
 *
 * `_states` (`UniverseStateStore`, see `universe_state.h`) holds
 * whichever per-cell states the attached solvers/codecs/measurers
 * currently need, the same only-pay-for-what-you-use pattern
 * `Fluid`/`FluidStateStore` uses for per-particle state (templated
 * `emplace_state`/`state`/`has_state`/`remove_state` accessors are
 * inline in the class body, per the shared `TypeStore`-backed
 * convention — see `docs/updates/updates.md` §2.12).
 */

namespace atlas {

using UniverseStateStore = TypeStore<UniverseState>;

/**
 * @brief The fixed uniform-grid simulation domain and its per-cell
 *        state store. See this file's top-of-file documentation for
 *        the grid-sizing derivation and state-store pattern.
 */
class Universe {
public:
    class Builder;

    Universe() = delete;

    ATLAS_HOST Universe(const Vector3& lower_corner,
                        const Vector3& upper_corner,
                        float cell_size);

    Universe(const Universe&) = delete;

    Universe(Universe&&) noexcept = default;

    ~Universe() = default;

    Universe&
    operator=(const Universe&)
        = delete;

    Universe&
    operator=(Universe&&) noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

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

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    template <typename StateT>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    /** @brief Total number of cells (`grid_size().x * .y * .z`). */
    ATLAS_HOST ATLAS_NODISCARD int
    cell_count() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3
    lower_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD Vector3
    upper_corner() const noexcept;

    /** @brief Per-axis cell counts; see this file's top-of-file
     *  documentation for the `compute_grid_size` derivation. */
    ATLAS_HOST ATLAS_NODISCARD Vector3i
    grid_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    cell_size() const noexcept;

    /** @brief `cell_size()^3`. */
    ATLAS_HOST ATLAS_NODISCARD float
    cell_volume() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD float
    inverse_cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD const ObserverHostPtr&
    observer() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD UniverseStateStore&
    states() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const UniverseStateStore&
    states() const noexcept;

    /**
     * @brief The boundary/region units owned centrally by the domain, one
     *        `UnitField` per consumer role. `Source`/`Sink`/`Collider`/
     *        `VolumeMeasurer` no longer own their units — they borrow the
     *        matching field through the `UniverseHostPtr` they are built
     *        with, so unit pose integration and world-bound caching happen
     *        once here rather than being re-implemented per role.
     */
    ATLAS_HOST ATLAS_NODISCARD UnitField&
    source_units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const UnitField&
    source_units() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD UnitField&
    sink_units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const UnitField&
    sink_units() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD UnitField&
    collider_units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const UnitField&
    collider_units() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD UnitField&
    measurer_units() noexcept;

    ATLAS_HOST ATLAS_NODISCARD const UnitField&
    measurer_units() const noexcept;

    /** @brief Serializes this universe's state to a binary snapshot at
     *  `path` (see `serialization/protobuf_snapshot.h`). */
    ATLAS_HOST void
    save(std::string_view path) const;

private:
    /** @brief `floor((upper - lower) * inverse_cell_size) + 1` per
     *  axis; see this file's top-of-file documentation. */
    ATLAS_HOST ATLAS_NODISCARD static Vector3i
    compute_grid_size(const Vector3& lower_corner,
                      const Vector3& upper_corner,
                      float inverse_cell_size) noexcept;

    Vector3 _lower_corner;

    Vector3 _upper_corner;

    Vector3i _grid_size = Vector3i(1, 1, 1);

    float _cell_size = 1.0f;

    float _cell_volume = 1.0f;

    float _inv_h = 1.0f;

    int _cell_count = 1;

    ObserverHostPtr _observer {};

    UniverseStateStore _states;

    UnitField _source_units;

    UnitField _sink_units;

    UnitField _collider_units;

    UnitField _measurer_units;
};

/**
 * @brief Fluent builder for `Universe`. `with_geometry` is a
 *        convenience alternative to `with_lower_corner`/`with_upper_corner`:
 *        it derives the domain bounds directly from a `Geometry`'s
 *        world-space AABB instead of specifying corners manually.
 *        `with_binary` restores per-cell state from a saved snapshot.
 */
class Universe::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_NODISCARD Universe
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<Universe>
    make_host_shared() const;

    /** @brief Sets `_lower_corner`/`_upper_corner` to `geometry`'s
     *  world-space bounding box, instead of setting them individually. */
    ATLAS_HOST Builder&
    with_geometry(const Geometry& geometry);

    ATLAS_HOST Builder&
    with_lower_corner(const Vector3& v) noexcept;

    ATLAS_HOST Builder&
    with_upper_corner(const Vector3& v) noexcept;

    /** @brief The uniform cube cell size (`h`); required, must be
     *  positive. */
    ATLAS_HOST Builder&
    with_cell_size(float h) noexcept;

    ATLAS_HOST Builder&
    with_observer(ObserverHostPtr observer) noexcept;

    /** @brief Registers the boundary units the `Source` will emit from;
     *  the built `Universe` owns them (see `Universe::source_units`). */
    ATLAS_HOST Builder&
    with_source_units(const HostBuffer<Unit>& units);

    /** @brief Registers the boundary units the `Sink` will remove
     *  particles at; the built `Universe` owns them. */
    ATLAS_HOST Builder&
    with_sink_units(const HostBuffer<Unit>& units);

    /** @brief Registers the boundary units the `Collider` will collide
     *  particles against; the built `Universe` owns them. */
    ATLAS_HOST Builder&
    with_collider_units(const HostBuffer<Unit>& units);

    /** @brief Registers the region units the `VolumeMeasurer` measures
     *  occupied volume for; the built `Universe` owns them. */
    ATLAS_HOST Builder&
    with_measurer_units(const HostBuffer<Unit>& units);

    /** @brief Restores per-cell state (temperature/bulk velocity/field
     *  force/...) from a saved binary snapshot at `path`. */
    ATLAS_HOST Builder&
    with_binary(const std::string& path);

private:
    ATLAS_HOST void
    validate() const;

private:
    Vector3 _lower_corner = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 _upper_corner = Vector3(1.0f, 1.0f, 1.0f);

    float _cell_size = 1.0f;

    ObserverHostPtr _observer {};

    HostBuffer<Unit> _source_units;

    HostBuffer<Unit> _sink_units;

    HostBuffer<Unit> _collider_units;

    HostBuffer<Unit> _measurer_units;

    std::optional<HostBuffer<float>> _temperature_state;

    std::optional<HostBuffer<Vector3>> _bulk_velocity_state;

    std::optional<HostBuffer<Vector3>> _field_force_state;

    std::optional<HostBuffer<float>> _max_relative_speed_state;

    std::optional<HostBuffer<float>> _thermal_energy_state;

    std::optional<HostBuffer<float>> _number_particle_state;

    std::optional<HostBuffer<int>> _collision_count_state;

    std::optional<HostBuffer<float>> _knudsen_number_state;
};

using UniverseHostPtr = atlas::host_shared_ptr<Universe>;

using UniverseDevicePtr = atlas::device_shared_ptr<Universe>;

}
