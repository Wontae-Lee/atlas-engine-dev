#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/container/type_store.h>
#include <atlas/core/macros.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace atlas {

/**
 * @brief Type-erased container of the per-cell field states owned by a Universe.
 *
 * A `TypeStore<UniverseState>` keyed by the concrete state type: at most one
 * instance of each `UniverseState` subclass (temperature, bulk velocity, the
 * DSMC counters, and so on) may live in the store at a time. Each entry is a
 * `std::unique_ptr<UniverseState>`, so the store owns the device buffers behind
 * every field. See `TypeStore` for the lookup/insertion semantics.
 */
using UniverseStateStore = TypeStore<UniverseState>;

/**
 * @brief The Eulerian background grid: a uniform, axis-aligned lattice of cubic
 *        cells covering the simulation domain, plus the per-cell field states.
 *
 * The Universe is the fixed grid counterpart to the Lagrangian particle set
 * (`Fluid`). It defines the cell geometry — a uniform grid of `cell_size`-edged
 * cubes spanning `[lower_corner, upper_corner]` — and owns an open-ended set of
 * per-cell field arrays (temperature, bulk velocity, DSMC collision statistics,
 * ...) stored one-per-type in a `UniverseStateStore`. Solvers reach the raw
 * device pointers through a trivially-copyable view (see `UniverseDsmcView`)
 * gathered on the host via `view<ViewT>()`.
 *
 * Move-only: it owns device buffers (via the state store) whose copy is
 * host-only and expensive, so copy construction/assignment are deleted while
 * move is defaulted.
 *
 * @note All grid scalars are host-side metadata; the field data itself lives in
 *       device memory inside the individual `UniverseState` leaves.
 */
class Universe {
public:
    /// Fluent builder that derives the grid from corners/cell size or a Geometry.
    class Builder;

    /// No default grid: the domain corners and cell size must always be supplied.
    Universe() = delete;

    /**
     * @brief Construct the grid from an explicit bounding box and cell size.
     *
     * Derives the cached grid metadata: `cell_volume = cell_size^3`, the inverse
     * cell size `1/cell_size`, the per-axis `grid_size` via `compute_grid_size`,
     * and the total `cell_count = grid_size.x * grid_size.y * grid_size.z`. The
     * state store starts empty; callers add field states afterwards.
     *
     * @param lower_corner Minimum corner of the domain, in world units.
     * @param upper_corner Maximum corner of the domain, in world units.
     * @param cell_size Edge length of a cubic cell, in world units; must be > 0.
     * @note Prefer `Builder`, which validates the arguments before constructing.
     */
    ATLAS_HOST
    Universe(const Float3& lower_corner,
             const Float3& upper_corner,
             float cell_size);

    /// Deleted: the Universe owns device buffers whose copy is host-only.
    Universe(const Universe&) = delete;

    /// Move construction is cheap (transfers the state store); defaulted.
    Universe(Universe&&) noexcept = default;

    /// Defaulted; the state store releases every owned device buffer.
    ~Universe() = default;

    /// Deleted: see the copy constructor.
    Universe&
    operator=(const Universe&)
        = delete;

    /// Move assignment transfers ownership of the grid and its states; defaulted.
    Universe&
    operator=(Universe&&) noexcept = default;

    /**
     * @brief Create an empty builder for the fluent construction path.
     * @return A default-initialized `Builder` (unit box, unit cell size).
     */
    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /**
     * @brief Construct a field state of type `StateT` in place and store it.
     *
     * Forwards `args` to `StateT`'s constructor and inserts the result into the
     * state store, replacing any existing instance of the same type. The typical
     * argument is the cell count, so the buffer is sized to one entry per cell.
     *
     * @tparam StateT Concrete `UniverseState` subclass to create.
     * @tparam Args Constructor argument types, forwarded to `StateT`.
     * @param args Constructor arguments (usually the cell count).
     * @return Reference to the newly stored state, valid until it is removed.
     */
    template <typename StateT, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE StateT&
    emplace_state(Args&&... args) {
        return _states.template emplace<StateT>(std::forward<Args>(args)...);
    }

    /**
     * @brief Move an already-built field state into the store.
     *
     * Replaces any existing instance of the same type. Ownership transfers to
     * the Universe.
     *
     * @tparam StateT Concrete `UniverseState` subclass.
     * @param state Non-null owning pointer; the underlying store throws on null.
     */
    template <typename StateT>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_state(std::unique_ptr<StateT> state) {
        _states.template set<StateT>(std::move(state));
    }

    /**
     * @brief Look up the stored field state of type `StateT`.
     * @tparam StateT Concrete `UniverseState` subclass to fetch.
     * @return Borrowed pointer to the state, or `nullptr` if none is stored. The
     *         pointer is owned by the store and must not be freed by the caller.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE StateT*
    state() noexcept {
        return _states.template get<StateT>();
    }

    /**
     * @brief Const overload of `state()`.
     * @tparam StateT Concrete `UniverseState` subclass to fetch.
     * @return Borrowed const pointer to the state, or `nullptr` if absent.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const StateT*
    state() const noexcept {
        return _states.template get<StateT>();
    }

    /**
     * @brief Test whether a field state of type `StateT` is present.
     * @tparam StateT Concrete `UniverseState` subclass to test for.
     * @return `true` if the store holds an instance of `StateT`.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    has_state() const noexcept {
        return _states.template contains<StateT>();
    }

    /**
     * @brief Detach and return the stored field state of type `StateT`.
     * @tparam StateT Concrete `UniverseState` subclass to remove.
     * @return Owning pointer to the removed state, or `nullptr` if none existed.
     *         Ownership passes to the caller; the store no longer references it.
     */
    template <typename StateT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::unique_ptr<StateT>
    remove_state() {
        return _states.template remove<StateT>();
    }

    /**
     * @brief Gather a device-capturable view of selected states.
     *
     * Delegates to `ViewT::make(*this)`, which collects raw device pointers and
     * grid scalars into a trivially-copyable struct so a device lambda can
     * capture it by value. Views are snapshots: they must be re-gathered after
     * any operation that reallocates a state buffer.
     *
     * @tparam ViewT A view type exposing `static ViewT make(Universe&)`.
     * @return The freshly gathered view, by value.
     */
    template <typename ViewT>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE ViewT
    view() {
        return ViewT::make(*this);
    }

    /**
     * @brief Total number of grid cells.
     * @return `grid_size.x * grid_size.y * grid_size.z`, guaranteed to fit `int`
     *         (the builder rejects grids that would overflow).
     */
    ATLAS_NODISCARD ATLAS_HOST int
    cell_count() const noexcept;

    /**
     * @brief Minimum corner of the domain bounding box, in world units.
     * @return The lower corner passed at construction.
     */
    ATLAS_NODISCARD ATLAS_HOST Float3
    lower_corner() const noexcept;

    /**
     * @brief Maximum corner of the domain bounding box, in world units.
     * @return The upper corner passed at construction.
     */
    ATLAS_NODISCARD ATLAS_HOST Float3
    upper_corner() const noexcept;

    /**
     * @brief Per-axis cell counts of the grid.
     * @return `(nx, ny, nz)`, each >= 1.
     */
    ATLAS_NODISCARD ATLAS_HOST Int3
    grid_size() const noexcept;

    /**
     * @brief Edge length of a cubic cell, in world units.
     * @return The cell size passed at construction.
     */
    ATLAS_NODISCARD ATLAS_HOST float
    cell_size() const noexcept;

    /**
     * @brief Volume of one cubic cell, in world units cubed.
     * @return `cell_size^3`, cached at construction; used e.g. as the number
     *         density denominator in the DSMC collision rate.
     */
    ATLAS_NODISCARD ATLAS_HOST float
    cell_volume() const noexcept;

    /**
     * @brief Reciprocal of the cell size.
     * @return `1 / cell_size`, cached to turn world-to-cell scaling into a
     *         multiply.
     */
    ATLAS_NODISCARD ATLAS_HOST float
    inverse_cell_size() const noexcept;

    /**
     * @brief Direct access to the underlying state store.
     * @return Mutable reference to the `UniverseStateStore`.
     */
    ATLAS_NODISCARD ATLAS_HOST UniverseStateStore&
    states() noexcept;

    /**
     * @brief Const access to the underlying state store.
     * @return Const reference to the `UniverseStateStore`.
     */
    ATLAS_NODISCARD ATLAS_HOST const UniverseStateStore&
    states() const noexcept;

private:
    /**
     * @brief Compute the per-axis cell counts covering the given box.
     *
     * `floor((upper - lower) * inverse_cell_size) + 1` on each axis, so a box
     * whose extent is not an exact multiple of the cell size is rounded up to a
     * full covering cell and every non-degenerate box yields at least one cell.
     *
     * @param lower_corner Minimum corner of the domain.
     * @param upper_corner Maximum corner of the domain.
     * @param inverse_cell_size Reciprocal of the cell size (`1 / cell_size`).
     * @return Per-axis cell counts `(nx, ny, nz)`.
     */
    ATLAS_NODISCARD ATLAS_HOST static Int3
    compute_grid_size(const Float3& lower_corner,
                      const Float3& upper_corner,
                      float inverse_cell_size) noexcept;

    Float3 _lower_corner; ///< Minimum corner of the domain, world units.

    Float3 _upper_corner; ///< Maximum corner of the domain, world units.

    Int3 _grid_size = Int3(1, 1, 1); ///< Per-axis cell counts, each >= 1.

    float _cell_size = 1.0f; ///< Cubic cell edge length, world units.

    float _cell_volume = 1.0f; ///< Cached `cell_size^3`, world units cubed.

    float _inv_h = 1.0f; ///< Cached `1 / cell_size`.

    int _cell_count = 1; ///< Cached product of `_grid_size` components.

    UniverseStateStore _states; ///< Owns the per-cell field states, one per type.
};

/**
 * @brief Fluent builder for `Universe`.
 *
 * Collects the domain corners and cell size — either set directly or derived
 * from a `Geometry`'s bounding box — plus optional initial cell fields,
 * validates them, and constructs the grid.
 * Defaults describe the unit box `[(0,0,0), (1,1,1)]` with a unit cell size.
 */
class Universe::Builder final {
public:
    /// Start from the default unit-box, unit-cell configuration.
    Builder() = default;

    /**
     * @brief Take the domain corners from a geometry's bounding box.
     *
     * Overwrites both corners with `geometry.bound().lower_corner` and
     * `upper_corner`; the cell size is left untouched.
     *
     * @param geometry Geometry whose axis-aligned bound defines the domain.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_geometry(const Geometry& geometry);

    /**
     * @brief Set the minimum corner of the domain.
     * @param v Lower corner in world units.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_lower_corner(const Float3& v) noexcept;

    /**
     * @brief Set the maximum corner of the domain.
     * @param v Upper corner in world units.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_upper_corner(const Float3& v) noexcept;

    /**
     * @brief Set the cubic cell edge length.
     * @param h Cell size in world units; `build()` requires it to be > 0.
     * @return `*this`, for chaining.
     */
    ATLAS_HOST Builder&
    with_cell_size(float h) noexcept;

    /** @brief Stage one temperature value per computed grid cell. */
    ATLAS_HOST Builder& with_temperature(HostBuffer<float> values);
    /** @brief Stage one bulk velocity per computed grid cell. */
    ATLAS_HOST Builder& with_bulk_velocity(HostBuffer<Float3> values);
    /** @brief Stage one field force per computed grid cell. */
    ATLAS_HOST Builder& with_field_force(HostBuffer<Float3> values);
    /** @brief Stage one gravity value per computed grid cell. */
    ATLAS_HOST Builder& with_gravity(HostBuffer<Float3> values);
    /** @brief Stage one thermal energy per computed grid cell. */
    ATLAS_HOST Builder& with_thermal_energy(HostBuffer<float> values);
    /** @brief Stage one Knudsen number per computed grid cell. */
    ATLAS_HOST Builder& with_knudsen_number(HostBuffer<float> values);

    /**
     * @brief Validate the configuration and construct the grid by value.
     * @return A fully-initialized `Universe`.
     * @throws std::invalid_argument if the cell size or domain is invalid, the
     *         resulting cell count overflows `int`, or a supplied cell field has
     *         a length different from the resulting cell count.
     */
    ATLAS_NODISCARD ATLAS_HOST Universe
    build() const;

    /**
     * @brief Build the grid and wrap it in a host-side unique pointer.
     * @return An owning `host_unique_ptr<Universe>`.
     * @throws std::invalid_argument on the same conditions as `build()`.
     */
    ATLAS_NODISCARD ATLAS_HOST atlas::host_unique_ptr<Universe>
    make_host_unique() const;

private:
    /**
     * @brief Check the pending configuration, throwing on any invalid field.
     *
     * Verifies finite positive cell size, finite ordered corners, a grid that
     * fits in `int`, and supplied cell fields whose lengths match its cell count.
     *
     * @throws std::invalid_argument describing the first failed constraint.
     */
    ATLAS_HOST void
    validate() const;

private:
    Float3 _lower_corner = Float3(0.0f, 0.0f, 0.0f); ///< Pending lower corner.

    Float3 _upper_corner = Float3(1.0f, 1.0f, 1.0f); ///< Pending upper corner.

    float _cell_size = 1.0f; ///< Pending cubic cell edge length.

    std::optional<HostBuffer<float>> _temperature; ///< Supplied cell temperatures.
    std::optional<HostBuffer<Float3>> _bulk_velocity; ///< Supplied cell bulk velocities.
    std::optional<HostBuffer<Float3>> _field_force; ///< Supplied cell forces.
    std::optional<HostBuffer<Float3>> _gravity; ///< Supplied cell gravity values.
    std::optional<HostBuffer<float>> _thermal_energy; ///< Supplied cell thermal energies.
    std::optional<HostBuffer<float>> _knudsen_number; ///< Supplied cell Knudsen numbers.
};

/// Owning host-side handle to a `Universe`, as produced by the builder.
using UniverseHostPtr = atlas::host_unique_ptr<Universe>;

}
