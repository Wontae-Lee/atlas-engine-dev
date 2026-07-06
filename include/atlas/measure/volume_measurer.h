#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/math/math.h>
#include <atlas/measure/measurer.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

/**
 * @file volume_measurer.h
 * @brief Estimates each cell's *free* (fluid-accessible) volume by
 *        sub-sampling how much of the cell is occluded by embedded
 *        solid geometry — a cut-cell volume-fraction correction needed
 *        wherever solid boundaries don't align with the simulation
 *        grid.
 *
 * @details
 * ### Background
 * DSMC/kinetic solvers derive number density from `n = particle_count *
 * F_N / cell_volume` (see e.g. `KnudsenCodec::knudsen_number`,
 * `dsmc_solver.h`'s NTC formula). That is only correct if `cell_volume`
 * is the volume actually *available* to particles; a cell partially
 * overlapped by a solid `Unit` (a wall, an embedded obstacle) has less
 * accessible volume than its nominal grid-cell volume, and using the
 * nominal volume would understate density/overstate collision rate in
 * every such cell. `VolumeMeasurer` corrects for this: it writes a
 * per-cell "free volume" into `UniverseVolumeState`, which
 * volume-dependent calculations elsewhere can read instead of the
 * uniform grid `cell_volume`.
 *
 * ### Operating principle — deterministic sub-grid volume-fraction estimate
 * With no embedded units, free volume is simply the nominal
 * `cell_volume` everywhere (cheap fast path). Otherwise:
 * 1. Per unit, computes a conservative cell-index range (`UnitRegion`)
 *    the unit's world-space AABB could possibly overlap (expanded by
 *    one cell in each direction as a safety margin), so the per-cell
 *    inner loop can skip units whose bounding box doesn't reach that
 *    cell at all.
 * 2. Per cell, per `samples_per_axis^3` regularly-spaced sample point
 *    within the cell (a deterministic grid, not Monte Carlo/random
 *    sampling — same "regular lattice at fixed spacing" idea as
 *    `detail::SourceCacheBuilder`'s candidate positions), tests whether
 *    that point falls inside any (region-relevant) unit's exact
 *    geometry (`Geometry::is_inside`, in the unit's local
 *    frame). A sample covered by *any* unit counts as occluded (breaks
 *    out of the per-sample unit loop on the first hit; stops sampling
 *    the whole cell early once every sample so far is occluded).
 * 3. `free_volume = cell_volume - occluded_sample_count * (cell_volume /
 *    samples_per_cell)` — i.e. the occluded fraction of samples scales
 *    the nominal cell volume down. Accuracy improves with
 *    `samples_per_axis` (more samples resolve the solid/fluid boundary
 *    more finely within a cell) at the cost of more work per cell
 *    (`O(samples_per_axis^3)`).
 */

namespace atlas {

/**
 * @brief Per-cell free-volume estimator for cut cells partially
 *        occluded by embedded solid geometry. See this file's
 *        top-of-file documentation for why free volume matters and the
 *        sub-sampling algorithm.
 */
class VolumeMeasurer final : public Measurer {
public:
    class Builder;

public:
    /** @brief Conservative cell-index bounding box a unit's world-space
     *  AABB could overlap, used to skip irrelevant units per cell
     *  during sampling. `active == false` marks a unit whose bound
     *  doesn't overlap the grid at all this step. */
    struct UnitRegion {
        Int3 begin {};
        Int3 end {};
        bool active {};

        /** @brief Whether `cell` falls within this unit's conservative
         *  index range (a necessary, not sufficient, condition for the
         *  cell to actually be occluded by this unit). */
        ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
        contains(const Int3& cell) const noexcept {
            return active && atlas::all((cell >= begin) & (cell <= end));
        }
    };

public:
    VolumeMeasurer() = default;

    ATLAS_HOST explicit VolumeMeasurer(UniverseHostPtr universe,
                                       int samples_per_axis = 4) noexcept;

    ~VolumeMeasurer() override = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    /** @brief `measure(0.0f)` — units are not advanced. */
    ATLAS_HOST void
    measure() override;

    /** @brief Advances the occluding units by `dt`, then recomputes
     *  every cell's free volume (`measure_volume()`); no-op if this
     *  measurer has no universe. */
    ATLAS_HOST void
    measure(float dt) override;

    /** @brief Always `MeasureModeType::field` — this measurer only
     *  writes per-cell `UniverseVolumeState`. */
    ATLAS_NODISCARD ATLAS_HOST MeasureModeType
    measure_mode() const noexcept override;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Unit>&
    units() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST int
    samples_per_axis() const noexcept;

private:
    /** @brief Allocates/resizes `UniverseVolumeState` to the universe's
     *  current cell count. */
    ATLAS_HOST void
    ensure_state();

public:
    /** @brief Advances every occluding unit's own motion by `dt`
     *  (no-op for `dt <= 0`). */
    ATLAS_HOST void
    update_units(float dt) noexcept;

    /** @brief Recomputes every cell's free volume via the sub-sampling
     *  algorithm in this file's top-of-file documentation. */
    ATLAS_HOST void
    measure_volume();

private:
    DeviceBuffer<UnitRegion> _unit_regions;

    int _samples_per_axis { 4 };
};

/**
 * @brief Fluent builder for `VolumeMeasurer`. Validation requires a
 *        non-null `_universe` and positive `_samples_per_axis`.
 */
class VolumeMeasurer::Builder final {
public:
    Builder() = default;

    /** @brief The domain that owns this measurer's region units
     *  (registered via `Universe::Builder::with_measurer_units`); required,
     *  non-null. */
    ATLAS_HOST Builder&
    with_universe(UniverseHostPtr universe) noexcept;

    ATLAS_HOST Builder&
    with_samples_per_axis(int samples_per_axis);

    ATLAS_NODISCARD ATLAS_HOST VolumeMeasurer
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<VolumeMeasurer>
    make_host_shared() const;

private:
    ATLAS_HOST void
    validate() const;

private:
    UniverseHostPtr _universe {};

    int _samples_per_axis { 4 };
};

using VolumeMeasurerHostPtr = atlas::host_shared_ptr<VolumeMeasurer>;

using VolumeMeasurerDevicePtr = atlas::device_shared_ptr<VolumeMeasurer>;

}
