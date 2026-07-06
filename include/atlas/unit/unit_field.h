#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>

/**
 * @file unit_field.h
 * @brief A set of `Unit`s owned as one device buffer, together with the
 *        shared per-step machinery every unit consumer needs: pose
 *        integration and a per-unit / scene-level world-space AABB cache.
 *
 * @details
 * `Source`, `Sink`, `Collider`, and `VolumeMeasurer` all hold a group of
 * boundary/region units and, each step, (a) advance every unit's own
 * motion (`Unit::update`) and (b) recompute each unit's world-space AABB
 * (for broad-phase reject tests). `UnitField` centralizes that duplicated
 * ownership + machinery in one place so those consumers hold a single
 * `UnitField` instead of a bare `DeviceBuffer<Unit>` plus an ad-hoc bound
 * cache; it is also the unit-store type `Universe` is intended to own.
 *
 * `advance(dt)` integrates every unit's kinematics; `refresh_bounds()`
 * recomputes `unit_bounds()` (each unit's world-space AABB via
 * `Unit::world_bound()`), the merged `scene_bound()`, and `covers_units()`
 * (whether `scene_bound()` is a valid superset of every unit bound — false
 * when any unit is unbounded, gating whether a scene-level early-out is
 * safe). Neither is auto-invalidated: `advance` then `refresh_bounds` must
 * be called in that order each step whenever units may have moved.
 */

namespace atlas {

/**
 * @brief Owns a group of `Unit`s and the per-step pose-integration and
 *        world-space AABB cache shared by every unit consumer. See this
 *        file's top-of-file documentation.
 */
class UnitField final {
public:
    using Bound = atlas::AABB;

    UnitField() = default;

    ATLAS_HOST explicit UnitField(DeviceBuffer<Unit> units);

    /** @brief Integrates every unit's own motion by `dt` (a no-op for
     *  `dt <= 0` or an empty field). */
    ATLAS_HOST void
    advance(float dt);

    /** @brief Recomputes every unit's world-space AABB (`unit_bounds()`),
     *  the merged `scene_bound()`, and `covers_units()`. Call after
     *  `advance()` whenever units may have moved. */
    ATLAS_HOST void
    refresh_bounds();

    /** @brief The units, as one device buffer. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Unit>&
    units() const noexcept;

    /** @brief Per-unit world-space bounds, indexed like `units()`. */
    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Bound>&
    unit_bounds() const noexcept;

    /** @brief Union of every unit's bound as of the last `refresh_bounds()`. */
    ATLAS_NODISCARD ATLAS_HOST const Bound&
    scene_bound() const noexcept;

    /** @brief Whether `scene_bound()` is a valid, up-to-date superset of
     *  every unit's bound (safe as a broad-phase reject test). */
    ATLAS_NODISCARD ATLAS_HOST bool
    covers_units() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST std::size_t
    size() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST bool
    empty() const noexcept;

private:
    DeviceBuffer<Unit> _units;

    DeviceBuffer<Bound> _unit_bounds;

    Bound _scene_bound {};

    bool _covers_units {};
};

}
