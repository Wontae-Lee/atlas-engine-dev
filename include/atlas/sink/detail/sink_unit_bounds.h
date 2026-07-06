#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

/**
 * @file sink_unit_bounds.h
 * @brief Per-unit AABB cache for `Sink`, the same broad-phase-reject-test
 *        role `atlas::UnitField` plays for `Collider` (see that file's
 *        top-of-file documentation for the general rationale), with one
 *        addition: bounds are expanded by a
 *        tolerance margin so a particle just outside a unit's exact
 *        geometry — but within the despawn tolerance — is not
 *        incorrectly culled by the box test before the exact query runs.
 */

namespace atlas::detail {

/** @brief Per-unit device functor: recomputes one unit's world-space
 *  AABB and optionally expands it by `expand` (the sink tolerance). */
struct SinkRefreshUnitBound final {
    const Unit* units {};
    atlas::AABB* bounds {};
    float expand {};

    ATLAS_ALL_DEVICE void
    operator()(const int unit_index) const {
        auto& world_bound = bounds[unit_index];

        world_bound = units[unit_index].world_bound();

        if (expand > 0.0f) {
            world_bound.expand(expand);
        }
    }
};

/** @brief Owns and refreshes the per-unit tolerance-expanded AABB cache
 *  used as `Sink`'s despawn broad-phase reject test. */
class SinkUnitBounds final {
public:
    /** @brief Recomputes every unit's world-space, tolerance-expanded
     *  AABB into `unit_bounds` (resized to match `units` if needed). */
    ATLAS_HOST void
    refresh(const DeviceBuffer<Unit>& units,
            DeviceBuffer<atlas::AABB>& unit_bounds,
            float tolerance) const noexcept;
};

}
