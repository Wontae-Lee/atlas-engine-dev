#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sink/detail/sink_unit_bounds.h>

namespace atlas::detail {

void
SinkUnitBounds::refresh(const DeviceBuffer<Unit>& units,
                        DeviceBuffer<atlas::AABB>& unit_bounds,
                        const float tolerance) const noexcept {
    if (units.empty()) {
        unit_bounds.clear();
        return;
    }

    if (unit_bounds.size() != units.size()) {
        unit_bounds.resize(units.size());
    }

    auto* units_ptr      = atlas::raw_pointer_cast(units.data());
    auto* bounds_ptr     = atlas::raw_pointer_cast(unit_bounds.data());
    const int unit_count = static_cast<int>(units.size());
    // A negative tolerance (an "inside-only" despawn margin) must not
    // shrink the broad-phase box below the unit's exact geometry, or the
    // AABB reject test could wrongly cull a particle the exact query would
    // still have accepted.
    const float expand   = tolerance > 0.0f ? tolerance : 0.0f;

    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        unit_count,
        SinkRefreshUnitBound {
            units_ptr,
            bounds_ptr,
            expand });
}

}
