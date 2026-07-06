#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

namespace atlas::detail {

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

class SinkUnitBounds final {
public:
    ATLAS_HOST void
    refresh(const DeviceBuffer<Unit>& units,
            DeviceBuffer<atlas::AABB>& unit_bounds,
            float tolerance) const noexcept;
};

}
