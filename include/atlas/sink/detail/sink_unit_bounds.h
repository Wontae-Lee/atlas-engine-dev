#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

namespace atlas::detail {

template <typename T>
class SinkUnitBounds final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    refresh(const DeviceBuffer<Unit<T>>& units,
            DeviceBuffer<atlas::AxisAlignedBoundingBox<T>>& unit_bounds,
            T tolerance) const noexcept;
};

}

#include <atlas/sink/detail/sink_unit_bounds.hpp>