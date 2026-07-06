#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>

namespace atlas {

class UnitField final {
public:
    using Bound = atlas::AABB;

    UnitField() = default;

    ATLAS_HOST explicit UnitField(DeviceBuffer<Unit> units);

    ATLAS_HOST void
    advance(float dt);

    ATLAS_HOST void
    refresh_bounds();

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Unit>&
    units() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const DeviceBuffer<Bound>&
    unit_bounds() const noexcept;

    ATLAS_NODISCARD ATLAS_HOST const Bound&
    scene_bound() const noexcept;

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
