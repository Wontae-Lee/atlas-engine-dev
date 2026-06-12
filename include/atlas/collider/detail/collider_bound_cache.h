#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <type_traits>

namespace atlas::detail {

template <typename T>
class ColliderBoundCache final {
    static_assert(std::is_floating_point_v<T>, "ColliderBoundCache requires a floating-point T");

public:
    using Bound = atlas::AxisAlignedBoundingBox<T>;

    ColliderBoundCache() = default;

    explicit ColliderBoundCache(std::size_t unit_count);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    refresh(const DeviceBuffer<Unit<T>>& units);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const DeviceBuffer<Bound>&
    unit_bounds() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Bound&
    scene_bound() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    covers_units() const noexcept;

private:
    DeviceBuffer<Bound> _unit_bounds;
    Bound _scene_bound {};
    bool _covers_units {};
};

}

#include <atlas/collider/detail/collider_bound_cache.hpp>