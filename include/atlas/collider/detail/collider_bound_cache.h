#pragma once

/**
 * @file collider_bound_cache.h
 * @brief Declares the world-space bound cache used by Collider.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <type_traits>

namespace atlas::detail {

/**
 * @brief Maintains per-unit and scene-level world-space collider bounds.
 *
 * @tparam T Floating-point scalar type used by collider geometry.
 */
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

} // namespace atlas::detail

#include <atlas/collider/detail/collider_bound_cache.hpp>
