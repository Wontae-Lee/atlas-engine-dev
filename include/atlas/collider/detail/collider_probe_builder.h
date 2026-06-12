#pragma once

/**
 * @file collider_probe_builder.h
 * @brief Declares helper routines that populate ColliderProbe.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/collider/collider_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstdint>
#include <type_traits>

namespace atlas::detail {

/**
 * @brief Populates the device-side collision probe from Collider-owned state.
 *
 * @tparam T Floating-point scalar type used by the collider.
 */
template <typename T>
class ColliderProbeBuilder final {
    static_assert(std::is_floating_point_v<T>, "ColliderProbeBuilder requires a floating-point T");

public:
    using Bound = atlas::AxisAlignedBoundingBox<T>;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    make(ColliderProbe<T>& probe,
         const DeviceBuffer<Unit<T>>& units,
         const DeviceBuffer<Bound>& unit_bounds,
         const DeviceBuffer<SurfaceInteractionKernel<T>>& surface_interactions,
         const DeviceBuffer<std::uint8_t>& flips,
         const Bound& scene_bound,
         bool scene_bound_covers_units,
         const FluidHostPtr<T>& fluid) noexcept;
};

} // namespace atlas::detail

#include <atlas/collider/detail/collider_probe_builder.hpp>
