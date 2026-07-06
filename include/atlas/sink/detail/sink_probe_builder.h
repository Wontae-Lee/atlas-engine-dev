#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/sink/sink_probe.h>
#include <atlas/unit/unit.h>

/**
 * @file sink_probe_builder.h
 * @brief Host-only factory that assembles a `SinkProbe` from a `Sink`'s
 *        buffers and its target `Fluid`.
 */

namespace atlas::detail {

/**
 * @brief Populates a `SinkProbe`'s raw-pointer/count fields from live
 *        buffers.
 */
class SinkProbeBuilder final {
public:
    /**
     * @brief Fills `probe` in place from `units`/`unit_bounds`/
     *        `despawn_operators`/`flip`/`tolerance`/`dt` and the
     *        position/velocity/active buffers owned by `fluid`.
     * @return `false` if `fluid` is null or currently holds zero
     *         particles, so callers can skip the despawn pass entirely;
     *         `true` otherwise.
     */
    ATLAS_HOST ATLAS_NODISCARD static bool
    make(const FluidHostPtr& fluid,
         const DeviceBuffer<Unit>& units,
         const DeviceBuffer<atlas::AABB>& unit_bounds,
         const DeviceBuffer<Despawn>& despawn_operators,
         bool flip,
         float tolerance,
         float dt,
         SinkProbe& probe) noexcept;
};

}
