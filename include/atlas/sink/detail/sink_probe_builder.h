#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/sink/sink_probe.h>
#include <atlas/unit/unit.h>

namespace atlas::detail {

/**
 * @brief Builds device-readable sink probes.
 */
template <typename T>
class SinkProbeBuilder final {
public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    build(const FluidHostPtr<T>& fluid,
          const DeviceBuffer<Unit<T>>& units,
          const DeviceBuffer<atlas::AxisAlignedBoundingBox<T>>& unit_bounds,
          const DeviceBuffer<DespawnOperator<T>>& despawn_operators,
          bool flip,
          T tolerance,
          T dt,
          SinkProbe<T>& probe) const noexcept;
};

} // namespace atlas::detail

#include <atlas/sink/detail/sink_probe_builder.hpp>
