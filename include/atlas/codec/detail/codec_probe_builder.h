#pragma once

/**
 * @file codec_probe_builder.h
 * @brief Declares helper routines that populate CodecProbe.
 */

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::detail {

/**
 * @brief Populates the device-side codec probe from codec-owned state.
 *
 * @tparam T Floating-point scalar type used by the codec.
 */
template <typename T>
class CodecProbeBuilder final {
    static_assert(std::is_floating_point_v<T>, "CodecProbeBuilder requires a floating-point T");

public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    make(CodecProbe<T>& probe,
         const UniverseHostPtr<T>& universe,
         const FluidHostPtr<T>& fluid,
         const SpatialHashingSearcherHostPtr<T>& searcher,
         DeviceBuffer<int>& allocated_solver,
         const DeviceBuffer<int>& fixed_solver,
         const DeviceBuffer<int>& fixed_region) noexcept;
};

} // namespace atlas::detail

#include <atlas/codec/detail/codec_probe_builder.hpp>
