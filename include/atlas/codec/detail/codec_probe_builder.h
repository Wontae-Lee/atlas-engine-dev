#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::detail {

template <typename T>
class CodecProbeBuilder final {
    static_assert(std::is_floating_point_v<T>, "CodecProbeBuilder requires a floating-point T");

public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    make(CodecProbe<T>& probe,
         const UniverseHostPtr<T>& universe,
         const FluidHostPtr<T>& fluid,
         const SearcherHostPtr<T>& searcher,
         DeviceBuffer<int>& allocated_solver,
         const DeviceBuffer<int>& fixed_solver,
         const DeviceBuffer<int>& fixed_region) noexcept;
};

}

#include <atlas/codec/detail/codec_probe_builder.hpp>