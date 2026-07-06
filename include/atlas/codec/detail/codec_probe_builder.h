#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/codec/codec_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

/**
 * @file codec_probe_builder.h
 * @brief Host-only factory that assembles a `CodecProbe` from a
 *        `Codec`'s universe/fluid/searcher and allocation buffers.
 */

namespace atlas::detail {

/**
 * @brief Populates a `CodecProbe`'s raw-pointer/count fields from live
 *        universe/fluid/searcher state and a `Codec`'s allocation
 *        buffers.
 */
class CodecProbeBuilder final {
public:
    /**
     * @brief Fills `probe` in place. Each of `temperature_ptr`/
     *        `number_particle_ptr`/`knudsen_number_ptr` is left null if
     *        the corresponding universe state doesn't exist (a codec
     *        that doesn't need one simply sees it as absent, rather than
     *        the whole build failing); `allocated_solver_ptr`/
     *        `fixed_solver_ptr`/`fixed_region_ptr` are left null if
     *        their buffer is empty.
     * @return `false` (leaving `probe` zeroed) if `universe`/`fluid`/
     *         `searcher` is null; otherwise `true` iff
     *         `universe->cell_count() > 0`.
     */
    ATLAS_HOST ATLAS_NODISCARD static bool
    make(CodecProbe& probe,
         const UniverseHostPtr& universe,
         const FluidHostPtr& fluid,
         const SearcherHostPtr& searcher,
         DeviceBuffer<int>& allocated_solver,
         const DeviceBuffer<int>& fixed_solver,
         const DeviceBuffer<int>& fixed_region) noexcept;
};

}
