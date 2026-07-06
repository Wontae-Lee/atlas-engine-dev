#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/collider/collider_probe.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/unit/unit.h>

#include <cstdint>

/**
 * @file collider_probe_builder.h
 * @brief Host-only factory that assembles a `ColliderProbe` from a
 *        `Collider`'s buffers and its target `Fluid`.
 */

namespace atlas::detail {

/**
 * @brief Populates a `ColliderProbe`'s raw-pointer/count fields from live
 *        buffers, failing safely if the target fluid has no particles.
 *
 * Runs on the host once per `Collider::collide` (or whenever the caller's
 * buffers may have been reallocated) — see `collider_probe.h` for why the
 * probe cannot be built once and cached across steps.
 */
class ColliderProbeBuilder final {
public:
    using Bound = atlas::AABB;

    /**
     * @brief Fills `probe` in place from `units`/`unit_bounds`/
     *        `surface_interactions`/`flips`/`scene_bound` and the
     *        position/velocity/species/internal-energy/material buffers
     *        owned by `fluid`.
     *
     * @return `false` (leaving `probe` zeroed) if there is nothing to
     *         collide against or with — `units`/`surface_interactions`
     *         empty, `fluid` null, or `fluid` currently holding zero
     *         particles — so callers can skip launching the collision
     *         kernel entirely; `true` otherwise. `internal_energies` /
     *         `species` / `materials` / `material_count` are populated
     *         only when the fluid tracks internal energy AND has a
     *         non-empty material table; they stay null/zero (rather than
     *         failing the whole build) when only translational collision
     *         response is available.
     */
    ATLAS_NODISCARD ATLAS_HOST static bool
    make(ColliderProbe& probe,
         const DeviceBuffer<Unit>& units,
         const DeviceBuffer<Bound>& unit_bounds,
         const DeviceBuffer<SurfaceInteractionKernel>& surface_interactions,
         const DeviceBuffer<std::uint8_t>& flips,
         const Bound& scene_bound,
         bool scene_bound_covers_units,
         const FluidHostPtr& fluid) noexcept;
};

}
