#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/universe/universe.h>

#include <cstdint>

/**
 * @file dsmc_probe_builder.h
 * @brief Host-only factory that assembles a `DsmcProbe` from a
 *        `DsmcSolver`'s universe/fluid/searcher, mirroring
 *        `detail::ColliderProbeBuilder`'s role for `Collider`.
 */

namespace atlas::detail {

/**
 * @brief Validates and populates a `DsmcProbe`'s raw-pointer/count
 *        fields from live universe/fluid/searcher state.
 */
class DsmcProbeBuilder final {
public:
    /** @brief Whether `universe`/`fluid`/`searcher` are non-null and
     *  every NTC bookkeeping state (`UniverseNumberParticleState`,
     *  `UniverseMaxRelativeSpeedState`, `UniverseMaxSigmaGState`,
     *  `UniverseCollisionRemainderState`, `UniverseCollisionCountState`)
     *  plus `FluidVelocityState`/`FluidSpeciesState` already exist —
     *  the precondition `make()` checks before touching any buffer. */
    ATLAS_HOST ATLAS_NODISCARD static bool
    ready(const UniverseHostPtr& universe,
          const FluidHostPtr& fluid,
          const SearcherHostPtr& searcher) noexcept;

    /**
     * @brief Fills `probe` in place from `universe`'s cell partition
     *        (searcher-built `indices`/`cell_start`/`cell_end`),
     *        `fluid`'s particle state, and `kernel` (the collision
     *        model to install into the probe).
     * @return `true` iff `ready()` holds (see that method for exactly
     *         which states are required); `false` (leaving `probe`
     *         zeroed) otherwise. Unlike `ColliderProbeBuilder::make`,
     *         this does not itself gate on the particle count being
     *         positive — an empty fluid still yields a valid, zero-work
     *         probe.
     */
    ATLAS_HOST ATLAS_NODISCARD static bool
    make(DsmcProbe& probe,
         const UniverseHostPtr& universe,
         const FluidHostPtr& fluid,
         const SearcherHostPtr& searcher,
         const DsmcKernel& kernel,
         std::uint64_t collision_seed) noexcept;
};

}
