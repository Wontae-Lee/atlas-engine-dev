#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>
#include <atlas/universe/universe.h>

/**
 * @file sph_probe_builder.h
 * @brief Host-only factory that assembles an `SphProbe` from an
 *        `SphSolver`'s universe/fluid/searcher, mirroring
 *        `detail::DsmcProbeBuilder`'s role for `DsmcSolver`.
 */

namespace atlas::detail {

/**
 * @brief Validates and populates an `SphProbe`'s raw-pointer/count
 *        fields from live universe/fluid/searcher state, including the
 *        searcher-built neighbor list SPH needs (see `sph_probe.h`).
 */
class SphProbeBuilder final {
public:
    /** @brief Whether `fluid` has `FluidPositionState`,
     *  `FluidVelocityState`, and `FluidSpeciesState` — the minimal
     *  particle state SPH needs before anything else is checked. */
    ATLAS_HOST ATLAS_NODISCARD static bool
    has_particle_states(const FluidHostPtr& fluid) noexcept;

    /** @brief Whether `universe`/`fluid`/`searcher` are non-null,
     *  `has_particle_states(fluid)` holds, and the required per-cell
     *  universe states exist — the precondition `make()` checks. */
    ATLAS_HOST ATLAS_NODISCARD static bool
    ready(const UniverseHostPtr& universe,
          const FluidHostPtr& fluid,
          const SearcherHostPtr& searcher) noexcept;

    /**
     * @brief Fills `probe` in place from `universe`'s grid, `fluid`'s
     *        particle state, `searcher`'s cell partition and
     *        precomputed neighbor list, and `kernel` (the SPH kernel to
     *        install into the probe).
     * @return `true` iff `ready()` holds; `false` (leaving `probe`
     *         zeroed) otherwise.
     */
    ATLAS_HOST ATLAS_NODISCARD static bool
    make(SphProbe& probe,
         const UniverseHostPtr& universe,
         const FluidHostPtr& fluid,
         const SearcherHostPtr& searcher,
         const SphKernel& kernel) noexcept;
};

}
