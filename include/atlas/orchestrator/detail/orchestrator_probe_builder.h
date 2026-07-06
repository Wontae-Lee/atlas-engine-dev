#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe.h>

namespace atlas::detail {

class OrchestratorProbeBuilder final {
public:
    ATLAS_HOST ATLAS_NODISCARD static bool
    make(const UniverseHostPtr& universe,
         const FluidHostPtr& fluid,
         const SearcherHostPtr& searcher,
         OrchestratorProbe& probe) noexcept;

private:
    ATLAS_HOST ATLAS_NODISCARD static bool
    load_common_data(const UniverseHostPtr& universe,
                     const FluidHostPtr& fluid,
                     const SearcherHostPtr& searcher,
                     OrchestratorProbe& probe) noexcept;

    ATLAS_HOST static void
    load_species_data(const FluidHostPtr& fluid, OrchestratorProbe& probe) noexcept;

    ATLAS_HOST static void
    load_field_force_data(const UniverseHostPtr& universe, OrchestratorProbe& probe) noexcept;

    ATLAS_HOST static void
    load_gravity_data(const UniverseHostPtr& universe, OrchestratorProbe& probe) noexcept;
};

}
