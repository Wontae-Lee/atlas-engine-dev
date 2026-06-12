#pragma once

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/orchestrator/orchestrator_probe.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

/**
 * @brief Builds device-readable orchestration probes from runtime owners.
 */
template <typename T>
class OrchestratorProbeBuilder final {
public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    build(const UniverseHostPtr<T>& universe,
          const FluidHostPtr<T>& fluid,
          const SpatialHashingSearcherHostPtr<T>& searcher,
          OrchestratorProbe<T>& probe) const noexcept;

private:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    load_common_data(const UniverseHostPtr<T>& universe,
                     const FluidHostPtr<T>& fluid,
                     const SpatialHashingSearcherHostPtr<T>& searcher,
                     OrchestratorProbe<T>& probe) const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    load_species_data(const FluidHostPtr<T>& fluid, OrchestratorProbe<T>& probe) const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    load_field_force_data(const UniverseHostPtr<T>& universe, OrchestratorProbe<T>& probe) const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    load_gravity_data(const UniverseHostPtr<T>& universe, OrchestratorProbe<T>& probe) const noexcept;
};

} // namespace atlas::detail

#include <atlas/orchestrator/detail/orchestrator_probe_builder.hpp>
