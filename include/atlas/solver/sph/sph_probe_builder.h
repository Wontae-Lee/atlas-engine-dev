#pragma once

/**
 * @file sph_probe_builder.h
 * @brief Declares helper routines that populate SphProbe.
 */

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/sph/sph_kernel.h>
#include <atlas/solver/sph/sph_probe.h>
#include <atlas/universe/universe.h>

#include <type_traits>

namespace atlas::detail {

/**
 * @brief Populates the device-side SPH probe from solver-owned state.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class SphProbeBuilder final {
    static_assert(std::is_floating_point_v<T>, "SphProbeBuilder requires a floating-point T");

public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    has_particle_states(const FluidHostPtr<T>& fluid) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    ready(const UniverseHostPtr<T>& universe,
          const FluidHostPtr<T>& fluid,
          const SearcherHostPtr<T>& searcher) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    make(SphProbe<T>& probe,
         const UniverseHostPtr<T>& universe,
         const FluidHostPtr<T>& fluid,
         const SearcherHostPtr<T>& searcher,
         const SphKernel<T>& kernel) noexcept;
};

} // namespace atlas::detail

#include <atlas/solver/sph/sph_probe_builder.hpp>
