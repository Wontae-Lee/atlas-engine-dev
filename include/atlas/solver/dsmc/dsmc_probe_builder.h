#pragma once

/**
 * @file dsmc_probe_builder.h
 * @brief Declares helper routines that populate DsmcProbe.
 */

#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/searcher/searcher.h>
#include <atlas/solver/dsmc/dsmc_kernel.h>
#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/universe/universe.h>

#include <cstdint>
#include <type_traits>

namespace atlas::detail {

/**
 * @brief Populates the device-side DSMC probe from solver-owned state.
 *
 * @tparam T Floating-point scalar type used by the solver.
 */
template <typename T>
class DsmcProbeBuilder final {
    static_assert(std::is_floating_point_v<T>, "DsmcProbeBuilder requires a floating-point T");

public:
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    ready(const UniverseHostPtr<T>& universe,
          const FluidHostPtr<T>& fluid,
          const SearcherHostPtr<T>& searcher) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    make(DsmcProbe<T>& probe,
         const UniverseHostPtr<T>& universe,
         const FluidHostPtr<T>& fluid,
         const SearcherHostPtr<T>& searcher,
         const DsmcKernel<T>& kernel,
         std::uint64_t collision_seed) noexcept;
};

} // namespace atlas::detail

#include <atlas/solver/dsmc/dsmc_probe_builder.hpp>
