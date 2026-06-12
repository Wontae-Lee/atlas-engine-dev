#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/buffer/host_buffer.h>
#include <atlas/core/macros.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/source/spawn_operator.h>
#include <atlas/unit/unit.h>

#include <cstddef>
#include <cstdint>

namespace atlas::detail {

/**
 * @brief Rebuilds cached local source samples and base species assignment.
 */
template <typename T>
class SourceCacheBuilder final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    rebuild(const DeviceBuffer<Unit<T>>& units,
            const DeviceBuffer<SpawnType>& spawn_types,
            const DeviceBuffer<SpawnOperator<T>>& spawn_operators,
            const FluidHostPtr<T>& fluid,
            bool flip,
            T spacing,
            T tolerance,
            HostBuffer<int>& local_unit_counts,
            DeviceBuffer<Vector3<T>>& flat_local_positions,
            DeviceBuffer<int>& flat_unit_indices,
            std::size_t& local_particle_count,
            DeviceBuffer<std::size_t>& species_cache,
            DeviceBuffer<std::size_t>& shuffled_species,
            DeviceBuffer<std::uint64_t>& shuffle_keys,
            std::uint64_t& shuffle_seed) const noexcept;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear(HostBuffer<int>& local_unit_counts,
          DeviceBuffer<Vector3<T>>& flat_local_positions,
          DeviceBuffer<int>& flat_unit_indices,
          std::size_t& local_particle_count,
          DeviceBuffer<std::size_t>& species_cache,
          DeviceBuffer<std::size_t>& shuffled_species,
          DeviceBuffer<std::uint64_t>& shuffle_keys,
          std::uint64_t& shuffle_seed) const noexcept;
};

} // namespace atlas::detail

#include <atlas/source/detail/source_cache_builder.hpp>
