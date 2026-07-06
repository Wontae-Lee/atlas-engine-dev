#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sampling/sampling.h>
#include <atlas/source/detail/source_cache_builder.h>

namespace atlas::detail {

void
SourceCacheBuilder::rebuild(const DeviceBuffer<Unit>& units,
                            const DeviceBuffer<SpawnType>& spawn_types,
                            const DeviceBuffer<Spawn>& spawn_operators,
                            const FluidHostPtr& fluid,
                            const bool flip,
                            const float spacing,
                            const float tolerance,
                            HostBuffer<int>& local_unit_counts,
                            DeviceBuffer<Vector3>& flat_local_positions,
                            DeviceBuffer<int>& flat_unit_indices,
                            std::size_t& local_particle_count,
                            DeviceBuffer<std::size_t>& species_cache,
                            DeviceBuffer<std::size_t>& shuffled_species,
                            DeviceBuffer<std::uint64_t>& shuffle_keys,
                            std::uint64_t& shuffle_seed) const noexcept {
    if (units.empty() || spawn_types.empty() || spawn_operators.empty() || !fluid || fluid->generators().empty()) {
        clear(local_unit_counts,
              flat_local_positions,
              flat_unit_indices,
              local_particle_count,
              species_cache,
              shuffled_species,
              shuffle_keys,
              shuffle_seed);
        return;
    }

    const auto& generators = fluid->generators();

    const HostBuffer<Unit> host_units(units.begin(), units.end());
    const HostBuffer<Spawn> host_spawn_operators(
        spawn_operators.begin(),
        spawn_operators.end());

    local_unit_counts.clear();
    local_unit_counts.resize(host_units.size(), 0);

    HostBuffer<Vector3> host_flat_positions;
    HostBuffer<int> host_flat_unit_indices;

    const std::size_t spawn_operator_count = host_spawn_operators.size();

    for (std::size_t unit_index = 0; unit_index < host_units.size(); ++unit_index) {
        const auto& geometry = host_units[unit_index].geometry();
        const auto bounds    = geometry.bound();
        const auto& lower    = bounds.lower_corner;
        const auto& upper    = bounds.upper_corner;
        const auto& spawn_op = host_spawn_operators[(spawn_operator_count == 1) ? 0 : unit_index];

        const int nx = atlas::sample_axis_count(lower.x, upper.x, spacing);
        const int ny = atlas::sample_axis_count(lower.y, upper.y, spacing);
        const int nz = atlas::sample_axis_count(lower.z, upper.z, spacing);

        std::size_t accepted_count = 0;

        // Regular Cartesian lattice over the unit's local bounding box, one
        // candidate every `spacing` along each axis (not Poisson-disk or
        // other blue-noise sampling — density is uniform but not
        // randomized). Each candidate is tested against the unit's own
        // shape via spawn_op; `flip` inverts accept/reject (same
        // convention as Sink's flip).
        if (nx > 0 && ny > 0 && nz > 0) {
            for (int iz = 0; iz < nz; ++iz) {
                for (int iy = 0; iy < ny; ++iy) {
                    for (int ix = 0; ix < nx; ++ix) {
                        const Vector3 sample(
                            lower.x + static_cast<float>(ix) * spacing,
                            lower.y + static_cast<float>(iy) * spacing,
                            lower.z + static_cast<float>(iz) * spacing);

                        const bool accepted = spawn_op.spawn(geometry, sample, tolerance);

                        if (flip ? !accepted : accepted) {
                            host_flat_positions.push_back(sample);
                            host_flat_unit_indices.push_back(static_cast<int>(unit_index));
                            ++accepted_count;
                        }
                    }
                }
            }
        }

        local_unit_counts[unit_index] = static_cast<int>(accepted_count);
    }

    species_cache.clear();
    shuffled_species.clear();
    shuffle_keys.clear();

    const std::size_t total_count = host_flat_positions.size();
    local_particle_count          = total_count;

    if (total_count == 0) {
        flat_local_positions.clear();
        flat_unit_indices.clear();
        shuffle_seed = 0;
        return;
    }

    flat_local_positions = DeviceBuffer<Vector3>(host_flat_positions.begin(), host_flat_positions.end());
    flat_unit_indices    = DeviceBuffer<int>(host_flat_unit_indices.begin(), host_flat_unit_indices.end());

    species_cache.resize(total_count);
    shuffled_species.resize(total_count);
    shuffle_keys.resize(total_count);

    const int count         = static_cast<int>(total_count);
    const int species_count = static_cast<int>(generators.size());
    auto* cache             = atlas::raw_pointer_cast(species_cache.data());

    // Round-robin species assignment keeps species proportions balanced
    // across the candidate pool; SourceSpeciesShuffler later randomizes
    // *which* candidate gets which species per emission without changing
    // this even distribution.
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        count,
        [=] ATLAS_ALL_DEVICE(const int i) {
            cache[i] = static_cast<std::size_t>(i % species_count);
        });

    shuffle_seed = 0;
}

void
SourceCacheBuilder::clear(HostBuffer<int>& local_unit_counts,
                          DeviceBuffer<Vector3>& flat_local_positions,
                          DeviceBuffer<int>& flat_unit_indices,
                          std::size_t& local_particle_count,
                          DeviceBuffer<std::size_t>& species_cache,
                          DeviceBuffer<std::size_t>& shuffled_species,
                          DeviceBuffer<std::uint64_t>& shuffle_keys,
                          std::uint64_t& shuffle_seed) const noexcept {
    local_unit_counts.clear();
    flat_local_positions.clear();
    flat_unit_indices.clear();
    local_particle_count = 0;
    species_cache.clear();
    shuffled_species.clear();
    shuffle_keys.clear();
    shuffle_seed = 0;
}

}
