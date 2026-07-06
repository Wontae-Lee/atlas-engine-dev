#include <atlas/codec/detail/codec_probe_builder.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/probe/probe_common.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>

namespace atlas::detail {

bool
CodecProbeBuilder::make(CodecProbe& probe,
                        const UniverseHostPtr& universe,
                        const FluidHostPtr& fluid,
                        const SearcherHostPtr& searcher,
                        DeviceBuffer<int>& allocated_solver,
                        const DeviceBuffer<int>& fixed_solver,
                        const DeviceBuffer<int>& fixed_region) noexcept {
    probe = {};

    if (!universe || !fluid || !searcher) {
        return false;
    }

    auto* temperature_state     = universe->state<UniverseTemperatureState>();
    auto* number_particle_state = universe->state<UniverseNumberParticleState>();
    auto* knudsen_number_state  = universe->state<UniverseKnudsenNumberState>();

    probe.temperature_ptr      = optional_state_ptr(temperature_state);
    probe.number_particle_ptr  = optional_state_ptr(number_particle_state);
    probe.knudsen_number_ptr   = optional_state_ptr(knudsen_number_state);
    probe.allocated_solver_ptr = optional_buffer_ptr(allocated_solver);
    probe.fixed_solver_ptr     = optional_buffer_ptr(fixed_solver);
    probe.fixed_region_ptr     = optional_buffer_ptr(fixed_region);
    fill_cell_partition(probe, searcher);
    probe.particle_count       = static_cast<int>(fluid->particle_count());
    probe.cell_count         = universe->cell_count();
    probe.cell_volume          = universe->cell_volume();
    probe.statistical_weight   = fluid->statistical_weight();

    return probe.cell_count > 0;
}

}
