#pragma once

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/universe/universe_state.h>

namespace atlas::detail {

template <typename T>
bool
CodecProbeBuilder<T>::make(CodecProbe<T>& probe,
                           const UniverseHostPtr<T>& universe,
                           const FluidHostPtr<T>& fluid,
                           const SpatialHashingSearcherHostPtr<T>& searcher,
                           DeviceBuffer<int>& allocated_solver,
                           const DeviceBuffer<int>& fixed_solver,
                           const DeviceBuffer<int>& fixed_region) noexcept {
    probe = {};

    if (!universe || !fluid || !searcher) {
        return false;
    }

    auto* temperature_state     = universe->template state<UniverseTemperatureState<T>>();
    auto* number_particle_state = universe->template state<UniverseNumberParticleState<T>>();
    auto* knudsen_number_state  = universe->template state<UniverseKnudsenNumberState<T>>();

    probe.temperature_ptr      = temperature_state != nullptr
             ? atlas::raw_pointer_cast(temperature_state->data().data())
             : nullptr;
    probe.number_particle_ptr  = number_particle_state != nullptr
         ? atlas::raw_pointer_cast(number_particle_state->data().data())
         : nullptr;
    probe.knudsen_number_ptr   = knudsen_number_state != nullptr
          ? atlas::raw_pointer_cast(knudsen_number_state->data().data())
          : nullptr;
    probe.allocated_solver_ptr = allocated_solver.empty()
        ? nullptr
        : atlas::raw_pointer_cast(allocated_solver.data());
    probe.fixed_solver_ptr     = fixed_solver.empty()
            ? nullptr
            : atlas::raw_pointer_cast(fixed_solver.data());
    probe.fixed_region_ptr     = fixed_region.empty()
            ? nullptr
            : atlas::raw_pointer_cast(fixed_region.data());
    probe.indices_ptr          = searcher->indices();
    probe.cell_start_ptr       = searcher->cell_start();
    probe.cell_end_ptr         = searcher->cell_end();
    probe.particle_count       = static_cast<int>(fluid->particle_count());
    probe.num_of_cells         = universe->number_of_cells();
    probe.cell_volume          = universe->cell_volume();
    probe.statistical_weight   = fluid->statistical_weight();

    return probe.num_of_cells > 0;
}

}