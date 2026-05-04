#pragma once
#include <atlas/memory/raw_pointer_cast.h>
namespace atlas::system {
template <typename T>
Measurer<T>::Measurer(UniverseHostPtr<T> universe,
                      FluidHostPtr<T> fluid,
                      SpatialHashingSearcherHostPtr<T> searcher) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
}

template <typename T>
bool
Measurer<T>::make_probe(MeasurerProbe& probe) noexcept {
    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* universe_temperature
        = _universe->template state<atlas::universe::UniverseTemperatureState<T>>();
    auto* universe_bulk_velocity
        = _universe->template state<atlas::universe::UniverseBulkVelocityState<T>>();
    auto* universe_thermal_energy
        = _universe->template state<atlas::universe::UniverseThermalEnergyState<T>>();
    auto* universe_number_particle
        = _universe->template state<atlas::universe::UniverseNumberParticleState<T>>();
    auto* fluid_velocity = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* fluid_temperature = _fluid->template state<atlas::fluid::FluidTemperatureState<T>>();

    if (universe_temperature == nullptr || universe_bulk_velocity == nullptr
        || universe_thermal_energy == nullptr || universe_number_particle == nullptr
        || fluid_velocity == nullptr) {
        return false;
    }

    probe.field_temperature_ptr = atlas::raw_pointer_cast(universe_temperature->data().data());
    probe.bulk_velocity_ptr = atlas::raw_pointer_cast(universe_bulk_velocity->data().data());
    probe.thermal_energy_ptr = atlas::raw_pointer_cast(universe_thermal_energy->data().data());
    probe.number_particle_ptr = atlas::raw_pointer_cast(universe_number_particle->data().data());
    probe.velocity_ptr = atlas::raw_pointer_cast(fluid_velocity->data().data());
    probe.particle_temperature_ptr = fluid_temperature != nullptr
        ? atlas::raw_pointer_cast(fluid_temperature->data().data())
        : nullptr;
    probe.indices_ptr = _searcher->indices();
    probe.cell_start_ptr = _searcher->cell_start();
    probe.cell_end_ptr = _searcher->cell_end();
    probe.particle_count = static_cast<int>(_fluid->particle_count());
    probe.num_of_cells = _universe->number_of_cells();

    return probe.particle_count >= 0
        && probe.num_of_cells > 0
        && probe.indices_ptr != nullptr
        && probe.cell_start_ptr != nullptr
        && probe.cell_end_ptr != nullptr;
}

} 
