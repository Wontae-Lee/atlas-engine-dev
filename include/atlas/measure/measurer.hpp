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
Measurer<T>::make_probe() noexcept {
    _probe = {};
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
    auto* fluid_velocity    = _fluid->template state<atlas::fluid::FluidVelocityState<T>>();
    auto* fluid_temperature = _fluid->template state<atlas::fluid::FluidTemperatureState<T>>();

    if (universe_temperature == nullptr || universe_bulk_velocity == nullptr
        || universe_thermal_energy == nullptr || universe_number_particle == nullptr
        || fluid_velocity == nullptr) {
        return false;
    }

    _probe.field_temperature_ptr    = atlas::raw_pointer_cast(universe_temperature->data().data());
    _probe.bulk_velocity_ptr        = atlas::raw_pointer_cast(universe_bulk_velocity->data().data());
    _probe.thermal_energy_ptr       = atlas::raw_pointer_cast(universe_thermal_energy->data().data());
    _probe.number_particle_ptr      = atlas::raw_pointer_cast(universe_number_particle->data().data());
    _probe.velocity_ptr             = atlas::raw_pointer_cast(fluid_velocity->data().data());
    _probe.particle_temperature_ptr = fluid_temperature != nullptr
        ? atlas::raw_pointer_cast(fluid_temperature->data().data())
        : nullptr;
    _probe.indices_ptr              = _searcher->indices();
    _probe.cell_start_ptr           = _searcher->cell_start();
    _probe.cell_end_ptr             = _searcher->cell_end();
    _probe.particle_count           = static_cast<int>(_fluid->particle_count());
    _probe.num_of_cells             = _universe->number_of_cells();

    return _probe.particle_count >= 0
        && _probe.num_of_cells > 0
        && _probe.indices_ptr != nullptr
        && _probe.cell_start_ptr != nullptr
        && _probe.cell_end_ptr != nullptr;
}

}
