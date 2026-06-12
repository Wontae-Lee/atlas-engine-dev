#pragma once

#include <atlas/memory/raw_pointer_cast.h>

namespace atlas {

template <typename T>
Measurer<T>::Measurer(UniverseHostPtr<T> universe,
                      FluidHostPtr<T> fluid,
                      SpatialHashingSearcherHostPtr<T> searcher) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
    // Store the shared simulation dependencies used to construct measurement probes.
}

template <typename T>
void
Measurer<T>::measure(const T) {
    measure();
}

template <typename T>
bool
Measurer<T>::make_probe() noexcept {
    _probe = {};

    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* fluid_temperature = _fluid->template state<atlas::FluidTemperatureState<T>>();

    _probe.field_temperature_ptr = atlas::raw_pointer_cast(_universe->template state<atlas::UniverseTemperatureState<T>>()->data().data());
    _probe.bulk_velocity_ptr     = atlas::raw_pointer_cast(_universe->template state<atlas::UniverseBulkVelocityState<T>>()->data().data());
    _probe.thermal_energy_ptr    = atlas::raw_pointer_cast(_universe->template state<atlas::UniverseThermalEnergyState<T>>()->data().data());
    _probe.number_particle_ptr   = atlas::raw_pointer_cast(_universe->template state<atlas::UniverseNumberParticleState<T>>()->data().data());
    _probe.velocity_ptr          = atlas::raw_pointer_cast(_fluid->template state<atlas::FluidVelocityState<T>>()->data().data());
    _probe.particle_temperature_ptr = fluid_temperature != nullptr
        ? atlas::raw_pointer_cast(fluid_temperature->data().data())
        : nullptr;
    _probe.indices_ptr    = _searcher->indices();
    _probe.cell_start_ptr = _searcher->cell_start();
    _probe.cell_end_ptr   = _searcher->cell_end();
    _probe.particle_count = static_cast<int>(_fluid->particle_count());
    _probe.num_of_cells   = _universe->number_of_cells();

    return _probe.particle_count >= 0
        && _probe.num_of_cells > 0
        && _probe.indices_ptr != nullptr
        && _probe.cell_start_ptr != nullptr
        && _probe.cell_end_ptr != nullptr;
}

} // namespace atlas
