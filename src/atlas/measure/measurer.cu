#include <atlas/measure/measurer.h>

#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/searcher/searcher.h>

#include <utility>

namespace atlas {

Measurer::Measurer(UniverseHostPtr universe,
                   FluidHostPtr fluid,
                   SearcherHostPtr searcher) noexcept
    : _universe(std::move(universe))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
}

void
Measurer::measure(const float) {
    measure();
}

bool
Measurer::make_probe() noexcept {
    _probe = {};

    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* fluid_temperature = _fluid->state<atlas::FluidTemperatureState>();

    _probe.field_temperature_ptr    = atlas::raw_pointer_cast(_universe->state<atlas::UniverseTemperatureState>()->data().data());
    _probe.bulk_velocity_ptr        = atlas::raw_pointer_cast(_universe->state<atlas::UniverseBulkVelocityState>()->data().data());
    _probe.thermal_energy_ptr       = atlas::raw_pointer_cast(_universe->state<atlas::UniverseThermalEnergyState>()->data().data());
    _probe.number_particle_ptr      = atlas::raw_pointer_cast(_universe->state<atlas::UniverseNumberParticleState>()->data().data());
    _probe.velocity_ptr             = atlas::raw_pointer_cast(_fluid->state<atlas::FluidVelocityState>()->data().data());
    _probe.particle_temperature_ptr = fluid_temperature != nullptr ? atlas::raw_pointer_cast(fluid_temperature->data().data()) : nullptr;
    _probe.indices_ptr    = _searcher->indices();
    _probe.cell_start_ptr = _searcher->cell_start();
    _probe.cell_end_ptr   = _searcher->cell_end();
    _probe.particle_count = static_cast<int>(_fluid->particle_count());
    _probe.cell_count     = _universe->cell_count();

    return _probe.particle_count >= 0
        && _probe.cell_count > 0
        && _probe.indices_ptr != nullptr
        && _probe.cell_start_ptr != nullptr
        && _probe.cell_end_ptr != nullptr;
}

}
