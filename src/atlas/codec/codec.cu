#include <atlas/codec/codec.h>

#include <atlas/logging/logging.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/searcher/searcher.h>
#include <atlas/universe/universe_state.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace atlas {

Codec::Codec(UniverseHostPtr domain,
             FluidHostPtr fluid,
             SearcherHostPtr searcher)
    : _universe(std::move(domain))
    , _fluid(std::move(fluid))
    , _searcher(std::move(searcher)) {
    atlas::check<std::invalid_argument>(static_cast<bool>(_universe))
        << "Codec: universe must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_fluid))
        << "Codec: fluid must not be null.";
    atlas::check<std::invalid_argument>(static_cast<bool>(_searcher))
        << "Codec: searcher must not be null.";

    reset();
}

void
Codec::update() {
    encode();
    decode();
}

void
Codec::reset() noexcept {
    const auto cell_count = _universe->cell_count();
    d_allocated_solver.resize(cell_count, 0);
    d_fixed_solver.resize(cell_count, 0);
    d_fixed_region.resize(cell_count, 0);
}

DeviceBuffer<int>&
Codec::allocated_solver() noexcept {
    return d_allocated_solver;
}

const DeviceBuffer<int>&
Codec::allocated_solver() const noexcept {
    return d_allocated_solver;
}

void
Codec::set_fixed_solver(DeviceBuffer<int> fixed_solver) {
    if (_universe && !fixed_solver.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_solver.size() == static_cast<std::size_t>(_universe->cell_count()))
            << "Codec: fixed_solver size must match universe cell count.";
    }
    d_fixed_solver = std::move(fixed_solver);
}

DeviceBuffer<int>&
Codec::fixed_solver() noexcept {
    return d_fixed_solver;
}

const DeviceBuffer<int>&
Codec::fixed_solver() const noexcept {
    return d_fixed_solver;
}

void
Codec::set_fixed_region(DeviceBuffer<int> fixed_region) {
    if (_universe && !fixed_region.empty()) {
        atlas::check<std::invalid_argument>(
            fixed_region.size() == static_cast<std::size_t>(_universe->cell_count()))
            << "Codec: fixed_region size must match universe cell count.";
    }
    d_fixed_region = std::move(fixed_region);
}

DeviceBuffer<int>&
Codec::fixed_region() noexcept {
    return d_fixed_region;
}

const DeviceBuffer<int>&
Codec::fixed_region() const noexcept {
    return d_fixed_region;
}

bool
Codec::make_probe() noexcept {
    _probe = {};

    if (!_universe || !_fluid || !_searcher) {
        return false;
    }

    auto* temperature_state     = _universe->state<UniverseTemperatureState>();
    auto* number_particle_state = _universe->state<UniverseNumberParticleState>();
    auto* knudsen_number_state  = _universe->state<UniverseKnudsenNumberState>();

    _probe.temperature_ptr      = temperature_state != nullptr ? atlas::raw_pointer_cast(temperature_state->data().data()) : nullptr;
    _probe.number_particle_ptr  = number_particle_state != nullptr ? atlas::raw_pointer_cast(number_particle_state->data().data()) : nullptr;
    _probe.knudsen_number_ptr   = knudsen_number_state != nullptr ? atlas::raw_pointer_cast(knudsen_number_state->data().data()) : nullptr;
    _probe.allocated_solver_ptr = d_allocated_solver.empty() ? nullptr : atlas::raw_pointer_cast(d_allocated_solver.data());
    _probe.fixed_solver_ptr     = d_fixed_solver.empty() ? nullptr : atlas::raw_pointer_cast(d_fixed_solver.data());
    _probe.fixed_region_ptr     = d_fixed_region.empty() ? nullptr : atlas::raw_pointer_cast(d_fixed_region.data());
    _probe.indices_ptr          = _searcher->indices();
    _probe.cell_start_ptr       = _searcher->cell_start();
    _probe.cell_end_ptr         = _searcher->cell_end();
    _probe.particle_count     = static_cast<int>(_fluid->particle_count());
    _probe.cell_count         = _universe->cell_count();
    _probe.cell_volume        = _universe->cell_volume();
    _probe.statistical_weight = _fluid->statistical_weight();

    return _probe.cell_count > 0;
}

}
